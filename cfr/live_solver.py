"""
Live subgame solver that runs MCCFR in a background thread while
the human player is thinking.

Node-locked continuation of the blueprint solver:
  - Hero's hand is fixed (known)
  - Opponent ranges are Bayesian-weighted from action history
  - Same MCCFR traversal and info set keys as the trainer
  - Warm-starts from blueprint with high warmth for stability
  - Runs continuously until the player acts
  - Strategies converge toward (and refine) the blueprint
"""

import random
import threading
import traceback

import numpy as np

from cfr.game_state import GameState, get_position_name, lookup_with_fallback
from cfr.information_set import InfoSet, sample_action
from cfr.preflop_ranges import (
    get_preflop_action, classify_scenario,
)


class LiveSolver:
    def __init__(self, blueprint_strategy, card_abstraction,
                 batch_size=100):
        self.blueprint = blueprint_strategy
        self.ca = card_abstraction
        self.batch_size = batch_size

        # Solving state
        self._thread = None
        self._stop_event = threading.Event()
        self._lock = threading.RLock()

        # Results
        self._info_sets = {}
        self._iterations_done = 0
        self._current_key = None
        self._solved_cache = {}
        self._current_actions = None
        self._solving_state = None
        self._preflop_probs = None

    def start_solving(self, state, player_seat, player_stats=None):
        """Begin background solving for the player's current decision.

        Preflop: returns static preflop range probabilities directly.
        Postflop: runs node-locked MCCFR in a background thread.
        """
        self.stop()

        stats = player_stats or {}
        actions = state.get_actions()

        # Preflop: use static ranges directly
        if state.street == 0:
            probs = self._preflop_heuristic_probs(state, player_seat, actions)
            with self._lock:
                self._info_sets = {}
                self._iterations_done = 0
                self._solving_state = state
                self._current_key = '__preflop__'
                self._current_actions = actions
                self._preflop_probs = probs or [1.0 / len(actions)] * len(actions)
            return

        # Postflop: run MCCFR in background
        with self._lock:
            self._info_sets = {}
            self._iterations_done = 0
            self._solving_state = state
            self._preflop_probs = None

            hand = state.hands[player_seat]
            vis = state.visible_board()
            bucket = self.ca.get_bucket(hand, vis)
            self._current_key = state.get_info_set_key(player_seat, bucket)
            self._current_actions = actions

        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._solve_loop,
            args=(state, player_seat, stats),
            daemon=True,
        )
        self._thread.start()

    def stop(self):
        """Stop the background solver and store results."""
        self._stop_event.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2.0)
        self._thread = None
        self._preflop_probs = None
        self._store_results()

    def get_recommendation(self):
        """Get the current best recommendation."""
        with self._lock:
            if not self._current_key or not self._current_actions:
                return None

            actions = self._current_actions

            # Preflop: return static range probs directly
            if self._preflop_probs is not None:
                return {
                    'actions': list(actions),
                    'probs': self._preflop_probs,
                    'iterations': 0,
                    'converged': True,
                }

            # Postflop: return MCCFR result
            iters = self._iterations_done

            if self._current_key in self._info_sets:
                iset = self._info_sets[self._current_key]
                probs = iset.get_average_strategy().tolist()
            else:
                probs = self._lookup_blueprint(self._current_key, len(actions))

            return {
                'actions': list(actions),
                'probs': probs,
                'iterations': iters,
                'converged': False,
            }

    def get_strategy_for_key(self, info_key, n_actions):
        """Get strategy for an info key from in-progress solver, or None."""
        with self._lock:
            if info_key in self._info_sets:
                iset = self._info_sets[info_key]
                total = sum(iset.strategy_sum)
                if total > 0:
                    return [s / total for s in iset.strategy_sum]
        return None

    @property
    def is_solving(self):
        return self._thread is not None and self._thread.is_alive()

    # ------------------------------------------------------------------
    # Strategy lookup & storage
    # ------------------------------------------------------------------

    def _lookup_blueprint(self, key, n_actions):
        """Look up a key in the solved cache then blueprint strategy."""
        with self._lock:
            if key in self._solved_cache:
                probs = self._solved_cache[key]
                if len(probs) == n_actions:
                    return list(probs)
        num_players = self._solving_state.num_players if self._solving_state else 6
        bp_probs = lookup_with_fallback(self.blueprint, key, num_players)
        if bp_probs is not None and len(bp_probs) == n_actions:
            return list(bp_probs)
        return [1.0 / n_actions] * n_actions

    def _store_results(self):
        """Store solved info sets in a separate cache (thread-safe)."""
        from config import LIVE_SOLVER_MERGE_THRESHOLD
        with self._lock:
            if not self._info_sets or self._iterations_done < LIVE_SOLVER_MERGE_THRESHOLD:
                return
            solved = {}
            for key, iset in self._info_sets.items():
                total = sum(iset.strategy_sum)
                if total > 0:
                    solved[key] = [s / total for s in iset.strategy_sum]
        if solved:
            with self._lock:
                self._solved_cache.update(solved)

    def lookup_solved(self, info_key, num_players):
        """Look up a key in solved cache, then blueprint."""
        with self._lock:
            if info_key in self._solved_cache:
                return self._solved_cache[info_key]
        return lookup_with_fallback(self.blueprint, info_key, num_players)

    # ------------------------------------------------------------------
    # Background solve loop
    # ------------------------------------------------------------------

    def _solve_loop(self, state, player_seat, player_stats):
        """Run MCCFR iterations until stopped."""
        try:
            self._solve_loop_inner(state, player_seat, player_stats)
        except Exception as e:
            print(f'[LiveSolver] Solve loop crashed: {e}')
            traceback.print_exc()

    def _solve_loop_inner(self, state, player_seat, player_stats):
        num_players = state.num_players
        hero_hand = state.hands[player_seat]

        # Active seats (not folded)
        active_seats = [i for i in range(num_players) if not state.folded[i]]

        # Build Bayesian opponent ranges (same as SubgameSolver)
        opponent_ranges = {}
        for opp in active_seats:
            if opp == player_seat:
                continue
            opponent_ranges[opp] = self._compute_opponent_range(
                hero_hand, state.board, state.history,
                player_seat, opp, state)

        if not opponent_ranges:
            print('[LiveSolver] No opponent ranges, aborting')
            return

        for opp, rng in opponent_ranges.items():
            if not rng:
                print(f'[LiveSolver] Empty range for opponent seat {opp}, aborting')
                return

        # Precompute numpy arrays for fast sampling
        opp_arrays = {}
        for opp, weights in opponent_ranges.items():
            hands = list(weights.keys())
            probs = np.array(list(weights.values()))
            opp_arrays[opp] = (hands, probs)

        iteration = 0
        while not self._stop_event.is_set():
            for _ in range(self.batch_size):
                if self._stop_event.is_set():
                    break

                # Sample opponent hands (hero hand is fixed)
                sampled = self._sample_opponent_hands(
                    hero_hand, opp_arrays, state.board)
                if sampled is None:
                    continue

                # Build hands array with hero fixed
                hands = list(state.hands)
                for opp_seat, opp_hand in sampled.items():
                    hands[opp_seat] = opp_hand

                try:
                    sub_state = GameState(
                        num_players=num_players,
                        hands=tuple(hands), board=state.board,
                        stacks=state.stacks, pot=state.pot, bets=state.bets,
                        current_player=state.current_player,
                        history=state.history, street=state.street,
                        raises_this_street=state.raises_this_street,
                        is_terminal=False, terminal_type=None,
                        min_raise=state.min_raise,
                        folded=state.folded, all_in=state.all_in,
                        has_acted=state.has_acted, last_raiser=state.last_raiser,
                    )

                    cache = self._bucket_cache(
                        tuple(hands), state.board, state.street, num_players)

                    # Run CFR outside lock to avoid blocking get_recommendation()
                    local_sets = {}
                    with self._lock:
                        local_sets = dict(self._info_sets)

                    for tp in active_seats:
                        self._cfr(sub_state, tp, cache,
                                  local_sets, iteration)

                    with self._lock:
                        self._info_sets.update(local_sets)
                except Exception:
                    continue

                iteration += 1

            with self._lock:
                self._iterations_done = iteration

        print(f'[LiveSolver] Stopped: {iteration} iterations')

    # ------------------------------------------------------------------
    # Bayesian opponent range (same algorithm as SubgameSolver)
    # ------------------------------------------------------------------

    def _compute_opponent_range(self, hero_hand, board, history,
                                hero_seat, opp_seat, state):
        """Weight each possible opponent hand by blueprint probability
        of the observed actions."""
        used = set(hero_hand) | set(board)

        # Start with uniform range over non-conflicting hands
        avail = [c for c in range(52) if c not in used]
        weights = {}
        for i in range(len(avail)):
            for j in range(i + 1, len(avail)):
                weights[(avail[i], avail[j])] = 1.0

        # Replay action history, weighting by blueprint action probabilities
        dummy_hands = list(state.hands)
        dummy_hands[opp_seat] = (avail[0], avail[1])
        try:
            replay = GameState.new_hand(tuple(dummy_hands), board)
        except Exception:
            return weights

        for street_actions in history:
            for action in street_actions:
                if replay.is_terminal:
                    break

                if replay.current_player == opp_seat:
                    actions = replay.get_actions()
                    if action not in actions:
                        break
                    act_idx = actions.index(action)
                    vis = replay.visible_board()

                    for hand in list(weights):
                        if weights[hand] < 1e-12:
                            continue

                        if replay.street == 0:
                            # Preflop: use preflop ranges
                            from cfr.preflop_solver import hand_to_class
                            hand_key = hand_to_class(hand)
                            pos = get_position_name(
                                state.num_players, opp_seat)
                            scenario = classify_scenario(
                                replay.history, pos, state.num_players)
                            action_probs = get_preflop_action(
                                hand_key, pos, scenario, actions,
                                num_players=state.num_players)
                            prob = action_probs.get(action, 0.0)
                            weights[hand] *= max(prob, 1e-6)
                        else:
                            # Postflop: use blueprint
                            bucket = self.ca.get_bucket(hand, vis)
                            key = replay.get_info_set_key(opp_seat, bucket)
                            if key in self.blueprint:
                                probs = self.blueprint[key]
                                if len(probs) == len(actions):
                                    weights[hand] *= max(
                                        probs[act_idx], 1e-6)
                                else:
                                    weights[hand] *= 1.0 / len(actions)
                            else:
                                weights[hand] *= 1.0 / len(actions)

                replay = replay.apply_action(action)

        # Prune and normalize
        weights = {h: w for h, w in weights.items() if w > 1e-12}
        total = sum(weights.values())
        if total <= 0:
            return {}
        return {h: w / total for h, w in weights.items()}

    # ------------------------------------------------------------------
    # Sampling (hero hand fixed, sample opponents only)
    # ------------------------------------------------------------------

    def _sample_opponent_hands(self, hero_hand, opp_arrays, board):
        """Sample one hand per opponent, rejecting card conflicts."""
        used = set(hero_hand) | set(board)
        sampled = {}

        for opp_seat, (hands, probs) in opp_arrays.items():
            for _ in range(50):
                idx = np.random.choice(len(hands), p=probs)
                hand = hands[idx]
                if hand[0] not in used and hand[1] not in used:
                    sampled[opp_seat] = hand
                    used.add(hand[0])
                    used.add(hand[1])
                    break
            else:
                return None

        return sampled

    # ------------------------------------------------------------------
    # MCCFR (same traversal as SubgameSolver / CFRTrainer)
    # ------------------------------------------------------------------

    def _get_or_create_info_set(self, info_key, n_actions, info_sets):
        """Get or create an info set, warm-starting from blueprint."""
        if info_key in info_sets:
            return info_sets[info_key]

        iset = InfoSet(n_actions)

        # Warm-start from blueprint: low regret warmth for fast adaptation,
        # high strategy_sum warmth to anchor average strategy to blueprint
        num_players = self._solving_state.num_players if self._solving_state else 6
        bp = lookup_with_fallback(self.blueprint, info_key, num_players)
        if bp is not None and len(bp) == n_actions:
            WARMTH = 10000.0
            REGRET_WARMTH = WARMTH * 0.1
            iset.cumulative_regret = [p * REGRET_WARMTH for p in bp]
            iset.strategy_sum = [p * WARMTH for p in bp]

        info_sets[info_key] = iset
        return iset

    def _cfr(self, state, traversing_player, cache, info_sets, iteration):
        if state.is_terminal:
            return state.get_terminal_utility(traversing_player)

        player = state.current_player
        bucket = cache.get((player, state.street))
        if bucket is None:
            return 0.0
        actions = state.get_actions()
        if not actions:
            return 0.0

        info_key = state.get_info_set_key(player, bucket)
        info_set = self._get_or_create_info_set(
            info_key, len(actions), info_sets)
        strategy = info_set.get_strategy()

        if player == traversing_player:
            utilities = [0.0] * len(actions)
            node_util = 0.0
            for i, action in enumerate(actions):
                utilities[i] = self._cfr(
                    state.apply_action(action), traversing_player,
                    cache, info_sets, iteration)
                node_util += strategy[i] * utilities[i]
            cr = info_set.cumulative_regret
            for i in range(len(actions)):
                cr[i] = max(0.0, cr[i] + utilities[i] - node_util)
            return node_util
        else:
            idx = sample_action(strategy)
            weight = iteration + 1
            ss = info_set.strategy_sum
            for i in range(len(actions)):
                ss[i] += strategy[i] * weight
            return self._cfr(
                state.apply_action(actions[idx]), traversing_player,
                cache, info_sets, iteration)

    def _bucket_cache(self, hands, board, start_street, num_players):
        cache = {}
        vis_map = {0: (), 1: board[:3], 2: board[:4], 3: board[:5]}
        for p in range(num_players):
            h = hands[p]
            for s in range(start_street, 4):
                cache[(p, s)] = self.ca.get_bucket(h, vis_map[s])
        return cache

    # ------------------------------------------------------------------
    # Preflop helper
    # ------------------------------------------------------------------

    def _preflop_heuristic_probs(self, state, player, actions):
        """Get preflop action probabilities from the ranges module."""
        if player is None or actions is None or not actions:
            return None
        try:
            from cfr.preflop_solver import hand_to_class
            hand = state.hands[player]
            hand_key = hand_to_class(hand)
            pos = get_position_name(state.num_players, player)
            scenario = classify_scenario(state.history, pos, state.num_players)
            action_probs = get_preflop_action(
                hand_key, pos, scenario, actions, history=state.history,
                num_players=state.num_players)
            probs = [action_probs.get(a, 0.0) for a in actions]
            total = sum(probs)
            if total > 0:
                return [p / total for p in probs]
            return None
        except Exception:
            return None
