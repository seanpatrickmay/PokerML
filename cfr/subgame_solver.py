"""
Real-time subgame solver with safe resolving (gadget game).

Improvements over naive resolving:
  1. Gadget game: opponents can "terminate" for their blueprint EV,
     guaranteeing the resolved strategy is never worse than the blueprint
     (Brown & Sandholm, "Safe and Nested Subgame Solving", NeurIPS 2017).
  2. MC equity at leaf nodes: samples board completions instead of crude
     bucket-based heuristics for depth-limited evaluation.
  3. Linear CFR: discounts regrets every iteration (not every 100).
  4. Bayesian opponent ranges from action history.

Performance optimizations:
  - FastState in-place mutation with undo (avoids GameState tuple copies)
  - Pre-computed sampling arrays (avoids dict→list per iteration)
  - Inline regret discounting (avoids list↔numpy conversion overhead)
  - Single base_key computation per node (eliminates duplicate work)
"""

import random
import time

import numpy as np

from cfr.fast_state import FastState
from cfr.game_state import GameState, STARTING_STACK, SMALL_BLIND, BIG_BLIND, get_position_name
from cfr.information_set import InfoSet
from cfr.depth_limited_solver import estimate_leaf_value, should_depth_limit
from cfr.evaluator import enumerate_river_equity
from cfr.numba_utils import weighted_sample, discount_regrets_array
from cfr.preflop_ranges import get_preflop_action, classify_scenario
from cfr.preflop_solver import hand_to_class
from config import (SUBGAME_ITERATIONS, GADGET_WARMTH, GADGET_REGRET_FRACTION,
                     BOT_DECISION_TIMEOUT,
                     SUBGAME_MAX_BETS, SUBGAME_PRUNE_THRESHOLD,
                     RBP_THRESHOLD, RBP_WARMUP,
                     REACH_PRUNE_THRESHOLD)


def _to_fast_state(gs):
    """Convert an immutable GameState to a mutable FastState."""
    n = gs.num_players
    fs = FastState(n, gs.hands, gs.board)
    fs.stacks = list(gs.stacks)
    fs.pot = gs.pot
    fs.bets = list(gs.bets)
    fs.current_player = gs.current_player
    # Deep-copy history (list of lists)
    fs.history = [list(s) for s in gs.history]
    fs.street = gs.street
    fs.raises_this_street = gs.raises_this_street
    fs.is_terminal = gs.is_terminal
    fs.terminal_type = gs.terminal_type
    fs.min_raise = gs.min_raise
    fs.last_raiser = gs.last_raiser
    # Convert folded/all_in/has_acted from tuple-of-bools to bitmask
    if isinstance(gs.folded, int):
        fs.folded = gs.folded
        fs.all_in = gs.all_in
        fs.has_acted = gs.has_acted
    else:
        fs.folded = sum((1 << i) for i in range(n) if gs.folded[i])
        fs.all_in = sum((1 << i) for i in range(n) if gs.all_in[i])
        fs.has_acted = sum((1 << i) for i in range(n) if gs.has_acted[i])
    return fs


def _is_folded(state, i):
    """Check if player i has folded, handling both bitmask and tuple."""
    if isinstance(state.folded, int):
        return bool((state.folded >> i) & 1)
    return state.folded[i]


class SubgameSolver:
    def __init__(self, blueprint_strategy, card_abstraction, iterations=None,
                 use_linear_cfr=True, max_depth=3, timeout=None):
        self.blueprint = blueprint_strategy
        self.ca = card_abstraction
        self.iterations = iterations or SUBGAME_ITERATIONS
        self.use_linear_cfr = use_linear_cfr
        self.max_depth = max_depth
        self._warmth = GADGET_WARMTH
        self.timeout = timeout or BOT_DECISION_TIMEOUT
        self._max_bets = SUBGAME_MAX_BETS
        self._prune_threshold = SUBGAME_PRUNE_THRESHOLD
        self._rbp_threshold = RBP_THRESHOLD
        self._rbp_warmup = RBP_WARMUP

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def solve(self, state, bot_position):
        """
        Solve the subgame rooted at *state* and return a dict of
        {info_set_key: [action_probabilities]} for every node in the subtree.

        Uses safe resolving: opponents get a "terminate" option at the
        subgame root that pays their blueprint EV.  This ensures the
        resolved strategy is never worse than the blueprint for the bot.
        """
        bot_hand = state.hands[bot_position]
        num_players = state.num_players

        # 1 ---- Bayesian opponent ranges from action history ------
        opponent_ranges = {}
        for opp in range(num_players):
            if opp == bot_position or _is_folded(state, opp):
                continue
            opponent_ranges[opp] = self._compute_opponent_range(
                bot_hand, state.board, state.history, bot_position, opp, state)

        if not opponent_ranges:
            return {}

        # 2 ---- Compute gadget values (blueprint EV per hand) -----
        gadget_values = self._compute_gadget_values(state, opponent_ranges)

        # Pre-compute sampling arrays (avoid dict→list every iteration)
        sampling_data = {}
        for opp_seat, range_weights in opponent_ranges.items():
            if range_weights:
                hands_list = list(range_weights.keys())
                probs_arr = np.array(list(range_weights.values()))
                sampling_data[opp_seat] = (hands_list, probs_arr)

        # Convert to FastState for in-place CFR traversal
        root_fs = _to_fast_state(state)

        # Pre-compute which players are not folded
        active_players = [p for p in range(num_players) if not _is_folded(state, p)]

        # Adaptive depth: river needs less depth (fewer remaining streets)
        effective_max_depth = self.max_depth
        if state.street == 3:
            effective_max_depth = min(self.max_depth, 2)  # bet-raise-respond
        elif state.street == 2:
            effective_max_depth = min(self.max_depth, 3)

        # 3 ---- Mini-MCCFR on the subtree --------------------------
        info_sets: dict[str, InfoSet] = {}
        self._equity_cache = {}  # (hand, board_tuple) → equity
        start_time = time.monotonic()
        deadline = start_time + max(self.timeout - 1.0, 1.0)

        prev_strategies = {}
        converge_check_interval = 50

        for it in range(self.iterations):
            # Time check every 25 iterations
            if it > 0 and it % 25 == 0 and time.monotonic() > deadline:
                break

            # Early convergence: check root strategy stability every N iterations
            if it > 0 and it % converge_check_interval == 0 and info_sets:
                max_delta = 0.0
                checked = 0
                for k, iset in info_sets.items():
                    if k.startswith('G:'):
                        continue
                    avg = iset.get_average_strategy()
                    prev = prev_strategies.get(k)
                    if prev is not None and len(prev) == len(avg):
                        delta = sum(abs(a - b) for a, b in zip(avg, prev))
                        max_delta = max(max_delta, delta)
                    prev_strategies[k] = avg.tolist()
                    checked += 1
                    if checked >= 20:  # only check first 20 info sets
                        break
                if max_delta < 0.01 and it >= 100:
                    break

            # Sample opponent hands with pre-computed arrays
            sampled = self._sample_opponent_hands_fast(
                bot_hand, sampling_data, state.board)
            if sampled is None:
                continue

            # Build hands tuple for this iteration
            hands = list(state.hands)
            for opp_seat, opp_hand in sampled.items():
                hands[opp_seat] = opp_hand
            hands = tuple(hands)

            # Update FastState hands for terminal evaluation
            root_fs.hands = hands

            cache = self._bucket_cache(hands, state.board,
                                       state.street, num_players)

            # Alternating updates: traverse for one player per iteration
            # (same convergence per traversal, halves wall-clock time)
            tp = active_players[it % len(active_players)]
            self._cfr_fast(root_fs, tp, cache, info_sets, depth=0,
                           gadget_values=gadget_values,
                           bot_position=bot_position, hands=hands,
                           iteration=it,
                           max_depth=effective_max_depth)

            # DCFR schedule: aggressive early, gentle late
            if self.use_linear_cfr:
                t = it + 1
                # Farina et al. 2024: start with aggressive discounting,
                # converge to standard linear CFR
                if t < 50:
                    discount = max(0.5, t / (t + 2))
                else:
                    discount = t / (t + 1)
                for iset in info_sets.values():
                    cr = iset.cumulative_regret
                    for j in range(len(cr)):
                        cr[j] *= discount

        # 4 ---- Extract average strategies --------------------------
        result = {}
        for k, v in info_sets.items():
            if k.startswith('G:'):
                real_key = k[2:]
                avg = v.get_average_strategy().tolist()
                real_probs = avg[:-1]
                total = sum(real_probs)
                if total > 0:
                    real_probs = [p / total for p in real_probs]
                else:
                    n = len(real_probs)
                    real_probs = [1.0 / n] * n
                result[real_key] = real_probs
            else:
                result[k] = v.get_average_strategy().tolist()
        return result

    # ------------------------------------------------------------------
    # Gadget game: compute blueprint EV for each opponent hand
    # ------------------------------------------------------------------

    def _compute_gadget_values(self, state, opponent_ranges):
        """Compute per-hand blueprint EV for each opponent at the subgame root.

        Uses equity realization discounting: medium-strength hands realize
        less equity with more streets remaining (they get outdrawn or bluffed
        off). This matches depth_limited_solver's approach and produces more
        accurate gadget values than raw equity * pot.

        Returns {opp_seat: {hand_tuple: float_ev}}.
        """
        gadget = {}
        vis = state.visible_board()
        pot = state.pot + sum(state.bets)
        streets_left = max(0, 3 - state.street)

        for opp, range_weights in opponent_ranges.items():
            hand_values = {}
            invested = STARTING_STACK - state.stacks[opp]
            for hand in range_weights:
                bucket = self.ca.get_bucket(hand, vis)
                equity = (bucket + 0.5) / max(self.ca.num_postflop_buckets, 1)

                # Equity realization discount for remaining streets
                if streets_left > 0:
                    if equity > 0.5:
                        realization = 1.0 - 0.05 * streets_left
                    else:
                        realization = 1.0 - 0.15 * streets_left
                    equity *= realization

                hand_values[hand] = equity * pot - invested
            gadget[opp] = hand_values

        return gadget

    # ------------------------------------------------------------------
    # Bayesian range tracking (per opponent)
    # ------------------------------------------------------------------

    def _compute_opponent_range(self, bot_hand, board, history,
                                 bot_position, opp_position, state):
        """
        Walk the action history and weight each possible opponent hand
        by the blueprint probability of the observed action.

        Uses GTO preflop opening ranges as priors instead of uniform,
        narrowing the range before postflop Bayesian updates.
        """
        used = set(bot_hand) | set(board)
        num_players = state.num_players
        opp_pos = get_position_name(num_players, opp_position)

        # Seed: weight by preflop opening probability (GTO prior)
        weights: dict[tuple, float] = {}
        avail = [c for c in range(52) if c not in used]
        for i in range(len(avail)):
            for j in range(i + 1, len(avail)):
                hand = (avail[i], avail[j])
                hand_key = hand_to_class(hand)
                # Get GTO opening probability for this hand class + position
                preflop_probs = get_preflop_action(
                    hand_key, opp_pos, 'first_in',
                    ['f', 'c', 'b75'], num_players=num_players)
                # Weight = probability of not folding (entering pot)
                fold_prob = preflop_probs.get('f', 0.5)
                weights[hand] = max(0.01, 1.0 - fold_prob)

        # Dummy hands for replaying actions through GameState
        dummy_hands = list(state.hands)
        dummy_opp = (avail[0], avail[1])
        dummy_hands[opp_position] = dummy_opp
        replay = GameState.new_hand(tuple(dummy_hands), board)

        for street_actions in history:
            for action in street_actions:
                if replay.is_terminal:
                    break

                if replay.current_player == opp_position:
                    actions = replay.get_actions()
                    if action not in actions:
                        break
                    act_idx = actions.index(action)
                    vis = replay.visible_board()

                    for hand in list(weights):
                        if weights[hand] < 1e-12:
                            continue
                        bucket = self.ca.get_bucket(hand, vis)
                        key = replay.get_info_set_key(opp_position, bucket)

                        if key in self.blueprint:
                            probs = self.blueprint[key]
                            if len(probs) == len(actions):
                                weights[hand] *= probs[act_idx]
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
    # Hand sampling with card-conflict rejection
    # ------------------------------------------------------------------

    def _sample_opponent_hands(self, bot_hand, opponent_ranges, board):
        """Sample one hand per opponent, rejecting card conflicts."""
        used = set(bot_hand) | set(board)
        sampled = {}

        for opp_seat, range_weights in opponent_ranges.items():
            if not range_weights:
                return None
            hands = list(range_weights.keys())
            probs = np.array(list(range_weights.values()))

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

    def _sample_opponent_hands_fast(self, bot_hand, sampling_data, board):
        """Sample using pre-computed arrays (avoids dict→list per call)."""
        used = set(bot_hand) | set(board)
        sampled = {}

        for opp_seat, (hands_list, probs_arr) in sampling_data.items():
            for _ in range(50):
                idx = np.random.choice(len(hands_list), p=probs_arr)
                hand = hands_list[idx]
                if hand[0] not in used and hand[1] not in used:
                    sampled[opp_seat] = hand
                    used.add(hand[0])
                    used.add(hand[1])
                    break
            else:
                return None

        return sampled

    # ------------------------------------------------------------------
    # CFR traversal with gadget game (original, for compatibility)
    # ------------------------------------------------------------------

    def _cfr(self, state, traversing_player, cache, info_sets, depth=0,
             gadget_values=None, bot_position=None, hands=None):
        if state.is_terminal:
            return state.get_terminal_utility(traversing_player)

        if should_depth_limit(depth, self.max_depth):
            return self._leaf_value(traversing_player, cache, state, hands)

        player = state.current_player
        bucket = cache.get((player, state.street))
        if bucket is None:
            return 0.0
        actions = state.get_actions()
        if not actions:
            return 0.0

        base_key = state.get_info_set_key(player, bucket)

        # Unified action filtering (same as _cfr_fast)
        actions, bp_map = self._filter_actions(actions, base_key, depth=depth)

        use_gadget = (depth == 0 and gadget_values
                      and bot_position is not None
                      and player != bot_position
                      and player in gadget_values)

        n_real = len(actions)
        n_total = n_real + (1 if use_gadget else 0)
        info_key = f"G:{base_key}" if use_gadget else base_key

        if info_key not in info_sets:
            iset = InfoSet(n_total)
            regret_w = self._warmth * GADGET_REGRET_FRACTION
            bp = self.blueprint.get(base_key)
            if bp is not None and bp_map is not None:
                mapped_probs = [bp[j] for j in bp_map if j < len(bp)]
                if len(mapped_probs) == n_real:
                    if use_gadget:
                        probs = mapped_probs + [0.02]
                        tot = sum(probs)
                        probs = [p / tot for p in probs]
                    else:
                        probs = mapped_probs
                        tot = sum(probs)
                        if tot > 0:
                            probs = [p / tot for p in probs]
                    iset.cumulative_regret = [p * regret_w for p in probs]
                    iset.strategy_sum = [p * self._warmth for p in probs]
            elif bp is not None and len(bp) == n_real:
                probs = list(bp)
                if use_gadget:
                    probs = probs + [0.02]
                    tot = sum(probs)
                    probs = [p / tot for p in probs]
                iset.cumulative_regret = [p * regret_w for p in probs]
                iset.strategy_sum = [p * self._warmth for p in probs]
            info_sets[info_key] = iset
        info_set = info_sets[info_key]
        strategy = info_set.get_strategy()

        if player == traversing_player:
            utilities = [0.0] * n_total
            node_util = 0.0

            for i in range(n_real):
                utilities[i] = self._cfr(
                    state.apply_action(actions[i]), traversing_player, cache,
                    info_sets, depth=depth + 1,
                    gadget_values=gadget_values,
                    bot_position=bot_position, hands=hands)
                node_util += strategy[i] * utilities[i]

            if use_gadget:
                opp_hand = hands[player] if hands else None
                utilities[n_real] = self._gadget_payoff(
                    traversing_player, player, opp_hand, gadget_values,
                    cache, state)
                node_util += strategy[n_real] * utilities[n_real]

            for i in range(n_total):
                info_set.cumulative_regret[i] = max(
                    0.0, info_set.cumulative_regret[i] + utilities[i] - node_util)
            return node_util
        else:
            idx = weighted_sample(np.asarray(strategy[:n_total]))
            for _si in range(len(strategy)):
                info_set.strategy_sum[_si] += strategy[_si]

            if use_gadget and idx == n_real:
                opp_hand = hands[player] if hands else None
                return self._gadget_payoff(
                    traversing_player, player, opp_hand, gadget_values,
                    cache, state)

            return self._cfr(
                state.apply_action(actions[idx]), traversing_player, cache,
                info_sets, depth=depth + 1,
                gadget_values=gadget_values,
                bot_position=bot_position, hands=hands)

    # ------------------------------------------------------------------
    # Action filtering for subgame solving
    # ------------------------------------------------------------------

    def _filter_actions(self, actions, base_key, depth=1):
        """Filter actions to reduce branching factor for subgame solving.

        At depth 0 (root): NO filtering — the bot needs probs for all actions.
        At depth > 0: blueprint pruning + bet size cap for tree reduction.

        Returns (filtered_actions, bp_index_map) where bp_index_map maps
        filtered action indices to original action indices for blueprint seeding.
        """
        # Never filter at root — bot reads root probs and expects full action set
        if depth == 0 or len(actions) <= 3:
            return actions, None

        orig_actions = actions
        orig_n = len(actions)

        # Blueprint-guided pruning
        bp = self.blueprint.get(base_key)
        if bp is not None and len(bp) == orig_n:
            keep = [i for i in range(orig_n)
                    if bp[i] >= self._prune_threshold
                    or not actions[i].startswith('b')]
            if len(keep) >= 2:
                actions = [orig_actions[i] for i in keep]
                bp_map = keep
            else:
                bp_map = list(range(orig_n))
        else:
            bp_map = list(range(orig_n))

        # Cap bet sizes: keep at most _max_bets bet sizes
        bets = [(i, a) for i, a in enumerate(actions) if a.startswith('b')]
        if len(bets) > self._max_bets:
            keep_idx = {bets[0][0], bets[len(bets) // 2][0], bets[-1][0]}
            new_actions = []
            new_bp_map = []
            for i, a in enumerate(actions):
                if not a.startswith('b') or i in keep_idx:
                    new_actions.append(a)
                    new_bp_map.append(bp_map[i])
            actions = new_actions
            bp_map = new_bp_map

        return actions, bp_map

    # ------------------------------------------------------------------
    # Fast CFR traversal using FastState apply/undo (avoids copies)
    # ------------------------------------------------------------------

    def _cfr_fast(self, state, traversing_player, cache, info_sets, depth=0,
                  gadget_values=None, bot_position=None, hands=None,
                  iteration=0, max_depth=None):
        """CFR traversal using in-place mutation with undo stack.

        Optimizations over baseline _cfr:
          1. FastState apply/undo instead of GameState copy
          2. Tighter action abstraction (max 3 bet sizes per node)
          3. Blueprint-guided pruning at ALL depths (not just depth > 0)
          4. Regret-based pruning (RBP) after warmup iterations
          5. Adaptive depth limiting (river uses depth 2, turn uses depth 3)
        """
        if state.is_terminal:
            return state.get_terminal_utility(traversing_player)

        md = max_depth if max_depth is not None else self.max_depth
        if should_depth_limit(depth, md):
            return self._leaf_value_fast(traversing_player, cache, state, hands)

        player = state.current_player
        bucket = cache.get((player, state.street))
        if bucket is None:
            return 0.0
        actions = state.get_actions()
        if not actions:
            return 0.0

        base_key = state.get_info_set_key(player, bucket)

        # Filter actions: blueprint pruning + bet size cap
        actions, bp_map = self._filter_actions(actions, base_key, depth=depth)

        use_gadget = (depth == 0 and gadget_values
                      and bot_position is not None
                      and player != bot_position
                      and player in gadget_values)

        n_real = len(actions)
        n_total = n_real + (1 if use_gadget else 0)
        info_key = f"G:{base_key}" if use_gadget else base_key

        if info_key not in info_sets:
            iset = InfoSet(n_total)
            regret_w = self._warmth * GADGET_REGRET_FRACTION
            # Seed from blueprint using index map
            bp = self.blueprint.get(base_key)
            if bp is not None and bp_map is not None:
                mapped_probs = [bp[j] for j in bp_map if j < len(bp)]
                if len(mapped_probs) == n_real:
                    if use_gadget:
                        probs = mapped_probs + [0.02]
                        tot = sum(probs)
                        probs = [p / tot for p in probs]
                    else:
                        probs = mapped_probs
                        tot = sum(probs)
                        if tot > 0:
                            probs = [p / tot for p in probs]
                    iset.cumulative_regret = [p * regret_w for p in probs]
                    iset.strategy_sum = [p * self._warmth for p in probs]
            elif bp is not None and len(bp) == n_real:
                probs = list(bp)
                if use_gadget:
                    probs = probs + [0.02]
                    tot = sum(probs)
                    probs = [p / tot for p in probs]
                iset.cumulative_regret = [p * regret_w for p in probs]
                iset.strategy_sum = [p * self._warmth for p in probs]
            info_sets[info_key] = iset
        info_set = info_sets[info_key]
        strategy = info_set.get_strategy()

        if player == traversing_player:
            utilities = [0.0] * n_total
            node_util = 0.0

            # Regret-based pruning + reach-probability pruning
            use_rbp = iteration >= self._rbp_warmup
            rbp_thresh = self._rbp_threshold
            reach_thresh = REACH_PRUNE_THRESHOLD

            for i in range(n_real):
                # RBP: skip actions with deeply negative regret
                if use_rbp and info_set.cumulative_regret[i] < rbp_thresh:
                    continue
                # Reach pruning: skip actions with near-zero strategy probability
                if use_rbp and strategy[i] < reach_thresh:
                    continue
                state.apply_action(actions[i])
                utilities[i] = self._cfr_fast(
                    state, traversing_player, cache, info_sets,
                    depth=depth + 1, gadget_values=gadget_values,
                    bot_position=bot_position, hands=hands,
                    iteration=iteration, max_depth=md)
                state.undo_action()
                node_util += strategy[i] * utilities[i]

            if use_gadget:
                opp_hand = hands[player] if hands else None
                utilities[n_real] = self._gadget_payoff(
                    traversing_player, player, opp_hand, gadget_values,
                    cache, state)
                node_util += strategy[n_real] * utilities[n_real]

            for i in range(n_total):
                info_set.cumulative_regret[i] = max(
                    0.0, info_set.cumulative_regret[i] + utilities[i] - node_util)
            return node_util
        else:
            idx = weighted_sample(np.asarray(strategy[:n_total]))
            for _si in range(len(strategy)):
                info_set.strategy_sum[_si] += strategy[_si]

            if use_gadget and idx == n_real:
                opp_hand = hands[player] if hands else None
                return self._gadget_payoff(
                    traversing_player, player, opp_hand, gadget_values,
                    cache, state)

            state.apply_action(actions[idx])
            val = self._cfr_fast(
                state, traversing_player, cache, info_sets,
                depth=depth + 1, gadget_values=gadget_values,
                bot_position=bot_position, hands=hands,
                iteration=iteration, max_depth=md)
            state.undo_action()
            return val

    def _gadget_payoff(self, traversing_player, terminator, terminator_hand,
                       gadget_values, cache, state):
        """Payoff for traversing_player when terminator takes the terminate action.

        In heads-up (zero-sum): tp gets -(terminator's blueprint EV).
        Multi-way: approximated the same way.
        """
        if traversing_player == terminator:
            # Terminator gets their own blueprint EV
            if terminator_hand and terminator in gadget_values:
                return gadget_values[terminator].get(terminator_hand, 0.0)
            return 0.0
        else:
            # Other player: approximate as -(terminator's EV) (zero-sum)
            if terminator_hand and terminator in gadget_values:
                return -gadget_values[terminator].get(terminator_hand, 0.0)
            return 0.0

    def _leaf_value(self, traversing_player, cache, state, hands):
        """Estimate value at depth limit using MC equity when possible."""
        bucket = cache.get((traversing_player, state.street), 0)
        pot = state.pot + sum(state.bets)
        hand = hands[traversing_player] if hands else None
        vis = state.visible_board() if state.street > 0 else None
        num_opps = sum(1 for i in range(state.num_players)
                       if not _is_folded(state, i) and i != traversing_player)
        return estimate_leaf_value(
            self.blueprint, traversing_player, bucket, pot,
            state.stacks[traversing_player],
            num_buckets=self.ca.num_postflop_buckets,
            hand=hand, board=vis,
            num_opponents=max(num_opps, 1),
            street=state.street,
        )

    def _leaf_value_fast(self, traversing_player, cache, state, hands):
        """Leaf value for FastState (bitmask folded).

        Caches equity (both exact river and MC flop/turn) to avoid redundant
        evaluations. Equity depends on (hand, board, num_opponents) which
        repeats heavily across iterations.
        """
        bucket = cache.get((traversing_player, state.street), 0)
        pot = state.pot + sum(state.bets)
        hand = hands[traversing_player] if hands else None
        vis = state.visible_board() if state.street > 0 else None
        num_opps = 0
        for i in range(state.num_players):
            if i != traversing_player and not ((state.folded >> i) & 1):
                num_opps += 1

        if hand is not None and vis is not None:
            eq_key = (hand, vis, max(num_opps, 1))
            equity = self._equity_cache.get(eq_key)
            if equity is None:
                # Always use MC for leaf nodes (they're approximations anyway).
                # MC is ~13x faster than exact enumeration and caches well
                # since many iterations share the same (hand, board) pair.
                from cfr.depth_limited_solver import _mc_equity
                equity = _mc_equity(hand, vis, max(num_opps, 1))
                self._equity_cache[eq_key] = equity
            invested = STARTING_STACK - state.stacks[traversing_player]
            ev = equity * pot - invested
            return max(-STARTING_STACK, min(STARTING_STACK, ev))

        return estimate_leaf_value(
            self.blueprint, traversing_player, bucket, pot,
            state.stacks[traversing_player],
            num_buckets=self.ca.num_postflop_buckets,
            hand=hand, board=vis,
            num_opponents=max(num_opps, 1),
            street=state.street,
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _bucket_cache(self, hands, board, start_street, num_players):
        cache = {}
        vis_map = {0: (), 1: board[:3], 2: board[:4], 3: board[:5]}
        for p in range(num_players):
            h = hands[p]
            for s in range(start_street, 4):
                cache[(p, s)] = self.ca.get_bucket(h, vis_map[s])
        return cache
