"""
Real-time subgame solver with Bayesian opponent range tracking.

When the bot encounters an info set missing from the blueprint strategy,
this solver:
  1. Reconstructs the opponent's range by weighting each possible hand
     by the probability the blueprint assigns to the observed actions.
  2. Runs MCCFR on the remaining subtree, sampling opponent hands from
     that weighted range instead of uniformly.
  3. Returns learned strategies for every info set in the subtree.
"""

import random

import numpy as np

from cfr.game_state import GameState, STARTING_STACK, SMALL_BLIND, BIG_BLIND
from cfr.information_set import InfoSet


class SubgameSolver:
    def __init__(self, blueprint_strategy, card_abstraction, iterations=300):
        self.blueprint = blueprint_strategy
        self.ca = card_abstraction
        self.iterations = iterations

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def solve(self, state, bot_position):
        """
        Solve the subgame rooted at *state* and return a dict of
        {info_set_key: [action_probabilities]} for every node in the subtree.
        """
        bot_hand = state.hands[bot_position]
        opp_position = 1 - bot_position

        # 1 ---- Bayesian opponent range from the action history ----------
        range_weights = self._compute_opponent_range(
            bot_hand, state.board, state.history, bot_position)

        if not range_weights:
            return {}

        opp_hands = list(range_weights.keys())
        opp_probs = np.array(list(range_weights.values()))

        # 2 ---- Mini-MCCFR on the subtree --------------------------------
        info_sets: dict[str, InfoSet] = {}

        for _ in range(self.iterations):
            idx = np.random.choice(len(opp_hands), p=opp_probs)
            opp_hand = opp_hands[idx]

            if bot_position == 0:
                hands = (bot_hand, opp_hand)
            else:
                hands = (opp_hand, bot_hand)

            sub_state = GameState(
                hands=hands, board=state.board,
                stacks=state.stacks, pot=state.pot, bets=state.bets,
                current_player=state.current_player,
                history=state.history, street=state.street,
                raises_this_street=state.raises_this_street,
                num_actions=state.num_actions,
                is_terminal=False, terminal_type=None,
                min_raise=state.min_raise,
            )

            cache = self._bucket_cache(hands, state.board, state.street)

            for tp in (0, 1):
                self._cfr(sub_state, tp, cache, info_sets)

        # 3 ---- Extract average strategies --------------------------------
        return {k: v.get_average_strategy().tolist()
                for k, v in info_sets.items()}

    # ------------------------------------------------------------------
    # Bayesian range tracking
    # ------------------------------------------------------------------

    def _compute_opponent_range(self, bot_hand, board, history, bot_position):
        """
        Walk the action history from hand start to present.  At every
        opponent decision point, weight each possible opponent hand by the
        blueprint probability of the observed action.
        """
        opp_pos = 1 - bot_position
        used = set(bot_hand) | set(board)

        # seed: every non-conflicting two-card combo, equal weight
        weights: dict[tuple, float] = {}
        avail = [c for c in range(52) if c not in used]
        for i in range(len(avail)):
            for j in range(i + 1, len(avail)):
                weights[(avail[i], avail[j])] = 1.0

        # dummy hands so we can replay actions through GameState
        dummy_opp = (avail[0], avail[1])
        dummy_hands = ((bot_hand, dummy_opp) if bot_position == 0
                       else (dummy_opp, bot_hand))
        replay = GameState.new_hand(dummy_hands, board)

        for street_actions in history:
            for action in street_actions:
                if replay.is_terminal:
                    break

                if replay.current_player == opp_pos:
                    actions = replay.get_actions()
                    if action not in actions:
                        break
                    act_idx = actions.index(action)
                    vis = replay.visible_board()

                    for hand in list(weights):
                        if weights[hand] < 1e-12:
                            continue
                        bucket = self.ca.get_bucket(hand, vis)
                        key = replay.get_info_set_key(opp_pos, bucket)

                        if key in self.blueprint:
                            probs = self.blueprint[key]
                            if len(probs) == len(actions):
                                weights[hand] *= probs[act_idx]
                            else:
                                weights[hand] *= 1.0 / len(actions)
                        else:
                            weights[hand] *= 1.0 / len(actions)

                replay = replay.apply_action(action)

        # prune and normalize
        weights = {h: w for h, w in weights.items() if w > 1e-12}
        total = sum(weights.values())
        if total <= 0:
            return {}
        return {h: w / total for h, w in weights.items()}

    # ------------------------------------------------------------------
    # CFR traversal (same algorithm as CFRTrainer._cfr)
    # ------------------------------------------------------------------

    def _cfr(self, state, traversing_player, cache, info_sets):
        if state.is_terminal:
            return state.get_terminal_utility(traversing_player)

        player = state.current_player
        bucket = cache[(player, state.street)]
        actions = state.get_actions()
        if not actions:
            return 0.0

        info_key = state.get_info_set_key(player, bucket)
        if info_key not in info_sets:
            # warm-start from blueprint if available
            n = len(actions)
            iset = InfoSet(n)
            if info_key in self.blueprint:
                bp = self.blueprint[info_key]
                if len(bp) == n:
                    iset.strategy_sum[:] = bp
            info_sets[info_key] = iset
        info_set = info_sets[info_key]
        strategy = info_set.get_strategy()

        if player == traversing_player:
            utilities = np.zeros(len(actions))
            node_util = 0.0
            for i, action in enumerate(actions):
                utilities[i] = self._cfr(
                    state.apply_action(action), traversing_player, cache, info_sets)
                node_util += strategy[i] * utilities[i]
            for i in range(len(actions)):
                info_set.cumulative_regret[i] = max(
                    0.0, info_set.cumulative_regret[i] + utilities[i] - node_util)
            return node_util
        else:
            idx = np.random.choice(len(actions), p=strategy)
            info_set.strategy_sum += strategy
            return self._cfr(
                state.apply_action(actions[idx]), traversing_player, cache, info_sets)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _bucket_cache(self, hands, board, start_street):
        cache = {}
        vis_map = {0: (), 1: board[:3], 2: board[:4], 3: board[:5]}
        for p in (0, 1):
            h = hands[p]
            for s in range(start_street, 4):
                cache[(p, s)] = self.ca.get_bucket(h, vis_map[s])
        return cache
