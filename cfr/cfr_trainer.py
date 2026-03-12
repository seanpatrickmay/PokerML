"""
External Sampling Monte Carlo CFR+ trainer for HUNL poker.

- External sampling: enumerate all actions for the traversing player,
  sample one action for the opponent.
- CFR+: clamp negative regrets to zero for faster convergence.
- Bucket cache: precompute card buckets once per deal (huge speedup).
"""

import random
import numpy as np

from cfr.game_state import GameState
from cfr.information_set import InfoSet
from cfr.card_abstraction import CardAbstraction


class CFRTrainer:
    def __init__(self, card_abstraction=None, iterations=100_000):
        self.card_abstraction = card_abstraction or CardAbstraction()
        self.iterations = iterations
        self.info_sets: dict[str, InfoSet] = {}

    def _get_info_set(self, key, num_actions):
        if key not in self.info_sets:
            self.info_sets[key] = InfoSet(num_actions)
        return self.info_sets[key]

    def train(self):
        ca = self.card_abstraction
        for i in range(self.iterations):
            deck = list(range(52))
            random.shuffle(deck)
            hands = ((deck[0], deck[1]), (deck[2], deck[3]))
            board = tuple(deck[4:9])

            # Precompute buckets: 2 players x 4 streets = 8 lookups total
            cache = {}
            for p in (0, 1):
                h = hands[p]
                cache[(p, 0)] = ca.get_bucket(h, ())
                cache[(p, 1)] = ca.get_bucket(h, board[:3])
                cache[(p, 2)] = ca.get_bucket(h, board[:4])
                cache[(p, 3)] = ca.get_bucket(h, board[:5])

            state = GameState.new_hand(hands, board)
            for tp in (0, 1):
                self._cfr(state, tp, cache)

            if (i + 1) % 10_000 == 0:
                avg_r = self._avg_positive_regret()
                print(f"  iter {i+1:>8}/{self.iterations}  |  "
                      f"info sets: {len(self.info_sets):>7}  |  "
                      f"avg regret: {avg_r:.4f}")

    def _cfr(self, state, traversing_player, cache):
        if state.is_terminal:
            return state.get_terminal_utility(traversing_player)

        player = state.current_player
        bucket = cache[(player, state.street)]
        actions = state.get_actions()

        if not actions:
            return 0.0

        info_key = state.get_info_set_key(player, bucket)
        info_set = self._get_info_set(info_key, len(actions))
        strategy = info_set.get_strategy()

        if player == traversing_player:
            utilities = np.zeros(len(actions))
            node_util = 0.0

            for i, action in enumerate(actions):
                utilities[i] = self._cfr(
                    state.apply_action(action), traversing_player, cache)
                node_util += strategy[i] * utilities[i]

            for i in range(len(actions)):
                info_set.cumulative_regret[i] = max(
                    0.0, info_set.cumulative_regret[i] + utilities[i] - node_util)

            return node_util
        else:
            idx = np.random.choice(len(actions), p=strategy)
            info_set.strategy_sum += strategy
            return self._cfr(
                state.apply_action(actions[idx]), traversing_player, cache)

    def _avg_positive_regret(self):
        if not self.info_sets:
            return 0.0
        total = sum(iset.cumulative_regret.sum() for iset in self.info_sets.values())
        return total / len(self.info_sets)

    def get_average_strategy(self):
        """Return {info_set_key: [action_probabilities]} for the converged strategy."""
        return {key: iset.get_average_strategy().tolist()
                for key, iset in self.info_sets.items()}
