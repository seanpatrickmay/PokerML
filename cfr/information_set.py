"""
Information set for CFR+ with optimized storage.

Uses plain Python lists instead of numpy arrays for the small action
counts typical in poker (2-7 actions). This avoids numpy overhead
for tiny arrays which dominates at scale.
"""

import numpy as np
from random import random

try:
    from cfr.cfr_core import regret_match_cy, update_regrets_cy
    _USE_CYTHON = True
except ImportError:
    _USE_CYTHON = False


class InfoSet:
    __slots__ = ('num_actions', 'cumulative_regret', 'strategy_sum')

    def __init__(self, num_actions):
        self.num_actions = num_actions
        self.cumulative_regret = [0.0] * num_actions
        self.strategy_sum = [0.0] * num_actions

    def get_strategy(self):
        """Current strategy via regret matching. Returns plain list."""
        if _USE_CYTHON:
            return regret_match_cy(self.cumulative_regret, self.num_actions)
        n = self.num_actions
        cr = self.cumulative_regret
        total = 0.0
        for i in range(n):
            if cr[i] > 0:
                total += cr[i]

        if total > 0:
            return [max(cr[i], 0.0) / total for i in range(n)]
        u = 1.0 / n
        return [u] * n

    def get_average_strategy(self):
        """Converged strategy from accumulated sums. Returns numpy array."""
        total = sum(self.strategy_sum)
        if total > 0:
            return np.array([s / total for s in self.strategy_sum])
        return np.full(self.num_actions, 1.0 / self.num_actions)

    def cap_regrets(self, max_regret=1e9):
        """Clamp cumulative regrets to [-max_regret, max_regret] for stability."""
        for i in range(len(self.cumulative_regret)):
            if self.cumulative_regret[i] > max_regret:
                self.cumulative_regret[i] = max_regret
            elif self.cumulative_regret[i] < -max_regret:
                self.cumulative_regret[i] = -max_regret


def sample_action(strategy):
    """Sample an action index from a strategy (list of probabilities)."""
    r = random()
    cumulative = 0.0
    for i, p in enumerate(strategy):
        cumulative += p
        if r < cumulative:
            return i
    return len(strategy) - 1
