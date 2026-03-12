import numpy as np


class InfoSet:
    """Stores cumulative regret and strategy sums for one information set (CFR+)."""

    __slots__ = ('num_actions', 'cumulative_regret', 'strategy_sum')

    def __init__(self, num_actions):
        self.num_actions = num_actions
        self.cumulative_regret = np.zeros(num_actions, dtype=np.float64)
        self.strategy_sum = np.zeros(num_actions, dtype=np.float64)

    def get_strategy(self):
        """Current strategy via regret matching. Uniform if all regrets <= 0."""
        positive = np.maximum(self.cumulative_regret, 0)
        total = positive.sum()
        if total > 0:
            return positive / total
        return np.full(self.num_actions, 1.0 / self.num_actions)

    def get_average_strategy(self):
        """Converged Nash equilibrium approximation from accumulated strategy sums."""
        total = self.strategy_sum.sum()
        if total > 0:
            return self.strategy_sum / total
        return np.full(self.num_actions, 1.0 / self.num_actions)
