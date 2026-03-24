"""
Numba-accelerated numerical routines for CFR.

JIT-compiles the hot-path math: regret matching, regret updates, and
linear CFR discounting. These functions are called thousands of times
per subgame solve.

Falls back to pure Python/numpy when numba is unavailable.
"""

import numpy as np

try:
    from numba import njit
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False


if NUMBA_AVAILABLE:
    @njit(cache=True)
    def regret_match(regrets):
        """Regret matching: convert cumulative regrets to current strategy.

        Args:
            regrets: numpy array of cumulative regrets

        Returns:
            numpy array of action probabilities
        """
        n = len(regrets)
        strategy = np.empty(n, dtype=np.float64)
        total = 0.0
        for i in range(n):
            if regrets[i] > 0:
                total += regrets[i]

        if total > 0:
            for i in range(n):
                strategy[i] = max(regrets[i], 0.0) / total
        else:
            uniform = 1.0 / n
            for i in range(n):
                strategy[i] = uniform
        return strategy

    @njit(cache=True)
    def update_regrets(regrets, utilities, node_util):
        """Update cumulative regrets using CFR+ (clamp to >= 0).

        Args:
            regrets: numpy array of cumulative regrets (modified in place)
            utilities: numpy array of action utilities
            node_util: float, weighted utility of the node
        """
        for i in range(len(regrets)):
            regrets[i] = max(0.0, regrets[i] + utilities[i] - node_util)

    @njit(cache=True)
    def discount_regrets_array(regrets, discount):
        """Discount an array of regrets by a scalar factor.

        Args:
            regrets: numpy array (modified in place)
            discount: float multiplier
        """
        for i in range(len(regrets)):
            regrets[i] *= discount

    @njit(cache=True)
    def normalize_strategy(probs):
        """Normalize a probability distribution. Returns uniform if sum <= 0."""
        n = len(probs)
        total = 0.0
        for i in range(n):
            total += probs[i]
        result = np.empty(n, dtype=np.float64)
        if total > 0:
            for i in range(n):
                result[i] = probs[i] / total
        else:
            uniform = 1.0 / n
            for i in range(n):
                result[i] = uniform
        return result

    @njit(cache=True)
    def weighted_sample(probs):
        """Sample an index from a probability distribution.

        Args:
            probs: numpy array of probabilities (must sum to ~1)

        Returns:
            int index
        """
        r = np.random.random()
        cumulative = 0.0
        for i in range(len(probs)):
            cumulative += probs[i]
            if r < cumulative:
                return i
        return len(probs) - 1

    # Pre-warm JIT compilation at import time (avoids 5s cold-start on first solve)
    _warmup_arr = np.array([0.3, 0.2, 0.5], dtype=np.float64)
    regret_match(_warmup_arr)
    weighted_sample(_warmup_arr)
    discount_regrets_array(_warmup_arr.copy(), 0.9)
    normalize_strategy(_warmup_arr)
    _warmup_u = np.array([1.0, -1.0, 0.5], dtype=np.float64)
    update_regrets(_warmup_u.copy(), _warmup_u, 0.0)
    del _warmup_arr, _warmup_u

else:
    # Pure Python fallbacks (identical behavior, no JIT)
    def regret_match(regrets):
        n = len(regrets)
        total = 0.0
        for i in range(n):
            if regrets[i] > 0:
                total += regrets[i]

        if total > 0:
            return np.array([max(regrets[i], 0.0) / total for i in range(n)])
        return np.full(n, 1.0 / n)

    def update_regrets(regrets, utilities, node_util):
        for i in range(len(regrets)):
            regrets[i] = max(0.0, regrets[i] + utilities[i] - node_util)

    def discount_regrets_array(regrets, discount):
        regrets *= discount

    def normalize_strategy(probs):
        total = probs.sum()
        if total > 0:
            return probs / total
        return np.full(len(probs), 1.0 / len(probs))

    def weighted_sample(probs):
        return np.random.choice(len(probs), p=probs)
