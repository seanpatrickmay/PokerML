"""
Centralizes tunable parameters and allows overrides via environment variables.

The defaults preserve the repository's original behaviour, while providing a
single place to reason about cross-module constants.
"""

import os
from typing import Iterable, Optional, Tuple


def _parse_float_list(value: Optional[str], fallback: Iterable[float]) -> Tuple[float, ...]:
    """
    Convert a comma-separated string into a tuple of floats.

    Falls back to the provided iterable whenever the value is undefined or
    cannot be parsed cleanly. This keeps runtime behaviour deterministic while
    still supporting `.env` overrides for experimentation.
    """
    if not value:
        return tuple(fallback)
    try:
        return tuple(float(part.strip()) for part in value.split(",") if part.strip())
    except ValueError:
        return tuple(fallback)


# --- Tree-search parameters -------------------------------------------------

# Expressed in number of pot sizes the agent can choose from when betting.
BET_SIZINGS = _parse_float_list(os.getenv("BET_SIZINGS"), (1 / 3, 1.0, 2.0, 10.0))

# Maximum recursion depth for the minimax search (keeps original value 20).
SEARCH_MAX_DEPTH = int(os.getenv("SEARCH_MAX_DEPTH", "20"))


# --- Heuristic subsystem ----------------------------------------------------

# Candidate bet sizings used by the heuristic range splitter.
HEURISTIC_BET_SIZINGS = _parse_float_list(
    os.getenv("HEURISTIC_BET_SIZINGS"),
    (0, 1 / 4, 1 / 3, 1 / 2, 3 / 4, 1, 4 / 3, 3 / 2, 2, 3, 5, 10),
)


# --- Equity simulation ------------------------------------------------------

# Monte Carlo samples scale with 10 ** (MONTE_CARLO_DECAY_START - board_length).
MONTE_CARLO_BASE = int(os.getenv("MONTE_CARLO_BASE", "10"))

# Starting exponent for Monte Carlo sampling; defaults to the previous constant "4".
MONTE_CARLO_DECAY_START = int(os.getenv("MONTE_CARLO_DECAY_START", "4"))

# Minimum number of simulations to keep edges deterministic.
MONTE_CARLO_MIN_SAMPLES = int(os.getenv("MONTE_CARLO_MIN_SAMPLES", "1"))
