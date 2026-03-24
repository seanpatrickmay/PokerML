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


# --- CFR solver ----------------------------------------------------------------

NUM_PLAYERS = int(os.getenv("NUM_PLAYERS", "6"))
CFR_ITERATIONS = int(os.getenv("CFR_ITERATIONS", "100000"))
CFR_PREFLOP_BUCKETS = int(os.getenv("CFR_PREFLOP_BUCKETS", "0"))   # 0 = auto
CFR_POSTFLOP_BUCKETS = int(os.getenv("CFR_POSTFLOP_BUCKETS", "0")) # 0 = auto
CFR_STRATEGY_FILE = os.getenv("CFR_STRATEGY_FILE", "strategy.json.gz")
CFR_CHECKPOINT_INTERVAL = int(os.getenv("CFR_CHECKPOINT_INTERVAL", "50000"))

# --- Card abstraction bucket counts per format --------------------------------
# Higher = finer abstraction = better strategy quality but more memory/slower convergence.
# NOTE: Changing bucket counts invalidates existing strategy files (different info set keys).
PREFLOP_BUCKETS_2MAX = int(os.getenv("PREFLOP_BUCKETS_2MAX", "169"))  # Full canonical
PREFLOP_BUCKETS_6MAX = int(os.getenv("PREFLOP_BUCKETS_6MAX", "50"))
PREFLOP_BUCKETS_9MAX = int(os.getenv("PREFLOP_BUCKETS_9MAX", "30"))
POSTFLOP_BUCKETS_2MAX = int(os.getenv("POSTFLOP_BUCKETS_2MAX", "200"))
POSTFLOP_BUCKETS_6MAX = int(os.getenv("POSTFLOP_BUCKETS_6MAX", "100"))
POSTFLOP_BUCKETS_9MAX = int(os.getenv("POSTFLOP_BUCKETS_9MAX", "50"))

# Postflop sampling — more samples = less variance in bucket assignment.
FLOP_SAMPLES = int(os.getenv("FLOP_SAMPLES", "20"))
TURN_SAMPLES = int(os.getenv("TURN_SAMPLES", "10"))

# EMD-based potential-aware bucketing (flop/turn only).
# Clusters hands by distribution of future hand strengths rather than current equity.
# NOTE: Changing this invalidates existing strategy files (different bucket assignments).
USE_EMD_BUCKETING = os.getenv("USE_EMD_BUCKETING", "true").lower() in ("1", "true", "yes")

# OCHS-based river bucketing (Outcome Cluster Hand Strength).
# Uses a stratified equity histogram against opponent strength buckets to
# distinguish pure air from weak-but-real hands at the river.
# NOTE: Changing this invalidates existing strategy files (different bucket assignments).
USE_OCHS_RIVER = os.getenv("USE_OCHS_RIVER", "true").lower() in ("1", "true", "yes")

# Number of opponent hands to sample for OCHS river bucketing.
OCHS_OPPONENT_SAMPLES = int(os.getenv("OCHS_OPPONENT_SAMPLES", "200"))

# --- Subgame solver -----------------------------------------------------------
SUBGAME_ITERATIONS = int(os.getenv("SUBGAME_ITERATIONS", "5000"))

# Max bet sizes per node during subgame solving (fewer = faster tree traversal).
# Training uses 5; subgame solving uses 3 (small, geometric, all-in).
SUBGAME_MAX_BETS = int(os.getenv("SUBGAME_MAX_BETS", "3"))

# Blueprint probability threshold for action pruning during subgame solving.
# Actions with blueprint probability below this are pruned (except non-bet actions).
SUBGAME_PRUNE_THRESHOLD = float(os.getenv("SUBGAME_PRUNE_THRESHOLD", "0.08"))

# Regret-based pruning: skip actions with regret below this after warmup iterations.
RBP_THRESHOLD = float(os.getenv("RBP_THRESHOLD", "-300.0"))
RBP_WARMUP = int(os.getenv("RBP_WARMUP", "100"))
LIVE_SOLVER_MERGE_THRESHOLD = int(os.getenv("LIVE_SOLVER_MERGE_THRESHOLD", "100"))

# Reach-probability pruning: skip traverser actions with strategy probability
# below this threshold. Complementary to RBP (which uses cumulative regret).
REACH_PRUNE_THRESHOLD = float(os.getenv("REACH_PRUNE_THRESHOLD", "0.001"))

# Parallel training: number of worker processes for multi-process CFR training.
# 0 = auto-detect (use all CPU cores). 1 = single-process (default, no overhead).
TRAINING_WORKERS = int(os.getenv("TRAINING_WORKERS", "1"))

# --- Leaf evaluation -----------------------------------------------------------
LEAF_MC_SAMPLES = int(os.getenv("LEAF_MC_SAMPLES", "15"))

# --- Always-resolve settings ---------------------------------------------------
# Streets >= this value always use real-time resolving (0=preflop, 1=flop, 2=turn, 3=river)
ALWAYS_RESOLVE_FROM_STREET = int(os.getenv("ALWAYS_RESOLVE_FROM_STREET", "2"))

# --- Safe subgame solving ------------------------------------------------------
GADGET_WARMTH = float(os.getenv("GADGET_WARMTH", "300.0"))

# Fraction of GADGET_WARMTH used for regret warm-start (vs. strategy_sum).
# Lower = faster adaptation from blueprint, higher = more stable.
# Strategy_sum always uses full GADGET_WARMTH to anchor average strategy.
GADGET_REGRET_FRACTION = float(os.getenv("GADGET_REGRET_FRACTION", "0.1"))

# --- Bot timeout ---------------------------------------------------------------
BOT_DECISION_TIMEOUT = float(os.getenv("BOT_DECISION_TIMEOUT", "5.0"))
