"""
Depth-limited subgame solving with improved leaf value estimation.

Two estimation modes:
  1. MC equity sampling (when hand + board are available): samples board
     completions and evaluates actual equity against random opponent hands.
  2. Bucket-based heuristic (fallback): uses hand-strength bucket as equity proxy.

Reference: Brown & Sandholm, "Depth-Limited Solving for Imperfect-
Information Games", NeurIPS 2018.
"""
import random

import numpy as np
from phevaluator import evaluate_cards

from cfr.evaluator import enumerate_river_equity
from cfr.game_state import STARTING_STACK
from cfr.nn_estimator import predict_equity as _nn_predict
from config import LEAF_MC_SAMPLES


def estimate_leaf_value(blueprint, player, bucket, pot, stack,
                        num_buckets=200, starting_stack=STARTING_STACK,
                        hand=None, board=None, num_opponents=1, street=None):
    """
    Estimate the value of a position at a depth limit.

    If hand and board are provided, uses Monte Carlo equity sampling for
    a more accurate estimate. Otherwise falls back to bucket-based heuristic.
    """
    if num_buckets <= 1:
        return 0.0

    invested = starting_stack - stack

    # River with full board: use exact enumeration (no sampling variance)
    if hand is not None and board is not None and len(board) == 5:
        equity = enumerate_river_equity(hand, board)
    elif hand is not None and board is not None:
        equity = _mc_equity(hand, board, num_opponents)
    else:
        # Try NN estimator first (single forward pass, ~0.1ms)
        # NN already has street as a feature, so skip realization discount
        nn_eq = _nn_predict(bucket, pot, stack, street, num_opponents,
                            num_buckets, starting_stack)
        if nn_eq is not None:
            equity = nn_eq
        else:
            # Bucket-based heuristic — apply realization discount for
            # remaining streets (players don't always check down)
            equity = (bucket + 0.5) / num_buckets
            if street is not None:
                streets_left = max(0, 3 - street)
                if streets_left > 0:
                    if equity > 0.5:
                        realization = 1.0 - 0.05 * streets_left
                    else:
                        realization = 1.0 - 0.15 * streets_left
                    equity *= realization

    ev = equity * pot - invested
    return float(np.clip(ev, -starting_stack, starting_stack))


def _mc_equity(hand, board, num_opponents=1, samples=None):
    """Estimate equity by sampling board completions and opponent hands."""
    if samples is None:
        samples = LEAF_MC_SAMPLES

    used = set(hand) | set(board)
    deck = [c for c in range(52) if c not in used]
    remaining = max(0, 5 - len(board))
    cards_needed = remaining + 2 * num_opponents

    if len(deck) < cards_needed or cards_needed <= 0:
        return 0.5

    wins = 0
    ties = 0
    total = 0

    for _ in range(samples):
        drawn = random.sample(deck, cards_needed)
        full_board = list(board) + drawn[:remaining]

        hero_score = evaluate_cards(*full_board, *hand)

        beat_all = True
        tied_all = True
        for opp in range(num_opponents):
            offset = remaining + opp * 2
            opp_hand = (drawn[offset], drawn[offset + 1])
            opp_score = evaluate_cards(*full_board, *opp_hand)
            if opp_score < hero_score:
                beat_all = False
                tied_all = False
                break
            elif opp_score == hero_score:
                beat_all = False
            else:
                tied_all = False

        if beat_all:
            wins += 1
        elif tied_all:
            ties += 1
        total += 1

    if total == 0:
        return 0.5

    return (wins + 0.5 * ties) / total


def should_depth_limit(depth, max_depth=3):
    """Check if we should stop expanding and use leaf estimation."""
    return depth >= max_depth
