"""
GTO preflop ranges for 6-max NLHE.

Two modes:
  1. Trained: loads a preflop strategy from preflop_solver.py CFR output
  2. Heuristic: equity-threshold fallback when no trained strategy exists

Supports VPIP/PFR-based opponent modeling to widen/narrow ranges.
"""

import gzip
import json
import math
import os

from cfr.preflop_solver import hand_to_class

# ---- Load equity data -------------------------------------------------------
# Try 9-max first (most general), fall back to 6-max

_PROJECT_ROOT = os.path.join(os.path.dirname(__file__), '..')
_EQUITY_PATH_9MAX = os.path.join(_PROJECT_ROOT, 'preFlopEquities9max.json')
_EQUITY_PATH_6MAX = os.path.join(_PROJECT_ROOT, 'preFlopEquities6max.json')

if os.path.exists(_EQUITY_PATH_9MAX):
    with open(_EQUITY_PATH_9MAX) as _f:
        EQUITIES = json.load(_f)
else:
    with open(_EQUITY_PATH_6MAX) as _f:
        EQUITIES = json.load(_f)

_HANDS_BY_EQUITY = sorted(EQUITIES.items(), key=lambda x: x[1], reverse=True)
_TOTAL_HANDS = len(EQUITIES)


# ---- Preflop strategy (loaded from trained file if available) ---------------

_PREFLOP_STRATEGIES = {}  # num_players -> strategy dict
_STRATEGY_DIR = os.path.join(os.path.dirname(__file__), '..')

_STRATEGY_FILES = {
    6: os.path.join(_STRATEGY_DIR, 'preflop_strategy.json.gz'),
    9: os.path.join(_STRATEGY_DIR, 'preflop_strategy_9max.json.gz'),
}


def load_preflop_strategy(path=None, num_players=6):
    """Load a trained preflop strategy file."""
    if path is None:
        path = _STRATEGY_FILES.get(num_players)
    if path and os.path.exists(path):
        with gzip.open(path, 'rt', encoding='utf-8') as f:
            _PREFLOP_STRATEGIES[num_players] = json.load(f)
        return True
    return False


def has_trained_strategy(num_players=6):
    return num_players in _PREFLOP_STRATEGIES


def _get_strategy_for_players(num_players):
    """Get the best matching strategy for the given player count."""
    if num_players in _PREFLOP_STRATEGIES:
        return _PREFLOP_STRATEGIES[num_players]
    # 7-9 players use 9-max strategy, 3-6 use 6-max, 2 uses 6-max (HU subset)
    if num_players > 6 and 9 in _PREFLOP_STRATEGIES:
        return _PREFLOP_STRATEGIES[9]
    if 6 in _PREFLOP_STRATEGIES:
        return _PREFLOP_STRATEGIES[6]
    return None


# Try to load all available strategies at import time
for _np in _STRATEGY_FILES:
    load_preflop_strategy(num_players=_np)


# ---- Scenario classification ------------------------------------------------

def classify_scenario(history, position, num_players=6):
    """
    Classify preflop scenario from action history.
    Returns: 'first_in', 'vs_raise', 'vs_3bet', 'vs_4bet'
    """
    if not history or not history[0]:
        return 'first_in'

    preflop_actions = history[0]
    raise_count = sum(1 for a in preflop_actions
                      if a.startswith('b') or a == 'a')
    non_fold_non_check = [a for a in preflop_actions if a not in ('f', 'k')]

    if raise_count <= 1:
        # Check if all actions so far are folds (first in) or there's a raise
        has_raise = any(a.startswith('b') or a == 'a' for a in preflop_actions)
        if not has_raise:
            return 'first_in'
        return 'vs_raise'
    elif raise_count == 2:
        return 'vs_3bet'
    else:
        return 'vs_4bet'


# ---- Trained strategy lookup ------------------------------------------------

def _get_trained_action(hand_key, position, history, legal_actions,
                        num_players=6):
    """
    Look up action from trained preflop strategy.
    Returns dict {action: probability} or None if not found.
    """
    strategy = _get_strategy_for_players(num_players)
    if strategy is None:
        return None

    history_str = ','.join(history[0]) if history and history[0] else ''
    info_key = f'PF|{position}:{hand_key}|{history_str}'

    if info_key in strategy:
        probs = strategy[info_key]
        if len(probs) == len(legal_actions):
            return {a: p for a, p in zip(legal_actions, probs)}

    return None


# ---- Heuristic fallback (equity thresholds) ---------------------------------

# GTO-approximate thresholds per position.
# rfi = raise first in %, 3bet = 3bet %, call_open = flat call vs open %,
# call_3bet = call a 3bet %
_GTO_THRESHOLDS = {
    'UTG':  {'rfi': 0.15, '3bet': 0.04, 'call_open': 0.06, 'call_3bet': 0.04},
    'UTG1': {'rfi': 0.16, '3bet': 0.045, 'call_open': 0.07, 'call_3bet': 0.045},
    'UTG2': {'rfi': 0.17, '3bet': 0.05, 'call_open': 0.07, 'call_3bet': 0.045},
    'MP':   {'rfi': 0.20, '3bet': 0.06, 'call_open': 0.08, 'call_3bet': 0.05},
    'HJ':   {'rfi': 0.24, '3bet': 0.07, 'call_open': 0.10, 'call_3bet': 0.06},
    'CO':   {'rfi': 0.28, '3bet': 0.08, 'call_open': 0.12, 'call_3bet': 0.07},
    'BTN':  {'rfi': 0.48, '3bet': 0.11, 'call_open': 0.18, 'call_3bet': 0.09},
    'SB':   {'rfi': 0.42, '3bet': 0.12, 'call_open': 0.06, 'call_3bet': 0.05},
    'BB':   {'rfi': 0.00, '3bet': 0.12, 'call_open': 0.45, 'call_3bet': 0.07},
}

# 3bet thresholds tighten when facing early position opens.
# Keyed by opener position → multiplier applied to defender's 3bet threshold.
_OPENER_TIGHTNESS = {
    'UTG':  0.50,  # vs UTG open → halve your 3bet range
    'UTG1': 0.55,
    'UTG2': 0.58,
    'MP':   0.65,
    'HJ':   0.75,
    'CO':   0.85,
    'BTN':  1.00,
    'SB':   1.00,
}

# IP positions prefer smaller 3bet, OOP prefer larger
_IP_POSITIONS = {'BTN', 'CO'}
_OOP_POSITIONS = {'SB', 'BB', 'UTG', 'UTG1', 'UTG2', 'MP', 'HJ'}


# Pre-compute equity rank as O(1) dict lookup
_EQUITY_RANK_CACHE = {
    k: 1.0 - i / _TOTAL_HANDS
    for i, (k, _) in enumerate(_HANDS_BY_EQUITY)
}


def _equity_rank(hand_key):
    return _EQUITY_RANK_CACHE.get(hand_key, 0.0)


def _sigmoid_prob(equity_rank, threshold, sharpness=25.0):
    x = (equity_rank - threshold) * sharpness
    return 1.0 / (1.0 + math.exp(-max(-500, min(500, x))))


def _detect_opener(history, num_players=6):
    """Determine which position opened from the action history."""
    if not history or not history[0]:
        return None
    from cfr.game_state import action_order, get_position_name
    order = action_order(num_players, 0)  # preflop action order
    for i, action in enumerate(history[0]):
        if action.startswith('b') or action == 'a':
            if i < len(order):
                return get_position_name(num_players, order[i])
            return None
    return None


def adjust_thresholds(base_thresholds, vpip=None, pfr=None):
    """Adjust GTO thresholds based on observed VPIP and PFR."""
    if vpip is None and pfr is None:
        return base_thresholds

    t = dict(base_thresholds)
    gto_vpip = t['rfi'] + t['call_open']
    gto_pfr = t['rfi']

    if vpip is not None and gto_vpip > 0:
        vpip_ratio = vpip / max(gto_vpip, 0.01)
        t['call_open'] = min(0.60, t['call_open'] * vpip_ratio)
        t['call_3bet'] = min(0.40, t['call_3bet'] * vpip_ratio)

    if pfr is not None and gto_pfr > 0:
        pfr_ratio = pfr / max(gto_pfr, 0.01)
        t['rfi'] = min(0.65, t['rfi'] * pfr_ratio)
        t['3bet'] = min(0.30, t['3bet'] * pfr_ratio)

    return t


def _heuristic_action(hand_key, position, scenario, legal_actions,
                      vpip=None, pfr=None, history=None, num_players=6):
    """Heuristic preflop action using equity thresholds."""
    base = _GTO_THRESHOLDS.get(position, _GTO_THRESHOLDS['CO'])
    thresholds = adjust_thresholds(base, vpip, pfr)
    rank = _equity_rank(hand_key)

    probs = {}
    bet_actions = [a for a in legal_actions if a.startswith('b')]
    has_allin = 'a' in legal_actions

    if scenario == 'first_in':
        rfi_p = _sigmoid_prob(rank, 1.0 - thresholds['rfi'], sharpness=30.0)
        if bet_actions:
            probs['f'] = max(0, 1.0 - rfi_p)
            probs[bet_actions[0]] = rfi_p
        elif has_allin:
            probs['f'] = max(0, 1.0 - rfi_p)
            probs['a'] = rfi_p
        else:
            probs['f'] = 1.0

    elif scenario == 'vs_raise':
        # Tighten 3bet range based on opener position
        opener = _detect_opener(history, num_players)
        opener_mult = _OPENER_TIGHTNESS.get(opener, 1.0) if opener else 1.0
        effective_3bet = thresholds['3bet'] * opener_mult

        three_bet_p = _sigmoid_prob(rank, 1.0 - effective_3bet, sharpness=25.0)
        call_threshold = effective_3bet + thresholds['call_open']
        call_p = max(0, _sigmoid_prob(rank, 1.0 - call_threshold, sharpness=25.0)
                     - three_bet_p)
        fold_p = max(0, 1.0 - three_bet_p - call_p)

        if 'c' in legal_actions and (bet_actions or has_allin):
            # All-in is rare: only ~2% of 3bet range for very top hands
            allin_p = three_bet_p * 0.02 * _sigmoid_prob(rank, 0.995) if has_allin else 0.0
            bet_p = three_bet_p - allin_p

            if bet_actions:
                probs = {'f': fold_p, 'c': call_p}
                if len(bet_actions) >= 2:
                    # IP prefers smaller 3bet, OOP prefers larger
                    if position in _IP_POSITIONS:
                        probs[bet_actions[0]] = bet_p * 0.75
                        probs[bet_actions[1]] = bet_p * 0.25
                    else:
                        probs[bet_actions[0]] = bet_p * 0.25
                        probs[bet_actions[1]] = bet_p * 0.75
                else:
                    probs[bet_actions[0]] = bet_p
                if has_allin:
                    probs['a'] = allin_p
            else:
                probs = {'f': fold_p, 'c': call_p, 'a': three_bet_p}
        elif 'c' in legal_actions:
            probs = {'f': fold_p, 'c': call_p + three_bet_p}
        else:
            probs['f'] = fold_p + call_p
            if bet_actions:
                probs[bet_actions[0]] = three_bet_p
            elif has_allin:
                probs['a'] = three_bet_p

    elif scenario == 'vs_3bet':
        four_bet_p = _sigmoid_prob(rank, 1.0 - thresholds['3bet'] * 0.5, sharpness=50.0)
        call_threshold = thresholds['3bet'] * 0.5 + thresholds['call_3bet']
        call_p = max(0, _sigmoid_prob(rank, 1.0 - call_threshold, sharpness=30.0)
                     - four_bet_p)
        fold_p = max(0, 1.0 - four_bet_p - call_p)

        if 'c' in legal_actions and (bet_actions or has_allin):
            # All-in rare: ~3% of 4bet range for AA/KK
            allin_p = four_bet_p * 0.03 * _sigmoid_prob(rank, 0.995) if has_allin else 0.0
            bet_p = four_bet_p - allin_p

            if bet_actions:
                probs = {'f': fold_p, 'c': call_p}
                for a in bet_actions:
                    probs[a] = bet_p / len(bet_actions)
                if has_allin:
                    probs['a'] = allin_p
            else:
                probs = {'f': fold_p, 'c': call_p, 'a': four_bet_p}
        elif 'c' in legal_actions:
            probs = {'f': fold_p, 'c': call_p + four_bet_p}
        else:
            probs['f'] = 1.0

    else:  # vs_4bet+
        # Top hands jam, next tier calls, rest folds
        jam_p = _sigmoid_prob(rank, 0.985, sharpness=80.0)  # AA, KK
        call_p = max(0, _sigmoid_prob(rank, 0.95, sharpness=50.0) - jam_p)  # QQ, AKs, JJ
        fold_p = max(0, 1.0 - jam_p - call_p)

        if has_allin and 'c' in legal_actions:
            probs = {'f': fold_p, 'c': call_p, 'a': jam_p}
        elif has_allin:
            probs = {'f': fold_p + call_p, 'a': jam_p}
        elif 'c' in legal_actions:
            probs = {'f': fold_p, 'c': call_p + jam_p}
        else:
            probs['f'] = 1.0

    # Normalize
    total = sum(probs.values())
    if total <= 0:
        return {a: 1.0 / len(legal_actions) for a in legal_actions}
    return {a: probs.get(a, 0.0) / total for a in legal_actions}


# ---- Main API ---------------------------------------------------------------

def get_preflop_action(hand_key, position, scenario, legal_actions,
                       vpip=None, pfr=None, history=None, num_players=6):
    """
    Get preflop action probabilities for a canonical hand.

    Uses trained CFR strategy when available, falls back to heuristic.
    VPIP/PFR adjustments are applied to modify opponent behavior.

    Returns dict {action_token: probability} for all legal actions.
    """
    # Try trained strategy first (no VPIP/PFR adjustment for trained)
    if vpip is None and pfr is None and history is not None:
        trained = _get_trained_action(hand_key, position, history,
                                       legal_actions, num_players=num_players)
        if trained is not None:
            return trained

    return _heuristic_action(hand_key, position, scenario, legal_actions,
                             vpip=vpip, pfr=pfr, history=history,
                             num_players=num_players)


def get_range_for_position(position, scenario, vpip=None, pfr=None):
    """Return {hand_key: play_probability} for all 169 hands."""
    base = _GTO_THRESHOLDS.get(position, _GTO_THRESHOLDS['CO'])
    thresholds = adjust_thresholds(base, vpip, pfr)

    result = {}
    for hand_key in EQUITIES:
        rank = _equity_rank(hand_key)
        if scenario == 'first_in':
            p = _sigmoid_prob(rank, 1.0 - thresholds['rfi'])
        elif scenario == 'vs_raise':
            total = thresholds['3bet'] + thresholds['call_open']
            p = _sigmoid_prob(rank, 1.0 - total)
        elif scenario == 'vs_3bet':
            total = thresholds['3bet'] * 0.5 + thresholds['call_3bet']
            p = _sigmoid_prob(rank, 1.0 - total)
        else:
            p = _sigmoid_prob(rank, 0.95)
        result[hand_key] = p
    return result


def hand_in_range(hand_key, position, scenario, vpip=None, pfr=None):
    """Check if a hand is in the playing range."""
    base = _GTO_THRESHOLDS.get(position, _GTO_THRESHOLDS['CO'])
    thresholds = adjust_thresholds(base, vpip, pfr)
    rank = _equity_rank(hand_key)

    if scenario == 'first_in':
        return rank >= (1.0 - thresholds['rfi'] - 0.05)
    elif scenario == 'vs_raise':
        total = thresholds['3bet'] + thresholds['call_open']
        return rank >= (1.0 - total - 0.05)
    elif scenario == 'vs_3bet':
        total = thresholds['3bet'] * 0.5 + thresholds['call_3bet']
        return rank >= (1.0 - total - 0.05)
    return rank >= 0.95
