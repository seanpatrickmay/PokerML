"""
Action abstraction for CFR with geometric bet sizing.

Actions are encoded as string tokens:
  'f'    = fold
  'k'    = check
  'c'    = call
  'bXX'  = bet/raise XX% of pot  (e.g. b33, b67, b100, b150)
  'a'    = all-in
"""

# Heads-up: 3 bet sizes, 4 max raises
HU_MAX_RAISES = 4
HU_BET_FRACTIONS = {
    0: [0.75, 1.00, 1.75],        # preflop  (2.5x / 3x / 4.5x opens)
}

# 6-max: per-street max raises
SIXMAX_MAX_RAISES = {
    0: 5,   # preflop: open, 3bet, 4bet, 5bet (BB counts as 1)
    1: 3,   # flop
    2: 3,   # turn
    3: 3,   # river
}

# 9-max: same postflop structure, tighter preflop cap
NINEMAX_MAX_RAISES = {
    0: 5,   # preflop: open, 3bet, 4bet, 5bet (BB counts as 1)
    1: 3,   # flop
    2: 3,   # turn
    3: 3,   # river
}

# Preflop multiplier-based sizing (multiples of current max_bet)
# raises_this_street == 1 means facing BB (opening), 2 means facing open (3bet), etc.
PREFLOP_RAISE_MULTS = {
    1: [2.5],          # open: raise to 2.5x BB
    2: [3.0, 4.5],     # 3bet: 3x (IP) / 4.5x (OOP) — solver learns which to use
    3: [3.0],          # 4bet: 3x the 3bet
}

# Compatibility alias for assistant/solver_bridge.py
SIXMAX_BET_FRACTIONS = {
    0: [0.75, 1.00, 1.75],
    1: [0.33, 0.50, 1.00],
    2: [0.50, 1.00],
    3: [0.50, 1.00],
}

# Active config (set by set_num_players)
MAX_RAISES_PER_STREET = HU_MAX_RAISES
USE_PREFLOP_MULTS = False


def set_num_players(num_players):
    """Switch action abstraction between heads-up, 6-max, and 9-max presets."""
    global MAX_RAISES_PER_STREET, USE_PREFLOP_MULTS
    if num_players > 6:
        MAX_RAISES_PER_STREET = NINEMAX_MAX_RAISES
        USE_PREFLOP_MULTS = True
    elif num_players > 2:
        MAX_RAISES_PER_STREET = SIXMAX_MAX_RAISES
        USE_PREFLOP_MULTS = True
    else:
        MAX_RAISES_PER_STREET = HU_MAX_RAISES
        USE_PREFLOP_MULTS = False


def get_bet_amount(fraction, pot):
    return round(fraction * pot, 1)


# ---- Geometric bet sizing ----

def _geometric_fractions(effective_pot, stack, street):
    """Compute 3 geometric bet/raise sizes as pot fractions.

    Picks a "geometric" size that, if bet each remaining street, gets
    all-in by the river. Then generates a small, medium, and large
    option around it. Sizes are quantized to 5% increments for clean
    action tokens and better caching.
    """
    if effective_pot <= 0 or stack <= 0:
        return [1.0]

    streets_left = max(1, 4 - street)  # flop=3, turn=2, river=1
    spr = stack / effective_pot

    if spr <= 0.5:
        # Very shallow — just pot
        return [1.0]

    # Geometric fraction: (1 + 2f)^N = 1 + 2*spr
    # => f = ((1 + 2*spr)^(1/N) - 1) / 2
    geo = ((1 + 2 * spr) ** (1.0 / streets_left) - 1) / 2

    # 5 sizes: tiny (blocking), small, geometric, large, overbet
    raw = [
        geo * 0.25,   # tiny / blocking bet
        geo * 0.5,    # small
        geo,          # geometric (default)
        geo * 1.5,    # large
        geo * 2.5,    # overbet
    ]

    # Quantize to nearest 5%, clamp to [20%, 300%]
    pcts = set()
    for f in raw:
        pct = max(20, min(round(f * 20) * 5, 300))
        pcts.add(pct)

    # Ensure sizes are at least 10 percentage points apart
    result = []
    for pct in sorted(pcts):
        if not result or pct >= result[-1] + 10:
            result.append(pct)

    return [p / 100 for p in result]


_LEGAL_ACTION_CACHE = {}
_LEGAL_ACTION_CACHE_MAX = 500_000


def get_legal_actions(pot, to_call, stack, raises_this_street, min_raise,
                      street=0, max_bet=None):
    """Return list of legal action tokens given the current state."""
    key = (pot, to_call, stack, raises_this_street, min_raise, street, max_bet)
    cached = _LEGAL_ACTION_CACHE.get(key)
    if cached is not None:
        return cached

    if isinstance(MAX_RAISES_PER_STREET, dict):
        max_raises = MAX_RAISES_PER_STREET.get(street, 3)
    else:
        max_raises = MAX_RAISES_PER_STREET

    use_mults = (USE_PREFLOP_MULTS and street == 0
                 and max_bet is not None and max_bet > 0 and to_call > 0)
    actions = []

    if to_call > 0:
        actions.append('f')
        if stack <= to_call:
            actions.append('c')
            if len(_LEGAL_ACTION_CACHE) < _LEGAL_ACTION_CACHE_MAX:
                _LEGAL_ACTION_CACHE[key] = actions
            return actions
        actions.append('c')
        if raises_this_street < max_raises:
            effective_pot = pot + to_call
            if use_mults:
                mults = PREFLOP_RAISE_MULTS.get(raises_this_street, [])
                # SB open (to_call < max_bet because SB already posted): 3x
                if raises_this_street == 1 and to_call < max_bet:
                    mults = [3.0]
                for mult in mults:
                    raise_to = mult * max_bet
                    raise_amount = raise_to - max_bet
                    total_cost = to_call + raise_amount
                    if total_cost < stack and raise_amount >= min_raise:
                        frac_pct = int(round(raise_amount / effective_pot * 100))
                        actions.append(f'b{frac_pct}')
            else:
                # Postflop: geometric sizing; HU preflop: fixed fractions
                if street > 0:
                    fractions = _geometric_fractions(effective_pot, stack, street)
                else:
                    fractions = HU_BET_FRACTIONS.get(0, [1.0])
                for frac in fractions:
                    raise_amount = get_bet_amount(frac, effective_pot)
                    total_cost = to_call + raise_amount
                    if total_cost < stack and raise_amount >= min_raise:
                        actions.append(f'b{int(round(frac * 100))}')
            if stack > to_call:
                actions.append('a')
    else:
        actions.append('k')
        if raises_this_street < max_raises:
            # Postflop: geometric sizing; HU preflop: fixed fractions
            if street > 0:
                fractions = _geometric_fractions(pot, stack, street)
            else:
                fractions = HU_BET_FRACTIONS.get(0, [1.0])
            for frac in fractions:
                bet_amount = get_bet_amount(frac, pot)
                if bet_amount >= min_raise and bet_amount < stack:
                    actions.append(f'b{int(round(frac * 100))}')
            if stack > 0:
                actions.append('a')

    if len(_LEGAL_ACTION_CACHE) < _LEGAL_ACTION_CACHE_MAX:
        _LEGAL_ACTION_CACHE[key] = actions
    return actions


def action_to_chips(action, pot, to_call, stack):
    """Convert an action token to (chips_spent, is_allin)."""
    if action == 'f':
        return 0, False
    if action == 'k':
        return 0, False
    if action == 'c':
        amount = min(to_call, stack)
        return amount, amount >= stack
    if action == 'a':
        return stack, True
    frac = int(action[1:]) / 100.0
    if to_call > 0:
        effective_pot = pot + to_call
        raise_size = get_bet_amount(frac, effective_pot)
        total = min(to_call + raise_size, stack)
        return total, total >= stack
    else:
        bet_size = min(get_bet_amount(frac, pot), stack)
        return bet_size, bet_size >= stack
