"""
Action abstraction for HUNL CFR with street-specific bet sizings.

Actions are encoded as string tokens:
  'f'    = fold
  'k'    = check
  'c'    = call
  'bXX'  = bet/raise XX% of pot  (e.g. b33, b67, b100, b150)
  'a'    = all-in
"""

MAX_RAISES_PER_STREET = 4

# Different streets call for different sizing options.
# Preflop: standard open / 3-bet sizes
# Preflop fractions chosen to produce standard BB-multiple opens:
#   0.75 → 2.5x    1.0 → 3x    1.75 → 4.5x
# These also give reasonable 3-bet / 4-bet sizes when applied to larger pots.
# Postflop fractions from standard GTO solver configurations.
STREET_BET_FRACTIONS = {
    0: [0.75, 1.00, 1.75],        # preflop  (2.5x / 3x / 4.5x opens)
    1: [0.33, 0.67, 1.00],        # flop     (probe / standard / pot)
    2: [0.67, 1.00, 1.50],        # turn     (standard / pot / overbet)
    3: [0.50, 1.00, 2.00],        # river    (thin / pot / big overbet)
}


def get_bet_amount(fraction, pot):
    return round(fraction * pot, 1)


def get_legal_actions(pot, to_call, stack, raises_this_street, min_raise,
                      street=0):
    """Return list of legal action tokens given the current state."""
    fractions = STREET_BET_FRACTIONS.get(street, STREET_BET_FRACTIONS[0])
    actions = []

    if to_call > 0:
        actions.append('f')
        if stack <= to_call:
            actions.append('c')
            return actions
        actions.append('c')
        if raises_this_street < MAX_RAISES_PER_STREET:
            effective_pot = pot + to_call
            for frac in fractions:
                raise_amount = get_bet_amount(frac, effective_pot)
                total_cost = to_call + raise_amount
                if total_cost < stack and raise_amount >= min_raise:
                    actions.append(f'b{int(frac * 100)}')
            if stack > to_call:
                actions.append('a')
    else:
        actions.append('k')
        if raises_this_street < MAX_RAISES_PER_STREET:
            for frac in fractions:
                bet_amount = get_bet_amount(frac, pot)
                if bet_amount >= min_raise and bet_amount < stack:
                    actions.append(f'b{int(frac * 100)}')
            if stack > 0:
                actions.append('a')

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
