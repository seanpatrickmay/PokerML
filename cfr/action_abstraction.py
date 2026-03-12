"""
Action abstraction for HUNL CFR.

Actions are encoded as string tokens:
  'f'    = fold
  'k'    = check
  'c'    = call
  'b50'  = bet/raise 50% pot
  'b100' = bet/raise 100% pot
  'a'    = all-in
"""

MAX_RAISES_PER_STREET = 4
BET_FRACTIONS = [0.5, 1.0]


def get_bet_amount(fraction, pot):
    """Chip amount for a bet of `fraction` of the pot."""
    return round(fraction * pot, 1)


def get_legal_actions(pot, to_call, stack, raises_this_street, min_raise):
    """
    Return list of legal action tokens given the current state.

    Parameters
    ----------
    pot : float           Total pot before this action.
    to_call : float       Amount needed to call (0 if no facing bet).
    stack : float         Active player's remaining stack.
    raises_this_street : int  How many raises have occurred this street.
    min_raise : float     Minimum legal raise size.
    """
    actions = []

    if to_call > 0:
        # Facing a bet
        actions.append('f')
        if stack <= to_call:
            # Can only call all-in
            actions.append('c')
            return actions
        actions.append('c')
        # Raises allowed if under cap
        if raises_this_street < MAX_RAISES_PER_STREET:
            effective_pot = pot + to_call  # pot after calling
            for frac in BET_FRACTIONS:
                raise_amount = get_bet_amount(frac, effective_pot)
                total_cost = to_call + raise_amount
                if total_cost < stack and raise_amount >= min_raise:
                    actions.append(f'b{int(frac * 100)}')
            # All-in is always an option if we have chips
            if stack > to_call:
                actions.append('a')
    else:
        # No facing bet
        actions.append('k')
        if raises_this_street < MAX_RAISES_PER_STREET:
            for frac in BET_FRACTIONS:
                bet_amount = get_bet_amount(frac, pot)
                if bet_amount >= min_raise and bet_amount < stack:
                    actions.append(f'b{int(frac * 100)}')
            if stack > 0:
                actions.append('a')

    return actions


def action_to_chips(action, pot, to_call, stack):
    """
    Convert an action token to the number of chips the player puts in.

    Returns (chips_spent, is_allin).
    """
    if action == 'f':
        return 0, False
    if action == 'k':
        return 0, False
    if action == 'c':
        amount = min(to_call, stack)
        return amount, amount >= stack
    if action == 'a':
        return stack, True
    # Bet/raise: 'bXX' where XX is pot fraction * 100
    frac = int(action[1:]) / 100.0
    if to_call > 0:
        effective_pot = pot + to_call
        raise_size = get_bet_amount(frac, effective_pot)
        total = to_call + raise_size
        total = min(total, stack)
        return total, total >= stack
    else:
        bet_size = get_bet_amount(frac, pot)
        bet_size = min(bet_size, stack)
        return bet_size, bet_size >= stack
