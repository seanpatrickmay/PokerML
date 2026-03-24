"""
Pseudo-harmonic action translation for off-tree bet sizes.

When a human bets a size not in our abstraction, this maps it to
the nearest tree sizes with probabilities that satisfy game-theoretic
axioms (scale invariance, action robustness, monotonicity).

Reference: Ganzfried & Sandholm, "Action Translation in Extensive-Form
Games with Large Action Spaces", IJCAI 2013.
"""


def translate_action(actual_frac, tree_fracs):
    """
    Map an off-tree bet (as fraction of pot) to tree actions.

    Args:
        actual_frac: The actual bet as a fraction of the pot (e.g., 0.73)
        tree_fracs: Sorted list of bet fractions in the abstraction tree

    Returns:
        List of (tree_frac, probability) pairs summing to 1.0
    """
    if not tree_fracs:
        return []

    tree_fracs = sorted(tree_fracs)

    for tf in tree_fracs:
        if abs(actual_frac - tf) < 0.01:
            return [(tf, 1.0)]

    if actual_frac <= tree_fracs[0]:
        return [(tree_fracs[0], 1.0)]

    if actual_frac >= tree_fracs[-1]:
        return [(tree_fracs[-1], 1.0)]

    lo = None
    hi = None
    for i in range(len(tree_fracs) - 1):
        if tree_fracs[i] < actual_frac < tree_fracs[i + 1]:
            lo = tree_fracs[i]
            hi = tree_fracs[i + 1]
            break

    if lo is None or hi is None:
        return [(tree_fracs[0], 1.0)]

    # Pseudo-harmonic mapping formula:
    # P(lo) = (hi - x)(lo + 1) / [(hi - x)(lo + 1) + (x - lo)(hi + 1)]
    num = (hi - actual_frac) * (lo + 1.0)
    den = num + (actual_frac - lo) * (hi + 1.0)

    if den < 1e-12:
        return [(lo, 0.5), (hi, 0.5)]

    p_lo = num / den
    p_hi = 1.0 - p_lo

    return [(lo, p_lo), (hi, p_hi)]


def get_tree_fractions(action_tokens):
    """Extract pot-fraction bet sizes from action token list."""
    fracs = []
    for tok in action_tokens:
        if tok.startswith('b'):
            try:
                pct = int(tok[1:])
                fracs.append(pct / 100.0)
            except ValueError:
                continue
    return sorted(fracs)


def translate_bet_to_actions(actual_pct, action_tokens):
    """
    High-level: given an actual bet percentage and available actions,
    return the action(s) to use with probabilities.
    """
    tree_fracs = get_tree_fractions(action_tokens)
    if not tree_fracs:
        return []

    mappings = translate_action(actual_pct / 100.0, tree_fracs)

    result = []
    for frac, prob in mappings:
        pct = round(frac * 100)
        token = f"b{pct}"
        if token in action_tokens:
            result.append((token, prob))

    return result
