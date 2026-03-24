def threshold_strategy(probs, threshold=0.10):
    """Zero out actions below threshold, renormalize."""
    filtered = [p if p >= threshold else 0.0 for p in probs]
    total = sum(filtered)
    if total <= 0:
        idx = max(range(len(probs)), key=lambda i: probs[i])
        result = [0.0] * len(probs)
        result[idx] = 1.0
        return result
    return [p / total for p in filtered]

def purify_strategy(probs):
    """Convert mixed strategy to pure (highest probability action only)."""
    idx = max(range(len(probs)), key=lambda i: probs[i])
    result = [0.0] * len(probs)
    result[idx] = 1.0
    return result


def blend_strategies(resolved, blueprint, resolve_weight=0.7):
    """Blend resolved strategy with blueprint for robustness.

    A pure resolved strategy may be wrong if the opponent range estimate
    was off. Blending with the blueprint provides a safety net.

    Args:
        resolved: list of probabilities from real-time solving
        blueprint: list of probabilities from offline training
        resolve_weight: weight for resolved strategy (0.0-1.0)

    Returns:
        Blended probability list summing to 1.0
    """
    if not resolved or not blueprint:
        return resolved or blueprint or []

    n = min(len(resolved), len(blueprint))
    if n == 0:
        return resolved or blueprint

    bp_weight = 1.0 - resolve_weight
    blended = [
        resolve_weight * resolved[i] + bp_weight * blueprint[i]
        for i in range(n)
    ]

    total = sum(blended)
    if total > 0:
        return [p / total for p in blended]
    return [1.0 / n] * n


def sharpen_strategy(probs, temperature=0.5):
    """Softmax sharpening: amplifies high-probability actions smoothly.

    Unlike hard thresholding, this preserves all actions in the strategy
    but pushes probability mass toward the dominant actions. Lower
    temperature = more aggressive sharpening (0.1 ≈ pure, 1.0 = no change).
    """
    import math
    if not probs or all(p == 0 for p in probs):
        return probs
    max_p = max(probs)
    # Shift for numerical stability, then apply temperature
    exps = [math.exp((p - max_p) / max(temperature, 0.01)) for p in probs]
    total = sum(exps)
    if total <= 0:
        return probs
    return [e / total for e in exps]


def adaptive_threshold(probs, num_actions_taken=0):
    """Hybrid strategy cleanup: soft sharpening + hard floor.

    Early in a hand: gentle sharpening preserves mixed play.
    Later in a hand: moderate sharpening + hard threshold for clarity.

    Uses conservative temperatures to avoid collapsing mixed strategies
    (important for OOP play where checking is strategically vital).
    """
    # Conservative temperature — preserve mixed play on later streets
    base_temp = 0.7
    temp = max(0.35, base_temp - 0.03 * num_actions_taken)
    probs = sharpen_strategy(probs, temperature=temp)

    # Hard floor only deep in the hand to remove residual noise
    if num_actions_taken >= 8:
        threshold = min(0.10, 0.03 + 0.01 * num_actions_taken)
        probs = threshold_strategy(probs, threshold=threshold)

    return probs
