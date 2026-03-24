"""
Potential-aware card abstraction using Earth Mover's Distance.

Instead of clustering hands by current equity alone, this computes
the distribution of future hand-strength buckets for each hand and
clusters hands based on the similarity of these distributions using
EMD (Wasserstein distance).

For 1D ordered histograms, EMD equals the L1 distance between CDFs,
so we convert histograms to CDFs and use K-means for efficient clustering.

Reference: Johanson et al., "Evaluating State-Space Abstractions in
Extensive-Form Games", AAMAS 2013.
"""
import random

import numpy as np
from sklearn.cluster import KMeans

from cfr.evaluator import evaluate_hand


def compute_hand_features(hand, board, num_future_buckets=20, samples=100):
    """
    Compute the distribution of future hand-strength buckets.

    For a hand on the flop/turn, samples random completions and computes
    what hand-strength bucket the hand falls into on each completion.
    Returns a histogram over future buckets.

    Args:
        hand: Tuple of 2 card indices
        board: Tuple of 3 (flop) or 4 (turn) card indices
        num_future_buckets: Number of histogram bins
        samples: Number of random completions to sample

    Returns:
        List of floats (histogram) summing to ~1.0
    """
    used = set(hand) | set(board)
    deck = [c for c in range(52) if c not in used]

    cards_needed = 5 - len(board)
    histogram = [0.0] * num_future_buckets

    count = 0
    for _ in range(samples):
        completion = random.sample(deck, cards_needed)
        full_board = list(board) + completion

        try:
            score = evaluate_hand(full_board, hand)
            percentile = 1.0 - (score / 7463.0)
            bucket = min(int(percentile * num_future_buckets),
                         num_future_buckets - 1)
            histogram[bucket] += 1.0
            count += 1
        except Exception:
            continue

    if count > 0:
        histogram = [h / count for h in histogram]

    return histogram


def cluster_hands_emd(features, num_clusters=50):
    """
    Cluster hands using Earth Mover's Distance (EMD / Wasserstein-1).

    For 1D ordered histograms, EMD = L1 distance between CDFs.
    We convert histograms to CDFs and use K-means for approximation.

    Args:
        features: np.ndarray of shape (num_hands, num_bins)
        num_clusters: Number of clusters

    Returns:
        List of cluster labels (length = num_hands)
    """
    cdfs = np.cumsum(features, axis=1)

    kmeans = KMeans(n_clusters=min(num_clusters, len(cdfs)),
                    random_state=42, n_init=10, max_iter=300)
    labels = kmeans.fit_predict(cdfs)
    return labels.tolist()
