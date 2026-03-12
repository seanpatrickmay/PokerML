"""
Card abstraction for HUNL CFR.

Preflop:  169 canonical hands clustered into N buckets by equity (k-means).
Postflop: hand-strength percentile bucketing via a precomputed score CDF.

The score CDF lets us convert any phevaluator score to a hand-strength
percentile with a single evaluation + binary search — no opponent enumeration.
"""

import bisect
import json
import os
import random

import numpy as np
from phevaluator import evaluate_cards
from sklearn.cluster import KMeans

_CDF_SAMPLES = 100_000


class CardAbstraction:
    def __init__(self, num_preflop_buckets=15, num_postflop_buckets=20,
                 flop_samples=5, turn_samples=3):
        self.num_preflop_buckets = num_preflop_buckets
        self.num_postflop_buckets = num_postflop_buckets
        self.flop_samples = flop_samples
        self.turn_samples = turn_samples
        self.preflop_key_to_bucket = {}
        self._build_preflop_table()
        self._score_cdf = self._build_score_cdf()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_bucket(self, hand, visible_board):
        if len(visible_board) == 0:
            return self._preflop_bucket(hand)

        if len(visible_board) == 5:
            # River: single evaluation, exact
            score = evaluate_cards(*visible_board, *hand)
            return self._score_to_bucket(score)

        # Flop / turn: sample board completions, average percentile
        used = set(visible_board) | set(hand)
        available = [c for c in range(52) if c not in used]
        remaining = 5 - len(visible_board)
        n = self.flop_samples if len(visible_board) == 3 else self.turn_samples

        total_pct = 0.0
        for _ in range(n):
            extra = random.sample(available, remaining)
            score = evaluate_cards(*visible_board, *extra, *hand)
            total_pct += self._score_to_percentile(score)

        avg = total_pct / n
        return min(int(avg * self.num_postflop_buckets),
                   self.num_postflop_buckets - 1)

    # ------------------------------------------------------------------
    # Score CDF (one-time precomputation)
    # ------------------------------------------------------------------

    def _build_score_cdf(self):
        """Sample random 7-card hands to build an empirical score CDF."""
        deck = list(range(52))
        scores = []
        for _ in range(_CDF_SAMPLES):
            random.shuffle(deck)
            scores.append(evaluate_cards(deck[0], deck[1], deck[2],
                                         deck[3], deck[4], deck[5], deck[6]))
        scores.sort()
        return scores

    def _score_to_percentile(self, score):
        """Lower phevaluator score = stronger hand = higher percentile."""
        idx = bisect.bisect_left(self._score_cdf, score)
        return 1.0 - idx / len(self._score_cdf)

    def _score_to_bucket(self, score):
        pct = self._score_to_percentile(score)
        return min(int(pct * self.num_postflop_buckets),
                   self.num_postflop_buckets - 1)

    # ------------------------------------------------------------------
    # Preflop (unchanged — lookup table via k-means)
    # ------------------------------------------------------------------

    def _build_preflop_table(self):
        json_path = os.path.join(os.path.dirname(__file__), '..', 'preFlopEquities.json')
        with open(json_path) as f:
            equities = json.load(f)

        keys = list(equities.keys())
        vals = np.array([equities[k] for k in keys]).reshape(-1, 1)

        km = KMeans(n_clusters=self.num_preflop_buckets, random_state=42, n_init=10)
        labels = km.fit_predict(vals)

        order = np.argsort(km.cluster_centers_.flatten())
        relabel = {old: new for new, old in enumerate(order)}
        self.preflop_key_to_bucket = {k: relabel[l] for k, l in zip(keys, labels)}

    def _hand_to_preflop_key(self, hand):
        r0, r1 = hand[0] // 4, hand[1] // 4
        s0, s1 = hand[0] % 4, hand[1] % 4
        if r0 < r1:
            r0, r1 = r1, r0
            s0, s1 = s1, s0
        if r0 == r1:
            return f"{r0} {r1}"
        return f"{r0} {r1} s" if s0 == s1 else f"{r0} {r1} o"

    def _preflop_bucket(self, hand):
        return self.preflop_key_to_bucket.get(self._hand_to_preflop_key(hand), 0)
