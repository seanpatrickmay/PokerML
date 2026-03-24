"""
Card abstraction for CFR.

Preflop:  169 canonical hands clustered into N buckets by equity (k-means).
Postflop: hand-strength percentile bucketing via a precomputed score CDF.

The score CDF lets us convert any phevaluator score to a hand-strength
percentile with a single evaluation + binary search — no opponent enumeration.

Supports both heads-up and 6-max via separate preflop equity files.
"""

import bisect
import json
import os
import random

import numpy as np
from phevaluator import evaluate_cards
from sklearn.cluster import KMeans

from cfr.emd_abstraction import compute_hand_features, cluster_hands_emd

from config import (
    CFR_PREFLOP_BUCKETS, CFR_POSTFLOP_BUCKETS,
    PREFLOP_BUCKETS_2MAX, PREFLOP_BUCKETS_6MAX, PREFLOP_BUCKETS_9MAX,
    POSTFLOP_BUCKETS_2MAX, POSTFLOP_BUCKETS_6MAX, POSTFLOP_BUCKETS_9MAX,
    FLOP_SAMPLES, TURN_SAMPLES, USE_EMD_BUCKETING,
    USE_OCHS_RIVER, OCHS_OPPONENT_SAMPLES,
)

_CDF_SAMPLES = 100_000

# Default bucket counts per player count
_DEFAULT_PREFLOP_BUCKETS = {
    2: PREFLOP_BUCKETS_2MAX,
    6: PREFLOP_BUCKETS_6MAX,
    9: PREFLOP_BUCKETS_9MAX,
}
_DEFAULT_POSTFLOP_BUCKETS = {
    2: POSTFLOP_BUCKETS_2MAX,
    6: POSTFLOP_BUCKETS_6MAX,
    9: POSTFLOP_BUCKETS_9MAX,
}


class CardAbstraction:
    def __init__(self, num_preflop_buckets=None, num_postflop_buckets=None,
                 flop_samples=None, turn_samples=None, num_players=2,
                 use_emd=None, use_ochs_river=None):
        self.num_players = num_players

        if num_preflop_buckets and num_preflop_buckets > 0:
            self.num_preflop_buckets = num_preflop_buckets
        elif CFR_PREFLOP_BUCKETS > 0:
            self.num_preflop_buckets = CFR_PREFLOP_BUCKETS
        else:
            self.num_preflop_buckets = _DEFAULT_PREFLOP_BUCKETS.get(num_players, PREFLOP_BUCKETS_6MAX)

        if num_postflop_buckets and num_postflop_buckets > 0:
            self.num_postflop_buckets = num_postflop_buckets
        elif CFR_POSTFLOP_BUCKETS > 0:
            self.num_postflop_buckets = CFR_POSTFLOP_BUCKETS
        else:
            self.num_postflop_buckets = _DEFAULT_POSTFLOP_BUCKETS.get(num_players, POSTFLOP_BUCKETS_6MAX)

        self.flop_samples = flop_samples or FLOP_SAMPLES
        self.turn_samples = turn_samples or TURN_SAMPLES
        self.preflop_key_to_bucket = {}
        self._build_preflop_table()
        self._score_cdf = self._build_score_cdf()
        self.use_emd = use_emd if use_emd is not None else USE_EMD_BUCKETING
        self.use_ochs_river = use_ochs_river if use_ochs_river is not None else USE_OCHS_RIVER
        self._emd_cache = {}
        self._ochs_cache = {}
        self._postflop_cache = {}
        self._postflop_cache_max = 500_000

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_bucket(self, hand, visible_board):
        if len(visible_board) == 0:
            return self._preflop_bucket(hand)

        if len(visible_board) == 5:
            if self.use_ochs_river:
                return self._ochs_river_bucket(hand, visible_board)
            score = evaluate_cards(*visible_board, *hand)
            return self._score_to_bucket(score)

        # EMD-based bucketing (potential-aware)
        if self.use_emd and len(visible_board) in (3, 4):
            return self._emd_bucket(hand, visible_board)

        # Flop / turn: sample board completions, average percentile
        # Check cache first (huge speedup during CFR training)
        cache_key = (hand[0], hand[1], *visible_board)
        cached = self._postflop_cache.get(cache_key)
        if cached is not None:
            return cached

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
        bucket = min(int(avg * self.num_postflop_buckets),
                     self.num_postflop_buckets - 1)

        if len(self._postflop_cache) < self._postflop_cache_max:
            self._postflop_cache[cache_key] = bucket
        return bucket

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
    # OCHS river bucketing
    # ------------------------------------------------------------------

    def _ochs_river_bucket(self, hand, board):
        """Bucket river hands using Outcome Cluster Hand Strength.

        Instead of a single hand-strength percentile, computes a stratified
        equity profile against opponent hands grouped by strength.  This
        separates pure air (0% equity everywhere) from weak-but-real hands
        that beat some portion of the opponent range.

        The combined score blends overall equity with "equity floor" (win rate
        vs the strongest 25% of opponents) and "equity ceiling" (win rate vs
        the weakest 25%).
        """
        cache_key = (hand[0], hand[1], *board)
        cached = self._ochs_cache.get(cache_key)
        if cached is not None:
            return cached

        used = set(hand) | set(board)
        available = [c for c in range(52) if c not in used]

        hero_score = evaluate_cards(*board, *hand)

        # Sample opponent hands and evaluate them
        num_available = len(available)
        num_possible = num_available * (num_available - 1) // 2
        n_samples = min(OCHS_OPPONENT_SAMPLES, num_possible)

        opp_results = []
        if n_samples >= num_possible:
            # Full enumeration (45 remaining cards -> 990 combos, but we cap
            # at OCHS_OPPONENT_SAMPLES which defaults to 200, so this branch
            # triggers only when very few cards remain).
            for i in range(num_available):
                for j in range(i + 1, num_available):
                    opp_score = evaluate_cards(*board, available[i], available[j])
                    opp_results.append(opp_score)
        else:
            seen = set()
            attempts = 0
            max_attempts = n_samples * 3
            while len(opp_results) < n_samples and attempts < max_attempts:
                attempts += 1
                idx = random.sample(range(num_available), 2)
                pair = (min(idx[0], idx[1]), max(idx[0], idx[1]))
                if pair in seen:
                    continue
                seen.add(pair)
                opp_score = evaluate_cards(*board, available[pair[0]], available[pair[1]])
                opp_results.append(opp_score)

        if not opp_results:
            bucket = self._score_to_bucket(hero_score)
            if len(self._ochs_cache) < self._postflop_cache_max:
                self._ochs_cache[cache_key] = bucket
            return bucket

        # Sort opponent scores (lower = stronger for phevaluator)
        opp_results.sort()
        total = len(opp_results)

        # Partition opponents into quartiles by strength.
        # opp_results is sorted ascending (strongest first for phevaluator).
        q1 = max(1, total // 4)       # strongest 25%
        q3 = max(q1 + 1, total * 3 // 4)  # weakest 25% start index

        # Win rate vs each segment (hero wins when hero_score < opp_score)
        def _win_rate(scores):
            if not scores:
                return 0.0
            wins = sum(1 for s in scores if hero_score < s)
            ties = sum(1 for s in scores if hero_score == s)
            return (wins + 0.5 * ties) / len(scores)

        equity = _win_rate(opp_results)
        equity_floor = _win_rate(opp_results[:q1])     # vs strongest opponents
        equity_ceiling = _win_rate(opp_results[q3:])    # vs weakest opponents

        combined = 0.6 * equity + 0.2 * equity_floor + 0.2 * equity_ceiling
        bucket = min(int(combined * self.num_postflop_buckets),
                     self.num_postflop_buckets - 1)

        if len(self._ochs_cache) < self._postflop_cache_max:
            self._ochs_cache[cache_key] = bucket
        return bucket

    # ------------------------------------------------------------------
    # EMD-based bucketing
    # ------------------------------------------------------------------

    def _emd_bucket(self, hand, visible_board):
        """Bucket using EMD-based equity distribution features."""
        # Cache key: canonical hand + board
        cache_key = (tuple(sorted(hand)), tuple(visible_board))
        cached = self._emd_cache.get(cache_key)
        if cached is not None:
            return cached

        features = compute_hand_features(
            hand, visible_board,
            num_future_buckets=20,
            samples=self.flop_samples if len(visible_board) == 3 else self.turn_samples,
        )
        # Convert to percentile using feature centroid
        # Weight features by bucket index to get expected future strength
        expected_strength = sum(i / 20.0 * f for i, f in enumerate(features))
        bucket = min(int(expected_strength * self.num_postflop_buckets),
                     self.num_postflop_buckets - 1)

        # Cache (limit size to prevent memory bloat)
        if len(self._emd_cache) < 500_000:
            self._emd_cache[cache_key] = bucket
        return bucket

    # ------------------------------------------------------------------
    # Preflop (unchanged — lookup table via k-means)
    # ------------------------------------------------------------------

    def _build_preflop_table(self):
        if self.num_players > 6:
            fname = 'preFlopEquities9max.json'
        elif self.num_players > 2:
            fname = 'preFlopEquities6max.json'
        else:
            fname = 'preFlopEquities.json'
        json_path = os.path.join(os.path.dirname(__file__), '..', fname)
        with open(json_path) as f:
            equities = json.load(f)

        keys = list(equities.keys())
        vals = np.array([equities[k] for k in keys]).reshape(-1, 1)

        km = KMeans(n_clusters=self.num_preflop_buckets, random_state=42, n_init=1)
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
