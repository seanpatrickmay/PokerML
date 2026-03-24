"""
Frequency-based opponent model for exploitative play.

Tracks per-opponent action frequencies across situations and computes
strategy adjustments to exploit observed deviations from GTO play.

Uses a Bayesian approach: starts with GTO prior, updates with observations,
and only deviates from GTO when confidence (sample count) is sufficient.
"""
from collections import defaultdict
import math

_GTO_BASELINES = {
    'fold_to_cbet': 0.45,
    'fold_to_3bet': 0.55,
    'cbet_freq': 0.55,
    'raise_freq': 0.08,
    'vpip': 0.25,
    'pfr': 0.20,
    'wtsd': 0.28,
    'agg': 0.40,
}

MIN_SAMPLES = 20


class OpponentModel:
    def __init__(self):
        self._actions = defaultdict(lambda: defaultdict(int))
        self._hands = defaultdict(int)
        self._street_actions = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
        self._stats = defaultdict(dict)

    def record_action(self, seat, street, situation, action, position=None):
        key = (seat, street, situation)
        self._street_actions[seat][(street, situation)][action] += 1
        self._actions[seat][action] += 1
        if street == 0 and situation == 'first_in':
            self._hands[seat] += 1

        # Also track per-street stats
        if 'by_street' not in self._stats[seat]:
            self._stats[seat]['by_street'] = {}
        street_key = str(street)
        if street_key not in self._stats[seat]['by_street']:
            self._stats[seat]['by_street'][street_key] = {'actions': 0, 'folds': 0, 'bets': 0}
        sd = self._stats[seat]['by_street'][street_key]
        sd['actions'] += 1
        if action == 'fold':
            sd['folds'] += 1
        elif action in ('bet', 'raise'):
            sd['bets'] += 1

    def record_hand(self, seat):
        self._hands[seat] += 1

    def get_stats(self, seat):
        total_hands = self._hands.get(seat, 0)
        actions = self._actions.get(seat, {})
        total_actions = sum(actions.values()) or 1

        return {
            'hands': total_hands,
            'vpip': actions.get('call', 0) + actions.get('raise', 0) + actions.get('bet', 0),
            'fold_count': actions.get('fold', 0),
            'raise_count': actions.get('raise', 0) + actions.get('bet', 0),
            'call_count': actions.get('call', 0),
            'check_count': actions.get('check', 0),
            'agg_freq': (actions.get('raise', 0) + actions.get('bet', 0)) / total_actions,
            'fold_to_cbet': self._get_situation_freq(seat, 1, 'vs_bet', 'fold'),
            'fold_to_3bet': self._get_situation_freq(seat, 0, 'vs_raise', 'fold'),
            'by_street': self._stats.get(seat, {}).get('by_street', {}),
        }

    def _get_situation_freq(self, seat, street, situation, action):
        sit_actions = self._street_actions.get(seat, {}).get((street, situation), {})
        total = sum(sit_actions.values())
        if total < 1:
            return _GTO_BASELINES.get(f'{action}_to_cbet' if situation == 'vs_bet' else action, 0.5)
        return sit_actions.get(action, 0) / total

    def get_exploit_adjustments(self, seat):
        stats = self.get_stats(seat)
        total_samples = sum(self._actions.get(seat, {}).values())
        confidence = 1.0 / (1.0 + math.exp(-(total_samples - MIN_SAMPLES) / 8.0))

        adjustments = {
            'bet_frequency_mult': 1.0,
            'call_threshold_mult': 1.0,
            'bluff_frequency_mult': 1.0,
            'raise_frequency_mult': 1.0,
        }

        if total_samples < 5:
            return adjustments

        fold_to_cbet = stats.get('fold_to_cbet', _GTO_BASELINES['fold_to_cbet'])
        fold_deviation = fold_to_cbet - _GTO_BASELINES['fold_to_cbet']
        adjustments['bet_frequency_mult'] = 1.0 + confidence * fold_deviation * 2.0
        adjustments['bluff_frequency_mult'] = 1.0 + confidence * fold_deviation * 3.0

        agg = stats.get('agg_freq', _GTO_BASELINES['agg'])
        agg_deviation = agg - _GTO_BASELINES['agg']
        adjustments['call_threshold_mult'] = 1.0 + confidence * agg_deviation * 1.5

        fold_freq = stats.get('fold_count', 0) / max(total_samples, 1)
        if fold_freq < 0.3:
            adjustments['raise_frequency_mult'] = 1.0 + confidence * 0.3
            adjustments['bluff_frequency_mult'] = max(0.3, 1.0 - confidence * 0.5)

        for key in adjustments:
            adjustments[key] = max(0.3, min(2.5, adjustments[key]))

        return adjustments

    def _confidence(self, hands):
        """Sigmoid confidence based on number of hands observed."""
        return 1.0 / (1.0 + math.exp(-(hands - MIN_SAMPLES) / 8.0))

    def get_street_adjustments(self, seat, street):
        """Get exploit adjustments specific to a street.

        Returns dict with multipliers for different action types,
        bounded by confidence to prevent over-exploitation.
        """
        stats = self.get_stats(seat)
        total_samples = sum(self._actions.get(seat, {}).values())
        confidence = self._confidence(total_samples)

        # Street-specific baselines
        street_baselines = {
            0: {'fold_rate': 0.60, 'agg': 0.25},  # preflop: most fold
            1: {'fold_rate': 0.45, 'agg': 0.40},  # flop: standard cbet
            2: {'fold_rate': 0.50, 'agg': 0.35},  # turn: tighter
            3: {'fold_rate': 0.55, 'agg': 0.30},  # river: polarized
        }
        baseline = street_baselines.get(street, {'fold_rate': 0.45, 'agg': 0.40})

        if total_samples < 10:
            return {
                'bet_frequency_mult': 1.0,
                'call_threshold_mult': 1.0,
                'bluff_frequency_mult': 1.0,
                'raise_frequency_mult': 1.0,
                'max_deviation': 0.0,
            }

        # Compute opponent's fold rate and aggression for this street
        street_key = str(street)
        street_data = self._stats.get(seat, {}).get('by_street', {}).get(street_key, {})
        total_actions = street_data.get('actions', 0)
        if total_actions > 0:
            opp_fold = street_data.get('folds', 0) / total_actions
            opp_agg = street_data.get('bets', 0) / total_actions
        else:
            opp_fold = baseline['fold_rate']
            opp_agg = baseline['agg']

        # Adjust: bet more vs folders, less vs callers
        fold_diff = opp_fold - baseline['fold_rate']
        agg_diff = opp_agg - baseline['agg']

        # Bound deviation by confidence (max 30% deviation from GTO)
        max_dev = min(0.30, confidence * 0.30)

        bet_mult = 1.0 + min(max_dev, max(-max_dev, fold_diff * 2.0)) * confidence
        call_mult = 1.0 - min(max_dev, max(-max_dev, agg_diff * 1.5)) * confidence
        bluff_mult = 1.0 + min(max_dev, max(-max_dev, fold_diff * 1.5)) * confidence
        raise_mult = 1.0 - min(max_dev, max(-max_dev, (opp_agg - 0.4) * 1.0)) * confidence

        return {
            'bet_frequency_mult': max(0.5, min(1.5, bet_mult)),
            'call_threshold_mult': max(0.5, min(1.5, call_mult)),
            'bluff_frequency_mult': max(0.5, min(1.5, bluff_mult)),
            'raise_frequency_mult': max(0.5, min(1.5, raise_mult)),
            'max_deviation': max_dev,
        }

    def reset(self, seat=None):
        if seat is not None:
            self._actions.pop(seat, None)
            self._hands.pop(seat, None)
            self._street_actions.pop(seat, None)
            self._stats.pop(seat, None)
        else:
            self._actions.clear()
            self._hands.clear()
            self._street_actions.clear()
            self._stats.clear()
