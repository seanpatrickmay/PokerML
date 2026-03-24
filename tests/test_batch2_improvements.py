"""Tests for Batch 2 improvements: VR-MCCFR baselines, strategy blending,
adaptive thresholding, improved exploitation."""
import random
import numpy as np
import pytest

from cfr.strategy_utils import (
    threshold_strategy, blend_strategies, adaptive_threshold,
)
from cfr.opponent_model import OpponentModel


# ── Strategy Blending ─────────────────────────────────────────────

class TestStrategyBlending:
    def test_blend_equal_weight(self):
        """50/50 blend should average the strategies."""
        a = [0.6, 0.4]
        b = [0.4, 0.6]
        result = blend_strategies(a, b, resolve_weight=0.5)
        assert abs(result[0] - 0.5) < 1e-9
        assert abs(result[1] - 0.5) < 1e-9

    def test_blend_full_resolve(self):
        """Weight 1.0 should return resolved strategy."""
        resolved = [0.8, 0.2]
        blueprint = [0.3, 0.7]
        result = blend_strategies(resolved, blueprint, resolve_weight=1.0)
        assert abs(result[0] - 0.8) < 1e-9
        assert abs(result[1] - 0.2) < 1e-9

    def test_blend_sums_to_one(self):
        """Blended strategy should always sum to 1."""
        for _ in range(50):
            n = random.randint(2, 6)
            a = [random.random() for _ in range(n)]
            b = [random.random() for _ in range(n)]
            ta = sum(a)
            tb = sum(b)
            a = [x / ta for x in a]
            b = [x / tb for x in b]
            w = random.random()
            result = blend_strategies(a, b, resolve_weight=w)
            assert abs(sum(result) - 1.0) < 1e-9

    def test_blend_empty_resolved(self):
        """Empty resolved should return blueprint."""
        result = blend_strategies([], [0.5, 0.5])
        assert result == [0.5, 0.5]

    def test_blend_empty_blueprint(self):
        """Empty blueprint should return resolved."""
        result = blend_strategies([0.5, 0.5], [])
        assert result == [0.5, 0.5]


# ── Adaptive Thresholding ────────────────────────────────────────

class TestAdaptiveThreshold:
    def test_early_hand_low_threshold(self):
        """Early in hand, threshold should be low (keep more mixing)."""
        probs = [0.06, 0.44, 0.50]
        result = adaptive_threshold(probs, num_actions_taken=0)
        # At 0 actions, threshold=0.05, so 0.06 survives
        assert result[0] > 0  # 0.06 > 0.05, should survive

    def test_late_hand_high_threshold(self):
        """Late in hand, sharpening adjusts but output remains valid distribution."""
        probs = [0.06, 0.44, 0.50]
        result = adaptive_threshold(probs, num_actions_taken=10)
        assert abs(sum(result) - 1.0) < 1e-9
        assert all(r >= 0 for r in result)

    def test_threshold_sums_to_one(self):
        """Output should always sum to 1."""
        for actions in range(0, 15):
            probs = [0.05, 0.3, 0.65]
            result = adaptive_threshold(probs, num_actions_taken=actions)
            assert abs(sum(result) - 1.0) < 1e-9

    def test_threshold_capped(self):
        """Threshold should be conservative — preserve mixed strategies."""
        probs = [0.14, 0.36, 0.50]
        result = adaptive_threshold(probs, num_actions_taken=100)
        # Conservative threshold (0.10 cap) preserves 0.14
        assert sum(result) > 0.99  # valid distribution


# ── VR-MCCFR Baselines ───────────────────────────────────────────

class TestVRMCCFR:
    def test_trainer_has_baseline_values(self):
        """CFRTrainer should have baseline_values dict."""
        from cfr.cfr_trainer import CFRTrainer
        from cfr.card_abstraction import CardAbstraction
        ca = CardAbstraction(num_players=2)
        trainer = CFRTrainer(card_abstraction=ca, iterations=10, num_players=2)
        assert hasattr(trainer, 'baseline_values')
        assert isinstance(trainer.baseline_values, dict)

    def test_baseline_populated_after_training(self):
        """Baseline values should be populated after some training."""
        from cfr.cfr_trainer import CFRTrainer
        from cfr.card_abstraction import CardAbstraction
        ca = CardAbstraction(num_players=2)
        trainer = CFRTrainer(card_abstraction=ca, iterations=100,
                             num_players=2, postflop_only=False)
        trainer.train()
        # After 100 iterations, some baseline values should exist
        assert len(trainer.baseline_values) > 0


# ── Improved Opponent Model ──────────────────────────────────────

class TestImprovedOpponentModel:
    def test_get_street_adjustments_no_data(self):
        """Street adjustments with no data should return neutral."""
        model = OpponentModel()
        adj = model.get_street_adjustments(0, 1)
        assert adj['bet_frequency_mult'] == 1.0
        assert adj['max_deviation'] == 0.0

    def test_get_street_adjustments_with_data(self):
        """With enough data, adjustments should deviate from 1.0."""
        model = OpponentModel()
        # Record lots of folds (passive opponent)
        for _ in range(50):
            model.record_action(0, 1, 'cbet', 'fold')
        adj = model.get_street_adjustments(0, 1)
        # Should suggest betting more vs a folder
        assert adj['bet_frequency_mult'] >= 1.0
        assert adj['max_deviation'] > 0.0

    def test_street_adjustments_bounded(self):
        """Adjustments should be bounded [0.5, 1.5]."""
        model = OpponentModel()
        for _ in range(200):
            model.record_action(0, 1, 'cbet', 'fold')
        adj = model.get_street_adjustments(0, 1)
        assert 0.5 <= adj['bet_frequency_mult'] <= 1.5
        assert 0.5 <= adj['call_threshold_mult'] <= 1.5
        assert adj['max_deviation'] <= 0.30

    def test_per_street_tracking(self):
        """Should track stats separately per street."""
        model = OpponentModel()
        model.record_action(0, 0, 'first_in', 'fold')
        model.record_action(0, 1, 'cbet', 'bet')
        model.record_action(0, 2, 'turn', 'fold')
        stats = model.get_stats(0)
        assert 'by_street' in stats


# ── Bot Apply Adjustments ────────────────────────────────────────

class TestBotAdjustments:
    def test_bounded_adjustments(self):
        """Adjustments should be bounded by max_deviation."""
        from server.bot import Bot
        bot = Bot({}, num_players=2)
        probs = [0.3, 0.3, 0.4]
        actions = ['f', 'c', 'b50']
        adj = {
            'bet_frequency_mult': 3.0,  # Extreme multiplier
            'call_threshold_mult': 0.1,
            'max_deviation': 0.10,
        }
        result = bot._apply_adjustments(probs, actions, adj)
        # No action should change by more than 0.10 from original
        for i in range(3):
            assert abs(result[i] - probs[i]) <= 0.10 + 0.01  # small tolerance
        assert abs(sum(result) - 1.0) < 1e-9

    def test_neutral_adjustments(self):
        """Neutral multipliers should not change strategy."""
        from server.bot import Bot
        bot = Bot({}, num_players=2)
        probs = [0.3, 0.3, 0.4]
        actions = ['f', 'c', 'b50']
        adj = {
            'bet_frequency_mult': 1.0,
            'call_threshold_mult': 1.0,
            'max_deviation': 0.15,
        }
        result = bot._apply_adjustments(probs, actions, adj)
        for i in range(3):
            assert abs(result[i] - probs[i]) < 0.01
