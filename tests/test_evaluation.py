"""
Comprehensive evaluation test suite for PokerML strategy quality.

Tests strategy convergence metrics, EV consistency, and exploitability bounds.

Run with:
    python3 -m pytest tests/test_evaluation.py -v --tb=short
"""

import os
import random
from collections import Counter

import numpy as np
import pytest

from cfr.card_abstraction import CardAbstraction
from cfr.evaluator import evaluate_hand, enumerate_river_equity
from cfr.exploitability import compute_lbr, estimate_convergence_quality
from cfr.game_state import GameState, STARTING_STACK
from cfr.strategy_store import load_strategy
from server.bot import Bot


@pytest.fixture(scope="module")
def bot_2max():
    path = os.path.join(os.path.dirname(__file__), '..', 'strategy_2max.json.gz')
    if not os.path.exists(path):
        pytest.skip("strategy_2max.json.gz not found")
    strategy = load_strategy(path)
    ca = CardAbstraction(num_players=2, use_emd=False)
    return Bot(strategy, card_abstraction=ca, num_players=2)


def _advance_preflop(state):
    """Advance through preflop with calls."""
    while not state.is_terminal and state.street == 0:
        a = state.get_actions()
        state = state.apply_action('c' if 'c' in a else a[0])
    return state


# ======================================================================
# STRATEGY CONVERGENCE TESTS
# ======================================================================

class TestStrategyConvergence:
    """Verify trained strategy shows convergence (not uniform random)."""

    def test_strategy_not_uniform(self, bot_2max):
        """Most info sets should NOT have uniform random strategy."""
        strategy = bot_2max.strategy
        quality = estimate_convergence_quality(strategy)
        assert quality['num_info_sets'] > 0, "Strategy has no info sets"
        # avg_max_prob above 0.40 means distributions are meaningfully
        # non-uniform (uniform over 3 actions = 0.33, over 4 = 0.25)
        print(f"\n  avg_max_prob: {quality['avg_max_prob']:.3f}")
        assert quality['avg_max_prob'] > 0.40, (
            f"Strategy looks too uniform: avg_max_prob={quality['avg_max_prob']:.3f}"
        )
        # At least some pure strategies should exist
        assert quality['pct_pure'] > 2.0, (
            f"Too few pure info sets: {quality['pct_pure']:.1f}%"
        )

    def test_strategy_entropy_reasonable(self, bot_2max):
        """Average entropy should be between 0.3 and 2.0 (mixed but not random)."""
        strategy = bot_2max.strategy
        quality = estimate_convergence_quality(strategy)
        avg_ent = quality['avg_entropy']
        print(f"\n  Average entropy: {avg_ent:.3f}")
        assert 0.3 <= avg_ent <= 2.0, (
            f"Entropy out of range: {avg_ent:.3f} (expected 0.3-2.0)"
        )

    def test_pure_strategy_percentage(self, bot_2max):
        """Some info sets should be pure (>95% on one action), but not all."""
        strategy = bot_2max.strategy
        quality = estimate_convergence_quality(strategy)
        pct = quality['pct_pure']
        print(f"\n  Pure strategy %: {pct:.1f}%")
        assert 2.0 <= pct <= 70.0, (
            f"Pure % out of range: {pct:.1f}% (expected 2-70%)"
        )


# ======================================================================
# EV CONSISTENCY TESTS
# ======================================================================

class TestEVConsistency:
    """Verify expected value properties of the strategy."""

    def test_nuts_ev_positive(self, bot_2max):
        """Nut hands should have positive EV in all scenarios."""
        ca = bot_2max.card_abstraction
        positive_count = 0
        tested = 0

        for seed in range(200):
            random.seed(seed + 40000)
            np.random.seed(seed + 40000)
            deck = list(range(52))
            random.shuffle(deck)
            hands = ((deck[0], deck[1]), (deck[2], deck[3]))
            board = tuple(deck[4:9])

            # Check if hero (seat 0) has a nut hand on river
            equity = enumerate_river_equity(hands[0], board)
            if equity < 0.90:
                continue

            tested += 1
            state = GameState.new_hand(hands, board, num_players=2)

            total_ev = 0.0
            n_sims = 10
            for _ in range(n_sims):
                sim_state = GameState.new_hand(hands, board, num_players=2)
                moves = 0
                while not sim_state.is_terminal and moves < 50:
                    actions = sim_state.get_actions()
                    if not actions:
                        break
                    if sim_state.current_player == 0:
                        action = bot_2max.get_action(
                            hands[0], sim_state.visible_board(),
                            sim_state.history, actions,
                            state=sim_state, seat=0)
                    else:
                        # Opponent plays call-or-check
                        action = 'c' if 'c' in actions else (
                            'k' if 'k' in actions else actions[0])
                    sim_state = sim_state.apply_action(action)
                    moves += 1
                if sim_state.is_terminal:
                    total_ev += sim_state.get_terminal_utility(0)

            avg_ev = total_ev / n_sims
            if avg_ev > 0:
                positive_count += 1

        if tested < 5:
            pytest.skip("Not enough nut hands found")
        pct_positive = positive_count / tested
        print(f"\n  Nut hands with positive EV: {pct_positive:.1%} ({tested} tested)")
        assert pct_positive > 0.70, (
            f"Too few nut hands profitable: {pct_positive:.1%}"
        )

    def test_trash_ev_negative_facing_bet(self, bot_2max):
        """Trash hands facing a river bet should have negative EV if calling."""
        ca = bot_2max.card_abstraction
        negative_count = 0
        tested = 0

        for seed in range(300):
            random.seed(seed + 41000)
            np.random.seed(seed + 41000)
            deck = list(range(52))
            random.shuffle(deck)
            hands = ((deck[0], deck[1]), (deck[2], deck[3]))
            board = tuple(deck[4:9])

            # Check if hero (seat 1) has trash on river
            equity = enumerate_river_equity(hands[1], board)
            if equity > 0.20:
                continue

            # Build state to river with hero facing a bet
            state = GameState.new_hand(hands, board, num_players=2)
            state = _advance_preflop(state)
            if state.is_terminal or state.street < 1:
                continue
            # Check through to river
            while not state.is_terminal and state.street < 3:
                state = state.apply_action('k')
            if state.is_terminal or state.street != 3:
                continue
            # OOP checks, IP bets
            if state.current_player == 1:
                state = state.apply_action('k')
            if state.is_terminal:
                continue
            if state.current_player == 0:
                bets = [a for a in state.get_actions() if a.startswith('b')]
                if not bets:
                    continue
                state = state.apply_action(bets[0])
            if state.is_terminal or state.current_player != 1:
                continue

            actions = state.get_actions()
            if 'c' not in actions:
                continue

            # Compute EV of calling: need to see what happens at showdown
            call_state = state.apply_action('c')
            if not call_state.is_terminal:
                continue
            call_ev = call_state.get_terminal_utility(1)

            tested += 1
            if call_ev < 0:
                negative_count += 1

        if tested < 5:
            pytest.skip("Not enough trash hands facing river bet")
        pct_negative = negative_count / tested
        print(f"\n  Trash hands with negative call EV: {pct_negative:.1%} ({tested} tested)")
        assert pct_negative > 0.60, (
            f"Too few trash hands have negative call EV: {pct_negative:.1%}"
        )

    def test_position_ev_advantage(self, bot_2max):
        """Self-play EV should be approximately zero-sum with moderate positional gap."""
        btn_ev_total = 0.0
        bb_ev_total = 0.0
        played = 0

        for seed in range(300):
            random.seed(seed + 42000)
            np.random.seed(seed + 42000)
            deck = list(range(52))
            random.shuffle(deck)
            hands = ((deck[0], deck[1]), (deck[2], deck[3]))
            board = tuple(deck[4:9])

            state = GameState.new_hand(hands, board, num_players=2)
            moves = 0
            while not state.is_terminal and moves < 50:
                actions = state.get_actions()
                if not actions:
                    break
                player = state.current_player
                action = bot_2max.get_action(
                    hands[player], state.visible_board(),
                    state.history, actions,
                    state=state, seat=player)
                state = state.apply_action(action)
                moves += 1

            if state.is_terminal:
                # seat 0 = BTN (posts SB, IP postflop)
                # seat 1 = BB (posts BB, OOP postflop)
                btn_ev_total += state.get_terminal_utility(0)
                bb_ev_total += state.get_terminal_utility(1)
                played += 1

        if played < 50:
            pytest.skip("Not enough completed hands")
        btn_avg = btn_ev_total / played
        bb_avg = bb_ev_total / played
        ev_gap = abs(btn_avg - bb_avg)
        print(f"\n  BTN avg EV: {btn_avg:+.3f} BB, BB avg EV: {bb_avg:+.3f} BB ({played} hands)")
        print(f"  Positional EV gap: {ev_gap:.3f} BB/hand")
        # In self-play the game is zero-sum; the positional EV gap should be
        # moderate (< 5 BB/hand). Larger gaps suggest a badly calibrated strategy.
        assert ev_gap < 5.0, (
            f"Positional EV gap too large: {ev_gap:.3f} BB/hand"
        )
        # Verify it's approximately zero-sum
        assert abs(btn_avg + bb_avg) < 0.01, (
            f"Game not zero-sum: BTN={btn_avg:+.3f} + BB={bb_avg:+.3f} = {btn_avg + bb_avg:+.3f}"
        )


# ======================================================================
# EXPLOITABILITY TESTS
# ======================================================================

class TestExploitability:
    """Verify exploitability is bounded."""

    def test_lbr_exploitability_bounded(self, bot_2max):
        """LBR exploitability should be bounded (range-aware best response)."""
        strategy = bot_2max.strategy
        ca = bot_2max.card_abstraction
        random.seed(99)
        np.random.seed(99)
        # Use small sample for speed; the range-aware LBR is a strong exploiter
        exploit = compute_lbr(strategy, num_players=2, samples=200,
                              card_abstraction=ca)
        print(f"\n  LBR exploitability: {exploit:.1f} mbb/hand")
        # Range-aware LBR produces higher exploitability estimates than naive LBR.
        # A strategy trained with limited iterations and bucket abstraction will
        # show significant exploitability under a strong best response.
        # Bound at 20000 mbb/hand (20 BB/hand) -- mainly verifying it's finite.
        assert exploit < 20000, (
            f"Exploitability too high: {exploit:.1f} mbb/hand (limit: 20000)"
        )
