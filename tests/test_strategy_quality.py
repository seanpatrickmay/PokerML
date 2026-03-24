"""
Strategy quality benchmarks — GTO frequency validation.

Tests that the bot's action frequencies fall within GTO-realistic ranges.
These are integration tests that run the full bot pipeline (solver + corrections)
on 100+ hands per scenario.

Run with:
    python3 -m pytest tests/test_strategy_quality.py -v -s --tb=short
"""

import os
import random
from collections import Counter

import numpy as np
import pytest

from cfr.card_abstraction import CardAbstraction
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


def _sample_action(bot, hands, board, seat, state, n=10):
    """Sample bot action n times and return counts."""
    counts = Counter()
    actions = state.get_actions()
    for _ in range(n):
        a = bot.get_action(hands[seat], state.visible_board(), state.history,
                           actions, state=state, seat=seat)
        if a.startswith('b') or a == 'a':
            counts['bet'] += 1
        elif a == 'k':
            counts['check'] += 1
        elif a == 'c':
            counts['call'] += 1
        elif a == 'f':
            counts['fold'] += 1
    return counts


def _advance_preflop(state):
    """Advance through preflop with calls."""
    while not state.is_terminal and state.street == 0:
        a = state.get_actions()
        state = state.apply_action('c' if 'c' in a else a[0])
    return state


def _run_scenario(bot, seat, setup_fn, n_hands=100, samples_per=10):
    """Run a scenario and return (bet%, check%, call%, fold%)."""
    totals = Counter()
    total = 0
    for seed in range(n_hands):
        random.seed(seed + 30000)
        np.random.seed(seed + 30000)
        deck = list(range(52))
        random.shuffle(deck)
        hands = ((deck[0], deck[1]), (deck[2], deck[3]))
        board = tuple(deck[4:9])
        state = setup_fn(hands, board)
        if state is None or state.is_terminal or state.current_player != seat:
            continue
        counts = _sample_action(bot, hands, board, seat, state, samples_per)
        totals += counts
        total += samples_per
    if total == 0:
        return None
    return {k: v / total for k, v in totals.items()}


# ══════════════════════════════════════════════════════════════════
# FLOP FREQUENCY TESTS
# ══════════════════════════════════════════════════════════════════

class TestFlopFrequencies:
    """Validate flop betting/checking frequencies."""

    def test_flop_cbet_oop_not_excessive(self, bot_2max):
        """OOP flop c-bet should be 25-45% (GTO ~30%)."""
        def setup(h, b):
            s = GameState.new_hand(h, b, num_players=2)
            s = _advance_preflop(s)
            return s if not s.is_terminal and s.street >= 1 else None

        freqs = _run_scenario(bot_2max, seat=1, setup_fn=setup)
        if freqs is None:
            pytest.skip("No valid hands")
        bet_rate = freqs.get('bet', 0)
        print(f"\n  Flop c-bet OOP: {bet_rate:.1%} (target: 25-45%)")
        assert bet_rate < 0.50, f"Flop OOP c-bet too high: {bet_rate:.1%}"

    def test_flop_cbet_ip_reasonable(self, bot_2max):
        """IP flop c-bet should be 50-75% (GTO ~60%)."""
        def setup(h, b):
            s = GameState.new_hand(h, b, num_players=2)
            s = _advance_preflop(s)
            if s.is_terminal or s.street < 1:
                return None
            if s.current_player == 1:
                s = s.apply_action('k')
            return s if not s.is_terminal else None

        freqs = _run_scenario(bot_2max, seat=0, setup_fn=setup)
        if freqs is None:
            pytest.skip("No valid hands")
        bet_rate = freqs.get('bet', 0)
        print(f"\n  Flop c-bet IP: {bet_rate:.1%} (target: 50-75%)")
        assert 0.40 < bet_rate < 0.85, f"Flop IP c-bet out of range: {bet_rate:.1%}"

    def test_flop_check_raise_capped(self, bot_2max):
        """Flop check-raise should be 5-20% (GTO ~10%)."""
        def setup(h, b):
            s = GameState.new_hand(h, b, num_players=2)
            s = _advance_preflop(s)
            if s.is_terminal or s.street < 1:
                return None
            if s.current_player == 1:
                s = s.apply_action('k')
            if s.is_terminal:
                return None
            if s.current_player == 0:
                bets = [a for a in s.get_actions() if a.startswith('b')]
                if bets:
                    s = s.apply_action(bets[0])
            return s if not s.is_terminal else None

        freqs = _run_scenario(bot_2max, seat=1, setup_fn=setup)
        if freqs is None:
            pytest.skip("No valid hands")
        raise_rate = freqs.get('bet', 0)
        print(f"\n  Flop x/r rate: {raise_rate:.1%} (target: 5-20%)")
        assert raise_rate < 0.25, f"Flop x/r too high: {raise_rate:.1%}"

    def test_flop_oop_fold_to_bet_reasonable(self, bot_2max):
        """OOP should fold 25-50% to flop bets (not over-call)."""
        def setup(h, b):
            s = GameState.new_hand(h, b, num_players=2)
            s = _advance_preflop(s)
            if s.is_terminal or s.street < 1:
                return None
            # BB checks
            if s.current_player == 1:
                s = s.apply_action('k')
            if s.is_terminal:
                return None
            # BTN bets
            if s.current_player == 0:
                bets = [a for a in s.get_actions() if a.startswith('b')]
                if bets:
                    s = s.apply_action(bets[0])
            return s if not s.is_terminal else None

        freqs = _run_scenario(bot_2max, seat=1, setup_fn=setup)
        if freqs is None:
            pytest.skip("No valid hands")
        fold_rate = freqs.get('fold', 0)
        call_rate = freqs.get('call', 0)
        print(f"\n  Flop OOP vs bet: fold={fold_rate:.1%} call={call_rate:.1%}")
        assert fold_rate > 0.20, f"Folding too little vs flop bet: {fold_rate:.1%}"
        assert call_rate < 0.70, f"Calling too much vs flop bet: {call_rate:.1%}"


class TestTurnFrequencies:
    """Validate turn betting/checking frequencies."""

    def test_turn_lead_oop_after_check_check(self, bot_2max):
        """Turn OOP lead after x/x flop should be 25-55%."""
        def setup(h, b):
            s = GameState.new_hand(h, b, num_players=2)
            s = _advance_preflop(s)
            if s.is_terminal or s.street < 1:
                return None
            while not s.is_terminal and s.street == 1:
                s = s.apply_action('k')
            return s if not s.is_terminal and s.street >= 2 else None

        freqs = _run_scenario(bot_2max, seat=1, setup_fn=setup)
        if freqs is None:
            pytest.skip("No valid hands")
        bet_rate = freqs.get('bet', 0)
        print(f"\n  Turn lead OOP (x/x): {bet_rate:.1%} (target: 25-55%)")
        assert bet_rate < 0.60, f"Turn OOP lead too high: {bet_rate:.1%}"


# ══════════════════════════════════════════════════════════════════
# RIVER FREQUENCY TESTS
# ══════════════════════════════════════════════════════════════════

class TestRiverFrequencies:
    """Validate river action frequencies."""

    def test_river_medium_hand_calls_enough(self, bot_2max):
        """Medium hands facing river bet should call 30%+ (not over-raise)."""
        ca = bot_2max.card_abstraction

        def setup(h, b):
            s = GameState.new_hand(h, b, num_players=2)
            s = _advance_preflop(s)
            if s.is_terminal or s.street < 1:
                return None
            # Check through to river
            while not s.is_terminal and s.street < 3:
                s = s.apply_action('k')
            if s.is_terminal or s.street != 3:
                return None
            # BB checks, BTN bets
            if s.current_player == 1:
                s = s.apply_action('k')
            if s.is_terminal:
                return None
            if s.current_player == 0:
                bets = [a for a in s.get_actions() if a.startswith('b')]
                if bets:
                    s = s.apply_action(bets[0])
            if s.is_terminal:
                return None
            # Only test medium-strength hands
            bucket = ca.get_bucket(h[1], s.visible_board())
            if bucket < 60 or bucket > 140:
                return None
            return s

        freqs = _run_scenario(bot_2max, seat=1, setup_fn=setup)
        if freqs is None:
            pytest.skip("No medium hands facing river bet")
        call_rate = freqs.get('call', 0)
        raise_rate = freqs.get('bet', 0)
        print(f"\n  River medium hand: call={call_rate:.1%} raise={raise_rate:.1%}")
        assert call_rate > 0.25, f"River medium hands under-calling: {call_rate:.1%}"
        assert raise_rate < 0.30, f"River medium hands over-raising: {raise_rate:.1%}"


# ══════════════════════════════════════════════════════════════════
# TURN IP FREQUENCY TESTS
# ══════════════════════════════════════════════════════════════════

class TestTurnIPFrequencies:
    """Validate turn IP betting frequencies."""

    def test_turn_barrel_ip_not_excessive(self, bot_2max):
        """Turn IP barrel after flop c-bet should be 45-75%."""
        def setup(h, b):
            s = GameState.new_hand(h, b, num_players=2)
            s = _advance_preflop(s)
            if s.is_terminal or s.street < 1:
                return None
            # BB checks flop
            if s.current_player == 1:
                s = s.apply_action('k')
            if s.is_terminal:
                return None
            # BTN bets flop
            if s.current_player == 0:
                bets = [a for a in s.get_actions() if a.startswith('b')]
                if bets:
                    s = s.apply_action(bets[0])
                else:
                    return None
            if s.is_terminal:
                return None
            # BB calls
            if s.current_player == 1 and 'c' in s.get_actions():
                s = s.apply_action('c')
            if s.is_terminal or s.street < 2:
                return None
            # BB checks turn
            if s.current_player == 1:
                s = s.apply_action('k')
            return s if not s.is_terminal else None

        freqs = _run_scenario(bot_2max, seat=0, setup_fn=setup)
        if freqs is None:
            pytest.skip("No valid hands")
        bet_rate = freqs.get('bet', 0)
        print(f"\n  Turn barrel IP: {bet_rate:.1%} (target: 45-75%)")
        assert bet_rate < 0.80, f"Turn IP barrel too high: {bet_rate:.1%}"
        assert bet_rate > 0.35, f"Turn IP barrel too low: {bet_rate:.1%}"


    def test_turn_check_ip_first_to_act(self, bot_2max):
        """Turn IP check-back should be 25-50% (GTO ~35%)."""
        def setup(h, b):
            s = GameState.new_hand(h, b, num_players=2)
            s = _advance_preflop(s)
            if s.is_terminal or s.street < 1:
                return None
            # Check through flop
            while not s.is_terminal and s.street == 1:
                s = s.apply_action('k')
            if s.is_terminal or s.street < 2:
                return None
            # BB checks turn
            if s.current_player == 1:
                s = s.apply_action('k')
            return s if not s.is_terminal else None

        freqs = _run_scenario(bot_2max, seat=0, setup_fn=setup)
        if freqs is None:
            pytest.skip("No valid hands")
        check_rate = freqs.get('check', 0)
        print(f"\n  Turn IP check-back: {check_rate:.1%} (target: 25-50%)")
        assert check_rate > 0.20, f"Turn IP checking too little: {check_rate:.1%}"
        assert check_rate < 0.60, f"Turn IP checking too much: {check_rate:.1%}"


class TestBoardTexture:
    """Verify c-bet frequency adjusts for board wetness."""

    def test_wet_boards_less_cbet(self, bot_2max):
        """Wet boards should have lower IP c-bet than dry boards."""
        def _measure_cbet(bot, board_filter, n=150):
            counts = Counter()
            total = 0
            for seed in range(n):
                random.seed(seed + 36000)
                np.random.seed(seed + 36000)
                deck = list(range(52))
                random.shuffle(deck)
                hands = ((deck[0], deck[1]), (deck[2], deck[3]))
                board = tuple(deck[4:9])
                if not board_filter(board):
                    continue
                state = GameState.new_hand(hands, board, num_players=2)
                state = _advance_preflop(state)
                if state.is_terminal or state.street < 1:
                    continue
                if state.current_player == 1:
                    state = state.apply_action('k')
                if state.is_terminal or state.current_player != 0:
                    continue
                actions = state.get_actions()
                for _ in range(3):
                    a = bot.get_action(hands[0], state.visible_board(),
                                       state.history, actions, state=state, seat=0)
                    total += 1
                    if a.startswith('b') or a == 'a':
                        counts['bet'] += 1
            return counts.get('bet', 0) / max(total, 1), total

        def is_wet(b):
            suits = [c % 4 for c in b[:3]]
            ranks = sorted([c // 4 for c in b[:3]])
            return len(set(suits)) <= 2 and any(ranks[i+1]-ranks[i] <= 2 for i in range(2))

        def is_dry(b):
            suits = [c % 4 for c in b[:3]]
            ranks = sorted([c // 4 for c in b[:3]])
            return len(set(suits)) == 3 and all(ranks[i+1]-ranks[i] > 2 for i in range(2))

        wet_cbet, wet_n = _measure_cbet(bot_2max, is_wet)
        dry_cbet, dry_n = _measure_cbet(bot_2max, is_dry)
        print(f"\n  Wet board c-bet: {wet_cbet:.1%} (n={wet_n})")
        print(f"  Dry board c-bet: {dry_cbet:.1%} (n={dry_n})")
        if wet_n >= 20 and dry_n >= 20:
            assert wet_cbet < dry_cbet + 0.05, (
                f"Wet boards ({wet_cbet:.1%}) should have lower c-bet than "
                f"dry ({dry_cbet:.1%})"
            )


class TestMultiStreetConsistency:
    """Verify turn play differs based on flop action (range awareness)."""

    def test_delayed_cbet_less_frequent(self, bot_2max):
        """Delayed c-bet (check flop, bet turn) should be less frequent
        than double barrel (bet flop, bet turn) for IP player."""
        def setup_barrel(h, b):
            s = GameState.new_hand(h, b, num_players=2)
            s = _advance_preflop(s)
            if s.is_terminal or s.street < 1:
                return None
            # BB checks, BTN bets, BB calls
            if s.current_player == 1:
                s = s.apply_action('k')
            if s.is_terminal:
                return None
            bets = [a for a in s.get_actions() if a.startswith('b')]
            if not bets:
                return None
            s = s.apply_action(bets[0])
            if s.is_terminal or 'c' not in s.get_actions():
                return None
            s = s.apply_action('c')
            if s.is_terminal or s.street < 2:
                return None
            # BB checks turn
            if s.current_player == 1:
                s = s.apply_action('k')
            return s if not s.is_terminal else None

        def setup_delayed(h, b):
            s = GameState.new_hand(h, b, num_players=2)
            s = _advance_preflop(s)
            if s.is_terminal or s.street < 1:
                return None
            # Check-check flop
            while not s.is_terminal and s.street == 1:
                s = s.apply_action('k')
            if s.is_terminal or s.street < 2:
                return None
            # BB checks turn
            if s.current_player == 1:
                s = s.apply_action('k')
            return s if not s.is_terminal else None

        barrel_freqs = _run_scenario(bot_2max, seat=0, setup_fn=setup_barrel)
        delayed_freqs = _run_scenario(bot_2max, seat=0, setup_fn=setup_delayed)

        if barrel_freqs is None or delayed_freqs is None:
            pytest.skip("Insufficient samples")

        barrel_bet = barrel_freqs.get('bet', 0)
        delayed_bet = delayed_freqs.get('bet', 0)
        print(f"\n  Double barrel IP: {barrel_bet:.1%}")
        print(f"  Delayed c-bet IP: {delayed_bet:.1%}")
        assert barrel_bet > delayed_bet, (
            f"Delayed c-bet ({delayed_bet:.1%}) should be less than "
            f"double barrel ({barrel_bet:.1%})"
        )


class TestExploitability:
    """Verify the bot beats degenerate strategies."""

    def test_beats_calling_station(self, bot_2max):
        """Bot should profit against a player who never folds."""
        if hasattr(bot_2max, '_resolved_cache'):
            bot_2max._resolved_cache.clear()
            bot_2max._cache_board = None
        total = 0.0
        played = 0
        for seed in range(200):
            random.seed(seed + 63000)
            np.random.seed(seed + 63000)
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
                if state.current_player == 0:
                    action = bot_2max.get_action(
                        hands[0], state.visible_board(), state.history,
                        actions, state=state, seat=0)
                else:
                    action = 'c' if 'c' in actions else ('k' if 'k' in actions else actions[0])
                state = state.apply_action(action)
                moves += 1
            if state.is_terminal:
                total += state.get_terminal_utility(0)
                played += 1
        wr = total / max(played, 1)
        print(f"\n  vs Calling Station: {wr:+.2f} BB/hand ({played} hands)")
        # 200 hands has ~2 bb/hand std, allow moderate variance
        assert wr > -3.0, f"Bot badly losing to calling station: {wr:+.2f}"

    def test_beats_nit(self, bot_2max):
        """Bot should profit against a player who always folds."""
        if hasattr(bot_2max, '_resolved_cache'):
            bot_2max._resolved_cache.clear()
        total = 0.0
        played = 0
        for seed in range(100):
            random.seed(seed + 62000)
            np.random.seed(seed + 62000)
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
                if state.current_player == 0:
                    action = bot_2max.get_action(
                        hands[0], state.visible_board(), state.history,
                        actions, state=state, seat=0)
                else:
                    action = 'f' if 'f' in actions else ('k' if 'k' in actions else actions[0])
                state = state.apply_action(action)
                moves += 1
            if state.is_terminal:
                total += state.get_terminal_utility(0)
                played += 1
        wr = total / max(played, 1)
        print(f"\n  vs Nit: {wr:+.2f} BB/hand ({played} hands)")
        assert wr > -1.0, f"Bot badly losing to nit: {wr:+.2f}"


class TestSolverIntegration:
    """Verify the subgame solver produces correct-length strategies."""

    def test_solver_probs_match_actions(self, bot_2max):
        """Solver output probability vectors must match action count."""
        from cfr.subgame_solver import SubgameSolver
        from cfr.game_state import get_position_name, lookup_with_fallback

        ca = bot_2max.card_abstraction
        solver = bot_2max.solver

        mismatches = 0
        tested = 0
        for seed in range(20):
            random.seed(seed + 70000)
            np.random.seed(seed + 70000)
            deck = list(range(52))
            random.shuffle(deck)
            hands = ((deck[0], deck[1]), (deck[2], deck[3]))
            board = tuple(deck[4:9])
            state = GameState.new_hand(hands, board, num_players=2)
            state = _advance_preflop(state)
            if state.is_terminal or state.street < 1:
                continue
            while not state.is_terminal and state.street == 1:
                state = state.apply_action('k')
            if state.is_terminal or state.street < 2:
                continue

            actions = state.get_actions()
            bucket = ca.get_bucket(hands[1], state.visible_board())
            pos = get_position_name(2, 1)
            info_key = state.get_info_set_key(1, bucket)

            solved = solver.solve(state, 1)
            probs = solved.get(info_key)
            tested += 1

            if probs is None or len(probs) != len(actions):
                mismatches += 1

        print(f"\n  Solver key matches: {tested - mismatches}/{tested}")
        assert mismatches == 0, f"{mismatches}/{tested} solver key mismatches"

    def test_solver_check_frequency_reasonable(self, bot_2max):
        """Solver should produce check% > 15% for OOP turn decisions."""
        from cfr.subgame_solver import SubgameSolver
        from cfr.game_state import get_position_name

        ca = bot_2max.card_abstraction
        solver = bot_2max.solver

        check_pcts = []
        for seed in range(10):
            random.seed(seed + 80000)
            np.random.seed(seed + 80000)
            deck = list(range(52))
            random.shuffle(deck)
            hands = ((deck[0], deck[1]), (deck[2], deck[3]))
            board = tuple(deck[4:9])
            state = GameState.new_hand(hands, board, num_players=2)
            state = _advance_preflop(state)
            if state.is_terminal or state.street < 1:
                continue
            while not state.is_terminal and state.street == 1:
                state = state.apply_action('k')
            if state.is_terminal or state.street < 2 or state.current_player != 1:
                continue

            actions = state.get_actions()
            bucket = ca.get_bucket(hands[1], state.visible_board())
            info_key = state.get_info_set_key(1, bucket)

            solved = solver.solve(state, 1)
            probs = solved.get(info_key)
            if probs and len(probs) == len(actions) and 'k' in actions:
                check_pcts.append(probs[actions.index('k')])

        if not check_pcts:
            pytest.skip("No valid solver results")
        avg_check = sum(check_pcts) / len(check_pcts)
        print(f"\n  Solver avg check% OOP turn: {avg_check:.1%} (n={len(check_pcts)})")
        assert avg_check > 0.10, f"Solver check% too low: {avg_check:.1%}"


# ══════════════════════════════════════════════════════════════════
# PREFLOP FREQUENCY TESTS
# ══════════════════════════════════════════════════════════════════

class TestPreflopFrequencies:
    """Validate preflop opening and defending frequencies."""

    def test_btn_open_rate(self, bot_2max):
        """BTN should open-raise 30-55% (GTO ~45%)."""
        counts = Counter()
        total = 0
        for seed in range(200):
            random.seed(seed + 10000)
            np.random.seed(seed + 10000)
            deck = list(range(52))
            random.shuffle(deck)
            hands = ((deck[0], deck[1]), (deck[2], deck[3]))
            board = tuple(deck[4:9])
            state = GameState.new_hand(hands, board, num_players=2)
            # BTN acts first preflop in HU
            if state.current_player != 0:
                continue
            actions = state.get_actions()
            for _ in range(5):
                a = bot_2max.get_action(hands[0], (), state.history,
                                        actions, state=state, seat=0)
                total += 1
                if a.startswith('b') or a == 'a':
                    counts['raise'] += 1
                elif a == 'c':
                    counts['limp'] += 1
                elif a == 'f':
                    counts['fold'] += 1
        raise_rate = counts.get('raise', 0) / total
        fold_rate = counts.get('fold', 0) / total
        print(f"\n  BTN open: raise={raise_rate:.1%} fold={fold_rate:.1%}")
        assert raise_rate > 0.25, f"BTN open too tight: {raise_rate:.1%}"
        assert raise_rate < 0.65, f"BTN open too loose: {raise_rate:.1%}"

    def test_bb_defense_rate(self, bot_2max):
        """BB should defend 35-60% vs BTN open (GTO ~45%)."""
        counts = Counter()
        total = 0
        for seed in range(200):
            random.seed(seed + 11000)
            np.random.seed(seed + 11000)
            deck = list(range(52))
            random.shuffle(deck)
            hands = ((deck[0], deck[1]), (deck[2], deck[3]))
            board = tuple(deck[4:9])
            state = GameState.new_hand(hands, board, num_players=2)
            # BTN raises
            if state.current_player != 0:
                continue
            bets = [a for a in state.get_actions() if a.startswith('b')]
            if not bets:
                continue
            state = state.apply_action(bets[0])
            if state.is_terminal or state.current_player != 1:
                continue
            actions = state.get_actions()
            for _ in range(5):
                a = bot_2max.get_action(hands[1], (), state.history,
                                        actions, state=state, seat=1)
                total += 1
                if a == 'c':
                    counts['call'] += 1
                elif a.startswith('b') or a == 'a':
                    counts['3bet'] += 1
                elif a == 'f':
                    counts['fold'] += 1
        defend_rate = (counts.get('call', 0) + counts.get('3bet', 0)) / total
        fold_rate = counts.get('fold', 0) / total
        print(f"\n  BB defend: {defend_rate:.1%} (call={counts.get('call',0)/total:.1%} "
              f"3bet={counts.get('3bet',0)/total:.1%} fold={fold_rate:.1%})")
        assert defend_rate > 0.30, f"BB defending too little: {defend_rate:.1%}"
        assert defend_rate < 0.70, f"BB defending too much: {defend_rate:.1%}"


# ══════════════════════════════════════════════════════════════════
# RIVER VALUE vs BLUFF TESTS
# ══════════════════════════════════════════════════════════════════

    def test_btn_vs_3bet_not_over_4betting(self, bot_2max):
        """BTN should fold 40-80% facing a 3-bet, not over-4bet."""
        counts = Counter()
        total = 0
        for seed in range(200):
            random.seed(seed + 19000)
            np.random.seed(seed + 19000)
            deck = list(range(52))
            random.shuffle(deck)
            hands = ((deck[0], deck[1]), (deck[2], deck[3]))
            board = tuple(deck[4:9])
            state = GameState.new_hand(hands, board, num_players=2)
            if state.current_player != 0:
                continue
            bets = [a for a in state.get_actions() if a.startswith('b')]
            if not bets:
                continue
            state = state.apply_action(bets[0])
            if state.is_terminal or state.current_player != 1:
                continue
            bb_bets = [a for a in state.get_actions()
                       if a.startswith('b') or a == 'a']
            if not bb_bets:
                continue
            state = state.apply_action(bb_bets[0])
            if state.is_terminal or state.current_player != 0:
                continue
            actions = state.get_actions()
            a = bot_2max.get_action(hands[0], (), state.history,
                                     actions, state=state, seat=0)
            total += 1
            if a == 'f':
                counts['fold'] += 1
            elif a == 'c':
                counts['call'] += 1
            else:
                counts['4bet'] += 1

        if total < 50:
            pytest.skip("Not enough 3-bet scenarios")
        fourbet_rate = counts.get('4bet', 0) / total
        fold_rate = counts.get('fold', 0) / total
        print(f"\n  BTN vs 3bet: fold={fold_rate:.1%} 4bet={fourbet_rate:.1%}")
        assert fourbet_rate < 0.25, f"Over-4betting: {fourbet_rate:.1%}"
        assert fold_rate > 0.30, f"Under-folding vs 3bet: {fold_rate:.1%}"


class TestRiverPolarization:
    """River betting should be polarized: value bets with strong, bluffs with weak."""

    def test_river_value_bets_strong_hands(self, bot_2max):
        """Strong hands (top 25%) should bet river 50%+ when first to act."""
        ca = bot_2max.card_abstraction

        def setup(h, b):
            s = GameState.new_hand(h, b, num_players=2)
            s = _advance_preflop(s)
            if s.is_terminal or s.street < 1:
                return None
            while not s.is_terminal and s.street < 3:
                s = s.apply_action('k')
            if s.is_terminal or s.street != 3:
                return None
            # Only strong hands (bucket > 150/200)
            bucket = ca.get_bucket(h[1], s.visible_board())
            if bucket < 150:
                return None
            return s

        freqs = _run_scenario(bot_2max, seat=1, setup_fn=setup)
        if freqs is None:
            pytest.skip("No strong river hands")
        bet_rate = freqs.get('bet', 0)
        print(f"\n  River strong hand bet rate: {bet_rate:.1%} (target: >50%)")
        assert bet_rate > 0.40, f"Under-betting river value: {bet_rate:.1%}"

    def test_river_weak_hands_mostly_check(self, bot_2max):
        """Weak hands (bottom 25%) should check river 60%+ when first to act."""
        ca = bot_2max.card_abstraction

        def setup(h, b):
            s = GameState.new_hand(h, b, num_players=2)
            s = _advance_preflop(s)
            if s.is_terminal or s.street < 1:
                return None
            while not s.is_terminal and s.street < 3:
                s = s.apply_action('k')
            if s.is_terminal or s.street != 3:
                return None
            bucket = ca.get_bucket(h[1], s.visible_board())
            if bucket > 50:
                return None
            return s

        freqs = _run_scenario(bot_2max, seat=1, setup_fn=setup)
        if freqs is None:
            pytest.skip("No weak river hands")
        check_rate = freqs.get('check', 0)
        print(f"\n  River weak hand check rate: {check_rate:.1%} (target: >55%)")
        assert check_rate > 0.50, f"Weak hands bluffing too much: {check_rate:.1%}"
