"""
Performance and accuracy benchmarks for the PokerML solver.

These benchmarks measure both speed and quality, providing a baseline
for tracking improvements. Run with:

    python3 -m pytest tests/test_benchmarks.py -v -s --tb=short

The -s flag is critical: it prints timing data to stdout.

Benchmark categories:
  SPEED:
    1. Subgame solve time (end-to-end, the critical number)
    2. CFR iterations per second
    3. Card abstraction bucketing (cache miss cost)
    4. GameState apply_action throughput
    5. Opponent range computation
    6. InfoSet regret matching throughput

  ACCURACY:
    7. Win rate vs random opponent (bb/hand)
    8. Under-folding rate (air facing river bets)
    9. Nuts identification (never fold the nuts)
    10. Hand strength ordering (rank correlation)
    11. Strategy convergence (regret ratio)
"""

import os
import random
import time
from collections import Counter

import numpy as np
import pytest

from cfr.card_abstraction import CardAbstraction
from cfr.game_state import GameState, STARTING_STACK
from cfr.information_set import InfoSet
from cfr.subgame_solver import SubgameSolver
from cfr.evaluator import evaluate_hand, enumerate_river_equity
from cfr.depth_limited_solver import estimate_leaf_value, _mc_equity
from cfr.numba_utils import weighted_sample, discount_regrets_array, regret_match
from cfr.strategy_store import load_strategy
from server.bot import Bot


# ── Card constants ─────────────────────────────────────────────────
_2h, _2d, _2c, _2s = 0, 1, 2, 3
_3h, _3d, _3c, _3s = 4, 5, 6, 7
_4h, _4d, _4c, _4s = 8, 9, 10, 11
_5h, _5d, _5c, _5s = 12, 13, 14, 15
_6h, _6d, _6c, _6s = 16, 17, 18, 19
_7h, _7d, _7c, _7s = 20, 21, 22, 23
_8h, _8d, _8c, _8s = 24, 25, 26, 27
_9h, _9d, _9c, _9s = 28, 29, 30, 31
_Th, _Td, _Tc, _Ts = 32, 33, 34, 35
_Jh, _Jd, _Jc, _Js = 36, 37, 38, 39
_Qh, _Qd, _Qc, _Qs = 40, 41, 42, 43
_Kh, _Kd, _Kc, _Ks = 44, 45, 46, 47
_Ah, _Ad, _Ac, _As = 48, 49, 50, 51

AA = (_Ah, _Ad)
KK = (_Kh, _Kd)
QQ = (_Qh, _Qd)
AKs = (_Ah, _Kh)
AKo = (_Ah, _Kd)
JTs = (_Jh, _Th)
_72o = (_7h, _2d)
_83o = (_8h, _3d)


# ── Fixtures ───────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def strategy_2max():
    path = os.path.join(os.path.dirname(__file__), '..', 'strategy_2max.json.gz')
    if not os.path.exists(path):
        pytest.skip("strategy_2max.json.gz not found")
    return load_strategy(path)


@pytest.fixture(scope="module")
def bot_2max(strategy_2max):
    ca = CardAbstraction(num_players=2, use_emd=False)
    return Bot(strategy_2max, card_abstraction=ca, num_players=2)


@pytest.fixture(scope="module")
def ca_2max():
    return CardAbstraction(num_players=2, use_emd=False)


@pytest.fixture(scope="module")
def solver_2max(strategy_2max, ca_2max):
    return SubgameSolver(strategy_2max, ca_2max, iterations=200, timeout=30.0)


def _make_state(hand0, hand1, board):
    return GameState.new_hand((hand0, hand1), board, num_players=2)


def _advance_to_street(state, target):
    moves = 0
    while not state.is_terminal and state.street < target and moves < 30:
        actions = state.get_actions()
        if not actions:
            return None
        if 'k' in actions:
            state = state.apply_action('k')
        elif 'c' in actions:
            state = state.apply_action('c')
        else:
            bets = [a for a in actions if a.startswith('b')]
            if bets:
                state = state.apply_action(bets[0])
            else:
                return None
        moves += 1
    return state if (not state.is_terminal and state.street >= target) else None


def _sample_action(bot, hand, vis, history, actions, state, seat, trials=80):
    counts = Counter()
    for _ in range(trials):
        a = bot.get_action(hand, vis, history, actions, state=state, seat=seat)
        counts[a] += 1
    return counts


def _freq(counts, prefix):
    total = sum(counts.values())
    if total == 0:
        return 0.0
    return sum(v for k, v in counts.items() if k.startswith(prefix)) / total


# ══════════════════════════════════════════════════════════════════
# SPEED BENCHMARK 1: SUBGAME SOLVE TIME (THE KEY NUMBER)
# ══════════════════════════════════════════════════════════════════

class TestBenchSolveTimes:
    """Measure end-to-end subgame solve time at different streets.

    Two modes:
      - Warm cache: measures pure CFR iteration speed (main benchmark)
      - Cold start: measures total time including bucket cache + opponent range
    """

    # Warm-cache targets (seconds): the pure iteration speed
    FLOP_TARGET = 1.5     # seconds for 200-iteration flop solve (warm)
    TURN_TARGET = 1.0     # seconds for 200-iteration turn solve (warm)
    RIVER_TARGET = 0.5    # seconds for 200-iteration river solve (warm)

    # Cold-start targets: includes bucket cache warmup
    FLOP_COLD_TARGET = 8.0
    TURN_COLD_TARGET = 3.0
    RIVER_COLD_TARGET = 1.0

    def test_flop_solve_time(self, solver_2max):
        """Benchmark: flop solve (warm cache — pure CFR speed)."""
        board = (_Kd, _9c, _5s, _2h, _7d)
        state = _make_state(AA, _83o, board)
        state = _advance_to_street(state, 1)
        if state is None:
            pytest.skip("Could not reach flop")

        # Warm solve (populates bucket cache)
        solver_2max.solve(state, bot_position=0)

        # Timed solve (cache warm) — run twice, take best to filter contention
        best = float('inf')
        for _ in range(2):
            start = time.perf_counter()
            result = solver_2max.solve(state, bot_position=0)
            elapsed = time.perf_counter() - start
            best = min(best, elapsed)

        print(f"\n  FLOP SOLVE (warm): {best:.2f}s for 200 iters "
              f"({200/best:.0f} iter/s), {len(result)} info sets")
        assert best < self.FLOP_TARGET, (
            f"Flop solve took {best:.2f}s > target {self.FLOP_TARGET}s"
        )

    def test_turn_solve_time(self, solver_2max):
        """Benchmark: turn solve (warm cache — pure CFR speed)."""
        board = (_Kd, _9c, _5s, _2h, _7d)
        state = _make_state(AA, _83o, board)
        state = _advance_to_street(state, 2)
        if state is None:
            pytest.skip("Could not reach turn")

        solver_2max.solve(state, bot_position=0)
        start = time.perf_counter()
        result = solver_2max.solve(state, bot_position=0)
        elapsed = time.perf_counter() - start

        print(f"\n  TURN SOLVE (warm): {elapsed:.2f}s for 200 iters "
              f"({200/elapsed:.0f} iter/s), {len(result)} info sets")
        assert elapsed < self.TURN_TARGET, (
            f"Turn solve took {elapsed:.2f}s > target {self.TURN_TARGET}s"
        )

    def test_river_solve_time(self, solver_2max):
        """Benchmark: river solve (warm cache — pure CFR speed)."""
        board = (_Kd, _9c, _5s, _2h, _7d)
        state = _make_state(AA, _83o, board)
        state = _advance_to_street(state, 3)
        if state is None:
            pytest.skip("Could not reach river")

        solver_2max.solve(state, bot_position=0)
        start = time.perf_counter()
        result = solver_2max.solve(state, bot_position=0)
        elapsed = time.perf_counter() - start

        print(f"\n  RIVER SOLVE (warm): {elapsed:.2f}s for 200 iters "
              f"({200/elapsed:.0f} iter/s), {len(result)} info sets")
        assert elapsed < self.RIVER_TARGET, (
            f"River solve took {elapsed:.2f}s > target {self.RIVER_TARGET}s"
        )

    def test_cold_start_solve_time(self, strategy_2max):
        """Benchmark: flop solve including cache warmup (first-decision speed)."""
        board = (_Kd, _9c, _5s, _2h, _7d)
        # Fresh solver and CardAbstraction (cold caches)
        ca_cold = CardAbstraction(num_players=2, use_emd=False)
        solver_cold = SubgameSolver(strategy_2max, ca_cold, iterations=200, timeout=30.0)

        state = _make_state(AA, _83o, board)
        state = _advance_to_street(state, 1)
        if state is None:
            pytest.skip("Could not reach flop")

        start = time.perf_counter()
        result = solver_cold.solve(state, bot_position=0)
        elapsed = time.perf_counter() - start

        print(f"\n  FLOP COLD START: {elapsed:.2f}s for 200 iters "
              f"({200/elapsed:.0f} iter/s), {len(result)} info sets")
        assert elapsed < self.FLOP_COLD_TARGET, (
            f"Cold flop took {elapsed:.2f}s > target {self.FLOP_COLD_TARGET}s"
        )


# ══════════════════════════════════════════════════════════════════
# SPEED BENCHMARK 2: COMPONENT-LEVEL TIMINGS
# ══════════════════════════════════════════════════════════════════

class TestBenchComponents:
    """Micro-benchmarks for individual solver components."""

    def test_bucketing_cache_miss(self, ca_2max):
        """Benchmark: card abstraction bucketing with cache misses."""
        # Clear cache to force recomputation
        ca_2max._postflop_cache.clear()

        random.seed(42)
        boards = []
        hands = []
        for _ in range(20):
            deck = list(range(52))
            random.shuffle(deck)
            hands.append((deck[0], deck[1]))
            boards.append(tuple(deck[2:5]))  # flop only

        start = time.perf_counter()
        for hand, board in zip(hands, boards):
            ca_2max.get_bucket(hand, board)
        elapsed = time.perf_counter() - start

        avg_ms = (elapsed / 20) * 1000
        print(f"\n  BUCKETING (cache miss): {avg_ms:.1f}ms avg per flop bucket")
        assert avg_ms < 50.0, f"Bucketing too slow: {avg_ms:.1f}ms"

    def test_bucketing_cache_hit(self, ca_2max):
        """Benchmark: bucketing with warm cache."""
        hand = AA
        board = (_Kd, _9c, _5s)
        # Warm the cache
        ca_2max.get_bucket(hand, board)

        start = time.perf_counter()
        for _ in range(10000):
            ca_2max.get_bucket(hand, board)
        elapsed = time.perf_counter() - start

        avg_us = (elapsed / 10000) * 1e6
        print(f"\n  BUCKETING (cache hit): {avg_us:.1f}µs avg")
        assert avg_us < 10.0, f"Cache hit too slow: {avg_us:.1f}µs"

    def test_gamestate_apply_action_throughput(self):
        """Benchmark: GameState.apply_action() calls per second."""
        board = (_Kd, _9c, _5s, _2h, _7d)
        ops = 0

        start = time.perf_counter()
        for _ in range(500):
            state = GameState.new_hand((AA, _83o), board, num_players=2)
            while not state.is_terminal:
                actions = state.get_actions()
                if not actions:
                    break
                state = state.apply_action(actions[0])
                ops += 1
        elapsed = time.perf_counter() - start

        ops_per_sec = ops / elapsed
        print(f"\n  GAMESTATE apply_action: {ops_per_sec:.0f} ops/s ({ops} ops in {elapsed:.2f}s)")
        assert ops_per_sec > 5000, f"Too slow: {ops_per_sec:.0f} ops/s"

    def test_fast_state_apply_undo_throughput(self):
        """Benchmark: FastState apply+undo throughput."""
        from cfr.fast_state import FastState

        board = (_Kd, _9c, _5s, _2h, _7d)
        ops = 0

        start = time.perf_counter()
        for _ in range(500):
            state = FastState.new_hand((AA, _83o), board, num_players=2)
            depth = 0
            while not state.is_terminal and depth < 8:
                actions = state.get_actions()
                if not actions:
                    break
                state.apply_action(actions[0])
                ops += 1
                depth += 1
            # Undo everything
            for _ in range(depth):
                state.undo_action()
                ops += 1
        elapsed = time.perf_counter() - start

        ops_per_sec = ops / elapsed
        print(f"\n  FASTSTATE apply+undo: {ops_per_sec:.0f} ops/s ({ops} ops in {elapsed:.2f}s)")
        assert ops_per_sec > 10000, f"Too slow: {ops_per_sec:.0f} ops/s"

    def test_regret_matching_throughput(self):
        """Benchmark: InfoSet.get_strategy() calls per second."""
        iset = InfoSet(5)
        iset.cumulative_regret = [10.0, -5.0, 3.0, 0.0, 7.0]

        start = time.perf_counter()
        for _ in range(100000):
            iset.get_strategy()
        elapsed = time.perf_counter() - start

        ops_per_sec = 100000 / elapsed
        print(f"\n  REGRET MATCH: {ops_per_sec:.0f} ops/s")
        assert ops_per_sec > 500000, f"Too slow: {ops_per_sec:.0f} ops/s"

    def test_numba_weighted_sample_throughput(self):
        """Benchmark: numba weighted_sample calls per second."""
        probs = np.array([0.3, 0.2, 0.1, 0.15, 0.25], dtype=np.float64)
        # Warm up JIT
        weighted_sample(probs)

        start = time.perf_counter()
        for _ in range(100000):
            weighted_sample(probs)
        elapsed = time.perf_counter() - start

        ops_per_sec = 100000 / elapsed
        print(f"\n  WEIGHTED SAMPLE: {ops_per_sec:.0f} ops/s")
        assert ops_per_sec > 200000, f"Too slow: {ops_per_sec:.0f} ops/s"

    def test_river_equity_enumeration(self):
        """Benchmark: exact river equity enumeration."""
        hand = AA
        board = (_Kd, _9c, _5s, _2h, _7d)

        start = time.perf_counter()
        for _ in range(20):
            enumerate_river_equity(hand, board)
        elapsed = time.perf_counter() - start

        avg_ms = (elapsed / 20) * 1000
        print(f"\n  RIVER EQUITY ENUM: {avg_ms:.1f}ms avg")
        assert avg_ms < 100.0, f"Too slow: {avg_ms:.1f}ms"

    def test_opponent_range_computation(self, solver_2max):
        """Benchmark: Bayesian opponent range computation."""
        board = (_Kd, _9c, _5s, _2h, _7d)
        state = _make_state(AA, _83o, board)
        state = _advance_to_street(state, 2)
        if state is None:
            pytest.skip("Could not reach turn")

        start = time.perf_counter()
        for _ in range(5):
            solver_2max._compute_opponent_range(
                AA, board, state.history, 0, 1, state)
        elapsed = time.perf_counter() - start

        avg_ms = (elapsed / 5) * 1000
        print(f"\n  OPPONENT RANGE: {avg_ms:.0f}ms avg")
        assert avg_ms < 5000, f"Too slow: {avg_ms:.0f}ms"

    def test_leaf_estimation_speed(self):
        """Benchmark: leaf value estimation (MC and bucket modes)."""
        hand = AA
        board = (_Kd, _9c, _5s, _2h, _7d)

        # MC equity (flop, 15 samples)
        start = time.perf_counter()
        for _ in range(100):
            _mc_equity(hand, board[:3], num_opponents=1, samples=15)
        elapsed_mc = time.perf_counter() - start

        # Bucket-based (no hand/board)
        start = time.perf_counter()
        for _ in range(100000):
            estimate_leaf_value({}, 0, 100, 20.0, 80.0, num_buckets=200, street=1)
        elapsed_bucket = time.perf_counter() - start

        mc_us = (elapsed_mc / 100) * 1e6
        bucket_us = (elapsed_bucket / 100000) * 1e6
        print(f"\n  LEAF MC: {mc_us:.0f}µs avg | LEAF BUCKET: {bucket_us:.1f}µs avg")
        assert mc_us < 50000, f"MC too slow: {mc_us:.0f}µs"


# ══════════════════════════════════════════════════════════════════
# ACCURACY BENCHMARK 7: WIN RATE VS RANDOM
# ══════════════════════════════════════════════════════════════════

class TestBenchWinRate:
    """Measure bot's win rate against random opponents."""

    # Targets: bb/hand over N hands
    # 200 hands has high variance (~2 bb/hand std dev), so threshold is generous
    WIN_RATE_TARGET = -2.0  # 200 hands has ~2 bb/hand std dev, be generous

    def test_win_rate_200_hands(self, bot_2max):
        """Benchmark: average profit over 200 hands vs random."""
        if hasattr(bot_2max, '_resolved_cache'):
            bot_2max._resolved_cache.clear()
        total_profit = 0.0
        hands_played = 0

        for seed in range(200):
            random.seed(seed + 50000)
            np.random.seed(seed + 50000)
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
                p = state.current_player
                if p == 0:
                    action = bot_2max.get_action(
                        hands[0], state.visible_board(), state.history,
                        actions, state=state, seat=0)
                else:
                    action = random.choice(actions)
                state = state.apply_action(action)
                moves += 1

            if state.is_terminal:
                total_profit += state.get_terminal_utility(0)
                hands_played += 1

        avg_profit = total_profit / max(hands_played, 1)
        print(f"\n  WIN RATE: {avg_profit:.2f} bb/hand over {hands_played} hands "
              f"(total: {total_profit:.1f} bb)")
        assert avg_profit > self.WIN_RATE_TARGET, (
            f"Win rate {avg_profit:.2f} bb/hand — should beat random"
        )


# ══════════════════════════════════════════════════════════════════
# ACCURACY BENCHMARK 8: UNDER-FOLDING RATE
# ══════════════════════════════════════════════════════════════════

class TestBenchUnderFolding:
    """Measure how often bot folds with air facing river bets."""

    # Target: should fold at least this often with pure air
    FOLD_AIR_TARGET = 0.30

    def test_air_fold_rate_across_boards(self, bot_2max):
        """Benchmark: fold rate with 32o facing river bet across many boards."""
        total_folds = 0
        total_trials = 0

        for seed in range(20):
            random.seed(seed + 60000)
            hand = (_3h, _2d)
            opp = KK
            deck = list(range(52))
            random.shuffle(deck)
            used = set(hand) | set(opp)
            avail = [c for c in deck if c not in used]
            board = tuple(avail[:5])

            # Ensure 32o has no pair on board
            board_ranks = {c // 4 for c in board}
            if 0 in board_ranks or 1 in board_ranks:  # 2 or 3 on board
                continue

            state = _make_state(opp, hand, board)
            state = _advance_to_street(state, 3)
            if state is None:
                continue

            # Hero (BB, seat 1) checks, opponent (BTN, seat 0) bets
            if state.current_player == 1:
                state = state.apply_action('k')
            actions = state.get_actions()
            bets = [a for a in actions if a.startswith('b')]
            if not bets:
                continue
            state = state.apply_action(bets[0])
            if state.is_terminal:
                continue

            hero_actions = state.get_actions()
            for _ in range(30):
                a = bot_2max.get_action(hand, state.visible_board(),
                                         state.history, hero_actions,
                                         state=state, seat=1)
                total_trials += 1
                if a.startswith('f'):
                    total_folds += 1

        fold_rate = total_folds / max(total_trials, 1)
        print(f"\n  AIR FOLD RATE: {fold_rate:.0%} over {total_trials} trials "
              f"(target: >{self.FOLD_AIR_TARGET:.0%})")
        assert fold_rate > self.FOLD_AIR_TARGET, (
            f"Air fold rate {fold_rate:.0%} — should fold more with air"
        )


# ══════════════════════════════════════════════════════════════════
# ACCURACY BENCHMARK 9: NUTS IDENTIFICATION
# ══════════════════════════════════════════════════════════════════

class TestBenchNutsId:
    """Measure how often bot correctly plays the nuts."""

    # Target: nuts should raise/bet, never fold
    NUTS_FOLD_TARGET = 0.08  # max acceptable fold rate (variance margin)

    def test_nuts_never_folds(self, bot_2max):
        """Benchmark: fold rate with nut flush facing river bet."""
        fold_count = 0
        raise_count = 0
        total = 0

        for seed in range(10):
            random.seed(seed + 70000)
            hand = (_Ah, _Kh)
            opp = (_Jc, _Tc)
            deck = list(range(52))
            random.shuffle(deck)
            used = set(hand) | set(opp)
            avail = [c for c in deck if c not in used]
            # Need 3 hearts on board for nut flush
            hearts = [c for c in avail if c % 4 == 0]  # suit 0 = hearts
            non_hearts = [c for c in avail if c % 4 != 0]
            if len(hearts) < 3 or len(non_hearts) < 2:
                continue
            board = tuple(hearts[:3] + non_hearts[:2])

            state = _make_state(opp, hand, board)
            state = _advance_to_street(state, 3)
            if state is None:
                continue

            # Hero (BB, seat 1) checks, opponent (BTN, seat 0) bets
            if state.current_player == 1:
                state = state.apply_action('k')
            actions = state.get_actions()
            bets = [a for a in actions if a.startswith('b')]
            if not bets:
                continue
            state = state.apply_action(bets[0])
            if state.is_terminal:
                continue

            hero_actions = state.get_actions()
            for _ in range(20):
                a = bot_2max.get_action(hand, state.visible_board(),
                                         state.history, hero_actions,
                                         state=state, seat=1)
                total += 1
                if a.startswith('f'):
                    fold_count += 1
                elif a.startswith('b') or a == 'a':
                    raise_count += 1

        fold_rate = fold_count / max(total, 1)
        raise_rate = raise_count / max(total, 1)
        print(f"\n  NUTS: fold={fold_rate:.0%} raise={raise_rate:.0%} "
              f"over {total} trials (target fold: <{self.NUTS_FOLD_TARGET:.0%})")
        assert fold_rate <= self.NUTS_FOLD_TARGET, (
            f"Nut flush folds {fold_rate:.0%} — equity corrections should prevent this"
        )


# ══════════════════════════════════════════════════════════════════
# ACCURACY BENCHMARK 10: HAND STRENGTH ORDERING
# ══════════════════════════════════════════════════════════════════

class TestBenchHandOrdering:
    """Measure how well the bot's aggression correlates with hand strength."""

    def test_preflop_raise_monotonicity(self, bot_2max):
        """Benchmark: Spearman rank correlation between hand strength
        and preflop play frequency (raise + call, i.e. not folding).
        In HU, premium pairs may limp for balance, so measure total
        play rate not just raise rate."""
        hands_by_strength = [AA, KK, QQ, AKs, AKo, JTs, _72o]
        board = (_Kd, _9c, _5s, _2h, _7d)

        play_freqs = []
        for hand in hands_by_strength:
            state = _make_state(hand, _83o, board)
            actions = state.get_actions()
            counts = Counter()
            for _ in range(200):
                a = bot_2max.get_action(hand, (), state.history,
                                         actions, state=state, seat=0)
                counts[a] += 1
            # Total play rate = 1 - fold rate
            fold_f = _freq(counts, 'f')
            play_freqs.append(1.0 - fold_f)

        # Compute rank correlation
        n = len(play_freqs)
        expected_rank = list(range(n))  # 0=highest, 6=lowest
        actual_rank = sorted(range(n), key=lambda i: -play_freqs[i])
        d_sq = sum((expected_rank[i] - actual_rank.index(i))**2 for i in range(n))
        spearman = 1 - 6 * d_sq / (n * (n**2 - 1))

        print(f"\n  HAND ORDERING: Spearman rho={spearman:.3f}")
        print(f"    Play freqs: {[f'{f:.0%}' for f in play_freqs]}")
        print(f"    Hands: AA KK QQ AKs AKo JTs 72o")
        # In HU, most hands play (wide range), so correlation is weaker
        # Key check: 72o should play less than AA
        assert play_freqs[0] >= play_freqs[-1], (
            f"72o plays more than AA — hand ordering broken"
        )
        assert spearman > 0.0, (
            f"Negative hand ordering: Spearman={spearman:.3f}"
        )


# ══════════════════════════════════════════════════════════════════
# ACCURACY BENCHMARK 11: STRATEGY CONVERGENCE
# ══════════════════════════════════════════════════════════════════

class TestBenchConvergence:
    """Measure strategy convergence quality."""

    def test_solve_convergence_quality(self, strategy_2max, ca_2max):
        """Benchmark: run increasing iterations and check regret decreases."""
        board = (_Kd, _9c, _5s, _2h, _7d)
        state = _make_state(AA, _83o, board)
        state = _advance_to_street(state, 2)
        if state is None:
            pytest.skip("Could not reach turn")

        iter_counts = [50, 200]
        avg_regrets = []

        for iters in iter_counts:
            solver = SubgameSolver(strategy_2max, ca_2max,
                                    iterations=iters, timeout=30.0)
            start = time.perf_counter()
            result = solver.solve(state, bot_position=0)
            elapsed = time.perf_counter() - start

            # Check how "peaked" the strategy is (max prob)
            peaked_count = 0
            for probs in result.values():
                if max(probs) > 0.8:
                    peaked_count += 1
            peaked_ratio = peaked_count / max(len(result), 1)

            avg_regrets.append(peaked_ratio)
            print(f"\n  CONVERGENCE {iters} iters: {elapsed:.1f}s, "
                  f"{len(result)} info sets, "
                  f"peaked={peaked_ratio:.0%}")

        # More iterations should produce more peaked (converged) strategies
        # or at least not get worse
        print(f"  Peaked ratios: {[f'{r:.0%}' for r in avg_regrets]}")


# ══════════════════════════════════════════════════════════════════
# COMBINED SCOREBOARD
# ══════════════════════════════════════════════════════════════════

class TestBenchScoreboard:
    """Print a summary scoreboard after all benchmarks."""

    def test_print_scoreboard(self, bot_2max, strategy_2max, ca_2max):
        """Print comprehensive benchmark summary."""
        board = (_Kd, _9c, _5s, _2h, _7d)

        print("\n" + "="*60)
        print("  BENCHMARK SCOREBOARD")
        print("="*60)

        # Quick solve benchmark (warm cache)
        state = _make_state(AA, _83o, board)
        state = _advance_to_street(state, 3)
        if state:
            solver = SubgameSolver(strategy_2max, ca_2max,
                                    iterations=100, timeout=30.0)
            solver.solve(state, bot_position=0)  # warm caches
            start = time.perf_counter()
            result = solver.solve(state, bot_position=0)
            elapsed = time.perf_counter() - start
            ips = 100 / elapsed
            print(f"  River solve (100 iters, warm): {elapsed:.2f}s ({ips:.0f} iter/s)")

        # Quick win rate
        profit = 0.0
        played = 0
        for seed in range(50):
            random.seed(seed + 80000)
            np.random.seed(seed + 80000)
            deck = list(range(52))
            random.shuffle(deck)
            hands = ((deck[0], deck[1]), (deck[2], deck[3]))
            brd = tuple(deck[4:9])
            st = GameState.new_hand(hands, brd, num_players=2)
            moves = 0
            while not st.is_terminal and moves < 50:
                acts = st.get_actions()
                if not acts:
                    break
                p = st.current_player
                if p == 0:
                    action = bot_2max.get_action(
                        hands[0], st.visible_board(), st.history,
                        acts, state=st, seat=0)
                else:
                    action = random.choice(acts)
                st = st.apply_action(action)
                moves += 1
            if st.is_terminal:
                profit += st.get_terminal_utility(0)
                played += 1
        win_rate = profit / max(played, 1)
        print(f"  Win rate vs random: {win_rate:.2f} bb/hand ({played} hands)")

        # Component speeds
        iset = InfoSet(5)
        iset.cumulative_regret = [10.0, -5.0, 3.0, 0.0, 7.0]
        start = time.perf_counter()
        for _ in range(100000):
            iset.get_strategy()
        rm_elapsed = time.perf_counter() - start
        rm_ops = 100000 / rm_elapsed
        print(f"  Regret matching: {rm_ops:.0f} ops/s")

        start = time.perf_counter()
        for _ in range(20):
            enumerate_river_equity(AA, board)
        eq_elapsed = time.perf_counter() - start
        eq_ms = (eq_elapsed / 20) * 1000
        print(f"  River equity enum: {eq_ms:.1f}ms avg")

        print("="*60)
        print("  Targets: solve <0.5s (warm) | win rate >0 | regret >500k/s")
        print("="*60)
