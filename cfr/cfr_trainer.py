"""
Enhanced External Sampling MCCFR+ trainer for N-player NLHE.

Improvements over baseline:
  - Postflop-only mode: uses fixed preflop ranges to sample who enters
    the pot, skipping the massive preflop tree entirely (~20-100x speedup)
  - Regret-based pruning: after warmup, skip exploring actions whose
    cumulative regret is deeply negative (Brown & Sandholm 2015)
  - DCFR discounting: discount regrets and strategy sums from earlier
    iterations using t^alpha / (t^alpha + 1) weighting
  - Warm-start: optionally load an existing strategy to continue training
  - Continuous mode: train indefinitely with periodic checkpoints
  - Bytes keys: compact binary info set keys for fast hashing at scale
  - Bitmask state: folded/all_in/has_acted as int bitmasks (fast undo)
"""

import math
import random
import time
from struct import pack, unpack

from cfr.game_state import GameState, get_position_name
from cfr.fast_state import (
    FastState, _get_action_code, _ACTION_CODES, _CODE_TO_ACTION,
    _SEPARATOR, _STREET_SEP,
)
from cfr.information_set import InfoSet, sample_action
from cfr.card_abstraction import CardAbstraction
from cfr.action_abstraction import set_num_players
from cfr.strategy_store import save_strategy, load_strategy
from cfr.preflop_ranges import (
    get_preflop_action, classify_scenario, has_trained_strategy,
)
from cfr.preflop_solver import hand_to_class

try:
    from cfr.cfr_core import (
        update_regrets_cy,
        accumulate_strategy_cy,
        dcfr_discount_cy,
    )
    _USE_CYTHON = True
except ImportError:
    _USE_CYTHON = False


# Regret-based pruning: scale factor for dynamic threshold.
# Threshold = -PRUNE_SCALE * sqrt(iteration). Actions with cumulative regret
# below this are skipped, matching the O(sqrt(T)) regret variance in MCCFR.
PRUNE_SCALE = 1000

# Position name to seat mapping
_POS_TO_SEAT = {'BTN': 0, 'SB': 1, 'BB': 2, 'UTG': 3, 'MP': 4, 'CO': 5}


def _train_worker(worker_id, iterations, config, result_queue):
    """Module-level worker function for parallel training (picklable)."""
    import random as _rng
    import numpy as _np
    _rng.seed(42 + worker_id * 1000)
    _np.random.seed(42 + worker_id * 1000)

    ca = CardAbstraction(num_players=config['num_players'])
    trainer = CFRTrainer(
        card_abstraction=ca,
        iterations=iterations,
        num_players=config['num_players'],
        postflop_only=config['postflop_only'],
        prune_after=config['prune_after'],
        dcfr_alpha=config['dcfr_alpha'],
        dcfr_beta=config['dcfr_beta'],
        dcfr_gamma=config['dcfr_gamma'],
        exploration=config['exploration'],
    )
    trainer.train()
    strategy = trainer.get_average_strategy()
    result_queue.put((worker_id, strategy))


class CFRTrainer:
    def __init__(self, card_abstraction=None, iterations=100_000,
                 num_players=2, checkpoint_interval=0, checkpoint_path=None,
                 postflop_only=True, prune_after=1000, dcfr_alpha=1.5,
                 dcfr_beta=0.0, dcfr_gamma=2.0, exploration=0.05):
        self.num_players = num_players
        self.card_abstraction = card_abstraction or CardAbstraction(
            num_players=num_players)
        self.iterations = iterations
        self.info_sets: dict[str, InfoSet] = {}
        self.current_iteration = 0
        self.checkpoint_interval = checkpoint_interval
        self.checkpoint_path = checkpoint_path
        self.postflop_only = postflop_only and num_players > 2
        self.prune_after = prune_after
        self.dcfr_alpha = dcfr_alpha
        self.dcfr_beta = dcfr_beta
        self.dcfr_gamma = dcfr_gamma
        self.exploration = exploration
        self.baseline_values = {}

        set_num_players(num_players)

    def _get_prune_threshold(self):
        """Dynamic pruning threshold scaled to MCCFR regret variance O(sqrt(T))."""
        return -PRUNE_SCALE * math.sqrt(max(1, self.current_iteration))

    def _get_exploration_rate(self, iteration):
        """Decay exploration from initial rate to a minimum over training."""
        min_rate = 0.01
        decay_iterations = 500_000
        rate = self.exploration * max(0.0, 1.0 - iteration / decay_iterations)
        return max(min_rate, rate)

    def _get_info_set(self, key, num_actions):
        iset = self.info_sets.get(key)
        if iset is None:
            iset = InfoSet(num_actions)
            self.info_sets[key] = iset
        return iset

    @staticmethod
    def _str_key_to_bytes(key):
        """Convert 'POS:bucket|a1,a2/a3,a4' string key to bytes key (uint16)."""
        pos_bucket, _, history_str = key.partition('|')
        pos, _, bucket_str = pos_bucket.partition(':')
        player = _POS_TO_SEAT.get(pos, 0)
        try:
            bucket = int(bucket_str)
        except ValueError:
            bucket = 0
        parts = [player, bucket, _SEPARATOR]
        streets = history_str.split('/') if history_str else ['']
        for i, street in enumerate(streets):
            if i > 0:
                parts.append(_STREET_SEP)
            if street:
                for a in street.split(','):
                    parts.append(_get_action_code(a))
        return pack(f'>{len(parts)}H', *parts)

    @staticmethod
    def _bytes_key_to_str(key, num_players=6):
        """Convert bytes key (uint16 packed) back to string for export."""
        n_elements = len(key) // 2
        codes = unpack(f'>{n_elements}H', key)
        player = codes[0]
        bucket = codes[1]
        pos = get_position_name(num_players, player)
        # Parse actions after the separator
        streets = []
        current = []
        for code in codes[3:]:  # skip player, bucket, separator
            if code == _STREET_SEP:
                streets.append(current)
                current = []
            else:
                current.append(_CODE_TO_ACTION.get(code, f'?{code}'))
        streets.append(current)
        history_str = '/'.join(','.join(s) for s in streets)
        return f'{pos}:{bucket}|{history_str}'

    def warm_start(self, strategy_path):
        """Load an existing strategy to continue training from."""
        try:
            strategy = load_strategy(strategy_path)
            for str_key, probs in strategy.items():
                n = len(probs)
                iset = InfoSet(n)
                iset.strategy_sum = list(probs)
                self.info_sets[str_key] = iset
            print(f"  Warm-started from {len(self.info_sets)} info sets",
                  flush=True)
        except Exception as e:
            print(f"  Warm-start failed: {e}", flush=True)

    def train(self):
        ca = self.card_abstraction
        n = self.num_players
        self._t0 = time.time()

        for i in range(self.iterations):
            self.current_iteration = i

            deck = list(range(52))
            random.shuffle(deck)
            hands = tuple(
                (deck[j * 2], deck[j * 2 + 1]) for j in range(n))
            board = tuple(deck[n * 2: n * 2 + 5])

            if self.postflop_only:
                state = self._sample_preflop_fast(hands, board)
                if state is None or state.is_terminal:
                    continue

                cache = {}
                vis_map = ((), board[:3], board[:4], board[:5])
                for p in range(n):
                    if (state.folded >> p) & 1:  # bitmask check
                        continue
                    h = hands[p]
                    for s in range(state.street, 4):
                        cache[(p, s)] = ca.get_bucket(h, vis_map[s])

                use_pruning = i >= self.prune_after
                for tp in range(n):
                    if not ((state.folded >> tp) & 1):  # bitmask check
                        self._cfr_fast(state, tp, cache, use_pruning)
            else:
                cache = {}
                vis_map = ((), board[:3], board[:4], board[:5])
                for p in range(n):
                    h = hands[p]
                    for s in range(4):
                        cache[(p, s)] = ca.get_bucket(h, vis_map[s])

                state = FastState.new_hand(hands, board)
                use_pruning = i >= self.prune_after
                for tp in range(n):
                    self._cfr_fast(state, tp, cache, use_pruning)

            if (i + 1) % 100_000 == 0:
                self._dcfr_discount(i + 1)
                # Cap regrets for numerical stability
                for iset in self.info_sets.values():
                    iset.cap_regrets(max_regret=1e9)

            if (i + 1) % 10_000 == 0:
                avg_r = self._avg_positive_regret()
                active = len(self.info_sets)
                elapsed = time.time() - self._t0 if hasattr(self, '_t0') else 0
                rate = (i + 1) / max(elapsed, 1)
                print(f"  iter {i+1:>8,}  |  "
                      f"info sets: {active:>9,}  |  "
                      f"avg regret: {avg_r:.4f}  |  "
                      f"{rate:.0f} it/s", flush=True)

            if (self.checkpoint_interval > 0
                    and (i + 1) % self.checkpoint_interval == 0
                    and self.checkpoint_path):
                print(f"  checkpoint at iter {i+1:,}...", flush=True)
                save_strategy(self.get_average_strategy(),
                              self.checkpoint_path)
                print(f"  saved.", flush=True)

    def _sample_preflop_fast(self, hands, board):
        n = self.num_players
        state = FastState.new_hand(hands, board)

        while not state.is_terminal and state.street == 0:
            seat = state.current_player
            hand = hands[seat]
            pos = get_position_name(n, seat)
            legal = state.get_actions()

            if not legal:
                break

            hand_key = hand_to_class(hand)
            scenario = classify_scenario(state.history, pos, n)
            action_probs = get_preflop_action(
                hand_key, pos, scenario, legal, history=state.history,
                num_players=n)

            probs = [action_probs.get(a, 0.0) for a in legal]
            total = sum(probs)
            if total > 0:
                probs = [p / total for p in probs]
            else:
                u = 1.0 / len(legal)
                probs = [u] * len(legal)

            idx = sample_action(probs)
            state.apply_action(legal[idx])

        if state.is_terminal or state.street == 0:
            return None

        return state

    def _cfr_fast(self, state, traversing_player, cache, use_pruning=False):
        """CFR traversal using mutable FastState with apply/undo and bytes keys."""
        if state.is_terminal:
            return state.get_terminal_utility(traversing_player)

        player = state.current_player
        bucket = cache.get((player, state.street))
        if bucket is None:
            return 0.0
        actions = state.get_actions()
        n_actions = len(actions)

        if n_actions == 0:
            return 0.0

        info_key = state.get_info_set_key_fast(player, bucket)
        info_set = self._get_info_set(info_key, n_actions)
        strategy = info_set.get_strategy()

        if player == traversing_player:
            utilities = [0.0] * n_actions
            node_util = 0.0

            for i in range(n_actions):
                if (use_pruning
                        and info_set.cumulative_regret[i] < self._get_prune_threshold()):
                    continue
                # Reach pruning: skip near-zero strategy actions
                if use_pruning and strategy[i] < 0.001:
                    continue
                state.apply_action(actions[i])
                utilities[i] = self._cfr_fast(
                    state, traversing_player, cache, use_pruning)
                state.undo_action()
                node_util += strategy[i] * utilities[i]

            if _USE_CYTHON:
                update_regrets_cy(info_set.cumulative_regret, utilities,
                                  node_util, n_actions)
            else:
                cr = info_set.cumulative_regret
                for i in range(n_actions):
                    cr[i] = max(0.0, cr[i] + utilities[i] - node_util)

            return node_util
        else:
            explore_rate = self._get_exploration_rate(self.current_iteration)
            if explore_rate > 0 and random.random() < explore_rate:
                idx = random.randrange(n_actions)
            else:
                idx = sample_action(strategy)
            weight = self.current_iteration + 1
            if _USE_CYTHON:
                accumulate_strategy_cy(info_set.strategy_sum, strategy,
                                       weight, n_actions)
            else:
                ss = info_set.strategy_sum
                for i in range(n_actions):
                    ss[i] += strategy[i] * weight

            # VR-MCCFR: use baseline to reduce variance
            state.apply_action(actions[idx])
            sampled_val = self._cfr_fast(
                state, traversing_player, cache, use_pruning)
            state.undo_action()

            # VR-MCCFR: use baseline to reduce variance (only during warmup)
            # After prune_after iterations, baselines are stable enough; skip
            # the overhead of tracking new baselines to save memory.
            bl = self.baseline_values.get(info_key)
            if bl is not None and len(bl) == n_actions:
                baseline_sum = sum(strategy[i] * bl[i] for i in range(n_actions))
                correction = sampled_val - bl[idx]
                val = baseline_sum + correction
                alpha = 0.01
                bl[idx] += alpha * (sampled_val - bl[idx])
            elif self.current_iteration < self.prune_after * 5:
                val = sampled_val
                self.baseline_values[info_key] = [0.0] * n_actions
                self.baseline_values[info_key][idx] = sampled_val
            else:
                val = sampled_val

            return val

    def _cfr(self, state, traversing_player, cache, use_pruning=False):
        if state.is_terminal:
            return state.get_terminal_utility(traversing_player)

        player = state.current_player
        bucket = cache.get((player, state.street))
        if bucket is None:
            return 0.0
        actions = state.get_actions()
        n_actions = len(actions)

        if n_actions == 0:
            return 0.0

        info_key = state.get_info_set_key(player, bucket)
        info_set = self._get_info_set(info_key, n_actions)
        strategy = info_set.get_strategy()

        if player == traversing_player:
            utilities = [0.0] * n_actions
            node_util = 0.0

            for i in range(n_actions):
                if (use_pruning
                        and info_set.cumulative_regret[i] < self._get_prune_threshold()):
                    continue
                utilities[i] = self._cfr(
                    state.apply_action(actions[i]), traversing_player,
                    cache, use_pruning)
                node_util += strategy[i] * utilities[i]

            if _USE_CYTHON:
                update_regrets_cy(info_set.cumulative_regret, utilities,
                                  node_util, n_actions)
            else:
                cr = info_set.cumulative_regret
                for i in range(n_actions):
                    cr[i] = max(0.0, cr[i] + utilities[i] - node_util)

            return node_util
        else:
            idx = sample_action(strategy)
            weight = self.current_iteration + 1
            if _USE_CYTHON:
                accumulate_strategy_cy(info_set.strategy_sum, strategy,
                                       weight, n_actions)
            else:
                ss = info_set.strategy_sum
                for i in range(n_actions):
                    ss[i] += strategy[i] * weight
            return self._cfr(
                state.apply_action(actions[idx]), traversing_player,
                cache, use_pruning)

    def _dcfr_discount(self, t):
        alpha = self.dcfr_alpha
        beta = self.dcfr_beta
        gamma = self.dcfr_gamma

        pos_discount = (t ** alpha) / (t ** alpha + 1)
        neg_discount = (t ** beta) / (t ** beta + 1)
        strat_discount = ((t / (t + 1)) ** gamma)

        if _USE_CYTHON:
            for iset in self.info_sets.values():
                dcfr_discount_cy(iset.cumulative_regret, iset.strategy_sum,
                                 pos_discount, neg_discount, strat_discount,
                                 iset.num_actions)
        else:
            for iset in self.info_sets.values():
                cr = iset.cumulative_regret
                ss = iset.strategy_sum
                for i in range(iset.num_actions):
                    if cr[i] > 0:
                        cr[i] *= pos_discount
                    else:
                        cr[i] *= neg_discount
                    ss[i] *= strat_discount

    def _avg_positive_regret(self):
        if not self.info_sets:
            return 0.0
        n = len(self.info_sets)
        if n <= 10000:
            total = sum(sum(iset.cumulative_regret) for iset in self.info_sets.values())
            return total / n
        import itertools
        sample = itertools.islice(self.info_sets.values(), 10000)
        total = sum(sum(iset.cumulative_regret) for iset in sample)
        return total / 10000

    def get_average_strategy(self):
        """Return {info_set_key: [action_probabilities]} for the converged strategy.

        Converts bytes keys (used internally for fast hashing) back to
        string keys (used by server/bot for lookup).
        """
        result = {}
        for key, iset in self.info_sets.items():
            if isinstance(key, bytes):
                str_key = self._bytes_key_to_str(key, self.num_players)
            else:
                str_key = key
            result[str_key] = iset.get_average_strategy().tolist()
        return result

    def merge_info_sets(self, other_info_sets):
        """Merge another info_sets dict into this trainer's info_sets.

        Strategy sums are added (correct for averaging strategies across
        independent CFR runs). Cumulative regrets are averaged.
        """
        for key, other_iset in other_info_sets.items():
            if key in self.info_sets:
                iset = self.info_sets[key]
                if iset.num_actions == other_iset.num_actions:
                    for i in range(iset.num_actions):
                        iset.strategy_sum[i] += other_iset.strategy_sum[i]
                        iset.cumulative_regret[i] = (
                            iset.cumulative_regret[i] + other_iset.cumulative_regret[i]
                        ) / 2.0
            else:
                self.info_sets[key] = other_iset

    def train_parallel(self, num_workers=None):
        """Train using multiple processes, then merge strategies.

        Each worker trains independently for iterations/num_workers iterations.
        Strategies are merged by summing strategy_sums (mathematically correct
        for CFR — averaging independent runs gives the same convergence guarantee).

        Falls back to single-process train() if num_workers <= 1.
        """
        from multiprocessing import Process, Queue
        import os

        if num_workers is None:
            from config import TRAINING_WORKERS
            num_workers = TRAINING_WORKERS
        if num_workers <= 0:
            num_workers = os.cpu_count() or 4

        if num_workers <= 1:
            self.train()
            return

        iters_per_worker = self.iterations // num_workers
        remaining = self.iterations - iters_per_worker * num_workers

        # Pack config as picklable dict for spawn-mode multiprocessing
        worker_config = {
            'num_players': self.num_players,
            'postflop_only': self.postflop_only,
            'prune_after': self.prune_after,
            'dcfr_alpha': self.dcfr_alpha,
            'dcfr_beta': self.dcfr_beta,
            'dcfr_gamma': self.dcfr_gamma,
            'exploration': self.exploration,
        }

        result_queue = Queue()

        print(f"  Launching {num_workers} parallel workers "
              f"({iters_per_worker} iters each)...", flush=True)
        t0 = time.time()

        processes = []
        for w in range(num_workers):
            iters = iters_per_worker + (1 if w < remaining else 0)
            p = Process(target=_train_worker,
                        args=(w, iters, worker_config, result_queue))
            p.start()
            processes.append(p)

        # Collect results
        strategies = {}
        for _ in range(num_workers):
            worker_id, strategy = result_queue.get()
            strategies[worker_id] = strategy
            print(f"  Worker {worker_id} done ({len(strategy)} info sets)",
                  flush=True)

        for p in processes:
            p.join()

        elapsed = time.time() - t0
        print(f"  All workers done in {elapsed:.1f}s. Merging...", flush=True)

        # Merge: average all strategies
        all_keys = set()
        for s in strategies.values():
            all_keys.update(s.keys())

        for key in all_keys:
            probs_list = [s[key] for s in strategies.values() if key in s]
            n_probs = len(probs_list[0])
            avg_probs = [0.0] * n_probs
            for probs in probs_list:
                for i in range(n_probs):
                    avg_probs[i] += probs[i]
            total = sum(avg_probs)
            if total > 0:
                avg_probs = [p / total for p in avg_probs]
            else:
                avg_probs = [1.0 / n_probs] * n_probs

            iset = InfoSet(n_probs)
            iset.strategy_sum = avg_probs
            self.info_sets[key] = iset

        print(f"  Merged {len(self.info_sets)} info sets from "
              f"{num_workers} workers.", flush=True)
