"""
Preflop-only CFR solver for 6-max NLHE.

Solves Nash-approximate preflop ranges by running CFR on just the
preflop decision tree. Two equity estimation modes:

1. Fast mode (default): Uses preflop equity lookup tables for ~instant
   terminal valuations. Equity between hands is estimated using each
   hand's raw equity share among active players.

2. MC mode: Monte Carlo board sampling for exact equity. Slower but
   more accurate for uncommon matchups.

Produces a preflop strategy: {info_set_key: [action_probabilities]}
where keys encode position:hand_class|preflop_actions.
"""

import json
import os
import random
import time

import numpy as np
from phevaluator import evaluate_cards

from cfr.game_state import (
    GameState, STARTING_STACK, SMALL_BLIND, BIG_BLIND,
    get_position_name, action_order, _find_next_actor,
)
from cfr.action_abstraction import (
    get_legal_actions, action_to_chips, set_num_players,
)
from cfr.information_set import InfoSet


# ---- Load preflop equity data for fast mode ---------------------------------
# Loaded lazily per player count; module-level default is 6-max.

_PROJECT_ROOT = os.path.join(os.path.dirname(__file__), '..')
_EQUITY_FILES = {
    2: 'preFlopEquities.json',
    6: 'preFlopEquities6max.json',
    9: 'preFlopEquities9max.json',
}
_EQUITIES_CACHE = {}


def _load_equities(num_players):
    """Load and cache the equity file for the given player count."""
    if num_players in _EQUITIES_CACHE:
        return _EQUITIES_CACHE[num_players]
    # Find best match: exact, then nearest lower
    fname = _EQUITY_FILES.get(num_players)
    if fname is None:
        if num_players > 6:
            fname = _EQUITY_FILES.get(9, _EQUITY_FILES[6])
        elif num_players > 2:
            fname = _EQUITY_FILES[6]
        else:
            fname = _EQUITY_FILES[2]
    path = os.path.join(_PROJECT_ROOT, fname)
    if os.path.exists(path):
        with open(path) as f:
            data = json.load(f)
        _EQUITIES_CACHE[num_players] = data
        return data
    # Fallback
    return _EQUITIES_CACHE.get(6, {})


# Default load for backward compatibility
_EQUITY_PATH = os.path.join(_PROJECT_ROOT, 'preFlopEquities6max.json')
_EQUITIES = {}
if os.path.exists(_EQUITY_PATH):
    with open(_EQUITY_PATH) as _f:
        _EQUITIES = json.load(_f)
_EQUITIES_CACHE[6] = _EQUITIES


# ---- Hand canonicalization --------------------------------------------------

def hand_to_class(hand):
    """Convert a 2-card hand (card indices) to a canonical class string."""
    r0, r1 = hand[0] // 4, hand[1] // 4
    s0, s1 = hand[0] % 4, hand[1] % 4
    if r0 < r1:
        r0, r1 = r1, r0
        s0, s1 = s1, s0
    if r0 == r1:
        return f"{r0} {r1}"
    return f"{r0} {r1} s" if s0 == s1 else f"{r0} {r1} o"


# ---- Fast equity estimation using lookup tables -----------------------------

def fast_equity(hands, active_seats, num_players=None):
    """
    Estimate equity using preflop equity lookup tables.

    For each active player, look up their hand's raw equity.
    Normalize these equities so they sum to 1.0 among active players.
    """
    eq_table = _load_equities(num_players or len(hands))
    default_eq = 1.0 / max(len(active_seats), 1)
    raw = {}
    for seat in active_seats:
        hc = hand_to_class(hands[seat])
        raw[seat] = eq_table.get(hc, default_eq)

    total = sum(raw.values())
    if total <= 0:
        eq = 1.0 / len(active_seats)
        return {s: eq for s in active_seats}

    return {s: raw[s] / total for s in active_seats}


# ---- Monte Carlo equity at flop transition ----------------------------------

def mc_equity(hands, active_seats, n_samples=200):
    """
    Estimate each active player's equity via Monte Carlo board sampling.
    More accurate but slower than fast_equity.
    """
    used = set()
    for seat in range(len(hands)):
        used.add(hands[seat][0])
        used.add(hands[seat][1])

    available = [c for c in range(52) if c not in used]
    wins = {s: 0.0 for s in active_seats}

    for _ in range(n_samples):
        board = random.sample(available, 5)
        best_score = float('inf')
        round_winners = []

        for seat in active_seats:
            score = evaluate_cards(*board, *hands[seat])
            if score < best_score:
                best_score = score
                round_winners = [seat]
            elif score == best_score:
                round_winners.append(seat)

        share = 1.0 / len(round_winners)
        for w in round_winners:
            wins[w] += share

    total = sum(wins.values())
    if total <= 0:
        eq = 1.0 / len(active_seats)
        return {s: eq for s in active_seats}

    return {s: wins[s] / total for s in active_seats}


# ---- Preflop-only game state ------------------------------------------------
# We reuse the main GameState but override terminal detection:
# when the street would advance past 0, we compute equity-based utility.

class PreflopState:
    """
    Lightweight preflop state for CFR traversal.
    Terminal when: everyone folds to one player, or preflop betting completes.
    """
    __slots__ = (
        'num_players', 'hands', 'stacks', 'pot', 'bets',
        'current_player', 'history', 'raises_this_street',
        'is_terminal', 'terminal_type', 'min_raise',
        'folded', 'all_in', 'has_acted', 'last_raiser',
    )

    def __init__(self, **kw):
        for k, v in kw.items():
            setattr(self, k, v)

    @staticmethod
    def new_hand(hands, num_players=None):
        if num_players is None:
            num_players = len(hands)

        stacks = [STARTING_STACK] * num_players
        bets = [0.0] * num_players
        folded = [False] * num_players
        all_in = [False] * num_players
        has_acted = [False] * num_players

        sb_seat = 1 if num_players > 2 else 0
        bb_seat = 2 if num_players > 2 else 1

        stacks[sb_seat] -= SMALL_BLIND
        bets[sb_seat] = SMALL_BLIND
        stacks[bb_seat] -= BIG_BLIND
        bets[bb_seat] = BIG_BLIND

        first_to_act = action_order(num_players, 0)[0]

        return PreflopState(
            num_players=num_players, hands=tuple(hands),
            stacks=tuple(stacks), pot=0.0, bets=tuple(bets),
            current_player=first_to_act, history=((),),
            raises_this_street=1, is_terminal=False,
            terminal_type=None, min_raise=BIG_BLIND,
            folded=tuple(folded), all_in=tuple(all_in),
            has_acted=tuple(has_acted), last_raiser=bb_seat,
        )

    def get_actions(self):
        if self.is_terminal:
            return []
        p = self.current_player
        max_bet = max(self.bets)
        to_call = max_bet - self.bets[p]
        return get_legal_actions(
            pot=self.pot + sum(self.bets),
            to_call=to_call,
            stack=self.stacks[p],
            raises_this_street=self.raises_this_street,
            min_raise=self.min_raise,
            street=0,
        )

    def apply_action(self, action):
        p = self.current_player
        max_bet = max(self.bets)
        to_call = max_bet - self.bets[p]
        total_pot = self.pot + sum(self.bets)

        chips, is_allin = action_to_chips(action, total_pot, to_call, self.stacks[p])

        new_stacks = list(self.stacks)
        new_stacks[p] -= chips
        new_bets = list(self.bets)
        new_bets[p] += chips
        new_folded = list(self.folded)
        new_all_in = list(self.all_in)
        new_has_acted = list(self.has_acted)
        new_has_acted[p] = True

        street_history = list(self.history[-1]) + [action]
        new_history = list(self.history)
        new_history[-1] = tuple(street_history)

        new_raises = self.raises_this_street
        new_min_raise = self.min_raise
        new_last_raiser = self.last_raiser

        if action == 'f':
            new_folded[p] = True
            active = [i for i in range(self.num_players) if not new_folded[i]]
            if len(active) == 1:
                return PreflopState(
                    num_players=self.num_players, hands=self.hands,
                    stacks=tuple(new_stacks), pot=self.pot, bets=tuple(new_bets),
                    current_player=active[0], history=tuple(new_history),
                    raises_this_street=new_raises,
                    is_terminal=True, terminal_type='fold',
                    min_raise=new_min_raise, folded=tuple(new_folded),
                    all_in=tuple(new_all_in), has_acted=tuple(new_has_acted),
                    last_raiser=new_last_raiser,
                )
        else:
            if action.startswith('b') or action == 'a':
                raise_amount = chips - to_call
                if raise_amount > 0:
                    new_raises += 1
                    new_min_raise = max(self.min_raise, raise_amount)
                    new_last_raiser = p
                    new_has_acted = [False] * self.num_players
                    new_has_acted[p] = True
                    for i in range(self.num_players):
                        if new_folded[i] or new_all_in[i]:
                            new_has_acted[i] = True

            if is_allin:
                new_all_in[p] = True

        next_p = _find_next_actor(
            self.num_players, 0, p,
            new_folded, new_all_in, new_has_acted)

        if next_p is None:
            # Preflop betting complete → terminal "flop" node
            return PreflopState(
                num_players=self.num_players, hands=self.hands,
                stacks=tuple(new_stacks),
                pot=self.pot + sum(new_bets),
                bets=tuple(0.0 for _ in range(self.num_players)),
                current_player=p, history=tuple(new_history),
                raises_this_street=0,
                is_terminal=True, terminal_type='flop_transition',
                min_raise=BIG_BLIND, folded=tuple(new_folded),
                all_in=tuple(new_all_in), has_acted=tuple(new_has_acted),
                last_raiser=new_last_raiser,
            )

        return PreflopState(
            num_players=self.num_players, hands=self.hands,
            stacks=tuple(new_stacks), pot=self.pot, bets=tuple(new_bets),
            current_player=next_p, history=tuple(new_history),
            raises_this_street=new_raises,
            is_terminal=False, terminal_type=None,
            min_raise=new_min_raise, folded=tuple(new_folded),
            all_in=tuple(new_all_in), has_acted=tuple(new_has_acted),
            last_raiser=new_last_raiser,
        )

    def get_terminal_utility(self, player, use_mc=False, mc_samples=100):
        invested = STARTING_STACK - self.stacks[player]

        if self.terminal_type == 'fold':
            active = [i for i in range(self.num_players) if not self.folded[i]]
            winner = active[0]
            if player == winner:
                return self.pot + sum(self.bets) - invested
            return -invested

        if self.terminal_type == 'flop_transition':
            active = [i for i in range(self.num_players) if not self.folded[i]]

            if len(active) == 1:
                if player == active[0]:
                    return self.pot - invested
                return -invested

            if use_mc:
                equities = mc_equity(self.hands, active, mc_samples)
            else:
                equities = fast_equity(self.hands, active, self.num_players)
            eq = equities.get(player, 0.0)
            return eq * self.pot - invested

        return 0.0

    def get_info_set_key(self, player, hand_class):
        pos = get_position_name(self.num_players, player)
        if self.num_players <= 6:
            # Full history for small tables
            history_str = ','.join(self.history[0]) if self.history[0] else ''
        else:
            # Abstracted history for large tables:
            # Compress action sequence to (raises, callers, folds, facing_amount)
            # Include legal action count to avoid size mismatches
            actions = self.history[0] if self.history[0] else ()
            n_raises = sum(1 for a in actions if a.startswith('b') or a == 'a')
            n_calls = sum(1 for a in actions if a == 'c')
            n_folds = sum(1 for a in actions if a == 'f')
            n_legal = len(self.get_actions())
            history_str = f'r{n_raises}c{n_calls}f{n_folds}L{n_legal}'
        return f'PF|{pos}:{hand_class}|{history_str}'


# ---- Preflop CFR Trainer ----------------------------------------------------

class PreflopCFRTrainer:
    """
    Trains Nash-approximate preflop ranges using external-sampling MCCFR+.

    Two equity modes:
    - fast (default): lookup table equity, ~100x faster, good convergence
    - mc: Monte Carlo equity, slower but exact

    For >6 players, uses equity buckets instead of all 169 hand classes
    to keep memory manageable. The bucket count is configurable.
    """

    def __init__(self, num_players=6, iterations=50_000, equity_samples=100,
                 use_mc=False, num_buckets=None):
        self.num_players = num_players
        self.iterations = iterations
        self.equity_samples = equity_samples
        self.use_mc = use_mc
        self.info_sets = {}
        self.current_iteration = 0

        set_num_players(num_players)

        # For large tables, bucket hands by equity to reduce info set count.
        # 169 classes × 9 positions × deep trees = OOM.
        # 20 buckets × 9 positions = manageable.
        self.use_buckets = num_players > 6
        self.num_buckets = num_buckets or (20 if self.use_buckets else 0)
        self._bucket_map = {}
        if self.use_buckets:
            self._build_bucket_map()

    def _build_bucket_map(self):
        """Map each of 169 hand classes to an equity bucket."""
        eq_table = _load_equities(self.num_players)
        if not eq_table:
            eq_table = _EQUITIES
        # Sort hands by equity
        hands_eq = sorted(eq_table.items(), key=lambda x: x[1])
        n = len(hands_eq)
        for i, (hand_key, _) in enumerate(hands_eq):
            bucket = min(int(i * self.num_buckets / n), self.num_buckets - 1)
            self._bucket_map[hand_key] = bucket
        print(f"  Preflop bucketing: {n} hands → {self.num_buckets} buckets")

    def _get_hand_key(self, hand):
        """Get info set hand identifier — full class or bucket."""
        hc = hand_to_class(hand)
        if self.use_buckets:
            return str(self._bucket_map.get(hc, 0))
        return hc

    def train(self):
        n = self.num_players

        for i in range(self.iterations):
            self.current_iteration = i

            # Deal hands
            deck = list(range(52))
            random.shuffle(deck)
            hands = tuple(
                (deck[j * 2], deck[j * 2 + 1]) for j in range(n))

            # Precompute hand keys for all players
            hand_classes = {}
            for p in range(n):
                hand_classes[p] = self._get_hand_key(hands[p])

            state = PreflopState.new_hand(hands)

            # For large tables, sample one traversing player per iteration
            # (external sampling MCCFR) to keep memory manageable
            if n > 6:
                tp = random.randint(0, n - 1)
                self._cfr(state, tp, hand_classes)
            else:
                for tp in range(n):
                    self._cfr(state, tp, hand_classes)

            if (i + 1) % 5000 == 0:
                avg_r = self._avg_positive_regret()
                print(f"  preflop iter {i+1:>8}/{self.iterations}  |  "
                      f"info sets: {len(self.info_sets):>7}  |  "
                      f"avg regret: {avg_r:.4f}", flush=True)

    def _cfr(self, state, traversing_player, hand_classes):
        if state.is_terminal:
            return state.get_terminal_utility(
                traversing_player, use_mc=self.use_mc,
                mc_samples=self.equity_samples)

        player = state.current_player
        hc = hand_classes[player]
        actions = state.get_actions()

        if not actions:
            return 0.0

        info_key = state.get_info_set_key(player, hc)

        if info_key not in self.info_sets:
            self.info_sets[info_key] = InfoSet(len(actions))
        info_set = self.info_sets[info_key]
        strategy = info_set.get_strategy()

        if player == traversing_player:
            utilities = np.zeros(len(actions))
            node_util = 0.0

            for i, action in enumerate(actions):
                utilities[i] = self._cfr(
                    state.apply_action(action), traversing_player, hand_classes)
                node_util += strategy[i] * utilities[i]

            for i in range(len(actions)):
                info_set.cumulative_regret[i] = max(
                    0.0,
                    info_set.cumulative_regret[i] + utilities[i] - node_util)

            return node_util
        else:
            idx = np.random.choice(len(actions), p=strategy)
            weight = self.current_iteration + 1
            ss = info_set.strategy_sum
            for i in range(len(actions)):
                ss[i] += strategy[i] * weight
            return self._cfr(
                state.apply_action(actions[idx]), traversing_player, hand_classes)

    def _avg_positive_regret(self):
        if not self.info_sets:
            return 0.0
        total = sum(sum(iset.cumulative_regret) for iset in self.info_sets.values())
        return total / len(self.info_sets)

    def get_average_strategy(self):
        return {key: iset.get_average_strategy().tolist()
                for key, iset in self.info_sets.items()}

    def get_range_table(self):
        """
        Extract a position-aware range table from the trained strategy.

        Returns dict:
        {
            position: {
                action_sequence: {
                    hand_class: {action: probability}
                }
            }
        }
        """
        table = {}
        for key, iset in self.info_sets.items():
            # Parse key: 'PF|POS:hand_class|action_history'
            parts = key.split('|')
            if len(parts) != 3:
                continue
            pos_hand = parts[1]
            action_history = parts[2]
            pos, hand_class = pos_hand.split(':', 1)

            probs = iset.get_average_strategy()
            # We need to know the actions — reconstruct from state
            # For now, store raw probabilities keyed by the full info set key
            if pos not in table:
                table[pos] = {}
            if action_history not in table[pos]:
                table[pos][action_history] = {}
            table[pos][action_history][hand_class] = probs.tolist()

        return table
