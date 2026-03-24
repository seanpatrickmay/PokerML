"""
Mutable game state optimized for CFR training.

Instead of creating new immutable GameState objects at each apply_action
(which requires copying all arrays), this uses in-place mutation with an
undo stack.

Optimizations over baseline:
  - Bitmasks for folded/all_in/has_acted (int copy vs list copy on undo)
  - Dynamic 2-byte action codes for compact bytes keys
  - Fast info set key construction via bytes (faster hash than strings)
"""

from struct import pack, unpack
import threading

from cfr.action_abstraction import get_legal_actions, action_to_chips
from cfr.evaluator import evaluate_all_hands, winners_from_scores
from cfr.game_state import (
    STARTING_STACK, SMALL_BLIND, BIG_BLIND,
    get_position_name, action_order, _SEAT_INDEX_CACHE,
)

# ---------------------------------------------------------------------------
# Dynamic action code registry (2-byte codes for bytes keys)
# ---------------------------------------------------------------------------
# Fixed codes for common actions
_ACTION_CODES = {'f': 1, 'k': 2, 'c': 3, 'a': 4}
_CODE_TO_ACTION = {1: 'f', 2: 'k', 3: 'c', 4: 'a'}
_NEXT_CODE = 10  # start bet codes at 10
_ACTION_CODE_LOCK = threading.Lock()

# Special markers (as uint16)
_SEPARATOR = 0xFFFF      # header separator (player|bucket|actions)
_STREET_SEP = 0xFFFE     # street separator


def _get_action_code(action):
    """Get or assign a 2-byte action code (thread-safe)."""
    global _NEXT_CODE
    code = _ACTION_CODES.get(action)
    if code is not None:
        return code
    with _ACTION_CODE_LOCK:
        # Double-check after acquiring lock
        code = _ACTION_CODES.get(action)
        if code is not None:
            return code
        code = _NEXT_CODE
        _NEXT_CODE += 1
        _ACTION_CODES[action] = code
        _CODE_TO_ACTION[code] = action
        return code


def get_action_code_maps():
    """Return (action_to_code, code_to_action) for strategy export."""
    return _ACTION_CODES, _CODE_TO_ACTION


# Pre-register common bet actions so codes are stable across runs
for _a in ('b33', 'b50', 'b67', 'b75', 'b100', 'b150', 'b175', 'b200',
           'b60', 'b77', 'b135', 'b159'):
    _get_action_code(_a)


# ---------------------------------------------------------------------------
# Bitmask helpers
# ---------------------------------------------------------------------------

def _popcount(mask):
    """Count set bits (uses int.bit_count on Python 3.10+)."""
    return mask.bit_count()


def _lowest_set_bit_index(mask):
    """Index of lowest set bit, or -1."""
    if mask == 0:
        return -1
    return (mask & -mask).bit_length() - 1


def _iter_set_bits(mask, n):
    """Yield indices of set bits up to n."""
    for i in range(n):
        if (mask >> i) & 1:
            yield i


def _full_mask(n):
    """Bitmask with bits 0..n-1 set."""
    return (1 << n) - 1


# ---------------------------------------------------------------------------
# Fast _find_next_actor using bitmasks
# ---------------------------------------------------------------------------

def _find_next_actor_bitmask(num_players, street, after_seat,
                             folded, all_in, has_acted):
    """Find next seat to act using bitmask state. Returns seat or None."""
    order = action_order(num_players, street)
    start_idx = _SEAT_INDEX_CACHE.get((num_players, street, after_seat))
    if start_idx is None:
        return None

    n = len(order)
    # Eligible = not folded, not all-in, not has_acted
    excluded = folded | all_in | has_acted
    for offset in range(1, n + 1):
        candidate = order[(start_idx + offset) % n]
        if not ((excluded >> candidate) & 1):
            return candidate
    return None


# ---------------------------------------------------------------------------
# FastState
# ---------------------------------------------------------------------------

class FastState:
    """Mutable game state with undo for CFR traversal. Uses bitmasks."""

    __slots__ = (
        'num_players', 'hands', 'board', 'stacks', 'pot', 'bets',
        'current_player', 'history', 'street',
        'raises_this_street', 'is_terminal', 'terminal_type',
        'min_raise', 'folded', 'all_in', 'has_acted', 'last_raiser',
        '_undo_stack', '_key_buf',
    )

    def __init__(self, num_players, hands, board):
        self.num_players = num_players
        self.hands = hands
        self.board = board
        self.stacks = [STARTING_STACK] * num_players
        self.pot = 0.0
        self.bets = [0.0] * num_players
        self.current_player = 0
        self.history = [[]]
        self.street = 0
        self.raises_this_street = 0
        self.is_terminal = False
        self.terminal_type = None
        self.min_raise = BIG_BLIND
        self.folded = 0          # bitmask
        self.all_in = 0          # bitmask
        self.has_acted = 0       # bitmask
        self.last_raiser = None
        self._undo_stack = []
        self._key_buf = []       # reusable buffer for key construction

    @staticmethod
    def new_hand(hands, board, num_players=None):
        if num_players is None:
            num_players = len(hands)

        s = FastState(num_players, hands, board)

        sb_seat = 1 if num_players > 2 else 0
        bb_seat = 2 if num_players > 2 else 1

        s.stacks[sb_seat] -= SMALL_BLIND
        s.bets[sb_seat] = SMALL_BLIND
        s.stacks[bb_seat] -= BIG_BLIND
        s.bets[bb_seat] = BIG_BLIND

        s.current_player = action_order(num_players, 0)[0]
        s.raises_this_street = 1
        s.last_raiser = bb_seat
        return s

    def visible_board(self):
        if self.street == 0:
            return ()
        if self.street == 1:
            return self.board[:3]
        if self.street == 2:
            return self.board[:4]
        return self.board[:5]

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
            street=self.street,
            max_bet=max_bet,
        )

    def apply_action(self, action):
        """Apply action in-place, push undo info onto stack."""
        p = self.current_player
        max_bet = max(self.bets)
        to_call = max_bet - self.bets[p]
        total_pot = self.pot + sum(self.bets)

        chips, is_allin = action_to_chips(action, total_pot, to_call, self.stacks[p])
        p_bit = 1 << p

        # Save undo state — bitmasks are just ints, no list copy needed
        undo = (
            p,                          # 0: old current_player
            self.stacks[p],             # 1: old stack
            self.bets[p],               # 2: old bet
            self.folded,                # 3: old folded (int)
            self.all_in,                # 4: old all_in (int)
            self.has_acted,             # 5: old has_acted (int) — cheap!
            self.raises_this_street,    # 6: old raises
            self.is_terminal,           # 7: old is_terminal
            self.terminal_type,         # 8: old terminal_type
            self.min_raise,             # 9: old min_raise
            self.last_raiser,           # 10: old last_raiser
            self.pot,                   # 11: old pot
            self.bets[:],               # 12: old bets (still need list copy for street advance)
            self.street,                # 13: old street
            None,                       # 14: history undo info
        )

        # Apply mutation
        self.stacks[p] -= chips
        self.bets[p] += chips
        self.has_acted |= p_bit
        self.history[-1].append(action)

        if action == 'f':
            self.folded |= p_bit
            n = self.num_players
            active_mask = _full_mask(n) & ~self.folded
            if _popcount(active_mask) == 1:
                self.current_player = _lowest_set_bit_index(active_mask)
                self.is_terminal = True
                self.terminal_type = 'fold'
                self._undo_stack.append(undo)
                return
        else:
            if action.startswith('b') or action == 'a':
                raise_amount = chips - to_call
                if raise_amount > 0:
                    self.raises_this_street += 1
                    self.min_raise = max(self.min_raise, raise_amount)
                    self.last_raiser = p
                    # Everyone must act again except raiser, folded, all-in
                    self.has_acted = p_bit | self.folded | self.all_in
            if is_allin:
                self.all_in |= p_bit

        # Find next actor
        next_p = _find_next_actor_bitmask(
            self.num_players, self.street, p,
            self.folded, self.all_in, self.has_acted)

        if next_p is None:
            # Advance street or showdown
            n = self.num_players
            new_pot = self.pot + sum(self.bets)
            active_mask = _full_mask(n) & ~self.folded
            can_act_mask = active_mask & ~self.all_in

            if self.street == 3 or _popcount(can_act_mask) <= 1:
                # Showdown
                self.pot = new_pot
                for i in range(n):
                    self.bets[i] = 0.0
                self.is_terminal = True
                self.terminal_type = 'showdown'
            else:
                # Next street
                self.pot = new_pot
                for i in range(n):
                    self.bets[i] = 0.0
                self.street += 1
                self.raises_this_street = 0
                self.min_raise = BIG_BLIND
                self.last_raiser = None
                # has_acted = folded | all_in (those can't act)
                self.has_acted = self.folded | self.all_in
                self.history.append([])
                undo = (*undo[:14], True)  # mark that we advanced street

                order = action_order(self.num_players, self.street)
                first = None
                for seat in order:
                    if not ((self.folded | self.all_in) >> seat & 1):
                        first = seat
                        break
                if first is None:
                    self.is_terminal = True
                    self.terminal_type = 'showdown'
                else:
                    self.current_player = first
        else:
            self.current_player = next_p

        self._undo_stack.append(undo)

    def undo_action(self):
        """Undo the last apply_action, restoring previous state."""
        undo = self._undo_stack.pop()
        p = undo[0]

        # Restore player state — bitmask restore is just int assignment
        self.stacks[p] = undo[1]
        self.bets[p] = undo[2]
        self.folded = undo[3]
        self.all_in = undo[4]
        self.has_acted = undo[5]       # int copy, not list slice!
        self.raises_this_street = undo[6]
        self.is_terminal = undo[7]
        self.terminal_type = undo[8]
        self.min_raise = undo[9]
        self.last_raiser = undo[10]
        self.pot = undo[11]
        self.current_player = p

        advanced_street = undo[14]
        if advanced_street:
            self.bets[:] = undo[12]
            self.street = undo[13]
            self.history.pop()
        else:
            self.bets[:] = undo[12]
            self.street = undo[13]

        # Pop the action from history
        self.history[-1].pop()

    def get_terminal_utility(self, player):
        invested = STARTING_STACK - self.stacks[player]
        n = self.num_players

        if self.terminal_type == 'fold':
            active_mask = _full_mask(n) & ~self.folded
            winner = _lowest_set_bit_index(active_mask)
            if player == winner:
                return self.pot + sum(self.bets) - invested
            return -invested

        if self.terminal_type == 'showdown':
            active = list(_iter_set_bits(_full_mask(n) & ~self.folded, n))

            if not self.all_in:
                scores = evaluate_all_hands(self.board[:5], self.hands, active)
                winners = winners_from_scores(scores, active)
                total_pot = self.pot + sum(self.bets)
                if player in winners:
                    return total_pot / len(winners) - invested
                return -invested

            # Side pots needed
            pots = self._calculate_side_pots()
            scores = evaluate_all_hands(self.board[:5], self.hands, active)
            winnings = 0.0
            for pot_amount, eligible in pots:
                winners = winners_from_scores(scores, eligible)
                if player in winners:
                    winnings += pot_amount / len(winners)
            return winnings - invested

        return 0.0

    def _calculate_side_pots(self):
        n = self.num_players
        invested = [STARTING_STACK - self.stacks[i] for i in range(n)]
        levels = sorted(set(invested))

        pots = []
        prev = 0.0
        for level in levels:
            if level <= prev:
                continue
            increment = level - prev
            contributors = [i for i in range(n) if invested[i] >= level]
            eligible = [i for i in contributors
                        if not ((self.folded >> i) & 1)]
            pot_amount = increment * len(contributors)
            if pot_amount > 0 and eligible:
                pots.append((pot_amount, eligible))
            prev = level
        return pots

    # ---- String key (for compatibility with server/live_solver) ----

    def get_info_set_key(self, player, card_bucket):
        pos = get_position_name(self.num_players, player)
        parts = []
        for s in self.history:
            parts.append(','.join(s))
        return f'{pos}:{card_bucket}|{"/".join(parts)}'

    # ---- Fast bytes key (for training) ----

    def get_info_set_key_fast(self, player, card_bucket):
        """Build a compact bytes key for fast dict lookup.

        Format: [player:uint16] [bucket:uint16] [SEPARATOR:uint16]
                [action_code:uint16]... [STREET_SEP:uint16] [action_code:uint16]...

        2 bytes per element. Much faster to hash than equivalent strings.
        """
        buf = self._key_buf
        buf.clear()
        buf.append(player)
        buf.append(card_bucket)
        buf.append(_SEPARATOR)
        for i, street in enumerate(self.history):
            if i > 0:
                buf.append(_STREET_SEP)
            for a in street:
                buf.append(_get_action_code(a))
        # Pack as uint16 array
        return pack(f'>{len(buf)}H', *buf)
