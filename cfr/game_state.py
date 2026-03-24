"""
Immutable game state for CFR traversal in N-player no-limit hold'em.

Supports 2-9 players with proper position tracking, side pots,
and multi-way showdown.

Canonical seating (seats always start BTN=0, SB=1, BB=2, then early→late):
  9-max: BTN SB BB UTG UTG1 UTG2 MP HJ CO
  8-max: BTN SB BB UTG UTG1 MP HJ CO
  7-max: BTN SB BB UTG MP HJ CO
  6-max: BTN SB BB UTG MP CO
  5-max: BTN SB BB MP CO
  4-max: BTN SB BB CO
  3-max: BTN SB BB
  2-max: BTN BB  (BTN is also SB)

Preflop action order:  UTG → ... → BTN → SB → BB
Postflop action order: SB → BB → UTG → ... → BTN
"""

from cfr.action_abstraction import get_legal_actions, action_to_chips
from cfr.evaluator import determine_winner, determine_winners

STARTING_STACK = 100.0
SMALL_BLIND = 0.5
BIG_BLIND = 1.0

# Position names by player count
_POSITION_NAMES = {
    2: {0: 'BTN', 1: 'BB'},
    3: {0: 'BTN', 1: 'SB', 2: 'BB'},
    4: {0: 'BTN', 1: 'SB', 2: 'BB', 3: 'CO'},
    5: {0: 'BTN', 1: 'SB', 2: 'BB', 3: 'MP', 4: 'CO'},
    6: {0: 'BTN', 1: 'SB', 2: 'BB', 3: 'UTG', 4: 'MP', 5: 'CO'},
    7: {0: 'BTN', 1: 'SB', 2: 'BB', 3: 'UTG', 4: 'MP', 5: 'HJ', 6: 'CO'},
    8: {0: 'BTN', 1: 'SB', 2: 'BB', 3: 'UTG', 4: 'UTG1', 5: 'MP', 6: 'HJ', 7: 'CO'},
    9: {0: 'BTN', 1: 'SB', 2: 'BB', 3: 'UTG', 4: 'UTG1', 5: 'UTG2', 6: 'MP', 7: 'HJ', 8: 'CO'},
}

# Map any position to its nearest 6-max equivalent for blueprint fallback
_6MAX_FALLBACK = {
    'BTN': 'BTN', 'SB': 'SB', 'BB': 'BB',
    'UTG': 'UTG', 'UTG1': 'UTG', 'UTG2': 'UTG',
    'MP': 'MP', 'HJ': 'MP', 'CO': 'CO',
}


def get_position_name(num_players, seat):
    """Map seat index to position name for a given table size."""
    return _POSITION_NAMES.get(num_players, {}).get(seat, str(seat))


def map_to_6max_position(position_name):
    """Map any position name to its nearest 6-max equivalent."""
    return _6MAX_FALLBACK.get(position_name, position_name)


def lookup_with_fallback(strategy, info_key, num_players):
    """Look up an info set key in the strategy, with multi-level fallback.

    Fallback chain:
    1. Exact key match
    2. Position mapping (e.g., UTG1 → UTG for 6-max strategies)
    3. Nearby bucket (±1, ±2) — handles finer bucket granularity mismatches

    Returns the strategy probs list or None if not found.
    """
    if info_key in strategy:
        return strategy[info_key]

    # Try position mapping fallback
    pos, _, rest = info_key.partition(':')
    if num_players != 6:
        fallback_pos = map_to_6max_position(pos)
        if fallback_pos != pos:
            fallback_key = f'{fallback_pos}:{rest}'
            if fallback_key in strategy:
                return strategy[fallback_key]

    # Try nearby buckets (±1, ±2) — helps when bucket granularity differs
    bucket_str, _, history = rest.partition('|')
    try:
        bucket = int(bucket_str)
    except ValueError:
        return None

    for delta in (1, -1, 2, -2):
        neighbor = bucket + delta
        if neighbor < 0:
            continue
        neighbor_key = f'{pos}:{neighbor}|{history}'
        if neighbor_key in strategy:
            return strategy[neighbor_key]

    # History truncation: keep only the current street's actions
    # This finds blueprint entries for similar spots even with novel earlier play
    if '/' in history:
        current_street = history.rsplit('/', 1)[-1]
        for p in (pos, map_to_6max_position(pos)):
            for d in (0, 1, -1, 2, -2):
                b = bucket + d
                if b < 0:
                    continue
                trunc_key = f'{p}:{b}|{current_street}'
                if trunc_key in strategy:
                    return strategy[trunc_key]

    return None


# Precomputed action orders: _ACTION_ORDER_CACHE[(num_players, street)] = list
_ACTION_ORDER_CACHE = {}
# Precomputed seat→index in action order: _SEAT_INDEX_CACHE[(num_players, street, seat)] = int
_SEAT_INDEX_CACHE = {}


def action_order(num_players, street):
    """Return list of seat indices in the order they act for this street."""
    key = (num_players, street)
    cached = _ACTION_ORDER_CACHE.get(key)
    if cached is not None:
        return cached

    if num_players == 2:
        result = [0, 1] if street == 0 else [1, 0]
    elif street == 0:
        bb_seat = 2
        result = [(bb_seat + i) % num_players for i in range(1, num_players)]
        result.append(bb_seat)
    else:
        sb_seat = 1
        result = [(sb_seat + i) % num_players for i in range(num_players)]

    _ACTION_ORDER_CACHE[key] = result
    for idx, seat in enumerate(result):
        _SEAT_INDEX_CACHE[(num_players, street, seat)] = idx
    return result


class GameState:
    __slots__ = (
        'num_players', 'hands', 'board', 'stacks', 'pot', 'bets',
        'current_player', 'history', 'street',
        'raises_this_street', 'is_terminal', 'terminal_type',
        'min_raise', 'folded', 'all_in', 'has_acted', 'last_raiser',
    )

    def __init__(self, num_players, hands, board, stacks, pot, bets,
                 current_player, history, street,
                 raises_this_street, is_terminal, terminal_type,
                 min_raise, folded, all_in, has_acted, last_raiser):
        self.num_players = num_players
        self.hands = hands
        self.board = board
        self.stacks = stacks
        self.pot = pot
        self.bets = bets
        self.current_player = current_player
        self.history = history
        self.street = street
        self.raises_this_street = raises_this_street
        self.is_terminal = is_terminal
        self.terminal_type = terminal_type
        self.min_raise = min_raise
        self.folded = folded
        self.all_in = all_in
        self.has_acted = has_acted
        self.last_raiser = last_raiser

    @staticmethod
    def new_hand(hands, board, num_players=None):
        """Create initial preflop state after blinds are posted."""
        if num_players is None:
            num_players = len(hands)

        stacks = [STARTING_STACK] * num_players
        bets = [0.0] * num_players
        folded = [False] * num_players
        all_in = [False] * num_players
        has_acted = [False] * num_players

        if num_players == 2:
            sb_seat, bb_seat = 0, 1
        else:
            sb_seat, bb_seat = 1, 2

        stacks[sb_seat] -= SMALL_BLIND
        bets[sb_seat] = SMALL_BLIND
        stacks[bb_seat] -= BIG_BLIND
        bets[bb_seat] = BIG_BLIND

        first_to_act = action_order(num_players, 0)[0]

        return GameState(
            num_players=num_players,
            hands=tuple(hands),
            board=board,
            stacks=tuple(stacks),
            pot=0.0,
            bets=tuple(bets),
            current_player=first_to_act,
            history=((),),
            street=0,
            raises_this_street=1,   # BB counts as a raise
            is_terminal=False,
            terminal_type=None,
            min_raise=BIG_BLIND,
            folded=tuple(folded),
            all_in=tuple(all_in),
            has_acted=tuple(has_acted),
            last_raiser=bb_seat,
        )

    # ------------------------------------------------------------------
    # Board visibility
    # ------------------------------------------------------------------

    def visible_board(self):
        """Board cards visible at the current street."""
        if self.street == 0:
            return ()
        if self.street == 1:
            return self.board[:3]
        if self.street == 2:
            return self.board[:4]
        return self.board[:5]

    # ------------------------------------------------------------------
    # Legal actions
    # ------------------------------------------------------------------

    def get_actions(self):
        """Legal abstract actions for the current player."""
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

    # ------------------------------------------------------------------
    # Apply action
    # ------------------------------------------------------------------

    def apply_action(self, action):
        """Return a new GameState with the action applied."""
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

        # ---- Fold ----
        if action == 'f':
            new_folded[p] = True
            active = [i for i in range(self.num_players) if not new_folded[i]]
            if len(active) == 1:
                return GameState(
                    self.num_players, self.hands, self.board,
                    tuple(new_stacks), self.pot, tuple(new_bets),
                    active[0], tuple(new_history), self.street,
                    new_raises, True, 'fold',
                    new_min_raise, tuple(new_folded), tuple(new_all_in),
                    tuple(new_has_acted), new_last_raiser,
                )
        else:
            # ---- Raise / bet tracking (non-fold actions only) ----
            if action.startswith('b') or action == 'a':
                raise_amount = chips - to_call
                if raise_amount > 0:
                    new_raises += 1
                    new_min_raise = max(self.min_raise, raise_amount)
                    new_last_raiser = p
                    # Reset has_acted — everyone must respond to the raise
                    new_has_acted = [False] * self.num_players
                    new_has_acted[p] = True
                    for i in range(self.num_players):
                        if new_folded[i] or new_all_in[i]:
                            new_has_acted[i] = True

            if is_allin:
                new_all_in[p] = True

        # ---- Find next actor or end street ----
        next_p = _find_next_actor(
            self.num_players, self.street, p,
            new_folded, new_all_in, new_has_acted)

        if next_p is None:
            return self._advance_or_showdown(
                new_stacks, new_bets, new_folded, new_all_in,
                new_has_acted, new_history, new_raises, new_min_raise,
                new_last_raiser,
            )

        return GameState(
            self.num_players, self.hands, self.board,
            tuple(new_stacks), self.pot, tuple(new_bets),
            next_p, tuple(new_history), self.street,
            new_raises, False, None,
            new_min_raise, tuple(new_folded), tuple(new_all_in),
            tuple(new_has_acted), new_last_raiser,
        )

    # ------------------------------------------------------------------
    # Street transitions
    # ------------------------------------------------------------------

    def _advance_or_showdown(self, new_stacks, new_bets, new_folded,
                              new_all_in, new_has_acted, new_history,
                              new_raises, new_min_raise, new_last_raiser):
        """Advance to the next street, or go to showdown."""
        new_pot = self.pot + sum(new_bets)
        zero_bets = tuple(0.0 for _ in range(self.num_players))

        active = [i for i in range(self.num_players) if not new_folded[i]]
        can_act = [i for i in active if not new_all_in[i]]

        # Showdown if: river complete, or <=1 player can still bet
        if self.street == 3 or len(can_act) <= 1:
            return GameState(
                self.num_players, self.hands, self.board,
                tuple(new_stacks), new_pot, zero_bets,
                self.current_player, tuple(new_history), self.street,
                0, True, 'showdown',
                BIG_BLIND, tuple(new_folded), tuple(new_all_in),
                tuple(True for _ in range(self.num_players)), None,
            )

        # Advance to next street
        new_history_adv = list(new_history) + [()]
        reset_has_acted = tuple(
            True if (new_folded[i] or new_all_in[i]) else False
            for i in range(self.num_players)
        )

        # Find first actor for next street (postflop order)
        order = action_order(self.num_players, self.street + 1)
        first_actor = None
        for s in order:
            if not new_folded[s] and not new_all_in[s]:
                first_actor = s
                break

        if first_actor is None:
            # Everyone is all-in or folded — showdown
            return GameState(
                self.num_players, self.hands, self.board,
                tuple(new_stacks), new_pot, zero_bets,
                self.current_player, tuple(new_history), self.street,
                0, True, 'showdown',
                BIG_BLIND, tuple(new_folded), tuple(new_all_in),
                tuple(True for _ in range(self.num_players)), None,
            )

        return GameState(
            self.num_players, self.hands, self.board,
            tuple(new_stacks), new_pot, zero_bets,
            first_actor, tuple(new_history_adv), self.street + 1,
            0, False, None,
            BIG_BLIND, tuple(new_folded), tuple(new_all_in),
            reset_has_acted, None,
        )

    # ------------------------------------------------------------------
    # Terminal utility
    # ------------------------------------------------------------------

    def get_terminal_utility(self, player):
        """Utility for *player* at a terminal node (profit/loss from this hand)."""
        invested = STARTING_STACK - self.stacks[player]

        if self.terminal_type == 'fold':
            active = [i for i in range(self.num_players) if not self.folded[i]]
            winner = active[0]
            if player == winner:
                total = self.pot + sum(self.bets)
                return total - invested
            return -invested

        if self.terminal_type == 'showdown':
            pots = self._calculate_side_pots()
            winnings = 0.0
            for pot_amount, eligible in pots:
                winners = determine_winners(self.board[:5], self.hands, eligible)
                if player in winners:
                    winnings += pot_amount / len(winners)
            return winnings - invested

        return 0.0

    def _calculate_side_pots(self):
        """Return list of (pot_amount, [eligible_seats]) for showdown."""
        invested = [STARTING_STACK - self.stacks[i] for i in range(self.num_players)]
        levels = sorted(set(invested))

        pots = []
        prev = 0.0
        for level in levels:
            if level <= prev:
                continue
            increment = level - prev
            contributors = [i for i in range(self.num_players) if invested[i] >= level]
            eligible = [i for i in contributors if not self.folded[i]]
            pot_amount = increment * len(contributors)
            if pot_amount > 0 and eligible:
                pots.append((pot_amount, eligible))
            prev = level

        return pots

    # ------------------------------------------------------------------
    # Information set key
    # ------------------------------------------------------------------

    def get_info_set_key(self, player, card_bucket):
        """Info set key: '<position>:<bucket>|<street0_actions/street1_actions/...>'"""
        pos = get_position_name(self.num_players, player)
        history_str = '/'.join(','.join(s) for s in self.history)
        return f'{pos}:{card_bucket}|{history_str}'


# ------------------------------------------------------------------
# Module-level helpers
# ------------------------------------------------------------------

def _find_next_actor(num_players, street, after_seat, folded, all_in, has_acted):
    """Return next seat to act, or None if the street is over."""
    order = action_order(num_players, street)
    start_idx = _SEAT_INDEX_CACHE.get((num_players, street, after_seat))
    if start_idx is None:
        return None

    n = len(order)
    for offset in range(1, n + 1):
        candidate = order[(start_idx + offset) % n]
        if not folded[candidate] and not all_in[candidate] and not has_acted[candidate]:
            return candidate
    return None
