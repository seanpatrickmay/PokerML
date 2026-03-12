"""
Immutable game state for CFR traversal in heads-up no-limit hold'em.

Player 0 = SB/BTN (acts first preflop, last postflop)
Player 1 = BB (acts second preflop, first postflop)
"""

from cfr.action_abstraction import get_legal_actions, action_to_chips
from cfr.evaluator import determine_winner

STARTING_STACK = 100.0
SMALL_BLIND = 0.5
BIG_BLIND = 1.0


class GameState:
    __slots__ = (
        'hands', 'board', 'stacks', 'pot', 'bets',
        'current_player', 'history', 'street',
        'raises_this_street', 'num_actions',
        'is_terminal', 'terminal_type', 'min_raise',
    )

    def __init__(self, hands, board, stacks, pot, bets,
                 current_player, history, street,
                 raises_this_street, num_actions,
                 is_terminal, terminal_type, min_raise):
        self.hands = hands
        self.board = board
        self.stacks = stacks
        self.pot = pot
        self.bets = bets
        self.current_player = current_player
        self.history = history
        self.street = street
        self.raises_this_street = raises_this_street
        self.num_actions = num_actions
        self.is_terminal = is_terminal
        self.terminal_type = terminal_type
        self.min_raise = min_raise

    @staticmethod
    def new_hand(hands, board):
        """Create initial preflop state after blinds are posted."""
        return GameState(
            hands=hands,
            board=board,
            stacks=(STARTING_STACK - SMALL_BLIND, STARTING_STACK - BIG_BLIND),
            pot=0.0,
            bets=(SMALL_BLIND, BIG_BLIND),
            current_player=0,
            history=((),),
            street=0,
            raises_this_street=1,   # BB counts as a raise
            num_actions=0,
            is_terminal=False,
            terminal_type=None,
            min_raise=BIG_BLIND,
        )

    def visible_board(self):
        """Board cards visible at the current street."""
        if self.street == 0:
            return ()
        if self.street == 1:
            return self.board[:3]
        if self.street == 2:
            return self.board[:4]
        return self.board[:5]

    def get_actions(self):
        """Legal abstract actions for the current player."""
        if self.is_terminal:
            return []
        to_call = self.bets[1 - self.current_player] - self.bets[self.current_player]
        return get_legal_actions(
            pot=self.pot + self.bets[0] + self.bets[1],
            to_call=to_call,
            stack=self.stacks[self.current_player],
            raises_this_street=self.raises_this_street,
            min_raise=self.min_raise,
        )

    def apply_action(self, action):
        """Return a new GameState with the action applied."""
        p = self.current_player
        opp = 1 - p
        to_call = self.bets[opp] - self.bets[p]
        total_pot = self.pot + self.bets[0] + self.bets[1]

        chips, _ = action_to_chips(action, total_pot, to_call, self.stacks[p])

        new_stacks = list(self.stacks)
        new_stacks[p] -= chips

        new_bets = list(self.bets)
        new_bets[p] += chips

        street_history = list(self.history[-1]) + [action]
        new_history = list(self.history)
        new_history[-1] = tuple(street_history)

        new_raises = self.raises_this_street
        new_min_raise = self.min_raise
        new_num_actions = self.num_actions + 1

        # ---- Fold ----
        if action == 'f':
            return GameState(
                self.hands, self.board, tuple(new_stacks),
                self.pot, tuple(new_bets), p,
                tuple(new_history), self.street, new_raises,
                new_num_actions, True, 'fold', new_min_raise,
            )

        # ---- Raise / bet tracking ----
        if action.startswith('b') or action == 'a':
            raise_amount = chips - to_call
            new_raises += 1
            new_min_raise = max(self.min_raise, raise_amount)

        # ---- Handle all-in call for less (return excess) ----
        if action == 'c' and new_bets[p] < new_bets[opp]:
            excess = new_bets[opp] - new_bets[p]
            new_bets[opp] -= excess
            new_stacks[opp] += excess

        # ---- Check if street / hand is over ----
        bets_matched = new_bets[0] == new_bets[1]
        someone_allin = new_stacks[0] <= 0 or new_stacks[1] <= 0
        street_over = bets_matched and new_num_actions >= 2

        if street_over or (someone_allin and bets_matched):
            new_pot = self.pot + new_bets[0] + new_bets[1]

            if self.street == 3 or someone_allin:
                return GameState(
                    self.hands, self.board, tuple(new_stacks),
                    new_pot, (0.0, 0.0), opp,
                    tuple(new_history), self.street, 0,
                    0, True, 'showdown', BIG_BLIND,
                )

            # Advance to next street (BB acts first postflop)
            new_history_adv = list(new_history) + [()]
            return GameState(
                self.hands, self.board, tuple(new_stacks),
                new_pot, (0.0, 0.0), 1,
                tuple(new_history_adv), self.street + 1, 0,
                0, False, None, BIG_BLIND,
            )

        # Continue same street, other player acts
        return GameState(
            self.hands, self.board, tuple(new_stacks),
            self.pot, tuple(new_bets), opp,
            tuple(new_history), self.street, new_raises,
            new_num_actions, False, None, new_min_raise,
        )

    def get_terminal_utility(self, player):
        """Utility for *player* at a terminal node (profit/loss from this hand)."""
        if self.terminal_type == 'fold':
            folder = self.current_player
            total_pot = self.pot + self.bets[0] + self.bets[1]
            if player == folder:
                return self.stacks[player] - STARTING_STACK
            return self.stacks[player] + total_pot - STARTING_STACK

        if self.terminal_type == 'showdown':
            winner = determine_winner(self.board[:5], self.hands[0], self.hands[1])
            invested = STARTING_STACK - self.stacks[player]
            if winner == -1:
                return self.pot / 2.0 - invested
            if winner == player:
                return self.pot - invested
            return -invested

        return 0.0

    def get_info_set_key(self, player, card_bucket):
        """Information set key: '<bucket>|<street0_actions/street1_actions/...>'"""
        history_str = '/'.join(','.join(s) for s in self.history)
        return f'{card_bucket}|{history_str}'
