"""Manages poker game sessions between a human player and the bot."""

import random
import uuid
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from cfr.game_state import GameState, STARTING_STACK
from cfr.evaluator import determine_winner
import CardUtils

RANKS = ['A', 'K', 'Q', 'J', 'T', '9', '8', '7', '6', '5', '4', '3', '2']


class GameManager:
    def __init__(self, bot):
        self.bot = bot
        self.sessions = {}

    def new_game(self, player_position=0):
        session_id = uuid.uuid4().hex[:8]
        deck = list(range(52))
        random.shuffle(deck)

        hand0 = (deck[0], deck[1])
        hand1 = (deck[2], deck[3])
        board = tuple(deck[4:9])

        player_hand = hand0 if player_position == 0 else hand1
        bot_hand = hand1 if player_position == 0 else hand0

        state = GameState.new_hand((hand0, hand1), board)
        session = {
            'id': session_id,
            'state': state,
            'player_position': player_position,
            'player_hand': player_hand,
            'bot_hand': bot_hand,
        }
        self.sessions[session_id] = session
        return session_id, self._maybe_bot_acts(session)

    def player_action(self, session_id, action):
        session = self.sessions.get(session_id)
        if not session:
            return None
        state = session['state']
        if state.is_terminal:
            return self._state_dict(session)
        if state.current_player != session['player_position']:
            return {'error': 'Not your turn'}
        legal = state.get_actions()
        if action not in legal:
            return {'error': f'Illegal action. Legal: {legal}'}
        session['state'] = state.apply_action(action)
        return self._maybe_bot_acts(session)

    def get_state(self, session_id):
        session = self.sessions.get(session_id)
        return self._state_dict(session) if session else None

    # ------------------------------------------------------------------
    # Range strategy for every canonical hand at current decision point
    # ------------------------------------------------------------------

    def get_range_strategy(self, session_id):
        session = self.sessions.get(session_id)
        if not session:
            return None

        state = session['state']
        if state.is_terminal:
            return {'status': 'terminal'}

        actions = state.get_actions()
        if not actions:
            return {'status': 'no_actions'}

        board_set = set(state.visible_board())
        visible = state.visible_board()
        ca = self.bot.card_abstraction
        strat_table = self.bot.strategy
        player = state.current_player
        pp = session['player_position']

        hands = {}
        for row in range(13):
            for col in range(13):
                rank_a = 12 - row      # row's rank index (A=12..2=0)
                rank_b = 12 - col

                if row == col:
                    label = RANKS[row] + RANKS[col]
                    combos = _pair_combos(rank_a, board_set)
                elif row < col:
                    label = RANKS[row] + RANKS[col] + 's'
                    combos = _suited_combos(rank_a, rank_b, board_set)
                else:
                    label = RANKS[col] + RANKS[row] + 'o'
                    combos = _offsuit_combos(rank_b, rank_a, board_set)

                if not combos:
                    hands[f'{row},{col}'] = {'label': label, 'probs': None}
                    continue

                # Use first valid combo as representative (fast)
                combo = combos[0]
                bucket = ca.get_bucket(combo, visible)
                info_key = state.get_info_set_key(player, bucket)

                if info_key in strat_table:
                    probs = strat_table[info_key]
                    if len(probs) != len(actions):
                        probs = [1.0 / len(actions)] * len(actions)
                else:
                    probs = [1.0 / len(actions)] * len(actions)

                hands[f'{row},{col}'] = {
                    'label': label,
                    'probs': [round(p, 4) for p in probs],
                    'bucket': bucket,
                    'combos': len(combos),
                }

        return {
            'status': 'ok',
            'actions': actions,
            'current_player': 'player' if player == pp else 'bot',
            'street_name': ['Preflop', 'Flop', 'Turn', 'River'][min(state.street, 3)],
            'pot': round(state.pot + state.bets[0] + state.bets[1], 1),
            'hands': hands,
        }

    # ------------------------------------------------------------------

    def _maybe_bot_acts(self, session):
        state = session['state']
        bot_pos = 1 - session['player_position']
        bot_actions = []

        while not state.is_terminal and state.current_player == bot_pos:
            hand = session['bot_hand']
            action = self.bot.get_action(
                hand, state.visible_board(), state.history, state.get_actions(),
                state=state, bot_position=bot_pos)
            bot_actions.append(action)
            state = state.apply_action(action)
            session['state'] = state

        result = self._state_dict(session)
        result['bot_actions'] = bot_actions
        return result

    def _state_dict(self, session):
        state = session['state']
        pp = session['player_position']

        d = {
            'session_id': session['id'],
            'player_hand': _cards(session['player_hand']),
            'board': _cards(state.visible_board()),
            'pot': round(state.pot + state.bets[0] + state.bets[1], 1),
            'player_stack': round(state.stacks[pp], 1),
            'bot_stack': round(state.stacks[1 - pp], 1),
            'player_bet': round(state.bets[pp], 1),
            'bot_bet': round(state.bets[1 - pp], 1),
            'street': state.street,
            'street_name': ['Preflop', 'Flop', 'Turn', 'River'][min(state.street, 3)],
            'is_over': state.is_terminal,
            'legal_actions': state.get_actions() if not state.is_terminal else [],
            'current_turn': 'player' if (not state.is_terminal and
                                          state.current_player == pp) else 'bot',
        }

        if state.is_terminal:
            d['bot_hand'] = _cards(session['bot_hand'])
            d['full_board'] = _cards(state.board[:5])

            if state.terminal_type == 'fold':
                if state.current_player == pp:
                    d['winner'] = 'bot'
                    d['result_text'] = 'You folded'
                else:
                    d['winner'] = 'player'
                    d['result_text'] = 'Bot folded'
            else:
                w = determine_winner(state.board[:5], state.hands[0], state.hands[1])
                if w == -1:
                    d['winner'] = 'tie'
                    d['result_text'] = 'Split pot'
                elif w == pp:
                    d['winner'] = 'player'
                    d['result_text'] = 'You win at showdown!'
                else:
                    d['winner'] = 'bot'
                    d['result_text'] = 'Bot wins at showdown'

            d['player_profit'] = round(state.get_terminal_utility(pp), 1)

        return d


# ---- helpers ----

def _cards(cards):
    return [CardUtils.numToCard(c) for c in cards]


def _pair_combos(rank, board_set):
    out = []
    for s1 in range(4):
        for s2 in range(s1 + 1, 4):
            c1, c2 = rank * 4 + s1, rank * 4 + s2
            if c1 not in board_set and c2 not in board_set:
                out.append((max(c1, c2), min(c1, c2)))
    return out


def _suited_combos(rank_hi, rank_lo, board_set):
    out = []
    for s in range(4):
        c1, c2 = rank_hi * 4 + s, rank_lo * 4 + s
        if c1 not in board_set and c2 not in board_set:
            out.append((max(c1, c2), min(c1, c2)))
    return out


def _offsuit_combos(rank_hi, rank_lo, board_set):
    out = []
    for s1 in range(4):
        for s2 in range(4):
            if s1 == s2:
                continue
            c1, c2 = rank_hi * 4 + s1, rank_lo * 4 + s2
            if c1 not in board_set and c2 not in board_set:
                out.append((max(c1, c2), min(c1, c2)))
    return out
