"""Manages poker game sessions with VPIP/PFR tracking, live solving, and opponent modeling."""

import random
import uuid
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from cfr.game_state import GameState, STARTING_STACK, BIG_BLIND, get_position_name
from cfr.evaluator import determine_winners
from cfr.action_abstraction import action_to_chips
from cfr.live_solver import LiveSolver
from cfr.opponent_model import OpponentModel
import CardUtils

RANKS = ['A', 'K', 'Q', 'J', 'T', '9', '8', '7', '6', '5', '4', '3', '2']


class PlayerStats:
    """Tracks VPIP and PFR for a single seat across hands."""

    def __init__(self):
        self.hands_dealt = 0
        self.vpip_count = 0  # voluntarily put $ in pot
        self.pfr_count = 0   # pre-flop raise

    @property
    def vpip(self):
        return self.vpip_count / max(1, self.hands_dealt)

    @property
    def pfr(self):
        return self.pfr_count / max(1, self.hands_dealt)

    def to_dict(self):
        return {
            'hands': self.hands_dealt,
            'vpip': round(self.vpip * 100, 1),
            'pfr': round(self.pfr * 100, 1),
        }


class GameManager:
    def __init__(self, bot, num_players=6):
        self.bot = bot
        self.num_players = num_players
        self.sessions = {}
        self.stats = {i: PlayerStats() for i in range(num_players)}
        self.live_solver = None
        self.opponent_model = OpponentModel()

        # Initialize live solver if possible
        if hasattr(bot, 'strategy') and hasattr(bot, 'card_abstraction'):
            self.live_solver = LiveSolver(
                bot.strategy, bot.card_abstraction)

    def new_game(self, player_seat=0, num_players=None):
        if num_players is None:
            num_players = self.num_players

        # Update action abstraction for this player count
        from cfr.action_abstraction import set_num_players
        set_num_players(num_players)

        # Ensure stats dict covers enough seats
        for i in range(num_players):
            if i not in self.stats:
                self.stats[i] = PlayerStats()

        session_id = uuid.uuid4().hex[:8]
        deck = list(range(52))
        random.shuffle(deck)

        hands = tuple(
            (deck[i * 2], deck[i * 2 + 1]) for i in range(num_players))
        board = tuple(deck[num_players * 2: num_players * 2 + 5])

        state = GameState.new_hand(hands, board)

        # Track hands dealt for all seats
        for i in range(num_players):
            self.stats[i].hands_dealt += 1

        # Track preflop actions for this hand
        # Clear resolved subgame cache for fresh hand
        if hasattr(self.bot, '_resolved_cache'):
            self.bot._resolved_cache.clear()

        session = {
            'id': session_id,
            'state': state,
            'num_players': num_players,
            'player_seat': player_seat,
            'hands': hands,
            'preflop_actions': {i: [] for i in range(num_players)},
        }
        self.sessions[session_id] = session
        return session_id, self._advance_bots(session)

    def player_action(self, session_id, action):
        session = self.sessions.get(session_id)
        if not session:
            return None
        state = session['state']
        if state.is_terminal:
            return self._state_dict(session)
        if state.current_player != session['player_seat']:
            return {'error': 'Not your turn'}
        legal = state.get_actions()
        if action not in legal:
            return {'error': f'Illegal action. Legal: {legal}'}

        # Stop live solver when player acts
        if self.live_solver:
            self.live_solver.stop()

        # Track preflop stats and opponent model
        seat = state.current_player
        if state.street == 0:
            self._track_preflop_action(session, seat, action)

        # Record action in opponent model (for human players)
        situation = self._classify_action_situation(state)
        action_type = self._normalize_action_type(action)
        pos_name = get_position_name(state.num_players, seat)
        self.opponent_model.record_action(
            seat, state.street, situation, action_type, pos_name)

        session['state'] = state.apply_action(action)
        return self._advance_bots(session)

    def get_state(self, session_id):
        session = self.sessions.get(session_id)
        return self._state_dict(session) if session else None

    def get_live_recommendation(self, session_id):
        """Get the live solver's current recommendation."""
        if not self.live_solver:
            return {'status': 'unavailable'}

        rec = self.live_solver.get_recommendation()
        if rec is None:
            return {'status': 'idle'}

        iters = rec['iterations']
        is_preflop = rec.get('converged', False) and iters == 0
        confidence = min(100, int(iters / 50)) if not is_preflop else 100

        return {
            'status': 'solving' if self.live_solver.is_solving else 'done',
            'actions': rec['actions'],
            'probs': [round(p, 4) for p in rec['probs']],
            'iterations': iters,
            'mode': 'preflop_ranges' if is_preflop else 'mccfr',
            'confidence': confidence,
        }

    def get_player_stats(self):
        """Return VPIP/PFR stats for all seats."""
        return {i: self.stats[i].to_dict() for i in range(self.num_players)}

    # ------------------------------------------------------------------
    # Preflop stat tracking
    # ------------------------------------------------------------------

    def _track_preflop_action(self, session, seat, action):
        """Update VPIP and PFR based on preflop action."""
        session['preflop_actions'][seat].append(action)
        stats = self.stats[seat]

        # VPIP: any voluntary call or raise (not checking BB, not posting blinds)
        if action in ('c',) or action.startswith('b') or action == 'a':
            if not (action == 'k'):  # check is never VPIP
                stats.vpip_count += 1

        # PFR: any raise or re-raise preflop
        if action.startswith('b') or action == 'a':
            stats.pfr_count += 1

    @staticmethod
    def _classify_action_situation(state):
        """Classify the current situation for opponent modeling."""
        max_bet = max(state.bets)
        player_bet = state.bets[state.current_player]
        if max_bet <= BIG_BLIND and state.street == 0:
            return 'first_in'
        elif max_bet > player_bet:
            return 'vs_bet' if state.street > 0 else 'vs_raise'
        else:
            return 'checked_to'

    @staticmethod
    def _normalize_action_type(action):
        """Normalize action token to category for opponent model."""
        if action == 'f':
            return 'fold'
        elif action == 'k':
            return 'check'
        elif action == 'c':
            return 'call'
        elif action.startswith('b') or action == 'a':
            return 'raise'
        return action

    def get_opponent_stats_for_solver(self, player_seat, num_players):
        """
        Return VPIP/PFR stats for opponents, but only if enough hands
        have been observed (500+). Returns None values for insufficient data.
        """
        MIN_HANDS = 500
        result = {}
        for i in range(num_players):
            if i == player_seat:
                continue
            s = self.stats[i]
            if s.hands_dealt >= MIN_HANDS:
                result[i] = {'vpip': s.vpip, 'pfr': s.pfr}
            else:
                result[i] = {'vpip': None, 'pfr': None}
        return result

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
        pp = session['player_seat']
        pos_name = get_position_name(state.num_players, player)

        # For preflop, use preflop ranges module
        use_preflop = (state.street == 0)

        hands = {}
        for row in range(13):
            for col in range(13):
                rank_a = 12 - row
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

                if use_preflop:
                    probs = self._get_preflop_range_probs(
                        combos[0], pos_name, state, actions)
                else:
                    combo = combos[0]
                    bucket = ca.get_bucket(combo, visible)
                    info_key = state.get_info_set_key(player, bucket)

                    # Try live solver first (in-progress MCCFR)
                    probs = None
                    if self.live_solver and self.live_solver.is_solving:
                        probs = self.live_solver.get_strategy_for_key(
                            info_key, len(actions))

                    # Fall back to blueprint
                    if probs is None:
                        if info_key in strat_table:
                            probs = strat_table[info_key]
                            if len(probs) != len(actions):
                                probs = [1.0 / len(actions)] * len(actions)
                        else:
                            probs = [1.0 / len(actions)] * len(actions)

                # Build suit combo list with cards
                suit_combos = []
                for combo in combos:
                    suit_combos.append(_cards(combo))

                hands[f'{row},{col}'] = {
                    'label': label,
                    'probs': [round(p, 4) for p in probs],
                    'bucket': None if use_preflop else ca.get_bucket(combos[0], visible),
                    'combos': len(combos),
                    'suit_combos': suit_combos,
                }

        return {
            'status': 'ok',
            'actions': actions,
            'current_player': 'player' if player == pp else 'bot',
            'position_name': pos_name,
            'street_name': ['Preflop', 'Flop', 'Turn', 'River'][min(state.street, 3)],
            'pot': round(state.pot + sum(state.bets), 1),
            'hands': hands,
        }

    def _get_preflop_range_probs(self, combo, pos_name, state, actions):
        """Get preflop action probabilities for a hand combo."""
        from cfr.preflop_ranges import get_preflop_action, classify_scenario
        from cfr.preflop_solver import hand_to_class

        hand_key = hand_to_class(combo)
        scenario = classify_scenario(state.history, pos_name, state.num_players)
        action_probs = get_preflop_action(
            hand_key, pos_name, scenario, actions, history=state.history,
            num_players=state.num_players)
        return [action_probs.get(a, 0.0) for a in actions]

    # ------------------------------------------------------------------
    # Action labels
    # ------------------------------------------------------------------

    def _action_labels_and_chips(self, state):
        """Compute display labels and chip amounts for all legal actions."""
        actions = state.get_actions()
        if not actions:
            return {}, {}
        p = state.current_player
        max_bet = max(state.bets)
        to_call = max_bet - state.bets[p]
        total_pot = state.pot + sum(state.bets)
        stack = state.stacks[p]
        labels = {}
        chips = {}
        for a in actions:
            c, is_allin = action_to_chips(a, total_pot, to_call, stack)
            chips[a] = round(c, 1)
            if a == 'f':
                labels[a] = 'Fold'
            elif a == 'k':
                labels[a] = 'Check'
            elif a == 'c':
                amt = round(min(to_call, stack), 1)
                labels[a] = f'Call {amt}'
            elif a == 'a':
                labels[a] = f'All In {round(stack, 1)}'
            elif a.startswith('b'):
                if state.street == 0:
                    raise_to = state.bets[p] + c
                    base = max_bet if max_bet > BIG_BLIND else BIG_BLIND
                    x = raise_to / base
                    if x == int(x):
                        labels[a] = f'Raise {int(x)}x ({round(c, 1)})'
                    else:
                        labels[a] = f'Raise {x:.1f}x ({round(c, 1)})'
                else:
                    frac = int(a[1:])
                    labels[a] = f'Bet {frac}% ({round(c, 1)})'
        return labels, chips

    # ------------------------------------------------------------------
    # Bot advance loop
    # ------------------------------------------------------------------

    def _advance_bots(self, session):
        state = session['state']
        player_seat = session['player_seat']
        bot_actions = []

        while not state.is_terminal and state.current_player != player_seat:
            seat = state.current_player
            hand = session['hands'][seat]
            pos_name = get_position_name(state.num_players, seat)

            # Get opponent exploit adjustments for this seat's opponents
            exploit_adj = self.opponent_model.get_exploit_adjustments(
                player_seat) if seat != player_seat else None
            action = self.bot.get_action(
                hand, state.visible_board(), state.history,
                state.get_actions(), state=state, seat=seat,
                exploit_adj=exploit_adj)

            if state.street == 0:
                self._track_preflop_action(session, seat, action)

            bot_actions.append({
                'seat': seat,
                'position': pos_name,
                'action': action,
            })
            state = state.apply_action(action)
            session['state'] = state

        # Start live solver when it becomes player's turn
        if (not state.is_terminal
                and state.current_player == player_seat
                and self.live_solver):
            stat_dict = self.get_opponent_stats_for_solver(
                player_seat, state.num_players)
            self.live_solver.start_solving(
                state, player_seat, stat_dict)

        result = self._state_dict(session)
        result['bot_actions'] = bot_actions
        return result

    # ------------------------------------------------------------------
    # State serialization
    # ------------------------------------------------------------------

    def _state_dict(self, session):
        state = session['state']
        pp = session['player_seat']
        num_players = state.num_players

        seats = []
        for i in range(num_players):
            is_human = (i == pp)
            cards = None
            if is_human:
                cards = _cards(session['hands'][i])
            elif state.is_terminal and not state.folded[i]:
                cards = _cards(session['hands'][i])

            seats.append({
                'seat': i,
                'position': get_position_name(num_players, i),
                'stack': round(state.stacks[i], 1),
                'bet': round(state.bets[i], 1),
                'active': not state.folded[i],
                'all_in': state.all_in[i],
                'cards': cards,
                'is_human': is_human,
            })

        is_player_turn = (not state.is_terminal and state.current_player == pp)
        if is_player_turn:
            action_labels, action_chips = self._action_labels_and_chips(state)
        else:
            action_labels, action_chips = {}, {}

        # Compute to_call and SPR for the human player
        max_bet = max(state.bets) if state.bets else 0
        to_call = round(max_bet - state.bets[pp], 1)
        total_pot = round(state.pot + sum(state.bets), 1)
        eff_stack = state.stacks[pp]
        spr = round(eff_stack / max(total_pot, 0.1), 1) if total_pot > 0 else None

        # Compute hand equity from card abstraction bucket
        hand_equity = None
        if not state.is_terminal:
            try:
                hand = session['hands'][pp]
                vis = state.visible_board()
                ca = self.bot.card_abstraction
                bucket = ca.get_bucket(hand, vis)
                if state.street == 0:
                    n_buckets = ca.num_preflop_buckets
                else:
                    n_buckets = ca.num_postflop_buckets
                hand_equity = round((bucket + 0.5) / max(n_buckets, 1) * 100, 1)
            except Exception:
                pass

        d = {
            'session_id': session['id'],
            'num_players': num_players,
            'player_seat': pp,
            'board': _cards(state.visible_board()),
            'pot': total_pot,
            'street': state.street,
            'street_name': ['Preflop', 'Flop', 'Turn', 'River'][min(state.street, 3)],
            'is_over': state.is_terminal,
            'current_player': state.current_player,
            'legal_actions': state.get_actions() if is_player_turn else [],
            'action_labels': action_labels,
            'action_chips': action_chips,
            'to_call': to_call if to_call > 0 else 0,
            'spr': spr,
            'hand_equity': hand_equity,
            'current_turn': 'player' if is_player_turn else 'bot',
            'seats': seats,
        }

        if state.is_terminal:
            d['full_board'] = _cards(state.board[:5])
            results = []
            for i in range(num_players):
                profit = round(state.get_terminal_utility(i), 1)
                r = {
                    'seat': i,
                    'position': get_position_name(num_players, i),
                    'profit': profit,
                }
                if not state.folded[i]:
                    r['hand'] = _cards(session['hands'][i])
                results.append(r)
            d['results'] = results

            if state.terminal_type == 'fold':
                active = [i for i in range(num_players) if not state.folded[i]]
                winner = active[0]
                winner_pos = get_position_name(num_players, winner)
                if winner == pp:
                    d['result_text'] = 'You win (opponents folded)'
                else:
                    d['result_text'] = f'{winner_pos} wins (others folded)'
            else:
                active_seats = [i for i in range(num_players) if not state.folded[i]]
                winners = determine_winners(
                    state.board[:5], session['hands'], active_seats)
                winner_names = [get_position_name(num_players, w) for w in winners]
                if pp in winners:
                    if len(winners) == 1:
                        d['result_text'] = 'You win at showdown!'
                    else:
                        d['result_text'] = f'Split pot (you + {", ".join(n for w, n in zip(winners, winner_names) if w != pp)})'
                else:
                    d['result_text'] = f'{", ".join(winner_names)} wins at showdown'

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
