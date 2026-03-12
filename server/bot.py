"""Bot that plays using a trained CFR strategy with real-time subgame solving."""

import numpy as np

from cfr.card_abstraction import CardAbstraction
from cfr.strategy_store import load_strategy
from cfr.subgame_solver import SubgameSolver


class Bot:
    def __init__(self, strategy_path, card_abstraction=None, subgame_iters=300):
        self.strategy = load_strategy(strategy_path)
        self.card_abstraction = card_abstraction or CardAbstraction()
        self.solver = SubgameSolver(
            self.strategy, self.card_abstraction, iterations=subgame_iters)

    def get_action(self, hand, visible_board, history, legal_actions,
                   state=None, bot_position=None):
        bucket = self.card_abstraction.get_bucket(hand, visible_board)
        history_str = '/'.join(','.join(s) for s in history)
        info_key = f'{bucket}|{history_str}'

        if info_key in self.strategy:
            probs = np.array(self.strategy[info_key])
            if len(probs) == len(legal_actions):
                return legal_actions[np.random.choice(len(legal_actions), p=probs)]

        # ---- Real-time subgame solve on miss ----------------------------
        if state is not None and bot_position is not None:
            solved = self.solver.solve(state, bot_position)
            self.strategy.update(solved)     # cache for future lookups

            if info_key in solved:
                probs = np.array(solved[info_key])
                if len(probs) == len(legal_actions):
                    return legal_actions[
                        np.random.choice(len(legal_actions), p=probs)]

        # last resort
        return np.random.choice(legal_actions)
