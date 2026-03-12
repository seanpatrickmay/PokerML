"""Bot that plays using a trained CFR strategy."""

import numpy as np

from cfr.card_abstraction import CardAbstraction
from cfr.strategy_store import load_strategy


class Bot:
    def __init__(self, strategy_path, card_abstraction=None):
        self.strategy = load_strategy(strategy_path)
        self.card_abstraction = card_abstraction or CardAbstraction()

    def get_action(self, hand, visible_board, history, legal_actions):
        bucket = self.card_abstraction.get_bucket(hand, visible_board)
        history_str = '/'.join(','.join(s) for s in history)
        info_key = f'{bucket}|{history_str}'

        if info_key in self.strategy:
            probs = np.array(self.strategy[info_key])
            if len(probs) == len(legal_actions):
                idx = np.random.choice(len(legal_actions), p=probs)
                return legal_actions[idx]

        # Fallback: uniform random
        return np.random.choice(legal_actions)
