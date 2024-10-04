#Class for representing a player in HU NLHE

class PokerPlayer:
    def __init__(self, chips, hand=None):
        self.chips = chips
        self.hand = hand
