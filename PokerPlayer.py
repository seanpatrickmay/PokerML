#Class for representing a player in HU NLHE
import Range

class PokerPlayer:
    def __init__(self, chips, hand=None, handRange=Range()):
        self.chips = chips
        self.hand = hand
        self.handRange = handRange
