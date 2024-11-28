from Range import Range

#Class for representing a player in HU NLHE

class PokerPlayer:
    def __init__(self, chips, hand=None, handRange=Range()):
        self.chips = chips
        self.hand = hand
        self.handRange = handRange

    def copy(self):
        return PokerPlayer(self.chips, self.hand, self.handRange.copy())

    def spendChips(self, chips):
        assert(chips <= self.chips)
        self.chips -= chips

    def shrinkRange(self, hands):
        for hand in hands:
            self.handRange.remove(hand)
