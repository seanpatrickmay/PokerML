from Range import Range
#Class for representing a poker strategy.

NUT_EQUITY_CUTOFF = .9


class EquityBasedPokerStrategy:
    def __init__(self, myRange=Range(), opponentRange=Range()):
        self.equity = 0
        self.myRange = myRange
        self.opponentRange = opponentRange

    #Returns the ratio of nuts in hero vs opponent range.
    #Nuts will be calculated by raw equity %.
    def getNutRatio(self, board):
        #DOES NOTHING CURRENTLY

    #Returns the equity ratio in hero vs opponent range.
    def getEquityRatio(self, board):
        #DOES NOTHING CURRENTLY

    def getBettingFrequency

    def getBettingSize
