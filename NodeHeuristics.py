import math
import CardUtils
from Range import Range
from config import HEURISTIC_BET_SIZINGS

# Expressed in # of pots
BET_SIZINGS = HEURISTIC_BET_SIZINGS


# Method gives a rough approximation of EV for betting hand with given sizing
def betExpectedValue(hand, heroRange, board, sizing):
    B = sizing
    E = heroRange.getEquity(hand)
    alpha = B / (B + 1)
    mindef = 1 - alpha
    # 1 / (B + 1)
    pwin = max(0, (E - alpha)) / mindef
    plose = 1 - pwin
    return (pwin * (1 + 2 * B) - plose * B) * mindef + 1 * alpha

# Test method, not used
def getTotalValueFromSizing(heroRange, board, sizing):
    totalValue = 0
    for hand in heroRange.hands:
        EV = betExpectedValue(hand, heroRange, board, sizing)
        #print(f"EV of hand {CardUtils.numsToCards(hand)} with sizing {sizing} is {EV}")
        totalValue += max(0, EV)
    return totalValue

# Test method, not used
def getBestSizingForRange(heroRange, opponentRange, board, sizings=BET_SIZINGS):
    bestSizing = -1
    bestValue = -1
    heroRange.setEquitiesAgainstRange(opponentRange, board)
    for sizing in sizings:
        sizingValue = getTotalValueFromSizing(heroRange, board, sizing)
        print(f"Sizing Value for {sizing} is: {sizingValue}")
        if sizingValue > bestValue:
            bestSizing = sizing
            bestValue = sizingValue
    return bestSizing

# Given a sizing, returns all hands that are profitable to bet
def getBetHands(heroRange, opponentRange, board, sizing):
    betHands = set()
    for hand in heroRange.hands:
        if betExpectedValue(hand, heroRange, board, sizing) > heroRange.getEquity(hand):
            betHands.add(hand)
    return betHands

# Method to find as many bluff hands as needed.
# For now, this will simply use the bottom of range to bluff.
def getBluffHands(heroRange, numNeeded, excludedHands):
    hands = list(heroRange.hands)
    hands.sort(key=lambda hand : heroRange.getEquity(hand))
    bluffs = set()
    for i in range(int(numNeeded)):
        if hands[i] not in excludedHands:
            bluffs.add(hands[i])
    return bluffs

# Method to find sets of both value bets and bluffs
def getBetAndBluffs(heroRange, opponentRange, board, sizing):
    heroRange.setEquitiesAgainstRange(opponentRange, board)
    bets = getBetHands(heroRange, opponentRange, board, sizing)
    bluffsNeeded = (1 - (sizing / (sizing + 1))) * len(bets)
    bluffs = getBluffHands(heroRange, bluffsNeeded, bets)
    return bets, bluffs

# Method that splits a range into two, bet and check
def splitBetCheckRanges(heroRange, opponentRange, board, sizing):
    bets, bluffs = getBetAndBluffs(heroRange, opponentRange, board, sizing)
    betsAndBluffs = bets | bluffs
    betRange = Range(empty=True)
    checkRange = heroRange.copy()
    for hand in betsAndBluffs:
        betRange.add(hand)
        checkRange.remove(hand)
    return betRange, checkRange

# Method to find call hands given sizing and range
# Currently gets based purely on equity. Not so balanced... Also, only defends mindef... Maybe need to def higher %.
# This isn't balanced because sometimes you should call hands just to bluff later, for example, but it's nuanced
def getCallRange(heroRange, opponentRange, board, sizing):
    alpha = sizing / (sizing + 1)
    mindef = 1 - alpha
    combosNeeded = len(heroRange.hands) * mindef
    heroRange.setEquitiesAgainstRange(opponentRange, board)
    handsList = list(heroRange.hands)
    handsList.sort(key=lambda hand: heroRange.equities[hand], reverse=True)
    return Range(hands=set(handsList[:int(math.ceil(combosNeeded))]))

# Returns what equity hero's range has against opponent's.
def simpleEquityEvaluation(heroRange, opponentRange, board):
    return heroRange.equityAgainstRange(opponentRange, board)


if __name__ == "__main__":

    heroRange = Range()
    heroRange.abstractify(CardUtils.suitedAbstraction)
    heroRange.abstractify(CardUtils.offsuitAbstraction)

    villainRange = Range(empty=True)
    villainRange.add((31, 30))
    villainRange.add((27, 26))
    print(villainRange)
    print(heroRange)

    board = [49, 45, 0]

    print(f"The best sizing for hero on {CardUtils.numsToCards(board)} is: {getBestSizingForRange(heroRange, villainRange, board)}")
