import math
from Range import Range
import CardUtils

# Expressed in # of pots                                                                                              
BET_SIZINGS = (0, 1/4, 1/3, 1/2, 3/4, 1, 4/3, 3/2, 2, 3, 5, 10)


# Given a bet sizing, need to determine if a hand should bet. (For value)

# MDF is defined as 1-Alpha, where Alpha is bet / bet + pot
# If I bet pot, opponent minDef = 1/2. I will assume opponent will call 1/2 of hands in range.
# MDFs: 10: 1/11, 3: 1/4, 1: 1/2, 1/2: 2/3, 1/4: 4/5

# What is the weakest hand I can bet for value? How much equity does it need?
# First: If I bet any bluff, ev should be 0.
# EV of checking marginal hands is >0, so I can't bet all hands with Equity > MDF % villains range
# I need to find where EV betting > EV checking

# Assume villain always checks, and we have a hand with 50% equity against villains range.
# If we check, our EV is 1/2
# If we bet 1/2, opponent calls 66%, only 16% of which we beat.
# So our equity in this case would be: 1/3 * 1 (opp folds) + 1/6 * 2 + 1/2 * -1/2 = 1/3 + 1/3 - 1/4 = 5/12
# Therefore, we get EV of bet is: P(fold) + P(win & call) * (1 + Bet * 2) - P(lose & call) * Bet
# P(win & call) can be calculated as max(0, (E - alpha))
# P(lose & call) is just mindef - P(win & call)
# P(fold) is simply alpha, since mindef is defined as 1 - alpha

# Putting this together, we get the formula for EV of betting size B given hand with equity E and pot size 1:
# alpha + max(0, E - alpha) * (1 + B * 2) - (mindef - max(0, E - alpha)) * B

# So, if this formula evaluates to a value > E * P, Betting is profitable.
# However, this formula assumes villain has no raises, ie: villain always either checks/calls/folds

# Now we need to find bluffs:
# First, we need to know how many bluffs we need.
# # of bluffs needed is equal to # of value combos * Alpha   (B / (B + P))
# Now, how do we select bluffs? 
# We don't want to select bluffs from middling equity hands, so we should select from the following (In order):
# Hands with highest equity squared (draws to the nuts) 
# Hands that minimize opponents equity against our range if we remove combos of our cards (good blockers)
# Hands that have the least showdown value

# So, simply, to find the best sizing, we choose which sizing value maximises the EV of the range

# This method currently works if bet size is all in. We need to recur elsewise
def betExpectedValue(hand, heroRange, board, sizing):
    B = sizing
    E = heroRange.getEquity(hand)
    alpha = B / (B + 1)
    mindef = 1 - alpha
    pwin = max(0, (E - alpha)) / mindef
    plose = 1 - pwin
    return (pwin * (1 + 2 * B) - plose * B) * mindef + 1 * alpha

# Given a bet sizing, need to determine if a hand should bet. (For value)

# MDF is defined as 1-Alpha, where Alpha is bet / bet + pot
# If I bet pot, opponent minDef = 1/2. I will assume opponent will call 1/2 of hands in range.
# MDFs: 10: 1/11, 3: 1/4, 1: 1/2, 1/2: 2/3, 1/4: 4/5

# What is the weakest hand I can bet for value? How much equity does it need?
# First: If I bet any bluff, ev should be 0.
# EV of checking marginal hands is >0, so I can't bet all hands with Equity > MDF % villains range
# I need to find where EV betting > EV checking

# Assume villain always checks, and we have a hand with 50% equity against villains range.
# If we check, our EV is 1/2
# If we bet 1/2, opponent calls 66%, only 16% of which we beat.
# So our equity in this case would be: 1/3 * 1 (opp folds) + 1/6 * 2 + 1/2 * -1/2 = 1/3 + 1/3 - 1/4 = 5/12
# Therefore, we get EV of bet is: P(fold) + P(win & call) * (1 + Bet * 2) - P(lose & call) * Bet
# P(win & call) can be calculated as max(0, (E - alpha))
# P(lose & call) is just mindef - P(win & call)
# P(fold) is simply alpha, since mindef is defined as 1 - alpha

# Putting this together, we get the formula for EV of betting size B given hand with equity E and pot size 1:
# alpha + max(0, E - alpha) * (1 + B * 2) - (mindef - max(0, E - alpha)) * B

# So, if this formula evaluates to a value > E * P, Betting is profitable.
# However, this formula assumes villain has no raises, ie: villain always either checks/calls/folds

# Now we need to find bluffs:
# First, we need to know how many bluffs we need.
# # of bluffs needed is equal to # of value combos * Alpha   (B / (B + P))
# Now, how do we select bluffs? 
# We don't want to select bluffs from middling equity hands, so we should select from the following (In order):
# Hands with highest equity squared (draws to the nuts) 
# Hands that minimize opponents equity against our range if we remove combos of our cards (good blockers)
# Hands that have the least showdown value

# So, simply, to find the best sizing, we choose which sizing value maximises the EV of the range

# This method currently works if bet size is all in. We need to recur elsewise
def betExpectedValue(hand, heroRange, board, sizing):
    B = sizing
    E = heroRange.getEquity(hand)
    alpha = B / (B + 1)
    mindef = 1 - alpha
    pwin = max(0, (E - alpha)) / mindef
    plose = 1 - pwin
    return (pwin * (1 + 2 * B) - plose * B) * mindef + 1 * alpha

def getTotalValueFromSizing(heroRange, board, sizing):
    totalValue = 0
    for hand in heroRange.hands:
        EV = betExpectedValue(hand, heroRange, board, sizing)
        #print(f"EV of hand {CardUtils.numsToCards(hand)} with sizing {sizing} is {EV}")
        totalValue += max(0, EV)
    return totalValue

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

def getBetHands(heroRange, opponentRange, board, sizing):
    heroRange.setEquitiesAgainstRange(opponentRange, board)
    betHands = set()
    for hand in heroRange.hands:
        if betExpectedValue(hand, heroRange, board, sizing) > heroRange.getEquity(hand):
            betHands.add(hand)
    return betHands

# For now, this will simply use the bottom of range to bluff
def getBluffHands(heroRange, numNeeded, excludedHands):
    hands = list(heroRange.hands)
    hands.sort(key=lambda hand : heroRange.getEquity(hand))
    bluffs = set()
    for i in range(int(numNeeded)):
        if hands[i] not in excludedHands:
            bluffs.add(hands[i])
    return bluffs

def getBetAndBluffs(heroRange, opponentRange, board, sizing):
    bets = getBetHands(heroRange, opponentRange, board, sizing)
    bluffsNeeded = (1 - (sizing / (sizing + 1))) * len(bets)
    print(f"Bluffs needed for {len(bets)} value bets with sizing {sizing}: {bluffsNeeded}")
    bluffs = getBluffHands(heroRange, bluffsNeeded, bets)
    return bets, bluffs

def splitBetCheckRanges(heroRange, opponentRange, board, sizing):
    bets, bluffs = getBetAndBluffs(heroRange, opponentRange, board, sizing)
    betsAndBluffs = bets | bluffs
    betRange = Range(empty=True)
    checkRange = heroRange.copy()
    for hand in betsAndBluffs:
        betRange.add(hand)
        checkRange.remove(hand)
    return betRange, checkRange

# Currently gets based purely on equity. Not so balanced... Also, only defends mindef... Maybe need to def higher %.
def getCallRange(heroRange, opponentRange, board, sizing):
    alpha = sizing / (sizing + 1)
    mindef = 1 - alpha
    combosNeeded = len(heroRange.hands) * mindef
    print(f"Defend combos needed: {combosNeeded}")
    heroRange.setEquitiesAgainstRange(opponentRange, board)
    handsList = list(heroRange.hands)
    handsList.sort(key=lambda hand: heroRange.equities[hand], reverse=True)
    return Range(hands=set(handsList[:int(math.ceil(combosNeeded))]))

# Returns what equity hero's range has against opponent's.
def simpleEquityEvaluation(heroRange, opponentRange, board):
    return heroRange.equityAgainstRange(opponentRange, board)



if __name__ == "__main__":

#    bluffCatcherToyRange = Range(empty=True)
#    bluffCatcherToyRange.add((47, 46))
#    polarizedToyRange = Range(empty=True)
#    polarizedToyRange.add((51, 50))
#    polarizedToyRange.add((43, 42))
#    board = [0, 1, 2, 12, 13]
    
    heroRange = Range()
    #for card in range(40):
    #    heroRange.removeCard(card)
    heroRange.abstractify(CardUtils.suitedAbstraction)
    heroRange.abstractify(CardUtils.offsuitAbstraction)

    villainRange = Range(empty=True)
    #villainRange.abstractify(CardUtils.suitedAbstraction)
    #villainRange.abstractify(CardUtils.offsuitAbstraction)
    #villainRange.removeCard(47)
    #villainRange.removeCard(51)
    villainRange.add((31, 30))
    villainRange.add((27, 26))
    print(villainRange)
    print(heroRange)

    board = [49, 45, 0]

    print(f"The best sizing for hero on {CardUtils.numsToCards(board)} is: {getBestSizingForRange(heroRange, villainRange, board)}")
