from Range import Range
import CardUtils
import PokerNode

# Expressed in # of pots
BET_SIZINGS = (0, 1/4, 1/3, 1/2, 3/4, 1, 4/3, 3/2, 2, 3, 5, 10)

# Class for representing minimax strategy
# Currently make assumption that we either bet entire range or check



# First, need a way to evalute EV of an end node, in poker, this would be on the river
def getNodeEV(pokerNode):
    board = pokerNode.board
    hero = pokerNode.hero
    villain = pokerNode.villain
    
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
# Therefore, our EV is 5/6 * -1/2 + 1/6 * 2 = -5/12 + 4/12 = -1/12. < 1/2: Bad bet.
# Similarly, if we bet 1/4: EV is 5/8 * -1/4 + 3/8 * 3/2 = -5/32 + 18/32 = 13/32: Bad bet.
# Therefore, we get EV of bet is: P(win) * (Pot + Bet * 2) - P(lose) * Bet
# P(win) can be calculated as max(0, (E - alpha)) / mindef
# or: 
# P(lose) is just 1 -  P(win), so: Bet / (Bet + Pot) + Equity

# Putting this together, we get the formula for EV of betting size B given hand with equity E and pot size 1:
# (1 - B / (B + 1) - E) * (1 + B * 2) - (B / (B + 1) + E) * B
# So, if this formula evaluates to a value > E * P, Betting is profitable.

# Now we need to find bluffs:
# First, we need to know how many bluffs we need.
# # of bluffs needed is equal to # of value combos * Alpha   (B / (B + P))
# Now, how do we select bluffs? 
# We don't want to select bluffs from middling equity hands, so we should select from the following (In order):
# Hands with highest equity squared (draws to the nuts) 
# Hands that minimize opponents equity against our range if we remove combos of our cards (good blockers)
# Hands that have the least showdown value

# So, simply, to find the best sizing, we choose which sizing value maximises the EV of the range


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
