from Range import Range
import sys
import CardUtils
import json

# A file for getting and storing pre-flop equities for all hands

# Our range that we will be finding equities from:
# (Note): this range only contains one combo of both suited and offsuit hands.
# Therefore, we will be evaluating all possible combos and averaging the equities.

simplifiedRange = Range()
simplifiedRange.abstractify(CardUtils.suitedAbstraction)
simplifiedRange.abstractify(CardUtils.offsuitAbstraction)

# Find equities for each hand
# We will store the equities in a table, and use JSON to write to disk
handEquities = dict()

# Lets do pocket pairs first:
for card in range(13):
    equities = 0
    for firstSuit in range(4):
        for secondSuit in range(firstSuit + 1, 4):
            hand = (card * 4 + firstSuit, card * 4 + secondSuit)
            print(CardUtils.numsToCards(hand))
            equities += simplifiedRange.equityAgainstHand(hand, equitySquared=False, giveHandEquity=True)
    equities /= 6
    handEquities[str(card) + ' ' + str(card)] = equities
    #After all found, output to file
    with open("preFlopEquities.json", "w") as file:
        json.dump(handEquities, file, indent=3)

#Next, suited hands:
for firstCard in range(12, -1, -1):
    for secondCard in range(firstCard -1, -1, -1):
        equities = 0
        for suit in range(4):
            hand = (firstCard * 4 + suit, secondCard * 4 + suit)
            print(CardUtils.numsToCards(hand))
            equities += simplifiedRange.equityAgainstHand(hand, equitySquared=False, giveHandEquity=True)
        equities /= 4
        handEquities[str(firstCard) + ' ' + str(secondCard) + ' s'] = equities
        #After all found, output to file
        with open("preFlopEquities.json", "w") as file:
            json.dump(handEquities, file, indent=3)

#Finally, offsuit hands:
for firstCard in range(12, -1, -1):
    for secondCard in range(firstCard -1, -1, -1):
        equities = 0
        for firstSuit in range(4):
            for secondSuit in range(4):
                if firstSuit == secondSuit:
                    continue
                hand = (firstCard * 4 + firstSuit, secondCard * 4 + secondSuit)
                print(CardUtils.numsToCards(hand))
                equities += simplifiedRange.equityAgainstHand(hand, equitySquared=False, giveHandEquity=True)
        equities /= 12
        handEquities[str(firstCard) + ' ' + str(secondCard) + ' o'] = equities
        #After all found, output to file
        with open("preFlopEquities.json", "w") as file:
            json.dump(handEquities, file, indent=3)

