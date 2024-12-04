#File for hand utils, such as string conversion

rankStrings = ['2', '3', '4', '5', '6', '7', '8', '9', 'T', 'J', 'Q', 'K', 'A'] 
suitStrings = ['♣', '♢', '♡', '♠']

# Given integer 0-51, gives String Card
def numToCard(num):
    return rankStrings[num // 4] + suitStrings[num % 4]

# Gives all cards given integers
def numsToCards(nums):
    cardsString = ''
    for num in nums:
        cardsString += numToCard(num)
    return cardsString

# Converts text to a hand ex: JJ -> all pocket jacks hands
def textToHand(text):
    firstRank = rankStrings.index(text[0])
    secondRank = rankStrings.index(text[1])
    return ranksToHands(firstRank, secondRank)

# Given two ranks, gives all hands
def ranksToHands(firstRank, secondRank):
    handSet = set()
    if firstRank != secondRank:
        for firstCard in range(3, -1, -1):
            for secondCard in range(3, -1, -1):
                handSet.add((firstRank * 4 + firstCard, secondRank * 4 + secondCard))
        return handSet
    else:
        for firstCard in range(3, -1, -1):
            for secondCard in range(firstCard - 1, -1, -1):
                handSet.add((firstRank * 4 + firstCard, firstRank * 4 + secondCard))
        return handSet

# Helpful method for making test ranges, JT+ gives suited connectors, and 22+ gives pocket pairs, any other gives A2-K
def textToHandSet(text):
    allHands = set()
    hands = text.split(",")
    for hand in hands:
        firstRank = rankStrings.index(hand[0])
        secondRank = rankStrings.index(hand[1])
        if "+" in hand:
            rankDifference = firstRank - secondRank
            if rankDifference <= 1:
                for rank in range(firstRank, 13):
                    allHands = allHands.union(ranksToHands(rank, rank - rankDifference))
            else:
                for rank in range(secondRank, firstRank):
                    allHands = allHands.union(ranksToHands(firstRank, rank))
        else:
            allHands = allHands.union(ranksToHands(firstRank, secondRank))
    return allHands


def handComparator(hand1, hand2):
    firstRankDiff = hand1[0]//4 - hand2[0]//4
    secondRankDiff = hand1[1]//4 - hand2[1]//4
    if firstRankDiff != 0:
        return firstRankDiff
    elif secondRankDiff != 0:
        return secondRankDiff
    firstSuitDiff = hand1[0]%4 - hand2[0]%4
    secondSuitDiff = hand1[1]%4 - hand2[1]%4
    if firstSuitDiff != 0:
        return firstSuitDiff
    elif secondRankDiff !=0:
        return secondSuitDiff

def handToValue(hand):
    return hand[0] * 100 + hand[1]

def listNumStringsToListHands(nums):
    newList = []
    for hand in nums:
        handSplit = hand.split(' ')
        firstCard = int(handSplit[0]) * 4 + 1
        secondCard = int(handSplit[1]) * 4 + 0
        if len(handSplit) > 2:
            if handSplit[2] == 's':
                secondCard += 1
        newList.append((firstCard, secondCard))
    newList.sort(key=handToValue)
    return [numsToCards(hand) for hand in newList]

# Some abstractions for compressing ranges
def suitDifferenceAbstraction(handSet, concentrationsSet):
    handList = list(handSet)
    handList.sort(key=handToValue, reverse=True)
    for hand1Index in range(len(handList)):
        hand1 = handList[hand1Index]
        if hand1 not in handSet:
            continue
        for hand2Index in range(hand1Index + 1, len(handList), 1):
            hand2 = handList[hand2Index]
            if hand2 not in handSet:
                continue
            #If ranks are the same and difference between suits are the same: True
            if hand1[0]//4 == hand2[0]//4 and hand1[1]//4 == hand2[1]//4:
                if abs(hand1[0]%4 - hand1[1]%4) == abs(hand2[0]%4 - hand2[1]%4):
                    handSet.remove(hand2)
                    concentrationsSet[hand1[0], hand1[1]] += concentrationsSet[hand2[0], hand2[1]]
                    concentrationsSet[hand2[0], hand2[1]] = 0

    return handSet, concentrationsSet

def suitedAbstraction(handSet, concentrationsSet):
    handList = list(handSet)
    handList.sort(key=handToValue, reverse=True)
    for hand1Index in range(len(handList)):
        hand1 = handList[hand1Index]
        if hand1 not in handSet:
            continue
        for hand2Index in range(hand1Index + 1, len(handList), 1):
            hand2 = handList[hand2Index]
            if hand2 not in handSet:
                continue
            if hand1[0]//4 == hand2[0]//4 and hand1[1]//4 == hand2[1]//4:
                if hand1[0]%4 == hand1[1]%4 and hand2[0]%4 == hand2[1]%4:
                    handSet.remove(hand2)
                    concentrationsSet[hand1[0], hand1[1]] += concentrationsSet[hand2[0], hand2[1]]
                    concentrationsSet[hand2[0], hand2[1]] = 0

    return handSet, concentrationsSet

def offsuitAbstraction(handSet, concentrationsSet):
    handList = list(handSet)
    handList.sort(key=handToValue, reverse=True)
    for hand1Index in range(len(handList)):
        hand1 = handList[hand1Index]
        if hand1 not in handSet:
            continue
        for hand2Index in range(hand1Index + 1, len(handList), 1):
            hand2 = handList[hand2Index]
            if hand2 not in handSet:
                continue
            if hand1[0]//4 == hand2[0]//4 and hand1[1]//4 == hand2[1]//4:
                if hand1[0]%4 != hand1[1]%4 and hand2[0]%4 != hand2[1]%4:
                    handSet.remove(hand2)
                    concentrationsSet[hand1[0], hand1[1]] += concentrationsSet[hand2[0], hand2[1]]
                    concentrationsSet[hand2[0], hand2[1]] = 0

    return handSet, concentrationsSet


if __name__ == "__main__":
    print(textToHand("AA"))
    print(textToHand("AK"))
    print(textToHandSet("AK,JJ+,AT+"))
    print(numsToCards((51, 50, 49, 48)))
