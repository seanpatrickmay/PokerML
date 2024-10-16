#File for hand utils, such as string conversion

rankStrings = ['2', '3', '4', '5', '6', '7', '8', '9', 'T', 'J', 'Q', 'K', 'A'] 
suitStrings = ['♣', '♢', '♡', '♠']

def numToCard(num):
    return rankStrings[num // 4] + suitStrings[num % 4]

def numsToCards(nums):
    cardsString = ''
    for num in nums:
        cardsString += numToCard(num)
    return cardsString

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
