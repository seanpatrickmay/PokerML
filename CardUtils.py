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
