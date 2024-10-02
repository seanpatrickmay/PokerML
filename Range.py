import CardUtils
#Class to reoresent a range of two-card hands

class Range:

    #Automatically initialize range to full
    def __init__(self, handsString='', empty=False):
        self.hands = set()
        if empty:
            #Do nothing!
            pass
        else:
            if handsString:
                #Implement this later for easier range building
                pass    
            else:
                #Full range
                for firstCard in range(51, -1, -1):
                    for secondCard in range(firstCard - 1, -1, -1):
                        self.hands.add((firstCard, secondCard))

    def contains(self, hand):
        return hand in self.hands

    #Removes a hand from the range
    def remove(self, hand):
        self.hands.remove(hand)

    #Adds a hand to the range
    def add(self, hand):
        self.hands.add(hand)

    def removeCard(self, card):
        for hand in self.hands.copy():
            if card in hand:
                self.hands.remove(hand)

    def __str__(self):
        matrix = []
        for x in range(52):
            matrix.append([])
            for y in range(52):
                if (x, y) in self.hands:
                    matrix[x].append('X')
                else:
                    matrix[x].append(' ')

        resultString = ''
        for y, row in enumerate(matrix):
            rowString = ''
            currentCard = CardUtils.numToCard(y)
            for x, entry in enumerate(row):
                rowString = f'{entry} ' + rowString
                if x % 4 == 3:
                    rowString = '|' + rowString
            resultString = currentCard + rowString + '\n' + resultString
            if y % 4 == 3:
                resultString = '   ' + ('-' * 115) + '\n' + resultString

        TopRow = '   '
        for num in range(51, -1, -1):
            TopRow += '' + CardUtils.numToCard(num)
            if num % 4 == 0:
                TopRow += ' '

        return TopRow + '\n' + resultString
                        

if __name__ == "__main__":
    testFullRange = Range()
    testFullRange.remove((51, 47))
    testFullRange.removeCard(43)
    testFullRange.add((43, 31))
    print(testFullRange)


