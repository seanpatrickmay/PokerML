from phevaluator import evaluate_cards
import CardUtils
from Deck import Deck
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

    #Returns true if given hand is in range, false otherwise
    def contains(self, hand):
        return hand in self.hands

    #Removes a hand from the range
    def remove(self, hand):
        self.hands.remove(hand)

    #Adds a hand to the range
    def add(self, hand):
        self.hands.add(hand)

    #Removes all hands containing given card from range
    def removeCard(self, card):
        for hand in self.hands.copy():
            if card in hand:
                self.remove(hand)

    #Given an iterable, removes all cards in that iterable from this range
    def removeCards(self, cards):
        for card in cards:
            self.removeCard(card)

    #Gives this ranges raw equity against a given hand, on a given board
    def equityAgainstHand(self, hand, board=[]):
        allCardsSet = set(board)
        for card in hand:
            allCardsSet.add(card)
        if len(allCardsSet) != len(board) + 2:
            print('Illegal board/hand combination!')
            return
        self.removeCards(hand)
        self.removeCards(board)

        #Deck only needed if board on turn, flop, or pre
        currentDeck = Deck()
        currentDeck.removeCards(board)
        currentDeck.removeCards(hand)

        #CURRENT ASSUMPTION: Given board is on the river
        #THOUGHTS: Will implement either full runouts for turn and flop, or monte-carlo-like simulations
        wins = 0
        losses = 0
        chops = 0
        for selfHand in self.hands:
            selfScore = evaluate_cards(board[0], board[1], board[2], board[3], board[4], selfHand[0], selfHand[1])
            againstScore = evaluate_cards(board[0], board[1], board[2], board[3], board[4], hand[0], hand[1])
            if selfScore < againstScore:
                wins += 1
            elif selfScore > againstScore:
                losses += 1
            else:
                chops += 1
        return (wins + chops/2)/(wins + chops + losses) * 100

    #For representing the board in a string interface
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

        return TopRow + '\n' + resultString[:-1]
                        

if __name__ == "__main__":
    testFullRange = Range()
    cardsToRemove = [31, 30, 29, 28, 11, 10, 9, 8]
    for num in range(0, 51, 4):
        cardsToRemove.append(num)
    for card in cardsToRemove:
        testFullRange.removeCard(card)
    print(testFullRange)
    print(testFullRange.equityAgainstHand((51, 50), [49, 24, 20, 16, 12]))

