import numpy as np
import sys
from phevaluator import evaluate_cards
import CardUtils
from Deck import Deck

#IDEAS FOR ABSTRACTION:
# If I can find hands that are identical, given a board, remove all but one from the range.
# Then, have a matrix same size as array, that provides concentration.
# The default vals for this array will all be 1. np.ones(52, 52)
# Implement abstractify function, takes in a function to compare hands, edits concentrations

#Class to reoresent a range of two-card hands
class Range:

    #Automatically initialize range to full
    def __init__(self, hands=None, concentrations=None, empty=False):
        self.hands = set()
        self.concentrations = np.zeros((52, 52), dtype=float)
        if empty:
            #Do nothing!
            pass
        else:
            if hands != None:
                self.hands = hands.copy()
            else:
                #Full range
                for firstCard in range(51, -1, -1):
                    for secondCard in range(firstCard - 1, -1, -1):
                        self.hands.add((firstCard, secondCard))
        for hand in self.hands:
            self.concentrations[hand[0], hand[1]] = 1


    def copy(self):
        return Range(hands=self.hands, concentrations=self.concentrations)

    #Returns true if given hand is in range, false otherwise
    def contains(self, hand):
        return hand in self.hands

    #Removes a hand from the range
    def remove(self, hand):
        self.hands.remove(hand)
        self.concentrations[hand[0], hand[1]] = 0

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

    #Given a hand comparator, simplifies range by removing duplicates, changes range concentrations to match
    def abstractify(self, handComparator):
        handsInRange = self.hands.copy()
        for hand in handsInRange:
            if hand not in self.hands:
                continue
            for otherHand in handsInRange:
                if hand == otherHand:
                    continue
                if otherHand not in self.hands:
                    continue
                if handComparator(hand, otherHand):
                    self.hands.remove(otherHand)
                    self.concentrations[hand[0], hand[1]] += self.concentrations[otherHand[0], otherHand[1]]
                    self.concentrations[otherHand[0], otherHand[1]] = 0

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
        if len(self.hands) == 0:
            return None

        #Set concentration val
        concentration = self.concentrations[hand[0], hand[1]]

        #Deck only needed if board on turn, flop, or pre
        currentDeck = Deck()
        currentDeck.removeCards(board)
        currentDeck.removeCards(hand)


        #FIX THIS !!!! CHANGE SO CONCENTRATION IS FOUND FOR HAND IN RANGE, NOT HAND GIVEN! MULTIPLY BY THAT INSTEAD!


        #CURRENT ASSUMPTION: Given board is on the river
        #THOUGHTS: Will implement either full runouts for turn and flop, or monte-carlo-like simulations
        wins = 0
        losses = 0
        chops = 0
      
        #If on turn:
        if len(board) == 4:
            while currentDeck.size() > 0:
                riverCard = currentDeck.dealCard()
                currentEquity = self.copy().equityAgainstHand(hand, board.copy() + [riverCard])
                if currentEquity == None:
                    continue
                wins += currentEquity / 100
                losses += (100 - currentEquity) / 100

        #If on river:
        else:
            for selfHand in self.hands:
                selfScore = evaluate_cards(board[0], board[1], board[2], board[3], board[4], selfHand[0], selfHand[1])
                againstScore = evaluate_cards(board[0], board[1], board[2], board[3], board[4], hand[0], hand[1])
                if selfScore < againstScore:
                    wins += 1
                elif selfScore > againstScore:
                    losses += 1
                else:
                    chops += 1

        #Multiply value by concentration of hand in range
        wins = int(wins * concentration)
        chops = int(chops * concentration)
        losses = int(losses * concentration)

        print(wins, chops, losses)
        #Return total equity for range against hand
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
    acesHand = (51, 50)
    exampleBoard = [49, 20, 16, 12]
    print(testFullRange)
    print('Equity of this range against', CardUtils.numsToCards(acesHand), 'on board', CardUtils.numsToCards(exampleBoard), 'is',  testFullRange.equityAgainstHand((51, 50), exampleBoard))
    testFullRange.abstractify(lambda hand1, hand2: hand1[0]//4 == hand2[0]//4 and hand1[1]//4 == hand2[1]//4)
    print(testFullRange)
    print(testFullRange.hands)
