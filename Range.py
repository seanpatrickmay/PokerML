import numpy as np
import random
import sys
from phevaluator import evaluate_cards
import CardUtils
from Deck import Deck
from itertools import combinations

#IDEAS FOR ABSTRACTION:
# If I can find hands that are identical, given a board, remove all but one from the range.
# Then, have a matrix same size as array, that provides concentration.
# The default vals for this array will all be 1. np.ones(52, 52)
# Implement abstractify function, takes in a function to compare hands, edits concentrations

#Class to represent a range of two-card hands
class Range:

    #Automatically initialize range to full
    def __init__(self, hands=None, concentrations=None, equities=None, empty=False):
        self.hands = set()
        self.concentrationPrint = False
        self.equities = np.zeros((52, 52))

        #Feature for hand bucketing. Will be a table for redirecting hands to bucket
        #implement this
        self.redirect = dict()

        if empty:
            self.concentrations = np.zeros((52, 52), dtype=float)
            pass
        else:
            if hands != None:
                self.hands = hands.copy()
                self.concentrations = concentrations.copy()
            else:
                #Full range
                for firstCard in range(51, -1, -1):
                    for secondCard in range(firstCard - 1, -1, -1):
                        self.hands.add((firstCard, secondCard))
                self.concentrations = np.zeros((52, 52), dtype=float)
                for hand in self.hands:
                    self.concentrations[hand[0], hand[1]] = 1


    def copy(self):
        return Range(hands=self.hands.copy(), concentrations=self.concentrations.copy())

    #Returns true if given hand is in range, false otherwise
    def contains(self, hand):
        return hand in self.hands

    #Removes a hand from the range
    def remove(self, hand):
        self.hands.remove(hand)
        concentration = self.concentrations[hand[0], hand[1]]
        self.concentrations[hand[0], hand[1]] = 0
        return concentration

    #Adds a hand to the range
    def add(self, hand, concentration=1):
        self.hands.add(hand)
        self.addToConcentration(concentration, hand)
        return self.concentrations[hand[0], hand[1]]

    #Adds given value to a hands concentration
    def addToConcentration(self, value, hand):
        self.concentrations[hand[0], hand[1]] += value

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
    def abstractify(self, handSimplifier):
        newHandsInRange, newConcentrations = handSimplifier(self.hands.copy(), self.concentrations.copy())
        self.hands = newHandsInRange
        self.concentrations = newConcentrations

    #Gives this ranges raw equity against a given hand, on a given board
    def equityAgainstHand(self, hand, board=set(), equitySquared = False, giveHandEquity = False):
        for card in board:
            if card in hand:
                return 0

        #Create deck to represent cards not in current scenario
        currentDeck = Deck()
        currentDeck.removeCards(board) #There may be room for abstraction here
        currentDeck.removeCards(hand)

        #Counters for results per hand
        equities = 0
        simulationsDone = 0

        #Monte-Carlo factor
        mcf = max(10**(2-len(board)), 1)

        #Gets all combinations from remaining cards, evaluates amount corresponding to MCF
        allMissingBoardCards = list(combinations(currentDeck.cards, 5 - len(board)))
        for simulationNum in range(len(allMissingBoardCards) // mcf):

            #Get board for current sim
            finalBoard = [card for card in random.sample(allMissingBoardCards, 1)[0]] + list(board)

            #Remove self hands containing board/hand cards from range
            copyRange = self.copy()
            copyRange.removeCards(hand)
            copyRange.removeCards(finalBoard)

            #If no hands in range, equity is zero.
            if len(copyRange.hands) == 0: continue #Maybe this should be 0.5?

            #Score for other hand
        #    for num in range(5):
        #        print(finalBoard[num])
        #    print(hand[0])
        #    print(hand[1])
        #    print("\n")
            againstScore = evaluate_cards(finalBoard[0], finalBoard[1], finalBoard[2], finalBoard[3], finalBoard[4], hand[0], hand[1])

            #Find result for every hand in range
            wins = 0
            totalHands = 0
            for selfHand in copyRange.hands:
                concentration = self.concentrations[selfHand]
                selfScore = evaluate_cards(finalBoard[0], finalBoard[1], finalBoard[2], finalBoard[3], finalBoard[4], selfHand[0], selfHand[1])
                if selfScore < againstScore: wins += 1 * concentration
                elif selfScore == againstScore: wins += 0.5 * concentration
                totalHands += concentration
            if giveHandEquity:
                wins = totalHands - wins
            if equitySquared:
                equities += (wins / totalHands)**2
            else:
                equities += wins / totalHands

            #Incr sim nums
            simulationsDone += 1

        #Return total equity for range against hand
        if (simulationsDone == 0):
            return equities
        return equities/simulationsDone

    def setEquitiesAgainstRange(self, otherRange, board=[]):
        self.equities = np.zeros((52, 52))
        for hand in self.hands:
            self.equities[hand] = otherRange.equityAgainstHand(hand, board, giveHandEquity=True)

    def getEquity(self, hand):
        return self.equities[hand]

    def printAsConcentration(self, toggle=True):
        self.concentrationPrint = toggle

    #For representing the board in a string interface
    def __str__(self):
        matrix = []
        for x in range(52):
            matrix.append([])
            for y in range(52):
                if (x, y) in self.hands:
                    if self.concentrationPrint:
                        matrix[x].append(str(int(self.concentrations[x, y])))
                    else:
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

    #TEST ENV:
    testFullRange = Range()
    acesHand = (51, 50)
    exampleBoard = (20, 16, 12)

    testFullRange.abstractify(CardUtils.suitedAbstraction)
    testFullRange.abstractify(CardUtils.offsuitAbstraction)
    print(testFullRange)
    testFullRange.printAsConcentration()
    print(testFullRange)
    print('Equity of this range against', CardUtils.numsToCards(acesHand), 'on board', CardUtils.numsToCards(exampleBoard), 'is',  testFullRange.equityAgainstHand(acesHand, exampleBoard))
    print('Equity of this range against', CardUtils.numsToCards(acesHand), 'pre-flop is',  testFullRange.equityAgainstHand(acesHand, ()))
