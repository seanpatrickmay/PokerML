import random
import CardUtils

#Represents a deck of 52 cards
#Cards will be represented by integers, 0-51
#A Card is the following: (Rank - 2) * 4 + Suit
#Suits are in the following order, 0-3: c, d, h, s

class Deck:

    #Sets up new deck with all cards
    def __init__(self, shuffled=True):
        self.cards = [num for num in range(52)]
        if shuffled:
            random.shuffle(self.cards)

    def removeCards(self, cards):
        for card in cards:
            self.cards.remove(card)

    def dealCard(self):
        return self.cards.pop()

    def size(self):
        return len(self.cards)

if __name__ == '__main__':
    newDeck = Deck()
    newDeck.removeCards([num for num in range(48)])
    for _ in range(newDeck.size()):
        print(CardUtils.numToCard(newDeck.dealCard()))
