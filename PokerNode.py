from Deck import Deck
from Range import Range
import CardUtils
#Class for representing a gamestate node in HU NLHE

class PokerNode:
    def __init__(self, hero, villain, turn, position, weight=1, parent=None, board=[], pot=0, deck=Deck(), debt=0, bets=0, name='Unnamed', EV=None):
        self.parent = parent
        self.board = board.copy()
        self.hero = hero.copy()
        self.villain = villain.copy()
        self.pot = pot
        self.deck = deck.copy()
        self.name = name
        # To limit recursion, only 3 bets allowed per round
        self.bets = bets
        # Stack-to-pot ratio. 0 if all in player.
        self.spr = min(hero.chips / pot, villain.chips / pot)
        # Represents a bet that isn't yet called. + is villain bet. - is hero bet.
        self.debt = debt
        # Who's turn is it? True = Hero, False = Villain
        self.turn = turn
        # Who's in position? True = Hero, False = Villain
        self.position = position
        # What is the chance of this node occuring?
        self.weight = weight
        # Set the EV initially to none
        self.EV = EV


    def __str__(self):
        turn = "Villain"
        if self.turn:
            turn = "Hero"
        position = "Villain"
        if self.position:
            position = "Hero"
        return f"Pot: {self.pot} + Bet: {self.debt} | {CardUtils.numsToCards(self.board)} | Action: {turn} | Position: {position}\nHero: {self.hero.chips}$ {CardUtils.numsToCards(self.hero.hand)} ||| Villain: {self.villain.chips}$ {CardUtils.numsToCards(self.villain.hand)} | Weight: {self.weight}\nNode name: {self.name} | Node EV: {self.EV}\n" 
        
