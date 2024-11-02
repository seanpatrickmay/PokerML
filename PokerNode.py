#Class for representing a gamestate node in HU NLHE

class PokerNode:
    def ___init__(self, hero, villain, board=[], pot=0):
        self.board = board
        self.hero = hero
        self.villain = villain
        self.pot = pot
