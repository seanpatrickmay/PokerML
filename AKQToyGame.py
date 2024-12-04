from Range import Range
from PokerNode import PokerNode
from PokerPlayer import PokerPlayer
from Deck import Deck
import CardUtils
from MiniMax import getBestAction

#Test file, tests toy game, shows strategy chosen when in position, and out of position

acesQueensRange = Range(CardUtils.textToHandSet("AA,QQ"))
kingsRange = Range(CardUtils.textToHandSet("KK"))
OOP = PokerPlayer(1, (0, 0), acesQueensRange)
IP = PokerPlayer(1, (0, 0), kingsRange)
Board = [0, 1, 2, 3, 4]
Deck = Deck()
Deck.removeCards(Board)
Node = PokerNode(OOP, IP, True, False, board=Board, pot=1, name="Toy game", deck=Deck)
bestActions = getBestAction(Node)
Node = PokerNode(IP, OOP, False, True, board=Board, pot=1, name="Toy game", deck=Deck)
bestActions = getBestAction(Node)
