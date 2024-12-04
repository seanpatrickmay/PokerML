from Range import Range
from PokerNode import PokerNode
from PokerPlayer import PokerPlayer
from Deck import Deck
import CardUtils
from MiniMax import getBestAction

#Test file, tests sample simple river node, shows strategy chosen when out of position with much stronger range

heroRange = Range(CardUtils.textToHandSet("98,AK,K9,K8,A9,A8,22+,QJ"))
villainRange = Range(CardUtils.textToHandSet("Q6+,K3+,43+,A2+"))
hero = PokerPlayer(50, (51, 47), heroRange)
villain = PokerPlayer(50, (0, 0), villainRange)
Board = [40, 36, 33, 11, 17]
Deck = Deck()
Deck.removeCards(Board)
Node = PokerNode(hero, villain, True, False, board=Board, pot=25, name="Root test node", deck=Deck)
bestActions = getBestAction(Node)
