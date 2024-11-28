from Range import Range
import numpy as np
import CardUtils
from PokerNode import PokerNode
from PokerPlayer import PokerPlayer
from Deck import Deck
import NodeHeuristics

# Expressed in # of pots
BET_SIZINGS = (1/3, 1.0, 2.0, 10.0)

# Max recursion depth for tree search
SEARCH_MAX_DEPTH = 5

# Class for representing minimax strategy
# Currently make assumption that we either bet entire range or check

# For a given node, what children exist?
# If no debt and oop, check
# If no debt and ip, check: next card / showdown
# If any debt, call: next card / showdown
# If any debt, raise: 1 node for each sizing

# A raise node will always be accompanied by a check node, with given weights for the range.
# Ranges will also be updated each time a raise/call is made.
# A pure check node will not update range. (Range check)

# Returns all child nodes of a given node
# If folded or showdown, terminal node, no children
def getChildNodes(node):
    #print(f"Getting children nodes of: {node}\n")
    if node.spr == 0 and node.debt == 0:
        return [getCheckCallNodes(node)]
    allChildren = []
    if node.bets==3:
        allChildren = getCheckCallNodes(node)
    else:
        checkNodes = getCheckCallNodes(node)
        if checkNodes:
            allChildren += [checkNodes]
        foldNodes = getFoldNodes(node)
        if foldNodes:
            allChildren += [foldNodes]
        betNodes = getBetNodes(node)
        if betNodes:
            #print(f"BetNodes: {betNodes}")
            allChildren += betNodes
    return allChildren


# Nodes needed:
# Check / call
# Raise for each size
# Fold
def getBetNodes(node):
    allBetNodes = []
    newBetSizings = [node.spr]
    for sizing in BET_SIZINGS:
        if sizing * node.pot >= 1 and sizing < node.spr:
            newBetSizings.append(sizing)
    
    print(f"BET SIZINGS: {newBetSizings}")

    if len(newBetSizings) == 0:
        return allBetNodes

    #print(newBetSizings)
    for size in newBetSizings:
        currentBetCheckSet = []
        if node.turn:
            betRange, checkRange = NodeHeuristics.splitBetCheckRanges(node.hero.handRange, node.villain.handRange, node.board, size)
        else:
            betRange, checkRange = NodeHeuristics.splitBetCheckRanges(node.villain.handRange, node.hero.handRange, node.board, size)
        print(f"New bet range:\n {betRange}")
        print(f"New check range:\n {checkRange}")
        betRangeSize = len(betRange.hands)
        betWeight = betRangeSize / (betRangeSize + len(checkRange.hands))
        if betWeight == 0:
            continue
        #print(f"Raise node for size {size} is: {getRaiseNode(node, size, betRange, betWeight)}")
        newRaiseNode = getRaiseNode(node, size, betRange, betWeight * node.weight)
        if newRaiseNode == None:
            continue
        currentBetCheckSet.append(newRaiseNode)
        if betWeight == 1:
            allBetNodes.append(currentBetCheckSet)
            continue
        newCheckNode = PokerNode(node.hero, node.villain, node.turn, node.position, board=node.board, pot = node.pot, deck=node.deck, debt=node.debt, weight=(1 - betWeight) * node.weight, name=f"Check node, along with betting {size}")
        if node.turn:
            newCheckNode.hero.handRange = checkRange
        else:
            newCheckNode.villain.handRange = checkRange
        newCheckNodes = getCheckCallNodes(newCheckNode)
        if newCheckNodes:
            currentBetCheckSet.append(newCheckNodes)
        allBetNodes.append(currentBetCheckSet)
    return allBetNodes
    

# Bet heuristic is a function that returns two ranges, one that checked, one that bet
def getRaiseNode(node, size, newRange, newWeight):
    newPot = node.pot + abs(node.debt) * 2
    newHero = node.hero.copy()
    newVillain = node.villain.copy()
    if node.turn:
        newHero.chips -= abs(node.debt)
    else:
        newVillain.chips -= abs(node.debt)
    newSpr = min(newHero.chips / newPot, newVillain.chips / newPot)
    # If raise is larger than spr, invalid sizing
    if size > newSpr:
        return None
    newDebt = newPot * size
    if node.turn:
        newHero.chips -= newDebt
        newHero.handRange = newRange
        newDebt *= -1
    else:
        newVillain.chips -= newDebt
        newVillain.handRange = newRange
    return PokerNode(newHero, newVillain, not node.turn, node.position, weight=newWeight * node.weight, board=node.board, pot=newPot, deck=node.deck, debt=newDebt, bets=node.bets + 1, name=f"Raising {size} pots")


def getFoldNodes(node):
    # Not allowed to fold if not in debt
    if node.debt == 0:
        return None
    # Return the value of the terminal node
    # If a player folded, however much is in pot / 2 is how much they lost
    # Hero folds = lose
    if node.turn:
        print(f"Folding! Reward is: {(node.pot / 2) * -1 * node.weight}")
        return (node.pot / 2) * -1 * node.weight
    # Villain folds = win
    else:
        print(f"Folding! Reward is: {(node.pot / 2) * node.weight}")
        return node.pot / 2 * node.weight

    
def getCheckCallNodes(node):
    # Not allowed to check in in debt
    if node.debt != 0:
        return getCallNode(node)
    # Returns all possible turn nodes if closing action
    if node.turn == node.position:
        return getAllNewCardNodes(node)
    # Else, same node but in position action now
    return [PokerNode(node.hero, node.villain, node.position, node.position, node.weight, node, node.board, node.pot, node.deck, name='Checking')]

def getCallNode(node):
    sizing = abs(node.debt) / node.pot
    if node.turn:
        callRange = NodeHeuristics.getCallRange(node.hero.handRange, node.villain.handRange, node.board, sizing)
    else:
        callRange = NodeHeuristics.getCallRange(node.villain.handRange, node.hero.handRange, node.board, sizing)
    newVillain = node.villain.copy()
    newHero = node.hero.copy()
    defFreq = 0
    print(f"Call hands: {len(callRange.hands)}")
    print(f"Hero hands: {len(node.hero.handRange.hands)}")
    print(f"Villain hands: {len(node.villain.handRange.hands)}")
    print(f"Call Range on board: {CardUtils.numsToCards(node.board)} with sizing: {sizing}: \n{callRange}")
    if node.turn:
        newHero.handRange = callRange
        newHero.chips -= abs(node.debt)
        defFreq = len(callRange.hands) / len(node.hero.handRange.hands)
        # What we won/lost for a fold: whatever we contributed to pot
        foldFreq = 1 - defFreq
        reward = foldFreq * node.pot / 2 * node.weight * -1
    else:
        newVillain.handRange = callRange
        newVillain.chips -= abs(node.debt)
        defFreq = len(callRange.hands) / len(node.villain.handRange.hands)
        foldFreq = 1 - defFreq
        reward = foldFreq * node.pot / 2 * node.weight
    newNode = PokerNode(newHero, newVillain, not node.position, node.position, weight=defFreq * node.weight, parent=node, board=node.board, pot=node.pot + abs(node.debt) * 2, deck = node.deck, name=f"Calling {sizing} bet with def freq {defFreq}")
    print(f"Defence Frequency on call node is: {defFreq}")
    print(f"Fold frequency on call node is: {foldFreq}")
    print(f"Fold reward on call node is: {reward}")
    print(f"Node weight is: {node.weight}")
    return [getAllNewCardNodes(newNode), reward]

# Get all the children of node
def getAllNewCardNodes(node):
    # If board already full, no new cards, so we go to showdown
    # EV at showdown is equal to pot * equity
    if len(node.board) == 5 or node.spr == 0:
        return node.weight * (node.pot * node.hero.handRange.equityAgainstRange(node.villain.handRange, node.board) - node.pot / 2)
    newNodes = []
    numNodes = len(node.deck.cards)
    for card in node.deck.cards:
        deckCopy = node.deck.copy()
        deckCopy.removeCards([card])
        newNodes.append(PokerNode(node.hero, node.villain, not node.position, node.position, node.weight/numNodes, node, node.board + [card], node.pot, deckCopy))
    return newNodes

def expandTree(tree):
    #print(f"Expanding tree on: {tree}")
    if isinstance(tree, list):
        return [expandTree(node) for node in tree]
    elif isinstance(tree, float):
        return tree
    else:
        return expandTree(getChildNodes(tree))
        #return [expandTree(child) for child in getChildNodes(tree)]

# First, need a way to evalute EV of an end node, in poker, this would be on the river
def alphaBeta(pokerNode, maximize, depth, alpha=float("-inf"), beta=float("inf")):
    if isinstance(pokerNode, float):
        return pokerNode
    if isinstance(pokerNode, list):
        subNodes = []
        for subNode in pokerNode:
            subNodes.append(alphaBeta(subNode, maximize, depth, alpha, beta))
        return sum(subNodes)
    if depth == 0:
        if pokerNode.debt != 0:
            if maximize:
                return float("-inf")
            else:
                return float("inf")
        # Could be some mistakes here, if node is right after a bet, range could be much stronger than reality
        return (NodeHeuristics.simpleEquityEvaluation(pokerNode.hero.handRange, pokerNode.villain.handRange, pokerNode.board) * pokerNode.pot - pokerNode.pot / 2) * pokerNode.weight
    # Else, its a node
    print(pokerNode)
    nextNodes = getChildNodes(pokerNode)
    print("Children:\n", nextNodes, '\n')
    best_node = None
    if maximize:
        best_node = float("-inf")
    else:
        best_node = float("inf")
    for node in nextNodes:
        current_value = alphaBeta(node, not maximize, depth - 1, alpha, beta)
        print(f"Current Value: {current_value}")
        print(f"Alpha: {alpha}")
        print(f"Beta: {beta}")
        print(f"Is it a maximize?: {maximize}")
        print(f"Previous best value: {best_node}")
        if maximize:
            alpha = max(alpha, current_value)
            best_node = max(best_node, current_value)
            if beta <= alpha:
                break
        else:
            beta = min(beta, current_value)
            best_node = min(best_node, current_value)
            if beta <= alpha:
                break
        print(f"Best node: {best_node}")
    # Returns the min/max value
    return best_node

def getBestAction(pokerNode):
    pokerNode.villain.handRange.removeCards(pokerNode.hero.hand)
    return alphaBeta(testNode, pokerNode.turn, SEARCH_MAX_DEPTH)


    

if __name__ == "__main__":
    '''testHeroRange = Range(empty=True)
    testVillainRange = Range(empty=True)
    for ace1 in range(51, 48, -1):
        for ace2 in range(ace1 - 1, 47, -1):
            testHeroRange.add((ace1, ace2))
            testHeroRange.add((ace1 - 8, ace2 - 8))
            testVillainRange.add((ace1 - 4, ace2 - 4))
    #print(testHeroRange)
    #print(testVillainRange)
    
    testBoard = [0, 9, 18, 27, 32]
    testHero = PokerPlayer(1, (51, 50), testHeroRange)
    testVillain = PokerPlayer(1, (4, 0), testVillainRange)
    testDeck = Deck()
    testDeck.removeCards(testBoard)
    testDeck.removeCards(testHero.hand)
    testDeck.removeCards(testVillain.hand)
    testNode = PokerNode(testHero, testVillain, True, True, pot=1, board=testBoard, deck=testDeck)
    '''
    testHeroRange = Range()
    testVillainRange = Range()
    testBoard = [51, 46, 9, 2, 30]
    testHero = PokerPlayer(50, (50, 47), testHeroRange)
    testVillain = PokerPlayer(50, (49, 45), testVillainRange)
    testDeck = Deck()
    testDeck.removeCards(testBoard)
    testDeck.removeCards(testHero.hand)
    testDeck.removeCards(testVillain.hand)
    testNode = PokerNode(testHero, testVillain, False, True, pot=75, board=testBoard, deck=testDeck)

    #print('-' * 50)
    #expandedNode = expandTree(testNode)
    #print(expandedNode)
    #print(minMax(expandedNode, False))
    #print(findLayers(expandedNode))
    #print(getBestAction(testNode))
    #print(testNode.villain.handRange)
    #print(testNode.hero.handRange)
    #print(testNode.hero.handRange.equityAgainstRange(testNode.villain.handRange, testNode.board))
    #betNodes = getBetNodes(testNode)
    #print(getCheckCallNodes(betNodes[0][0]))
    print(getBestAction(testNode))
