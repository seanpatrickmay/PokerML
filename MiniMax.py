from Range import Range
import numpy as np
import CardUtils
from PokerNode import PokerNode
from PokerPlayer import PokerPlayer
from Deck import Deck
import NodeHeuristics

# Expressed in # of pots
BET_SIZINGS = (1/3, 1.0, 2.0, 10.0)

# Max recursion depth for tree search. Experimenting with multiple values.
SEARCH_MAX_DEPTH = 20

# Class for representing minimax strategy

# For a given node, what children exist?
# If no debt and oop, check
# If no debt and ip, check: next card / showdown
# If any debt, call: next card / showdown
# Always, raise: 1 node for each sizing

# A raise node will almost always be accompanied by a check node, with given weights for the range.
# If raise frequency is 100%, there will not be a check node, as no hands are choosing to check.
# Ranges will also be updated each time a raise/call is made.
# A pure check/raise node will not update range. (Range check)

# Returns all child nodes of a given node
# If folded or showdown, terminal node, no children
def getChildNodes(node):
    allChildren = []
    checkNodes = getCheckCallNodes(node)
    if checkNodes:
        allChildren += [checkNodes]
    foldNodes = getFoldNodes(node)
    if foldNodes:
        allChildren += [foldNodes]
    betNodes = getBetNodes(node)
    if betNodes:
        allChildren += betNodes
    return allChildren

# Returns all the bet children nodes from given node
def getBetNodes(node):
    allBetNodes = []

    # Find all viable sizings
    newBetSizings = [node.spr]
    for sizing in BET_SIZINGS:
        if sizing * node.pot >= 1 and sizing < node.spr:
            newBetSizings.append(sizing)
    
    # If no bet sizings, no bet nodes
    if len(newBetSizings) == 0:
        return allBetNodes

    # Iterate through each bet sizing, make new node
    for size in newBetSizings:
        # We don't bet our entire range when we bet, so we split the range into bet/check (or call)
        # For this, we assign weights based on how many hands bet/check
        currentBetCheckSet = []

        # Set our new bet/check ranges using basic Heuristic equity function
        if node.turn:
            betRange, checkRange = NodeHeuristics.splitBetCheckRanges(node.hero.handRange, node.villain.handRange, node.board, size)
        else:
            betRange, checkRange = NodeHeuristics.splitBetCheckRanges(node.villain.handRange, node.hero.handRange, node.board, size)

        # Find the weights
        betRangeSize = len(betRange.hands)
        betWeight = betRangeSize / (betRangeSize + len(checkRange.hands))
        # If never betting, no bet node for this sizing, so continue
        if betWeight == 0:
            continue

        # Helper method to create new node
        newRaiseNode = getRaiseNode(node, size, betRange, betWeight)
        # If unable to make raise node, if SPR too low, for example
        if newRaiseNode == None:
            continue

        # Add the node to current bet/check set, we will make check node next
        currentBetCheckSet.append(newRaiseNode)

        # If range betting, (betting entire range) we don't have a check node.
        if betWeight == 1:
            allBetNodes.append(currentBetCheckSet)
            continue

        # Create new check node with appropriate ranges
        newCheckNode = PokerNode(node.hero, node.villain, node.turn, node.position, board=node.board, pot = node.pot, deck=node.deck, debt=node.debt, weight=1 - betWeight, name=f"Check node, along with betting {size}")
        if node.turn:
            newCheckNode.hero.handRange = checkRange
        else:
            newCheckNode.villain.handRange = checkRange

        # Helper function that updates node to check/call
        currentBetCheckSet.append(getCheckCallNodes(newCheckNode))
        
        # Done with current sizing, add to all bets list, continue
        allBetNodes.append(currentBetCheckSet)

    return allBetNodes
    

# Helper function for making new raise node
def getRaiseNode(node, size, newRange, newWeight):
    # Add debt to the pot, takes away chips from players accordingly
    newPot = node.pot + abs(node.debt) * 2
    newHero = node.hero.copy()
    newVillain = node.villain.copy()
    if node.turn:
        newHero.spendChips(abs(node.debt))
    else:
        newVillain.spendChips(abs(node.debt))

    # Calculates new spr of pot based on player chip values, to check for invalid raise size    
    newSpr = min(newHero.chips / newPot, newVillain.chips / newPot)
    # If raise is larger than spr, invalid sizing, so no node
    if size > newSpr:
        return None

    # Calculate new debt, apply to betting player
    newDebt = newPot * size
    if node.turn:
        newHero.spendChips(newDebt)
        newHero.handRange = newRange
        newDebt *= -1
    else:
        newVillain.spendChips(newDebt)
        newVillain.handRange = newRange

    # Return new node with all needed information
    return PokerNode(newHero, newVillain, not node.turn, node.position, weight=newWeight, board=node.board, pot=newPot, deck=node.deck, debt=newDebt, name=f"Raising {size} pots")

# Helper method to get fold child node
def getFoldNodes(node):
    # Not allowed to fold if not in debt (technically you are but would be stupid, check is free)
    if node.debt == 0:
        return None

    # If a player folded, however much is in pot / 2 is how much they lost, since that is how much the contributed
    # Hero folds = lose
    if node.turn:
        reward = (node.pot / 2) * -1

    # Villain folds = win
    else:
        reward = node.pot / 2
        
    # Return new node with EV of folding
    return PokerNode(node.hero, node.villain, not node.turn, node.position, weight=1, parent=node, board=node.board, pot=node.pot, deck=node.deck, debt=node.debt, bets=node.bets, name=f"Folding range to a bet", EV=reward)

    
# Method to get all check/call child nodes
def getCheckCallNodes(node):
    # Not allowed to check if in debt, call instead
    if node.debt != 0:
        return getCallNode(node)

    # Returns all possible turn nodes if closing action, either new card, or showdown
    if node.turn == node.position:
        return getAllNewCardNodes(node)

    # Else, same node but in position action now
    return PokerNode(node.hero, node.villain, node.position, node.position, node.weight, node, node.board, node.pot, node.deck, name='Checking')

# Method to get call node, including calling only certain part of range
def getCallNode(node):
    # Determine the sizing based off previous bet
    sizing = abs(node.debt) / node.pot

    # Use heuristic function to get range of hands to call with
    if node.turn:
        callRange = NodeHeuristics.getCallRange(node.hero.handRange, node.villain.handRange, node.board, sizing)
    else:
        callRange = NodeHeuristics.getCallRange(node.villain.handRange, node.hero.handRange, node.board, sizing)

    # Copies of players to use in new nodes
    newVillain = node.villain.copy()
    newHero = node.hero.copy()
    foldHero = node.hero.copy()
    foldVillain = node.villain.copy()

    # Edit ranges and chip counts for players
    if node.turn:
        defFreq = len(callRange.hands) / len(newHero.handRange.hands)
        foldFreq = 1 - defFreq
        foldHero.shrinkRange(callRange.hands)
        newHero.handRange = callRange
        newHero.spendChips(abs(node.debt))
        reward = node.pot / 2 * -1
    else:
        defFreq = len(callRange.hands) / len(newVillain.handRange.hands)
        foldFreq = 1 - defFreq
        foldVillain.shrinkRange(callRange.hands)
        newVillain.handRange = callRange
        newVillain.spendChips(abs(node.debt))
        reward = node.pot / 2
    newPot = node.pot + abs(node.debt) * 2

    # Make new call node
    newNode = PokerNode(newHero, newVillain, not node.position, node.position, weight=defFreq, parent=node, board=node.board, pot=newPot, deck = node.deck, name=f"Calling {sizing} bet with def freq {defFreq}")

    # Make new fold node
    rewardNode = PokerNode(foldHero, foldVillain, not node.position, node.position, weight=foldFreq, parent=node, board=node.board, pot=node.pot, deck=node.deck, name=f"Folding to {sizing} bet with {foldFreq * 100} % of range", EV=reward)

    # Return the two new nodes as a pair, calling method to edit call node
    return [getAllNewCardNodes(newNode), rewardNode]

# Get all the children of node that prompts a new card, or goes to showdown
def getAllNewCardNodes(node):
    # EV at showdown is equal to pot * equity - pot / 2, or, what we win - what we put in
    # If board already full, or not more betting possible, we go to showdown
    if len(node.board) == 5 or node.spr == 0:
        # Calculate equity and EV using heuristic functions
        equity = NodeHeuristics.simpleEquityEvaluation(node.hero.handRange, node.villain.handRange, node.board)
        EV = node.pot * equity - node.pot / 2
        
        # Return new node with found EV
        return PokerNode(node.hero, node.villain, node.turn, node.position, 1, node.parent, node.board, node.pot, node.deck, node.debt, node.bets, name="Checking back / calling, going to showdown", EV=EV)

    # Find all possible new nodes from deck
    newNodes = []
    numNodes = len(node.deck.cards)
    for card in node.deck.cards:
        # New deck for node without new card
        deckCopy = node.deck.copy()
        deckCopy.removeCards([card])

        # New node with weight of P(card), and new deck
        newNodes.append(PokerNode(node.hero, node.villain, not node.position, node.position, 1/numNodes, node, node.board + [card], node.pot, deckCopy))

    return newNodes

# Helper method to recursively find EVs in nested lists of nodes
def sumEVWeights(node):
    if isinstance(node, PokerNode):
        return node.EV * node.weight
    # Otherwise, its a list
    else:
        return sum([sumEVWeights(subNode) for subNode in node])

# Alpha-beta pruning method. Recurs on entire tree, setting EV of each node
def alphaBeta(pokerNode, maximize, depth, alpha=float("-inf"), beta=float("inf")):
    # For grouped nodes, such as bet/check nodes, run alpha-beta on each
    if isinstance(pokerNode, list):
        for subNode in pokerNode:
            alphaBeta(subNode, maximize, depth, alpha, beta)
        return

    # If EV already found, leaf node, no further recursion needed
    if pokerNode.EV != None:
        return

    # If max depth reached, stop recurring, use heuristic to calculate EV
    if depth == 0:
        # Could be some mistakes here, if node is right after a bet, range could be much stronger than reality
        pokerNode.EV = (NodeHeuristics.simpleEquityEvaluation(pokerNode.hero.handRange, pokerNode.villain.handRange, pokerNode.board) * pokerNode.pot - pokerNode.pot / 2)
        return

    # Now handling case where pokerNode is actually a node, not list
    nextNodes = getChildNodes(pokerNode)

    # Initialize best_node to worst case based on if this is a max or min layer
    if maximize:
        best_ev = float("-inf")
    else:
        best_ev = float("inf")
    best_node = None

    # Alpha-beta on each child node
    for node in nextNodes:
        alphaBeta(node, not maximize, depth - 1, alpha, beta)
        current_value = sumEVWeights(node) # Helper method to find EV sum if node is combo (ex. check/bet)
        # Standard alpha-beta implementation
        if maximize:
            alpha = max(alpha, current_value)
            if current_value > best_ev:
                best_ev = current_value
                best_node = node
            if beta <= alpha:
                break
        else:
            beta = min(beta, current_value)
            if current_value < best_ev:
                best_ev = current_value
                best_node = node
            if beta <= alpha:
                break

    # Sets this nodes EV to best of all children
    pokerNode.EV = best_ev
    return best_node

# Returns the best action for a given node
def getBestAction(pokerNode):
    print(f"Searching for the best action for the following node:\n{pokerNode}")
    # Uses hero's blockers (cards) to remove hands from villain range
    pokerNode.villain.handRange.removeCards(pokerNode.hero.hand)

    # Alpha-beta on given node, which returns best child node
    bestNode = alphaBeta(pokerNode, pokerNode.turn, SEARCH_MAX_DEPTH)
    
    # If single node
    if isinstance(bestNode, PokerNode):
        print(f"Best action is: {bestNode.name}")
        print(f"Total EV is: {bestNode.EV}")
        return bestNode
    # Else, its a pairing of nodes
    else:
        if len(bestNode) == 1:
            print(f"Best action is: {bestNode[0].name} with entire range")
            print(f"Total EV is: {bestNode[0].EV}")
            return bestNode[0]
        print(f"Best action is to split range")
        if not bestNode[0].turn: # Find the bet/(check/call) ranges
            betRange = bestNode[0].hero.handRange
            checkRange = bestNode[1].hero.handRange
        else:
            betRange = bestNode[0].villain.handRange
            checkRange = bestNode[1].villain.handRange

        # Print both bet and check ranges
        print(f"{bestNode[0].name} with the following Range:\n{betRange}")
        print(f"{bestNode[1].name} with the following Range:\n{checkRange}")
        print(f"Total EV is: {sumEVWeights(bestNode)}")
        return bestNode

if __name__ == "__main__":
    print("ROAR")
