from phevaluator import evaluate_cards


def evaluate_hand(board, hand):
    """Return phevaluator score for a 5-card board + 2-card hand. Lower is better."""
    cards = list(board) + list(hand)
    return evaluate_cards(*cards)


def determine_winner(board, hand0, hand1):
    """Return 0 if player 0 wins, 1 if player 1 wins, -1 for tie."""
    score0 = evaluate_cards(*board, *hand0)
    score1 = evaluate_cards(*board, *hand1)
    if score0 < score1:
        return 0
    elif score1 < score0:
        return 1
    return -1
