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


def determine_winners(board, hands, eligible_seats):
    """Return list of winning seat indices among eligible players. Ties possible."""
    best_score = float('inf')
    winners = []
    for seat in eligible_seats:
        score = evaluate_cards(*board, *hands[seat])
        if score < best_score:
            best_score = score
            winners = [seat]
        elif score == best_score:
            winners.append(seat)
    return winners


def evaluate_all_hands(board, hands, active_seats):
    """Evaluate all active hands once, return dict of seat -> score."""
    scores = {}
    for seat in active_seats:
        scores[seat] = evaluate_cards(*board, *hands[seat])
    return scores


def enumerate_river_equity(hand, board):
    """Compute exact equity on the river by enumerating all possible opponent hands.

    With 5 board cards and 2 hero cards, 45 cards remain → C(45,2) = 990
    possible opponent hands. Evaluating all of them gives exact equity with
    no sampling variance, replacing MC estimation at the river.

    Returns a float in [0, 1].
    """
    used = set(hand) | set(board)
    available = [c for c in range(52) if c not in used]

    hero_score = evaluate_cards(*board, *hand)

    wins = 0
    ties = 0
    total = 0

    for i in range(len(available)):
        for j in range(i + 1, len(available)):
            opp_score = evaluate_cards(*board, available[i], available[j])
            if hero_score < opp_score:
                wins += 1
            elif hero_score == opp_score:
                ties += 1
            total += 1

    if total == 0:
        return 0.5
    return (wins + 0.5 * ties) / total


def batch_river_equity(hand, board, opponent_hands):
    """Compute equity against a specific set of opponent hands on the river.

    Args:
        hand: Hero's 2-card hand (tuple of ints)
        board: 5-card board (tuple of ints)
        opponent_hands: List of (card1, card2) tuples

    Returns:
        Float equity in [0, 1] against the given opponent range.
    """
    if not opponent_hands:
        return 0.5

    hero_score = evaluate_cards(*board, *hand)

    wins = 0
    ties = 0
    total = len(opponent_hands)

    for opp_hand in opponent_hands:
        opp_score = evaluate_cards(*board, opp_hand[0], opp_hand[1])
        if hero_score < opp_score:
            wins += 1
        elif hero_score == opp_score:
            ties += 1

    return (wins + 0.5 * ties) / total


def winners_from_scores(scores, eligible_seats):
    """Find winners from pre-computed scores. Avoids re-evaluation."""
    best_score = float('inf')
    winners = []
    for seat in eligible_seats:
        score = scores[seat]
        if score < best_score:
            best_score = score
            winners = [seat]
        elif score == best_score:
            winners.append(seat)
    return winners
