"""
Neural network leaf value estimator for depth-limited solving.

Replaces Monte Carlo equity sampling with a single forward pass through
a small MLP. Trains on (features → MC equity) pairs generated offline.

Architecture:
  Input (5 features) → Linear(32) → ReLU → Linear(16) → ReLU → Linear(1) → Sigmoid

Features:
  1. hand_strength: bucket / num_buckets (normalized current strength)
  2. pot_ratio: pot / (2 * starting_stack) (how much is at stake)
  3. invested_ratio: invested / starting_stack (sunk cost)
  4. street_progress: street / 3.0 (0=preflop, 1=river)
  5. opponent_density: num_opponents / 9.0 (multi-way factor)

Falls back gracefully to MC equity when torch is unavailable or model
hasn't been trained yet.
"""

import os
import random
import numpy as np

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from cfr.game_state import STARTING_STACK

_MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', 'leaf_estimator.pt')
_TRAINING_DATA_PATH = os.path.join(os.path.dirname(__file__), '..', 'leaf_training_data.npz')

_cached_model = None


def _build_model():
    """Build the small MLP for leaf value estimation."""
    if not TORCH_AVAILABLE:
        return None
    return nn.Sequential(
        nn.Linear(5, 32),
        nn.ReLU(),
        nn.Linear(32, 16),
        nn.ReLU(),
        nn.Linear(16, 1),
        nn.Sigmoid(),
    )


def _make_features(bucket, pot, stack, street, num_opponents,
                   num_buckets=200, starting_stack=STARTING_STACK):
    """Convert game state to normalized feature vector."""
    invested = starting_stack - stack
    return np.array([
        (bucket + 0.5) / max(num_buckets, 1),
        pot / (2 * starting_stack),
        invested / starting_stack,
        street / 3.0 if street is not None else 0.5,
        num_opponents / 9.0,
    ], dtype=np.float32)


def load_model():
    """Load a trained model from disk. Returns None if unavailable."""
    global _cached_model
    if _cached_model is not None:
        return _cached_model
    if not TORCH_AVAILABLE:
        return None
    if not os.path.exists(_MODEL_PATH):
        return None

    model = _build_model()
    model.load_state_dict(torch.load(_MODEL_PATH, weights_only=True))
    model.eval()
    _cached_model = model
    return model


def predict_equity(bucket, pot, stack, street=None, num_opponents=1,
                   num_buckets=200, starting_stack=STARTING_STACK):
    """Predict equity using the NN model. Returns None if model unavailable."""
    model = load_model()
    if model is None:
        return None

    features = _make_features(bucket, pot, stack, street, num_opponents,
                              num_buckets, starting_stack)
    with torch.no_grad():
        x = torch.from_numpy(features).unsqueeze(0)
        equity = model(x).item()
    return equity


def generate_training_data(card_abstraction, num_samples=50_000,
                           mc_samples=200):
    """Generate training data from MC equity sampling across random game states.

    Produces (features, equity) pairs by sampling random hands/boards at
    various streets and computing ground-truth equity via MC.
    """
    from phevaluator import evaluate_cards

    features_list = []
    equity_list = []

    deck = list(range(52))

    for _ in range(num_samples):
        random.shuffle(deck)
        hand = (deck[0], deck[1])
        street = random.choice([1, 2, 3])  # flop, turn, river
        board_len = street + 2  # 3, 4, or 5
        board = tuple(deck[2:2 + board_len])

        num_opponents = random.randint(1, 5)
        pot = random.uniform(10, 400)
        stack = random.uniform(50, STARTING_STACK)
        invested = STARTING_STACK - stack

        # Get bucket
        bucket = card_abstraction.get_bucket(hand, board)
        num_buckets = card_abstraction.num_postflop_buckets

        # Compute MC equity (ground truth)
        used = set(hand) | set(board)
        remaining_deck = [c for c in range(52) if c not in used]
        remaining_board = 5 - len(board)
        cards_needed = remaining_board + 2 * num_opponents

        if len(remaining_deck) < cards_needed:
            continue

        wins = 0
        ties_count = 0
        total = 0

        for _ in range(mc_samples):
            drawn = random.sample(remaining_deck, cards_needed)
            full_board = list(board) + drawn[:remaining_board]

            hero_score = evaluate_cards(*full_board, *hand)

            beat_all = True
            tied_all = True
            for opp in range(num_opponents):
                offset = remaining_board + opp * 2
                opp_hand = (drawn[offset], drawn[offset + 1])
                opp_score = evaluate_cards(*full_board, *opp_hand)
                if opp_score < hero_score:
                    beat_all = False
                    tied_all = False
                    break
                elif opp_score == hero_score:
                    beat_all = False
                else:
                    tied_all = False

            if beat_all:
                wins += 1
            elif tied_all:
                ties_count += 1
            total += 1

        if total == 0:
            continue

        equity = (wins + 0.5 * ties_count) / total

        feat = _make_features(bucket, pot, stack, street, num_opponents,
                              num_buckets, STARTING_STACK)
        features_list.append(feat)
        equity_list.append(equity)

    X = np.array(features_list, dtype=np.float32)
    y = np.array(equity_list, dtype=np.float32)

    np.savez(_TRAINING_DATA_PATH, X=X, y=y)
    print(f"Saved {len(X)} training samples to {_TRAINING_DATA_PATH}")
    return X, y


def train_model(X=None, y=None, epochs=100, lr=0.001, batch_size=256):
    """Train the leaf estimator MLP on (features, equity) data.

    If X and y are not provided, loads from the saved training data file.
    """
    if not TORCH_AVAILABLE:
        print("torch not available, skipping NN training")
        return None

    if X is None or y is None:
        if not os.path.exists(_TRAINING_DATA_PATH):
            print(f"No training data at {_TRAINING_DATA_PATH}. "
                  "Run generate_training_data() first.")
            return None
        data = np.load(_TRAINING_DATA_PATH)
        X, y = data['X'], data['y']

    model = _build_model()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    X_t = torch.from_numpy(X)
    y_t = torch.from_numpy(y).unsqueeze(1)

    n = len(X_t)
    indices = list(range(n))

    for epoch in range(epochs):
        random.shuffle(indices)
        epoch_loss = 0.0
        batches = 0

        for start in range(0, n, batch_size):
            batch_idx = indices[start:start + batch_size]
            xb = X_t[batch_idx]
            yb = y_t[batch_idx]

            pred = model(xb)
            loss = loss_fn(pred, yb)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            batches += 1

        if (epoch + 1) % 20 == 0:
            avg_loss = epoch_loss / batches
            print(f"  Epoch {epoch+1}/{epochs}  loss={avg_loss:.6f}")

    torch.save(model.state_dict(), _MODEL_PATH)
    print(f"Model saved to {_MODEL_PATH}")

    global _cached_model
    _cached_model = model
    model.eval()
    return model
