# PokerML

> A **simulation-driven**, **range-aware** Heads-Up NLHE engine that blends heuristic game-tree search with configurable abstractions for rapid strategy iteration.

## Overview
PokerML models Heads-Up No-Limit Hold'em decision-making as a configurable minimax search over betting trees. Player ranges are represented as weighted combos, equities are estimated with Monte Carlo rollouts, and betting decisions are guided by heuristic policies that split ranges into value, bluff, and checking buckets. The toolkit is designed to help researchers prototype solver ideas—ranging from toy games to full-street scenarios—without wrestling with low-level poker plumbing.

## Features
- Full-deck combinatorics, range management, and board-aware equity evaluation powered by `phevaluator`.
- Configurable minimax search with alpha-beta pruning and heuristic bet-size selection for tractable lookaheads.
- Modular architecture (deck, nodes, heuristics, clustering) that makes it easy to plug in new abstractions or training loops.

## Quickstart
```bash
# Setup
python3 -m venv .venv
source .venv/bin/activate
pip install numpy phevaluator scipy

# Run
python SampleRiverTest.py
```

## Configuration
This project uses environment variables for flexibility and portability.  
Copy `.env.example` to `.env` in the project root with entries like:

```bash
BET_SIZINGS=0.3333333333,1.0,2.0,10.0
HEURISTIC_BET_SIZINGS=0,0.25,0.3333333333,0.5,0.75,1,1.3333333333,1.5,2,3,5,10
SEARCH_MAX_DEPTH=20
MONTE_CARLO_BASE=10
MONTE_CARLO_DECAY_START=4
MONTE_CARLO_MIN_SAMPLES=1
```

## Architecture
- `PokerNode`, `PokerPlayer`, and `Deck` express the game state and enforce stack/pot transitions as the tree unfolds.
- `Range` (plus `CardUtils`) manages combinations, abstractions, and Monte Carlo equity rollouts against opponents and boards.
- `MiniMax` orchestrates alpha-beta search, while `NodeHeuristics` provides range-splitting logic for betting, checking, and calling.
- `KMeans` and pre-flop equity tables illustrate how to bucket hands for future large-scale abstractions.

## Next Steps
- Extend the solver with counterfactual regret minimization to generate equilibrium strategies beyond heuristic play.
- Introduce board-texture-aware hand bucketing (K-means + equity features) to scale to full-game decision trees.
- Plug in reinforcement learning or policy gradient fine-tuning to adapt strategies against specific opponent models.

## Tech Highlights
- **Type safety / clarity:** Deterministic range maths built on `numpy` with explicit combo bookkeeping.
- **Maintainability:** Centralized config (`config.py` + `.env`) and clearly named bet-sizing constants.
- **Scalability:** Stateless modules and seedable Monte Carlo sampling for reproducible experiments.

## Example
```bash
# Run a solved river spot and print the recommended action mix
python SampleRiverTest.py
```

## License
TODO
