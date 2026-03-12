# PokerML

> A **simulation-driven**, **range-aware** Heads-Up NLHE engine that blends heuristic game-tree search with configurable abstractions for rapid strategy iteration.

## Overview

PokerML models heads-up no-limit hold'em decision making as a configurable minimax search over betting trees. Player ranges are represented as weighted combos, equities are estimated with Monte Carlo rollouts, and betting decisions are guided by heuristic policies that split ranges into value, bluff, and checking buckets.

This is a research engine rather than a production-grade solver. Its value is in making poker abstractions easy to inspect and modify while you experiment with search depth, bet sizing, and range handling.

## Status

Implemented today:

- range representation and combo utilities
- board-aware equity calculations
- alpha-beta style search over poker nodes
- heuristic action selection and bet sizing
- sample river test harness for quick inspection

Not implemented yet:

- equilibrium solving through CFR
- polished experiment runners for multiple benchmark spots
- packaged dependency management beyond inline setup instructions

## Project layout

- `PokerNode.py`, `PokerPlayer.py`, `Deck.py` core game-state and card abstractions
- `Range.py` range parsing, weighting, and equity helpers
- `MiniMax.py` search orchestration
- `NodeHeuristics.py` heuristic betting and action logic
- `KMeans.py` clustering and abstraction experiments
- `SampleRiverTest.py` simple river scenario used as a smoke test
- `.env.example` configurable solver parameters

## Quickstart

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install numpy phevaluator scipy
cp .env.example .env
python SampleRiverTest.py
```

## Configuration

The solver reads optional values from `.env`:

```bash
BET_SIZINGS=0.3333333333,1.0,2.0,10.0
HEURISTIC_BET_SIZINGS=0,0.25,0.3333333333,0.5,0.75,1,1.3333333333,1.5,2,3,5,10
SEARCH_MAX_DEPTH=20
MONTE_CARLO_BASE=10
MONTE_CARLO_DECAY_START=4
MONTE_CARLO_MIN_SAMPLES=1
```

These values control the search tree shape and how aggressively equity sampling is pruned as recursion deepens.

## Architecture

- `Deck` and `CardUtils` manage card state and textual hand parsing
- `Range` stores weighted combinations and supports matchup calculations
- `PokerNode` expresses the current board, pot, stacks, and turn state
- `MiniMax` traverses the decision tree while `NodeHeuristics` proposes candidate actions
- `KMeans` and preflop equity tables support future abstraction experiments

## Example workflow

`SampleRiverTest.py` sets up a single river node where the out-of-position player has a stronger range and asks the search engine for the best action mix. That makes it the fastest way to verify the solver is wired correctly after setup.

Run:

```bash
python SampleRiverTest.py
```

## Limitations

- This is a heuristic engine, not an equilibrium solver
- Dependency installation is documented inline because there is no `requirements.txt` or `pyproject.toml`
- There is no benchmark suite or formal regression harness yet
- Example outputs are not documented in the README yet

## Why this repo is interesting

PokerML sits in the middle ground between toy poker scripts and industrial solvers. It is useful for prototyping ideas quickly because the abstractions are visible and hackable:

- hand ranges are explicit
- search behavior is configurable
- sampling tradeoffs are easy to inspect
- future solver ideas such as CFR can be layered on top of the same primitives

## License status

No license file is currently included in the repository. If this project is going to be shared more broadly, that should be resolved explicitly.
