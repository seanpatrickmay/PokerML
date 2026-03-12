#!/usr/bin/env python3
"""Train a CFR solver for heads-up no-limit hold'em."""

import argparse
import time

from cfr.card_abstraction import CardAbstraction
from cfr.cfr_trainer import CFRTrainer
from cfr.strategy_store import save_strategy


def main():
    ap = argparse.ArgumentParser(description="Train HUNL poker CFR solver")
    ap.add_argument("--iterations", type=int, default=100_000)
    ap.add_argument("--preflop-buckets", type=int, default=10)
    ap.add_argument("--postflop-buckets", type=int, default=10)
    ap.add_argument("--output", default="strategy.json.gz")
    args = ap.parse_args()

    print(f"Card abstraction: {args.preflop_buckets} preflop / "
          f"{args.postflop_buckets} postflop buckets")
    abstraction = CardAbstraction(
        num_preflop_buckets=args.preflop_buckets,
        num_postflop_buckets=args.postflop_buckets,
    )

    trainer = CFRTrainer(card_abstraction=abstraction, iterations=args.iterations)

    print(f"Training {args.iterations:,} iterations …")
    t0 = time.time()
    trainer.train()
    elapsed = time.time() - t0

    print(f"\nDone in {elapsed:.1f}s  —  {len(trainer.info_sets):,} info sets")

    strategy = trainer.get_average_strategy()
    save_strategy(strategy, args.output)
    print(f"Strategy saved → {args.output}")


if __name__ == "__main__":
    main()
