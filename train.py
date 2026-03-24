#!/usr/bin/env python3
"""
Train a CFR solver for N-player no-limit hold'em.

Features:
  - Postflop-only mode (default for 6-max): uses fixed preflop ranges
  - DCFR discounting for faster convergence
  - Regret-based pruning after warmup
  - Warm-start from existing strategy
  - Continuous mode: train indefinitely with periodic checkpoints
"""

import argparse
import signal
import sys
import time

from cfr.card_abstraction import CardAbstraction
from cfr.cfr_trainer import CFRTrainer
from cfr.strategy_store import save_strategy


_stop = False


def _handle_sigint(sig, frame):
    global _stop
    if _stop:
        print("\nForce quit.")
        sys.exit(1)
    print("\nGraceful shutdown — saving checkpoint...")
    _stop = True


def main():
    ap = argparse.ArgumentParser(description="Train poker CFR solver")
    ap.add_argument("--num-players", type=int, default=6)
    ap.add_argument("--iterations", type=int, default=0,
                    help="Total iterations (0 = continuous/indefinite)")
    ap.add_argument("--preflop-buckets", type=int, default=0,
                    help="Preflop buckets (0 = auto)")
    ap.add_argument("--postflop-buckets", type=int, default=0,
                    help="Postflop buckets (0 = auto)")
    ap.add_argument("--output", default="strategy_6max.json.gz")
    ap.add_argument("--checkpoint-interval", type=int, default=25_000,
                    help="Save every N iterations")
    ap.add_argument("--warm-start", default=None,
                    help="Path to existing strategy to continue from")
    ap.add_argument("--no-warm-start", action="store_true",
                    help="Start fresh, don't load existing strategy")
    ap.add_argument("--full-tree", action="store_true",
                    help="Train full tree including preflop (slower)")
    ap.add_argument("--use-emd", action="store_true", default=None,
                    help="Enable EMD-based bucketing (potential-aware)")
    ap.add_argument("--no-emd", action="store_true",
                    help="Disable EMD bucketing")
    ap.add_argument("--workers", type=int, default=1,
                    help="Parallel training workers (0=auto, 1=single)")
    args = ap.parse_args()

    signal.signal(signal.SIGINT, _handle_sigint)

    # Determine EMD setting: explicit flag > config default
    use_emd = None  # use config default
    if args.use_emd:
        use_emd = True
    elif args.no_emd:
        use_emd = False

    abstraction = CardAbstraction(
        num_preflop_buckets=args.preflop_buckets or None,
        num_postflop_buckets=args.postflop_buckets or None,
        num_players=args.num_players,
        use_emd=use_emd,
    )

    continuous = args.iterations == 0
    batch = 1_000_000 if continuous else args.iterations
    mode = "full tree" if args.full_tree else "postflop-only (fixed preflop ranges)"

    print(f"Training {args.num_players}-player NLHE CFR solver")
    print(f"  Mode: {mode}")
    print(f"  EMD bucketing: {'enabled' if abstraction.use_emd else 'disabled'}")
    print(f"  Buckets: {abstraction.num_preflop_buckets} preflop / "
          f"{abstraction.num_postflop_buckets} postflop")
    if continuous:
        print(f"  Iterations: continuous (Ctrl+C to stop)")
    else:
        print(f"  Iterations: {args.iterations:,}")
    print(f"  Checkpoint every {args.checkpoint_interval:,} iterations")
    print(f"  DCFR discounting: enabled")
    print(f"  Regret pruning: after 1000 iterations")
    print()

    trainer = CFRTrainer(
        card_abstraction=abstraction,
        iterations=batch,
        num_players=args.num_players,
        checkpoint_interval=args.checkpoint_interval,
        checkpoint_path=args.output,
        postflop_only=not args.full_tree,
    )

    # Warm-start
    import os
    if not args.no_warm_start:
        warm_path = args.warm_start or args.output
        if os.path.exists(warm_path):
            print(f"  Loading existing strategy from {warm_path}...", flush=True)
            trainer.warm_start(warm_path)

    t0 = time.time()
    total_iters = 0

    while not _stop:
        if args.workers > 1 or args.workers == 0:
            trainer.train_parallel(num_workers=args.workers)
        else:
            trainer.train()
        total_iters += batch
        elapsed = time.time() - t0

        print(f"\n  Batch done — {total_iters:,} total iters in {elapsed:.0f}s  "
              f"({total_iters/elapsed:.0f} iter/s)  —  "
              f"{len(trainer.info_sets):,} info sets")

        # Save
        strategy = trainer.get_average_strategy()
        save_strategy(strategy, args.output)
        print(f"  Strategy saved → {args.output}")

        # Evaluate exploitability for heads-up
        if args.num_players == 2:
            try:
                from cfr.exploitability import compute_lbr
                exploit = compute_lbr(strategy, num_players=2, samples=5000,
                                      card_abstraction=abstraction)
                print(f"  LBR exploitability: {exploit:.1f} mbb/hand")
            except Exception as e:
                print(f"  LBR evaluation failed: {e}")

        if not continuous:
            break

        # Reset for next batch
        trainer.current_iteration = 0

    # Final save on shutdown
    if _stop:
        strategy = trainer.get_average_strategy()
        save_strategy(strategy, args.output)
        print(f"  Final strategy saved → {args.output}")
        print(f"  Total: {total_iters + trainer.current_iteration:,} iterations")


if __name__ == "__main__":
    main()
