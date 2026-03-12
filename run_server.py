#!/usr/bin/env python3
"""Start the poker web server.  Usage: python3 run_server.py [--port 8000]"""

import argparse
import os
import sys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=5050)
    ap.add_argument("--strategy", default="strategy.json.gz")
    args = ap.parse_args()

    if not os.path.exists(args.strategy):
        print(f"Strategy file not found: {args.strategy}")
        print("Run `python3 train.py` first to train the CFR solver.")
        sys.exit(1)

    os.environ["STRATEGY_PATH"] = os.path.abspath(args.strategy)

    import uvicorn
    uvicorn.run("server.app:app", host="0.0.0.0", port=args.port, reload=False)


if __name__ == "__main__":
    main()
