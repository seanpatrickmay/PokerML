"""Serialize / deserialize trained CFR strategies."""

import gzip
import json


def save_strategy(strategy: dict, path: str):
    with gzip.open(path, 'wt', encoding='utf-8') as f:
        json.dump(strategy, f)


def load_strategy(path: str) -> dict:
    with gzip.open(path, 'rt', encoding='utf-8') as f:
        return json.load(f)
