"""Retrain the pairwise ranker from human review decisions: python -m app.train"""
from pathlib import Path

from .config import settings
from .db import Database
from .scoring.ranker import FEATURES, Ranker


def main() -> None:
    db = Database(settings.db_path)
    pairs = db.training_pairs()
    ranker = Ranker(Path(settings.db_path).parent / "ranker.json")
    n = ranker.fit(pairs)
    if n < 30:
        print(f"only {n} pairs, weights will be noisy (aim for 100+)")
    for k, w in zip(FEATURES, ranker.w):
        print(f"{k:16s} {w:+.3f}")
    print(f"saved {ranker.path} from {n} pairs; re-run `python -m app.sync` to rescore")


if __name__ == "__main__":
    main()
