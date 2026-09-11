"""Pull stacks from Immich, score every frame, store suggestions: python -m app.sync [--apply]"""
import argparse
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from .classify import classify
from .config import settings
from .db import Database
from .immich import Immich
from .scoring.ranker import Ranker
from .scoring.technical import FEATURE_VERSION, technical_features


class Scorer:
    def __init__(self):
        self.ranker = Ranker(Path(settings.db_path).parent / "ranker.json")
        self.aesthetic = None
        if settings.use_aesthetic:
            from .scoring.aesthetic import AestheticScorer

            self.aesthetic = AestheticScorer(settings.cache_dir)

    def features(self, im: Immich, asset: dict, cached: dict | None) -> dict:
        wants_aesthetic = self.aesthetic is not None
        if cached and cached.get("v") == FEATURE_VERSION and (not wants_aesthetic or "aesthetic" in cached):
            return cached
        data = im.thumbnail(asset["id"], "preview")
        feats = technical_features(data, im.faces(asset["id"]))
        if wants_aesthetic:
            feats["aesthetic"] = self.aesthetic.score(data)
        return feats


def sync(apply: bool, apply_kinds: set[str]) -> None:
    im = Immich(settings.immich_url, settings.api_key, settings.cache_dir)
    db = Database(settings.db_path)
    scorer = Scorer()
    print(f"ranker weights: {scorer.ranker.source}")
    stacks = im.stacks()
    seen, applied = [], 0
    with ThreadPoolExecutor(settings.workers) as pool:
        for i, st in enumerate(stacks, 1):
            assets = [a for a in st["assets"] if not a.get("isTrashed")]
            if len(assets) < 2:
                continue
            seen.append(st["id"])
            kind = classify(assets, settings.burst_window_s)
            cached = db.cached_features([a["id"] for a in assets])
            feats = list(pool.map(lambda a: scorer.features(im, a, cached.get(a["id"])), assets))
            scores = scorer.ranker.score(feats)
            best = assets[max(range(len(assets)), key=scores.__getitem__)]["id"]
            existing = db.get_stack(st["id"])
            db.upsert_stack(
                st["id"], kind, min(a["fileCreatedAt"] for a in assets), st["primaryAssetId"], best
            )
            db.replace_assets(
                st["id"],
                [
                    {"id": a["id"], "file_name": a["originalFileName"], "created_at": a["fileCreatedAt"],
                     "features": f, "score": s}
                    for a, f, s in zip(assets, feats, scores)
                ],
            )
            pending = existing is None or existing["status"] == "pending"
            if apply and pending and kind in apply_kinds and st["primaryAssetId"] != best:
                im.set_primary(st["id"], best)
                db.add_decision(st["id"], "auto", best, st["primaryAssetId"], [])
                db.set_status(st["id"], "pending", primary=best)
                applied += 1
            print(f"\r{i}/{len(stacks)} stacks", end="", file=sys.stderr)
    gone = db.mark_gone(seen)
    print(f"\nsynced {len(seen)} stacks, applied {applied} primaries, {gone} stacks gone")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--apply", action="store_true", help="write suggested primary to Immich for unreviewed stacks")
    p.add_argument("--kinds", default="burst", help="comma list of stack kinds to apply to (burst,format,other)")
    args = p.parse_args()
    sync(args.apply, set(args.kinds.split(",")))
