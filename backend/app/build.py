"""Propose stacks from time, GPS and CLIP similarity: python -m app.build [--apply]"""
import argparse
import sys
from concurrent.futures import ThreadPoolExecutor

import httpx
import numpy as np

from .config import settings
from .db import Database
from .group import candidate_groups, format_pairs, similarity_clusters
from .immich import Immich
from .scoring.embed import Embedder
from .sync import Scorer, ingest_stack


def embed_group(im: Immich, db: Database, embedder: Embedder, ids: list[str], pool: ThreadPoolExecutor) -> np.ndarray | None:
    cached = db.cached_embeddings(ids, embedder.spec)
    missing = [i for i in ids if i not in cached]
    if missing:
        def fetch(aid: str) -> bytes | None:
            try:
                return im.thumbnail(aid, "preview")
            except httpx.HTTPStatusError:
                return None
        blobs = list(pool.map(fetch, missing))
        ok = [(aid, b) for aid, b in zip(missing, blobs) if b is not None]
        if ok:
            vecs = embedder.embed([b for _, b in ok])
            fresh = {aid: v.astype(np.float32).tobytes() for (aid, _), v in zip(ok, vecs)}
            db.put_embeddings(embedder.spec, fresh)
            cached.update(fresh)
    if len(cached) < 2:
        return None
    return np.vstack([np.frombuffer(cached[i], dtype=np.float32) if i in cached else np.zeros(512, np.float32) for i in ids])


def propose(taken_after: str | None, taken_before: str | None) -> dict:
    im = Immich(settings.immich_url, settings.api_key, settings.cache_dir)
    db = Database(settings.db_path)
    embedder = Embedder(settings.embed_model)
    scorer = Scorer()
    print("listing assets...", file=sys.stderr)
    assets = list(im.iter_assets(taken_after, taken_before))
    groups = candidate_groups(assets, settings.time_window_s, settings.location_radius_m)
    print(f"{len(assets)} assets, {len(groups)} candidate groups ({sum(map(len, groups))} assets)", file=sys.stderr)

    proposals, sims = [], []
    with ThreadPoolExecutor(settings.workers) as pool:
        for gi, group in enumerate(groups, 1):
            ids = [a["id"] for a in group]
            embs = embed_group(im, db, embedder, ids, pool)
            if embs is None:
                continue
            for idx, min_sim in similarity_clusters(embs, settings.sim_threshold, format_pairs(group)):
                cluster = [ids[i] for i in idx]
                _, scores = scorer.score_assets(im, db, cluster, pool)
                primary = cluster[max(range(len(cluster)), key=scores.__getitem__)]
                proposals.append({"asset_ids": cluster, "primary_asset_id": primary, "min_sim": min_sim})
                sims.append(min_sim)
            if gi % 200 == 0:
                print(f"{gi}/{len(groups)} groups, {len(proposals)} proposed stacks", file=sys.stderr)

    # reconcile with what Immich has today
    existing = im.stacks()
    by_set = {frozenset(a["id"] for a in st["assets"]): st for st in existing}
    by_asset = {a["id"]: st["id"] for st in existing for a in st["assets"]}
    matched, rows = set(), []
    for pr in proposals:
        key = frozenset(pr["asset_ids"])
        if key in by_set:
            st = by_set[key]
            matched.add(st["id"])
            rows.append({**pr, "action": "keep", "existing_stack_id": st["id"]})
        else:
            overlaps = sorted({by_asset[i] for i in pr["asset_ids"] if i in by_asset})
            matched.update(overlaps)
            rows.append({**pr, "action": "create", "replaces": overlaps})
    in_scope = {a["id"] for a in assets}
    for st in existing:
        if st["id"] in matched or not any(a["id"] in in_scope for a in st["assets"]):
            continue
        rows.append({"action": "dissolve", "asset_ids": [a["id"] for a in st["assets"]],
                     "primary_asset_id": st["primaryAssetId"], "existing_stack_id": st["id"]})
    db.reset_proposals()
    db.add_proposals(rows)
    counts = {k: sum(r["action"] == k for r in rows) for k in ("keep", "create", "dissolve")}
    if sims:
        q = np.percentile(sims, [10, 50, 90])
        print(f"min-sim percentiles p10 {q[0]:.3f} p50 {q[1]:.3f} p90 {q[2]:.3f}", file=sys.stderr)
    print(f"proposals: {counts}")
    return counts


def apply_proposal(im: Immich, db: Database, scorer: Scorer, pid: int, pool: ThreadPoolExecutor) -> dict:
    pr = db.get_proposal(pid)
    if not pr or pr["status"] != "proposed":
        return pr or {}
    if pr["action"] == "keep":
        db.set_proposal(pid, "applied")
        return db.get_proposal(pid)
    for sid in pr["replaces"] + ([pr["existing_stack_id"]] if pr["action"] == "dissolve" else []):
        im.delete_stack(sid)
        db.mark_stack_gone(sid)
    created = None
    if pr["action"] == "create":
        ordered = [pr["primary_asset_id"]] + [i for i in pr["asset_ids"] if i != pr["primary_asset_id"]]
        created = im.create_stack(ordered)
        ingest_stack(im, db, scorer, im.stack(created["id"]), pool)
    db.set_proposal(pid, "applied", created["id"] if created else None)
    return db.get_proposal(pid)


def apply_all() -> int:
    im = Immich(settings.immich_url, settings.api_key, settings.cache_dir)
    db = Database(settings.db_path)
    scorer = Scorer()
    ids = db.proposal_ids()
    with ThreadPoolExecutor(settings.workers) as pool:
        for n, pid in enumerate(ids, 1):
            apply_proposal(im, db, scorer, pid, pool)
            if n % 100 == 0:
                print(f"{n}/{len(ids)} applied", file=sys.stderr)
    return len(ids)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--apply", action="store_true", help="apply every proposal still in 'proposed' state")
    p.add_argument("--taken-after", help="ISO date, limit the scan (existing stacks outside are left alone)")
    p.add_argument("--taken-before", help="ISO date")
    args = p.parse_args()
    propose(args.taken_after, args.taken_before)
    if args.apply:
        print(f"applied {apply_all()} proposals")
