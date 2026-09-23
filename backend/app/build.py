"""Propose stacks from time, GPS and CLIP similarity: python -m app.build [--apply]"""
import argparse
import sys
from concurrent.futures import ThreadPoolExecutor

from pathlib import Path

import httpx
import numpy as np

from .config import settings
from .db import Database
from .group import CLUSTERERS, candidate_groups, format_pairs, gps, haversine_m, split_by_identity, ts, upload_order
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


def face_records(im: Immich, db: Database, fe, ids: list[str], pool: ThreadPoolExecutor) -> dict[str, list[dict]]:
    """Identity vector, detector score, relative size and Immich person per face, cached per asset."""
    cached = db.cached_faces(ids, fe.spec)
    missing = [i for i in ids if i not in cached]

    def fetch(aid: str) -> tuple[list[dict], bytes | None]:
        boxes = im.faces(aid)
        try:
            return boxes, im.thumbnail(aid, "preview") if boxes else None
        except httpx.HTTPStatusError:
            return [], None

    fresh = {}
    for aid, (boxes, data) in zip(missing, pool.map(fetch, missing)):
        if not boxes or data is None:
            continue  # Immich may not have run face detection yet, look again next run
        meta, vecs = [], []
        for f, hit in zip(boxes, fe.embed(data, boxes)):
            if hit is None:
                continue
            side = max(f["boundingBoxX2"] - f["boundingBoxX1"], f["boundingBoxY2"] - f["boundingBoxY1"])
            meta.append({"det": hit[1], "yaw": hit[2], "size": side / min(f["imageWidth"], f["imageHeight"]),
                         "person": (f.get("person") or {}).get("id")})
            vecs.append(hit[0])
        fresh[aid] = (meta, np.asarray(vecs, np.float32).tobytes())
    if fresh:
        db.put_faces(fe.spec, fresh)
        cached.update(fresh)
    out = {}
    for aid in ids:
        meta, blob = cached.get(aid, ([], b""))
        vecs = np.frombuffer(blob, dtype=np.float32).reshape(-1, 512)
        out[aid] = [{**m, "vec": v} for m, v in zip(meta, vecs)]
    return out


def dissolve_reason(db: Database, model: str, assets: list[dict]) -> str:
    """Why an existing stack fails: widest time gap, GPS spread, or lowest similarity."""
    times = sorted(ts(a) for a in assets)
    span = max((b - a).total_seconds() for a, b in zip(times, times[1:]))
    if span > settings.time_window_s:
        return f"{span / 60:.0f} min between consecutive frames" if span >= 120 else f"{span:.0f} s between consecutive frames"
    pts = [p for p in (gps(a) for a in assets) if p]
    if len(pts) > 1:
        spread = max(haversine_m(p, q) for p in pts for q in pts)
        if spread > settings.location_radius_m:
            return f"{spread:.0f} m GPS spread"
    ids = [a["id"] for a in assets]
    vecs = db.cached_embeddings(ids, model)
    if len(vecs) == len(ids):
        m = np.vstack([np.frombuffer(vecs[i], dtype=np.float32) for i in ids])
        return f"similarity {float((m @ m.T).min()):.3f} below threshold"
    return "frames not similar enough"


def propose(taken_after: str | None, taken_before: str | None) -> dict:
    im = Immich(settings.immich_url, settings.api_key, settings.cache_dir)
    db = Database(settings.db_path)
    embedder = Embedder(settings.embed_model, cache_dir=str(Path(settings.cache_dir) / 'models'))
    scorer = Scorer()
    clusterer = CLUSTERERS[settings.cluster_mode]
    faces = None
    if settings.face_split:
        from .scoring.faces import FaceEmbedder

        faces = FaceEmbedder(str(Path(settings.cache_dir) / "models"))
    print("listing assets...", file=sys.stderr)
    assets = list(im.iter_assets(taken_after, taken_before))
    groups = candidate_groups(assets, settings.time_window_s, settings.location_radius_m)
    print(f"{len(assets)} assets, {len(groups)} candidate groups ({sum(map(len, groups))} assets)", file=sys.stderr)

    proposals, sims, face_splits = [], [], 0
    with ThreadPoolExecutor(settings.workers) as pool:
        for gi, group in enumerate(groups, 1):
            ids = [a["id"] for a in group]
            embs = embed_group(im, db, embedder, ids, pool)
            if embs is None:
                continue
            for idx, min_sim in clusterer(embs, settings.sim_threshold, format_pairs(group), upload_order(group)):
                parts = [idx]
                if faces is not None:
                    recs = face_records(im, db, faces, [ids[i] for i in idx], pool)
                    parts = split_by_identity(idx, {i: recs[ids[i]] for i in idx}, settings.face_min_size, settings.face_threshold)
                    face_splits += parts != [idx]
                for part in parts:
                    sim = min_sim if part == idx else float((embs[part] @ embs[part].T).min())
                    cluster = [ids[i] for i in part]
                    _, scores = scorer.score_assets(im, db, cluster, pool)
                    primary = cluster[max(range(len(cluster)), key=scores.__getitem__)]
                    proposals.append({"asset_ids": cluster, "primary_asset_id": primary, "min_sim": sim})
                    sims.append(sim)
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
                     "primary_asset_id": st["primaryAssetId"], "existing_stack_id": st["id"],
                     "reason": dissolve_reason(db, embedder.spec, st["assets"])})
    db.reset_proposals()
    db.add_proposals(rows)
    counts = {k: sum(r["action"] == k for r in rows) for k in ("keep", "create", "dissolve")}
    if sims:
        q = np.percentile(sims, [10, 50, 90])
        print(f"min-sim percentiles p10 {q[0]:.3f} p50 {q[1]:.3f} p90 {q[2]:.3f}", file=sys.stderr)
    if faces is not None:
        print(f"{face_splits} clusters split by face identity", file=sys.stderr)
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
    return apply_ids(Database(settings.db_path).proposal_ids())


def apply_ids(ids: list[int]) -> int:
    im = Immich(settings.immich_url, settings.api_key, settings.cache_dir)
    db = Database(settings.db_path)
    scorer = Scorer()
    failed = 0
    with ThreadPoolExecutor(settings.workers) as pool:
        for n, pid in enumerate(ids, 1):
            try:
                apply_proposal(im, db, scorer, pid, pool)
            except Exception as e:  # one bad stack must not stop the batch
                failed += 1
                db.set_proposal(pid, "failed")
                print(f"proposal {pid} failed: {e}", file=sys.stderr)
            if n % 100 == 0:
                print(f"{n}/{len(ids)} applied, {failed} failed", file=sys.stderr)
    return len(ids) - failed


def unstacked_by_user(db: Database, live: set[str]) -> list[set[str]]:
    """Asset sets of stacks we created that are gone from Immich without this app deleting them."""
    ours, ours_deleted = db.created_stacks(), db.deleted_by_app()
    return [ids for sid, ids in ours.items() if sid not in live and sid not in ours_deleted]


def periodic(window_days: int) -> int:
    """Scan recent photos and stack the ones nobody has stacked yet. Replace and dissolve stay manual."""
    from datetime import datetime, timedelta, timezone

    since = (datetime.now(timezone.utc) - timedelta(days=window_days)).strftime("%Y-%m-%dT%H:%M:%S.000Z")
    propose(since, None)
    db = Database(settings.db_path)
    live = {st["id"] for st in Immich(settings.immich_url, settings.api_key, settings.cache_dir).stacks()}
    rejected = unstacked_by_user(db, live)
    # re-pairing two frames the user split apart would undo their unstack
    ids = [pid for pid in db.proposal_ids("proposed", "create", only_new=True)
           if not any(len(rej & set(db.get_proposal(pid)["asset_ids"])) >= 2 for rej in rejected)]
    return apply_ids(ids)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--apply", action="store_true", help="apply every proposal still in 'proposed' state")
    p.add_argument("--taken-after", help="ISO date, limit the scan (existing stacks outside are left alone)")
    p.add_argument("--taken-before", help="ISO date")
    p.add_argument("--periodic", type=int, metavar="DAYS",
                   help="scan the last DAYS days and create stacks for unstacked photos only (cron friendly)")
    args = p.parse_args()
    if args.periodic:
        print(f"created {periodic(args.periodic)} stacks")
        raise SystemExit
    propose(args.taken_after, args.taken_before)
    if args.apply:
        print(f"applied {apply_all()} proposals")
