import math
from datetime import datetime
from pathlib import Path

import numpy as np


def ts(asset: dict) -> datetime:
    exif = asset.get("exifInfo") or {}
    raw = exif.get("dateTimeOriginal") or asset.get("fileCreatedAt") or asset.get("localDateTime")
    return datetime.fromisoformat(raw.replace("Z", "+00:00"))


def gps(asset: dict) -> tuple[float, float] | None:
    exif = asset.get("exifInfo") or {}
    lat, lon = exif.get("latitude"), exif.get("longitude")
    return (lat, lon) if lat is not None and lon is not None else None


def haversine_m(a: tuple[float, float], b: tuple[float, float]) -> float:
    lat1, lon1, lat2, lon2 = map(math.radians, (*a, *b))
    h = math.sin((lat2 - lat1) / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin((lon2 - lon1) / 2) ** 2
    return 2 * 6371000 * math.asin(math.sqrt(h))


def links(prev: dict, cur: dict, window_s: float, radius_m: float) -> bool:
    if (ts(cur) - ts(prev)).total_seconds() > window_s:
        return False
    a, b = gps(prev), gps(cur)
    return a is None or b is None or haversine_m(a, b) <= radius_m


def candidate_groups(assets: list[dict], window_s: float, radius_m: float) -> list[list[dict]]:
    """Chain consecutive shots (per owner) that are close in time and, when both have GPS, in space."""
    groups: list[list[dict]] = []
    by_owner: dict[str, list[dict]] = {}
    for a in assets:
        by_owner.setdefault(a.get("ownerId", ""), []).append(a)
    for owned in by_owner.values():
        cur: list[dict] = []
        for a in sorted(owned, key=ts):
            if cur and links(cur[-1], a, window_s, radius_m):
                cur.append(a)
            else:
                if len(cur) > 1:
                    groups.append(cur)
                cur = [a]
        if len(cur) > 1:
            groups.append(cur)
    return groups


def format_pairs(assets: list[dict]) -> list[tuple[int, int]]:
    """Indices of same-stem, different-extension files (RAW+JPG): always the same picture."""
    by_stem: dict[str, list[int]] = {}
    for i, a in enumerate(assets):
        by_stem.setdefault(Path(a["originalFileName"]).stem.lower(), []).append(i)
    pairs = []
    for idx in by_stem.values():
        pairs += [(idx[0], j) for j in idx[1:]]
    return pairs


def upload_order(assets: list[dict]) -> list[int]:
    """Indices sorted by Immich upload time, so frames from one device stay consecutive."""
    return sorted(range(len(assets)), key=lambda i: assets[i].get("createdAt") or "")


def _sim_matrix(embs: np.ndarray, forced: list[tuple[int, int]]) -> np.ndarray:
    sim = embs @ embs.T
    for i, j in forced:
        sim[i, j] = sim[j, i] = 1.0
    return sim


def _finish(sim: np.ndarray, clusters: list[list[int]], forced: list[tuple[int, int]]) -> list[tuple[list[int], float]]:
    # RAW+JPG pairs must end up together even when the chain walk separated them
    where = {i: n for n, cl in enumerate(clusters) for i in cl}
    for i, j in forced:
        a, b = where[i], where[j]
        if a != b:
            clusters[a] += clusters[b]
            for k in clusters[b]:
                where[k] = a
            clusters[b] = []
    return [(sorted(cl), float(sim[np.ix_(cl, cl)].min())) for cl in clusters if len(cl) >= 2]


def chain_clusters(embs: np.ndarray, threshold: float, forced: list[tuple[int, int]], order: list[int]) -> list[tuple[list[int], float]]:
    """Walk frames in `order`; a frame joins the most recent chain whose last frame it matches."""
    sim = _sim_matrix(embs, forced)
    chains: list[list[int]] = []
    for i in order:
        for cl in reversed(chains):
            if sim[i, cl[-1]] >= threshold:
                cl.append(i)
                break
        else:
            chains.append([i])
    return _finish(sim, chains, forced)


def complete_clusters(embs: np.ndarray, threshold: float, forced: list[tuple[int, int]], order: list[int]) -> list[tuple[list[int], float]]:
    """Greedy complete-linkage: a frame joins a cluster only if it matches every member."""
    sim = _sim_matrix(embs, forced)
    clusters: list[list[int]] = []
    for i in order:
        for cl in clusters:
            if all(sim[i, j] >= threshold for j in cl):
                cl.append(i)
                break
        else:
            clusters.append([i])
    return _finish(sim, clusters, forced)


MIN_DET_SCORE = 0.6
MAX_YAW = 0.25


def same_subject(a: list[dict], b: list[dict], min_size: float, threshold: float) -> bool | None:
    """Whether the main faces (large, frontal) of either frame show up anywhere in the other; None if one has none."""
    def main(fs):
        return [f for f in fs if f["size"] >= min_size and f["det"] >= MIN_DET_SCORE and f["yaw"] <= MAX_YAW]
    main_a, main_b = main(a), main(b)
    if not main_a or not main_b:
        return None
    # Immich sometimes files one person twice, so a shared id proves identity but disjoint ids do not
    ids_a, ids_b = {f["person"] for f in a if f["person"]}, {f["person"] for f in b if f["person"]}
    if {f["person"] for f in main_a} & ids_b or {f["person"] for f in main_b} & ids_a:
        return True
    va, vb = np.stack([f["vec"] for f in a]), np.stack([f["vec"] for f in b])
    ma, mb = np.stack([f["vec"] for f in main_a]), np.stack([f["vec"] for f in main_b])
    return float((ma @ vb.T).max()) >= threshold or float((mb @ va.T).max()) >= threshold


def split_by_identity(idx: list[int], faces: dict[int, list[dict]], min_size: float, threshold: float) -> list[list[int]]:
    """Split a cluster where main subjects differ; a frame joins the first part it conflicts with less than half of."""
    parts: list[list[int]] = []
    for i in idx:
        for part in parts:
            votes = [v for v in (same_subject(faces[i], faces[j], min_size, threshold) for j in part) if v is not None]
            if votes.count(False) * 2 <= len(votes):
                part.append(i)
                break
        else:
            parts.append([i])
    return [p for p in parts if len(p) >= 2]


CLUSTERERS = {"chain": chain_clusters, "complete": complete_clusters}
