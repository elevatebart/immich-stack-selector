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


def similarity_clusters(embs: np.ndarray, threshold: float, forced: list[tuple[int, int]]) -> list[tuple[list[int], float]]:
    """Connected components of pairwise cosine >= threshold. Returns (indices, min pairwise sim) for size >= 2."""
    n = len(embs)
    sim = embs @ embs.T
    adj = sim >= threshold
    for i, j in forced:
        adj[i, j] = adj[j, i] = True
    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for i in range(n):
        for j in range(i + 1, n):
            if adj[i, j]:
                parent[find(i)] = find(j)
    comps: dict[int, list[int]] = {}
    for i in range(n):
        comps.setdefault(find(i), []).append(i)
    out = []
    for idx in comps.values():
        if len(idx) < 2:
            continue
        sub = sim[np.ix_(idx, idx)]
        out.append((idx, float(sub.min())))
    return out
