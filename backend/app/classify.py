from datetime import datetime
from pathlib import Path

RAW_EXTS = {".arw", ".cr2", ".cr3", ".nef", ".nrw", ".dng", ".raf", ".orf", ".rw2", ".pef", ".srw", ".3fr"}


def _ts(asset: dict) -> datetime:
    raw = asset.get("fileCreatedAt") or asset.get("localDateTime")
    return datetime.fromisoformat(raw.replace("Z", "+00:00"))


def classify(assets: list[dict], window_s: float) -> str:
    """burst: same extension, tight timing. format: mixed extensions (RAW+JPG). loose: rest."""
    exts = {Path(a["originalFileName"]).suffix.lower() for a in assets}
    if len(exts) > 1 or exts & RAW_EXTS:
        return "format"
    times = sorted(_ts(a) for a in assets)
    gap = max((b - a).total_seconds() for a, b in zip(times, times[1:]))
    # a chain may skip a dissimilar middle frame, so allow two windows between kept frames
    return "burst" if gap <= 2 * window_s else "loose"
