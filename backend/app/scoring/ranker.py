import json
import warnings
from pathlib import Path

import numpy as np

FEATURES = ["sharpness", "face_sharpness", "face_area", "shadow_clip", "highlight_clip", "lum_dev", "aesthetic"]
DEFAULT_WEIGHTS = {
    "sharpness": 1.0, "face_sharpness": 1.5, "face_area": 0.3, "shadow_clip": -0.3,
    "highlight_clip": -0.5, "lum_dev": 0.3, "aesthetic": 1.0,
}


def stack_matrix(features: list[dict]) -> np.ndarray:
    """Within-stack z-scores. Missing values take the stack mean, so they contribute nothing."""
    m = np.array([[f.get(k) if f.get(k) is not None else np.nan for k in FEATURES] for f in features], dtype=float)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        mean = np.nanmean(m, axis=0)
    mean = np.where(np.isnan(mean), 0.0, mean)
    m = np.where(np.isnan(m), mean, m)
    std = m.std(axis=0)
    return (m - mean) / np.where(std < 1e-9, 1.0, std)


class Ranker:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.w = np.array([DEFAULT_WEIGHTS[k] for k in FEATURES])
        self.source = "default"
        if self.path.exists():
            saved = json.loads(self.path.read_text())
            if saved.get("features") == FEATURES:
                self.w = np.array(saved["weights"])
                self.source = f"trained({saved.get('pairs')} pairs)"

    def score(self, features: list[dict]) -> list[float]:
        if len(features) == 1:
            return [0.0]
        return (stack_matrix(features) @ self.w).tolist()

    def fit(self, pairs: list[tuple[dict, list[dict]]], l2: float = 1.0, epochs: int = 500, lr: float = 0.1) -> int:
        """Logistic regression on z(chosen) - z(rejected). Mirrored pairs keep the bias at zero."""
        xs = []
        for chosen, rejected in pairs:
            z = stack_matrix([chosen, *rejected])
            xs.extend(z[0] - z[i] for i in range(1, len(z)))
        if not xs:
            return 0
        x = np.array(xs)
        x = np.vstack([x, -x])
        y = np.concatenate([np.ones(len(xs)), np.zeros(len(xs))])
        w = self.w.copy()
        for _ in range(epochs):
            p = 1 / (1 + np.exp(-(x @ w)))
            grad = x.T @ (p - y) / len(y) + l2 * w / len(y)
            w -= lr * grad
        self.w = w
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps({"features": FEATURES, "weights": w.tolist(), "pairs": len(xs)}, indent=1))
        return len(xs)
