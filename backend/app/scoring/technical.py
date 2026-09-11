import io
import math

import numpy as np
from PIL import Image, ImageOps

FEATURE_VERSION = 1


def load_gray(data: bytes, max_side: int = 1024) -> np.ndarray:
    img = ImageOps.exif_transpose(Image.open(io.BytesIO(data))).convert("L")
    img.thumbnail((max_side, max_side))
    return np.asarray(img, dtype=np.float32)


def laplacian_var(g: np.ndarray) -> float:
    if g.shape[0] < 3 or g.shape[1] < 3:
        return 0.0
    lap = -4 * g[1:-1, 1:-1] + g[:-2, 1:-1] + g[2:, 1:-1] + g[1:-1, :-2] + g[1:-1, 2:]
    return float(lap.var())


def face_crops(g: np.ndarray, faces: list[dict]):
    h, w = g.shape
    for f in faces:
        sx, sy = w / f["imageWidth"], h / f["imageHeight"]
        x1, y1 = max(int(f["boundingBoxX1"] * sx), 0), max(int(f["boundingBoxY1"] * sy), 0)
        x2, y2 = int(f["boundingBoxX2"] * sx), int(f["boundingBoxY2"] * sy)
        crop = g[y1:y2, x1:x2]
        if crop.size:
            yield crop


def technical_features(data: bytes, faces: list[dict]) -> dict:
    g = load_gray(data)
    crops = list(face_crops(g, faces))
    face_sharp = float(np.mean([laplacian_var(c) for c in crops])) if crops else None
    return {
        "v": FEATURE_VERSION,
        "sharpness": math.log1p(laplacian_var(g)),
        "face_sharpness": math.log1p(face_sharp) if face_sharp is not None else None,
        "faces": len(crops),
        "face_area": float(sum(c.size for c in crops) / g.size),
        "shadow_clip": float((g < 8).mean()),
        "highlight_clip": float((g > 247).mean()),
        "lum_dev": -abs(float(g.mean()) / 255 - 0.45),
    }
