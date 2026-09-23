import hashlib
import io
from pathlib import Path

import httpx
import numpy as np
from PIL import Image, ImageOps

# ArcFace from InsightFace buffalo_l, as mirrored by Immich (insightface licence: non-commercial)
BASE_URL = "https://huggingface.co/immich-app/buffalo_l/resolve/main"
MODELS = {
    "detection": "5838f7fe053675b1c7a08b633df49e7af5495cee0493c7dcf6697200b85b5b91",
    "recognition": "4c06341c33c2ca1f86781dab0e829f88ad5b64be9fba56e56bc9ebdefc619e43",
}
MODEL_SPEC = "buffalo_l/w600k_r50/aligned-yaw"
# where ArcFace expects eyes, nose tip and mouth corners in its 112x112 input
ARCFACE_DST = np.array([[38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366], [41.5493, 92.3655], [70.7299, 92.2041]])
DET_SIZE = 160


def fetch_model(models_dir: str, name: str) -> Path:
    path = Path(models_dir) / "buffalo_l" / f"{name}.onnx"
    if path.exists():
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    url, tmp, digest = f"{BASE_URL}/{name}/model.onnx", path.with_suffix(".part"), hashlib.sha256()
    with httpx.stream("GET", url, follow_redirects=True, timeout=120) as r, tmp.open("wb") as f:
        r.raise_for_status()
        for chunk in r.iter_bytes(1 << 20):
            digest.update(chunk)
            f.write(chunk)
    if digest.hexdigest() != MODELS[name]:
        tmp.unlink()
        raise RuntimeError(f"{url} checksum mismatch")
    tmp.rename(path)
    return path


def region(img: Image.Image, face: dict, scale: float, size: int) -> tuple[Image.Image, float, float, float]:
    """Square crop `scale` times the face box, resized to `size`; returns crop, origin and px per crop px."""
    w, h = img.size
    sx, sy = w / face["imageWidth"], h / face["imageHeight"]
    cx = (face["boundingBoxX1"] + face["boundingBoxX2"]) / 2 * sx
    cy = (face["boundingBoxY1"] + face["boundingBoxY2"]) / 2 * sy
    side = max((face["boundingBoxX2"] - face["boundingBoxX1"]) * sx, (face["boundingBoxY2"] - face["boundingBoxY1"]) * sy) * scale
    x0, y0 = cx - side / 2, cy - side / 2
    crop = img.crop((int(x0), int(y0), int(x0 + side), int(y0 + side))).resize((size, size), Image.BILINEAR)
    return crop, int(x0), int(y0), side / size


def yaw(pts: np.ndarray) -> float:
    """Nose offset from the eye midpoint in eye distances: 0 frontal, about 0.5 and up in profile."""
    eyes = pts[1] - pts[0]
    return float(abs(np.dot(pts[2] - (pts[0] + pts[1]) / 2, eyes)) / max(np.dot(eyes, eyes), 1e-6))


def similarity_to(src: np.ndarray, dst: np.ndarray) -> tuple:
    """Least-squares similarity mapping dst points onto src, as PIL AFFINE coefficients."""
    a = np.zeros((2 * len(dst), 4))
    a[0::2] = np.c_[dst[:, 0], -dst[:, 1], np.ones(len(dst)), np.zeros(len(dst))]
    a[1::2] = np.c_[dst[:, 1], dst[:, 0], np.zeros(len(dst)), np.ones(len(dst))]
    p, q, tx, ty = np.linalg.lstsq(a, src.reshape(-1), rcond=None)[0]
    return (p, -q, tx, q, p, ty)


class FaceEmbedder:
    """ArcFace vectors (512-d, L2-normalised) for Immich's face boxes.
    SCRFD re-finds five landmarks per box so ArcFace gets the aligned crop it was trained on."""

    spec = MODEL_SPEC

    def __init__(self, models_dir: str):
        import onnxruntime as ort

        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 2
        opts.log_severity_level = 3  # SCRFD declares 640x640 outputs, we feed smaller crops
        self.det = ort.InferenceSession(str(fetch_model(models_dir, "detection")), opts, providers=["CPUExecutionProvider"])
        self.rec = ort.InferenceSession(str(fetch_model(models_dir, "recognition")), opts, providers=["CPUExecutionProvider"])
        self.anchors = {}

    def _centers(self, stride: int) -> np.ndarray:
        if stride not in self.anchors:
            n = DET_SIZE // stride
            xy = np.stack(np.meshgrid(np.arange(n), np.arange(n)), -1).reshape(-1, 2) * stride
            self.anchors[stride] = np.repeat(xy, 2, axis=0).astype(np.float32)
        return self.anchors[stride]

    def landmarks(self, crop: Image.Image) -> tuple[np.ndarray | None, float]:
        """Five landmarks and score of the most confident face near the crop centre, in crop pixels."""
        x = ((np.asarray(crop, np.float32) - 127.5) / 128).transpose(2, 0, 1)[None]
        outs = self.det.run(None, {self.det.get_inputs()[0].name: x})
        best, best_score = None, 0.3
        for k, stride in enumerate((8, 16, 32)):
            scores, kps = outs[k][:, 0], outs[k + 6] * stride
            centers = self._centers(stride)
            pts = (np.repeat(centers, 5, axis=0).reshape(-1, 5, 2) + kps.reshape(-1, 5, 2))
            # the nose must sit in the middle half of the crop, otherwise it is a neighbour's face
            near = np.all(np.abs(pts[:, 2] - DET_SIZE / 2) < DET_SIZE / 4, axis=1)
            for i in np.flatnonzero(near & (scores > best_score)):
                best, best_score = pts[i], scores[i]
        return best, float(best_score)

    def embed(self, data: bytes, faces: list[dict]) -> list[tuple[np.ndarray, float, float] | None]:
        """(vector, detector score, yaw) per face box, None when the landmarks could not be found."""
        if not faces:
            return []
        img = ImageOps.exif_transpose(Image.open(io.BytesIO(data))).convert("RGB")
        aligned, keep, scores = [], [], []
        for n, f in enumerate(faces):
            crop, x0, y0, px = region(img, f, 2.0, DET_SIZE)
            pts, score = self.landmarks(crop)
            if pts is None:
                continue
            scores.append((score, yaw(pts)))
            coeffs = similarity_to(pts * px + (x0, y0), ARCFACE_DST)
            aligned.append(np.asarray(img.transform((112, 112), Image.AFFINE, coeffs, Image.BILINEAR), np.float32))
            keep.append(n)
        out: list[tuple[np.ndarray, float, float] | None] = [None] * len(faces)
        if aligned:
            x = ((np.stack(aligned) - 127.5) / 127.5).transpose(0, 3, 1, 2)
            e = self.rec.run(None, {self.rec.get_inputs()[0].name: x})[0]
            for n, v, (sc, yw) in zip(keep, e / np.linalg.norm(e, axis=1, keepdims=True), scores):
                out[n] = (v.astype(np.float32), sc, yw)
        return out
