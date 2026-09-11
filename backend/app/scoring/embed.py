import io

import numpy as np


class Embedder:
    """open_clip image embeddings, L2-normalised. Model spec is "arch/pretrained"."""

    def __init__(self, spec: str, batch_size: int = 32):
        import open_clip
        import torch

        arch, pretrained = spec.split("/", 1)
        self.torch = torch
        self.device = "mps" if torch.backends.mps.is_available() else "cpu"
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(arch, pretrained=pretrained, device=self.device)
        self.model.eval()
        self.batch_size = batch_size
        self.spec = spec

    def embed(self, images: list[bytes]) -> np.ndarray:
        from PIL import Image, ImageOps

        out = []
        for i in range(0, len(images), self.batch_size):
            batch = [
                self.preprocess(ImageOps.exif_transpose(Image.open(io.BytesIO(b))).convert("RGB"))
                for b in images[i : i + self.batch_size]
            ]
            with self.torch.no_grad():
                x = self.torch.stack(batch).to(self.device)
                e = self.model.encode_image(x).float()
                e = e / e.norm(dim=-1, keepdim=True)
            out.append(e.cpu().numpy())
        return np.vstack(out) if out else np.zeros((0, 512), dtype=np.float32)
