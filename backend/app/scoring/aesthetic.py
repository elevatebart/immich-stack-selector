import io
import urllib.request
from pathlib import Path

# Weights from github.com/christophschuhmann/improved-aesthetic-predictor (CLIP ViT-L/14 head)
HEAD_URL = "https://github.com/christophschuhmann/improved-aesthetic-predictor/raw/main/sac%2Blogos%2Bava1-l14-linearMSE.pth"


class AestheticScorer:
    def __init__(self, cache_dir: str):
        import open_clip
        import torch
        from torch import nn

        self.torch = torch
        self.device = "mps" if torch.backends.mps.is_available() else "cpu"
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            "ViT-L-14", pretrained="openai", device=self.device
        )
        self.model.eval()
        head_path = Path(cache_dir) / "aesthetic_head.pth"
        if not head_path.exists():
            head_path.parent.mkdir(parents=True, exist_ok=True)
            urllib.request.urlretrieve(HEAD_URL, head_path)
        self.head = nn.Sequential(
            nn.Linear(768, 1024), nn.Dropout(0.2), nn.Linear(1024, 128), nn.Dropout(0.2),
            nn.Linear(128, 64), nn.Dropout(0.1), nn.Linear(64, 16), nn.Linear(16, 1),
        )
        state = torch.load(head_path, map_location="cpu")
        self.head.load_state_dict({k.removeprefix("layers."): v for k, v in state.items()})
        self.head.eval().to(self.device)

    def score(self, data: bytes) -> float:
        from PIL import Image

        img = self.preprocess(Image.open(io.BytesIO(data)).convert("RGB")).unsqueeze(0).to(self.device)
        with self.torch.no_grad():
            emb = self.model.encode_image(img)
            emb = emb / emb.norm(dim=-1, keepdim=True)
            return float(self.head(emb.float()).item())
