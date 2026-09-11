import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

# .env and relative data paths resolve against the repo root, one level above backend/
ROOT = Path(__file__).resolve().parents[2]
load_dotenv(ROOT / ".env")


def _path(env: str, default: str) -> str:
    return str(ROOT / os.environ.get(env, default))


@dataclass(frozen=True)
class Settings:
    immich_url: str = os.environ.get("IMMICH_URL", "http://localhost:2283").rstrip("/")
    api_key: str = os.environ.get("IMMICH_API_KEY", "")
    db_path: str = _path("DB_PATH", "data/selector.sqlite")
    cache_dir: str = _path("CACHE_DIR", "data/thumbs")
    burst_window_s: float = float(os.environ.get("BURST_WINDOW_S", "5"))
    use_aesthetic: bool = os.environ.get("USE_AESTHETIC", "0") == "1"
    workers: int = int(os.environ.get("WORKERS", "8"))


settings = Settings()
