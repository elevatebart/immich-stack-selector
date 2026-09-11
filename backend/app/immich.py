from pathlib import Path

import httpx


class Immich:
    """Thin client over the Immich REST API. httpx.Client is thread-safe."""

    def __init__(self, base_url: str, api_key: str, cache_dir: str):
        self.c = httpx.Client(
            base_url=f"{base_url}/api",
            headers={"x-api-key": api_key, "accept": "application/json"},
            timeout=60,
        )
        self.cache = Path(cache_dir)

    def stacks(self) -> list[dict]:
        r = self.c.get("/stacks")
        r.raise_for_status()
        return r.json()

    def stack(self, stack_id: str) -> dict:
        r = self.c.get(f"/stacks/{stack_id}")
        r.raise_for_status()
        return r.json()

    def set_primary(self, stack_id: str, asset_id: str) -> None:
        self.c.put(f"/stacks/{stack_id}", json={"primaryAssetId": asset_id}).raise_for_status()

    def faces(self, asset_id: str) -> list[dict]:
        r = self.c.get("/faces", params={"id": asset_id})
        return r.json() if r.is_success else []

    def thumbnail(self, asset_id: str, size: str = "preview") -> bytes:
        path = self.cache / size / f"{asset_id}.jpg"
        if path.exists():
            return path.read_bytes()
        r = self.c.get(f"/assets/{asset_id}/thumbnail", params={"size": size})
        r.raise_for_status()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(r.content)
        return r.content

    def trash(self, asset_ids: list[str]) -> None:
        # force=False keeps the assets in Immich's trash (30 day default retention)
        self.c.request("DELETE", "/assets", json={"ids": asset_ids, "force": False}).raise_for_status()

    def restore(self, asset_ids: list[str]) -> None:
        self.c.post("/trash/restore/assets", json={"ids": asset_ids}).raise_for_status()
