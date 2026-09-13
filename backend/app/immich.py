import time
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

    def iter_assets(self, taken_after: str | None = None, taken_before: str | None = None):
        """All of the key owner's images, oldest first, with EXIF (GPS, capture time)."""
        body = {"type": "IMAGE", "withExif": True, "size": 1000, "order": "asc", "page": 1}
        # Immich validates these as full datetimes, so pad bare dates
        if taken_after:
            body["takenAfter"] = taken_after if "T" in taken_after else f"{taken_after}T00:00:00.000Z"
        if taken_before:
            body["takenBefore"] = taken_before if "T" in taken_before else f"{taken_before}T00:00:00.000Z"
        while body["page"]:
            r = self._post_retry("/search/metadata", body)
            page = r.json()["assets"]
            yield from page["items"]
            nxt = page.get("nextPage")
            body["page"] = int(nxt) if nxt else None

    def _post_retry(self, path: str, body: dict, tries: int = 4) -> httpx.Response:
        for n in range(tries):
            try:
                r = self.c.post(path, json=body)
                if r.status_code < 500:
                    if r.is_error:
                        raise RuntimeError(f"{path} {r.status_code}: {r.text[:300]} body={body}")
                    return r
            except httpx.TransportError:
                if n == tries - 1:
                    raise
            time.sleep(2 ** n)
        raise RuntimeError(f"{path} kept failing: {r.status_code} {r.text[:300]}")

    def create_stack(self, asset_ids: list[str]) -> dict:
        r = self.c.post("/stacks", json={"assetIds": asset_ids})
        r.raise_for_status()
        return r.json()

    def delete_stack(self, stack_id: str) -> None:
        r = self.c.delete(f"/stacks/{stack_id}")
        if r.status_code != 404:
            r.raise_for_status()

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
