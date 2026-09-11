from pathlib import Path

from fastapi import BackgroundTasks, FastAPI, HTTPException, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from .config import settings
from .db import Database
from .immich import Immich
from . import sync as sync_mod

app = FastAPI(title="immich-stack-selector")
db = Database(settings.db_path)
im = Immich(settings.immich_url, settings.api_key, settings.cache_dir)


class DecisionIn(BaseModel):
    chosen_asset_id: str
    trash_others: bool = False


@app.get("/api/stats")
def stats():
    return db.stats()


@app.get("/api/stacks")
def list_stacks(status: str | None = "pending", kind: str | None = None, limit: int = 50, offset: int = 0):
    return db.list_stacks(status or None, kind or None, limit, offset)


@app.get("/api/stacks/{stack_id}")
def get_stack(stack_id: str):
    st = db.get_stack(stack_id)
    if not st:
        raise HTTPException(404)
    return st


@app.post("/api/stacks/{stack_id}/decision")
def decide(stack_id: str, body: DecisionIn):
    st = db.get_stack(stack_id)
    if not st:
        raise HTTPException(404)
    ids = [a["id"] for a in st["assets"]]
    if body.chosen_asset_id not in ids:
        raise HTTPException(400, "asset not in stack")
    if body.trash_others and st["kind"] != "burst":
        raise HTTPException(400, f"refusing to trash siblings of a '{st['kind']}' stack")
    previous = st["primary_asset_id"]
    if previous != body.chosen_asset_id:
        im.set_primary(stack_id, body.chosen_asset_id)
    trashed = [i for i in ids if i != body.chosen_asset_id] if body.trash_others else []
    if trashed:
        im.trash(trashed)
    db.add_decision(stack_id, "pick_trash" if trashed else "pick", body.chosen_asset_id, previous, trashed)
    db.set_status(stack_id, "reviewed", primary=body.chosen_asset_id)
    return db.get_stack(stack_id)


@app.post("/api/stacks/{stack_id}/skip")
def skip(stack_id: str):
    if not db.get_stack(stack_id):
        raise HTTPException(404)
    db.add_decision(stack_id, "skip", None, None, [])
    db.set_status(stack_id, "skipped")
    return db.get_stack(stack_id)


@app.post("/api/stacks/{stack_id}/undo")
def undo(stack_id: str):
    last = db.last_decision(stack_id)
    if not last:
        raise HTTPException(404, "nothing to undo")
    if last["trashed_asset_ids"]:
        im.restore(last["trashed_asset_ids"])
    if last["previous_primary_id"] and last["previous_primary_id"] != last["chosen_asset_id"]:
        im.set_primary(stack_id, last["previous_primary_id"])
    db.add_decision(stack_id, "undo", None, None, [])
    db.set_status(stack_id, "pending", primary=last["previous_primary_id"])
    return db.get_stack(stack_id)


@app.get("/api/thumb/{asset_id}")
def thumb(asset_id: str, size: str = "thumbnail"):
    if size not in ("thumbnail", "preview"):
        raise HTTPException(400)
    return Response(im.thumbnail(asset_id, size), media_type="image/jpeg",
                    headers={"cache-control": "private, max-age=86400"})


@app.post("/api/sync")
def run_sync(tasks: BackgroundTasks, apply: bool = False):
    tasks.add_task(sync_mod.sync, apply, {"burst"})
    return {"started": True}


dist = Path(__file__).resolve().parents[2] / "frontend" / "dist"
if dist.exists():
    app.mount("/", StaticFiles(directory=dist, html=True), name="ui")
