from pathlib import Path

import httpx
from fastapi import BackgroundTasks, FastAPI, HTTPException, Request, Response
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from .config import settings
from .db import Database
from .immich import Immich
from . import build as build_mod
from . import sync as sync_mod
from concurrent.futures import ThreadPoolExecutor

app = FastAPI(title="immich-stack-selector")
db = Database(settings.db_path)
im = Immich(settings.immich_url, settings.api_key, settings.cache_dir)


@app.exception_handler(httpx.HTTPStatusError)
def immich_error(_: Request, e: httpx.HTTPStatusError):
    # surface Immich's own message (usually a missing API key permission) instead of a 500
    try:
        detail = e.response.json().get("message", e.response.text)
    except ValueError:
        detail = e.response.text
    return JSONResponse({"detail": f"immich {e.response.status_code}: {detail}"}, status_code=502)


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


# -- proposals (stack builder output) ---------------------------------------

_scorer: sync_mod.Scorer | None = None


def scorer() -> sync_mod.Scorer:
    global _scorer
    if _scorer is None:
        _scorer = sync_mod.Scorer()
    return _scorer


def hydrate(pr: dict) -> dict:
    """Attach file names and scores for the proposal's assets, from the local cache where possible."""
    feats = db.cached_features(pr["asset_ids"])
    names = db.asset_names(pr["asset_ids"])
    pr["assets"] = [{"id": a, "file_name": names.get(a), "features": feats.get(a, {})} for a in pr["asset_ids"]]
    return pr


@app.get("/api/proposals")
def list_proposals(status: str | None = "proposed", action: str | None = None, limit: int = 50, offset: int = 0):
    return [hydrate(p) for p in db.list_proposals(status or None, action or None, limit, offset)]


@app.get("/api/proposals/stats")
def proposal_stats():
    return db.proposal_stats()


@app.post("/api/proposals/{pid}/apply")
def apply_proposal(pid: int):
    if not db.get_proposal(pid):
        raise HTTPException(404)
    with ThreadPoolExecutor(settings.workers) as pool:
        return hydrate(build_mod.apply_proposal(im, db, scorer(), pid, pool))


@app.post("/api/proposals/{pid}/reject")
def reject_proposal(pid: int):
    pr = db.get_proposal(pid)
    if not pr:
        raise HTTPException(404)
    if pr["status"] != "proposed":
        raise HTTPException(400, f"proposal is {pr['status']}")
    db.set_proposal(pid, "rejected")
    return hydrate(db.get_proposal(pid))


@app.post("/api/proposals/apply-all")
def apply_all_proposals(tasks: BackgroundTasks, action: str | None = None):
    ids = db.proposal_ids("proposed", action or None)
    tasks.add_task(build_mod.apply_ids, ids)
    return {"started": True, "count": len(ids)}


@app.post("/api/build")
def run_build(tasks: BackgroundTasks, taken_after: str | None = None, taken_before: str | None = None):
    tasks.add_task(build_mod.propose, taken_after or None, taken_before or None)
    return {"started": True}


def _every(minutes: int, name: str, fn) -> None:
    import threading
    import time

    def loop():
        while True:
            time.sleep(minutes * 60)
            try:
                print(f"periodic {name}: {fn()}")
            except Exception as e:  # keep the server alive, log and retry next tick
                print(f"periodic {name} failed: {e}")

    threading.Thread(target=loop, daemon=True, name=name).start()


@app.on_event("startup")
def schedule():
    """Long-running container mode: re-score stacks and stack new photos on timers."""
    if settings.sync_every_min > 0:
        _every(settings.sync_every_min, "sync", lambda: sync_mod.sync(False, set()))
    if settings.build_every_min > 0:
        _every(settings.build_every_min, "build", lambda: build_mod.periodic(settings.build_window_days))


dist = Path(__file__).resolve().parents[2] / "frontend" / "dist"
if dist.exists():
    app.mount("/", StaticFiles(directory=dist, html=True), name="ui")
