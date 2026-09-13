import json
import sqlite3
import threading
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS stacks (
  id TEXT PRIMARY KEY,
  kind TEXT NOT NULL,
  captured_at TEXT,
  primary_asset_id TEXT,
  suggested_asset_id TEXT,
  status TEXT NOT NULL DEFAULT 'pending',
  synced_at TEXT NOT NULL,
  reviewed_at TEXT
);
CREATE TABLE IF NOT EXISTS assets (
  id TEXT PRIMARY KEY,
  stack_id TEXT NOT NULL,
  file_name TEXT NOT NULL,
  created_at TEXT,
  features TEXT NOT NULL DEFAULT '{}',
  score REAL
);
CREATE INDEX IF NOT EXISTS assets_stack ON assets(stack_id);
CREATE TABLE IF NOT EXISTS feature_cache (
  asset_id TEXT PRIMARY KEY,
  features TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS embeddings (
  asset_id TEXT PRIMARY KEY,
  model TEXT NOT NULL,
  vec BLOB NOT NULL
);
CREATE TABLE IF NOT EXISTS proposals (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  action TEXT NOT NULL,
  asset_ids TEXT NOT NULL,
  primary_asset_id TEXT,
  min_sim REAL,
  existing_stack_id TEXT,
  replaces TEXT NOT NULL DEFAULT '[]',
  status TEXT NOT NULL DEFAULT 'proposed',
  created_stack_id TEXT,
  created_at TEXT NOT NULL,
  applied_at TEXT
);
CREATE TABLE IF NOT EXISTS decisions (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  stack_id TEXT NOT NULL,
  action TEXT NOT NULL,
  chosen_asset_id TEXT,
  previous_primary_id TEXT,
  trashed_asset_ids TEXT NOT NULL DEFAULT '[]',
  created_at TEXT NOT NULL
);
"""


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class Database:
    def __init__(self, path: str):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self._lock = threading.Lock()
        with self.conn() as c:
            c.executescript(SCHEMA)

    @contextmanager
    def conn(self):
        c = sqlite3.connect(self.path)
        c.row_factory = sqlite3.Row
        try:
            with self._lock:
                yield c
                c.commit()
        finally:
            c.close()

    # -- sync -------------------------------------------------------------

    def upsert_stack(self, id: str, kind: str, captured_at: str | None, primary: str, suggested: str) -> None:
        with self.conn() as c:
            c.execute(
                """INSERT INTO stacks(id, kind, captured_at, primary_asset_id, suggested_asset_id, synced_at)
                   VALUES(?,?,?,?,?,?)
                   ON CONFLICT(id) DO UPDATE SET kind=excluded.kind, captured_at=excluded.captured_at,
                     primary_asset_id=excluded.primary_asset_id, suggested_asset_id=excluded.suggested_asset_id,
                     synced_at=excluded.synced_at""",
                (id, kind, captured_at, primary, suggested, now()),
            )

    def replace_assets(self, stack_id: str, rows: list[dict]) -> None:
        with self.conn() as c:
            c.execute("DELETE FROM assets WHERE stack_id=?", (stack_id,))
            c.executemany(
                "INSERT OR REPLACE INTO assets(id, stack_id, file_name, created_at, features, score) VALUES(?,?,?,?,?,?)",
                [(r["id"], stack_id, r["file_name"], r["created_at"], json.dumps(r["features"]), r["score"]) for r in rows],
            )

    def cached_features(self, asset_ids: list[str]) -> dict[str, dict]:
        if not asset_ids:
            return {}
        with self.conn() as c:
            q = f"SELECT asset_id, features FROM feature_cache WHERE asset_id IN ({','.join('?' * len(asset_ids))})"
            return {r["asset_id"]: json.loads(r["features"]) for r in c.execute(q, asset_ids)}

    def put_features(self, feats: dict[str, dict]) -> None:
        with self.conn() as c:
            c.executemany(
                "INSERT OR REPLACE INTO feature_cache(asset_id, features) VALUES(?,?)",
                [(k, json.dumps(v)) for k, v in feats.items()],
            )

    def cached_embeddings(self, asset_ids: list[str], model: str) -> dict[str, bytes]:
        if not asset_ids:
            return {}
        with self.conn() as c:
            q = f"SELECT asset_id, vec FROM embeddings WHERE model=? AND asset_id IN ({','.join('?' * len(asset_ids))})"
            return {r["asset_id"]: r["vec"] for r in c.execute(q, [model, *asset_ids])}

    def put_embeddings(self, model: str, vecs: dict[str, bytes]) -> None:
        with self.conn() as c:
            c.executemany(
                "INSERT OR REPLACE INTO embeddings(asset_id, model, vec) VALUES(?,?,?)",
                [(k, model, v) for k, v in vecs.items()],
            )

    # -- proposals --------------------------------------------------------

    def reset_proposals(self) -> None:
        with self.conn() as c:
            c.execute("DELETE FROM proposals WHERE status='proposed'")

    def add_proposals(self, rows: list[dict]) -> None:
        with self.conn() as c:
            c.executemany(
                """INSERT INTO proposals(action, asset_ids, primary_asset_id, min_sim, existing_stack_id, replaces, created_at)
                   VALUES(?,?,?,?,?,?,?)""",
                [(r["action"], json.dumps(r["asset_ids"]), r.get("primary_asset_id"), r.get("min_sim"),
                  r.get("existing_stack_id"), json.dumps(r.get("replaces", [])), now()) for r in rows],
            )

    def _proposal(self, row) -> dict:
        return {**dict(row), "asset_ids": json.loads(row["asset_ids"]), "replaces": json.loads(row["replaces"])}

    def list_proposals(self, status: str | None, action: str | None, limit: int, offset: int) -> list[dict]:
        q, params = "SELECT * FROM proposals WHERE 1=1", []
        if status:
            q += " AND status=?"
            params.append(status)
        if action:
            q += " AND action=?"
            params.append(action)
        q += " ORDER BY min_sim ASC, id LIMIT ? OFFSET ?"
        params += [limit, offset]
        with self.conn() as c:
            return [self._proposal(r) for r in c.execute(q, params).fetchall()]

    def get_proposal(self, pid: int) -> dict | None:
        with self.conn() as c:
            row = c.execute("SELECT * FROM proposals WHERE id=?", (pid,)).fetchone()
            return self._proposal(row) if row else None

    def proposal_ids(self, status: str = "proposed", action: str | None = None) -> list[int]:
        q, params = "SELECT id FROM proposals WHERE status=?", [status]
        if action:
            q += " AND action=?"
            params.append(action)
        with self.conn() as c:
            return [r[0] for r in c.execute(q + " ORDER BY id", params)]

    def asset_names(self, asset_ids: list[str]) -> dict[str, str]:
        if not asset_ids:
            return {}
        with self.conn() as c:
            q = f"SELECT id, file_name FROM assets WHERE id IN ({','.join('?' * len(asset_ids))})"
            return {r["id"]: r["file_name"] for r in c.execute(q, asset_ids)}

    def set_proposal(self, pid: int, status: str, created_stack_id: str | None = None) -> None:
        with self.conn() as c:
            c.execute("UPDATE proposals SET status=?, applied_at=?, created_stack_id=COALESCE(?, created_stack_id) WHERE id=?",
                      (status, now(), created_stack_id, pid))

    def proposal_stats(self) -> list[dict]:
        with self.conn() as c:
            return [dict(r) for r in c.execute("SELECT action, status, COUNT(*) n FROM proposals GROUP BY action, status")]

    def mark_stack_gone(self, stack_id: str) -> None:
        with self.conn() as c:
            c.execute("UPDATE stacks SET status='gone' WHERE id=?", (stack_id,))

    def mark_gone(self, seen_ids: list[str]) -> int:
        with self.conn() as c:
            placeholders = ",".join("?" * len(seen_ids)) or "''"
            cur = c.execute(f"UPDATE stacks SET status='gone' WHERE status!='gone' AND id NOT IN ({placeholders})", seen_ids)
            return cur.rowcount

    # -- read -------------------------------------------------------------

    def _hydrate(self, c, row) -> dict:
        assets = c.execute(
            "SELECT * FROM assets WHERE stack_id=? ORDER BY score DESC, file_name", (row["id"],)
        ).fetchall()
        return {**dict(row), "assets": [{**dict(a), "features": json.loads(a["features"])} for a in assets]}

    def list_stacks(self, status: str | None, kind: str | None, limit: int, offset: int) -> list[dict]:
        q, params = "SELECT * FROM stacks WHERE 1=1", []
        if status:
            q += " AND status=?"
            params.append(status)
        if kind:
            q += " AND kind=?"
            params.append(kind)
        q += " ORDER BY captured_at DESC, id LIMIT ? OFFSET ?"
        params += [limit, offset]
        with self.conn() as c:
            return [self._hydrate(c, r) for r in c.execute(q, params).fetchall()]

    def get_stack(self, stack_id: str) -> dict | None:
        with self.conn() as c:
            row = c.execute("SELECT * FROM stacks WHERE id=?", (stack_id,)).fetchone()
            return self._hydrate(c, row) if row else None

    def stats(self) -> dict:
        with self.conn() as c:
            rows = c.execute("SELECT kind, status, COUNT(*) n FROM stacks GROUP BY kind, status").fetchall()
            pairs = c.execute("SELECT COUNT(*) FROM decisions WHERE action IN ('pick','pick_trash')").fetchone()[0]
        return {"by_kind_status": [dict(r) for r in rows], "human_decisions": pairs}

    # -- write ------------------------------------------------------------

    def set_status(self, stack_id: str, status: str, primary: str | None = None) -> None:
        with self.conn() as c:
            c.execute(
                "UPDATE stacks SET status=?, reviewed_at=?, primary_asset_id=COALESCE(?, primary_asset_id) WHERE id=?",
                (status, now(), primary, stack_id),
            )

    def add_decision(self, stack_id: str, action: str, chosen: str | None, previous: str | None, trashed: list[str]) -> None:
        with self.conn() as c:
            c.execute(
                "INSERT INTO decisions(stack_id, action, chosen_asset_id, previous_primary_id, trashed_asset_ids, created_at) VALUES(?,?,?,?,?,?)",
                (stack_id, action, chosen, previous, json.dumps(trashed), now()),
            )

    def last_decision(self, stack_id: str) -> dict | None:
        with self.conn() as c:
            row = c.execute(
                "SELECT * FROM decisions WHERE stack_id=? AND action!='undo' ORDER BY id DESC LIMIT 1", (stack_id,)
            ).fetchone()
        if not row:
            return None
        return {**dict(row), "trashed_asset_ids": json.loads(row["trashed_asset_ids"])}

    def training_pairs(self) -> list[tuple[dict, list[dict]]]:
        """Human picks only: (features of chosen, features of every rejected sibling)."""
        out = []
        with self.conn() as c:
            decisions = c.execute(
                "SELECT stack_id, chosen_asset_id FROM decisions WHERE action IN ('pick','pick_trash')"
            ).fetchall()
            for d in decisions:
                assets = c.execute("SELECT id, features FROM assets WHERE stack_id=?", (d["stack_id"],)).fetchall()
                feats = {a["id"]: json.loads(a["features"]) for a in assets}
                if d["chosen_asset_id"] not in feats:
                    continue
                chosen = feats.pop(d["chosen_asset_id"])
                if feats:
                    out.append((chosen, list(feats.values())))
        return out
