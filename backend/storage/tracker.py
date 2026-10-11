"""
Application tracker: a small SQLite store for the postings you've analyzed or
applied to, with their match scores and status, so you can compare postings
and see your pipeline at a glance.

sqlite3 is in the standard library — no extra dependency, no server. Each call
opens its own short-lived connection, which keeps it safe under FastAPI's
thread pool without any locking of our own.

Note: on free PaaS tiers (Render, Railway) the filesystem is ephemeral, so the
database resets on redeploy. Point COPILOT_DB_PATH at a persistent disk to keep it.
"""

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

STATUSES = ("saved", "applied", "interviewing", "offer", "rejected", "withdrawn")

_SCHEMA = """
CREATE TABLE IF NOT EXISTS applications (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    company       TEXT NOT NULL DEFAULT '',
    role          TEXT NOT NULL DEFAULT '',
    url           TEXT NOT NULL DEFAULT '',
    status        TEXT NOT NULL DEFAULT 'saved',
    notes         TEXT NOT NULL DEFAULT '',
    deadline      TEXT,
    applied_on    TEXT,
    match_score   INTEGER,
    coverage_pct  REAL,
    missing_skills TEXT NOT NULL DEFAULT '[]',
    analysis      TEXT,
    created_at    TEXT NOT NULL,
    updated_at    TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_applications_status ON applications(status);
"""

_UPDATABLE = {"company", "role", "url", "status", "notes", "deadline", "applied_on",
              "match_score", "coverage_pct", "missing_skills", "analysis"}


class Tracker:
    def __init__(self, db_path: Path | str):
        # A file path, not ":memory:" — every call opens its own connection, and an
        # in-memory database would be a fresh empty one each time.
        self.db_path = str(db_path)
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            conn.executescript(_SCHEMA)

    @contextmanager
    def _connect(self):
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    @staticmethod
    def _now() -> str:
        return datetime.now(timezone.utc).isoformat(timespec="seconds")

    @staticmethod
    def _row_to_dict(row: sqlite3.Row, include_analysis: bool = False) -> dict:
        item = dict(row)
        item["missing_skills"] = json.loads(item.get("missing_skills") or "[]")
        analysis = item.pop("analysis", None)
        if include_analysis:
            item["analysis"] = json.loads(analysis) if analysis else None
        return item

    @staticmethod
    def _encode(fields: dict) -> dict:
        out = dict(fields)
        if "missing_skills" in out and not isinstance(out["missing_skills"], str):
            out["missing_skills"] = json.dumps(out["missing_skills"] or [])
        if "analysis" in out and out["analysis"] is not None and not isinstance(out["analysis"], str):
            out["analysis"] = json.dumps(out["analysis"])
        if "status" in out and out["status"] not in STATUSES:
            raise ValueError(f"status must be one of {', '.join(STATUSES)}")
        return out

    def create(self, **fields) -> dict:
        fields = self._encode({k: v for k, v in fields.items() if k in _UPDATABLE and v is not None})
        if fields.get("status") == "applied" and not fields.get("applied_on"):
            fields["applied_on"] = datetime.now(timezone.utc).date().isoformat()
        now = self._now()
        fields.update(created_at=now, updated_at=now)
        cols = ", ".join(fields)
        placeholders = ", ".join("?" for _ in fields)
        with self._connect() as conn:
            cur = conn.execute(f"INSERT INTO applications ({cols}) VALUES ({placeholders})", list(fields.values()))
            new_id = cur.lastrowid
        return self.get(new_id)

    def get(self, app_id: int, include_analysis: bool = True) -> dict | None:
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM applications WHERE id = ?", (app_id,)).fetchone()
        return self._row_to_dict(row, include_analysis) if row else None

    def list(self, status: str | None = None, query: str | None = None) -> list[dict]:
        sql, params = "SELECT * FROM applications WHERE 1=1", []
        if status:
            sql += " AND status = ?"
            params.append(status)
        if query:
            sql += " AND (company LIKE ? OR role LIKE ? OR notes LIKE ?)"
            params += [f"%{query}%"] * 3
        sql += " ORDER BY updated_at DESC, id DESC"
        with self._connect() as conn:
            rows = conn.execute(sql, params).fetchall()
        return [self._row_to_dict(r) for r in rows]

    def update(self, app_id: int, **fields) -> dict | None:
        fields = self._encode({k: v for k, v in fields.items() if k in _UPDATABLE})
        if not fields:
            return self.get(app_id)
        current = self.get(app_id, include_analysis=False)
        if current is None:
            return None
        if fields.get("status") == "applied" and not current.get("applied_on") and "applied_on" not in fields:
            fields["applied_on"] = datetime.now(timezone.utc).date().isoformat()
        fields["updated_at"] = self._now()
        assignments = ", ".join(f"{k} = ?" for k in fields)
        with self._connect() as conn:
            conn.execute(f"UPDATE applications SET {assignments} WHERE id = ?", [*fields.values(), app_id])
        return self.get(app_id)

    def delete(self, app_id: int) -> bool:
        with self._connect() as conn:
            cur = conn.execute("DELETE FROM applications WHERE id = ?", (app_id,))
        return cur.rowcount > 0

    def stats(self) -> dict:
        with self._connect() as conn:
            by_status = {s: 0 for s in STATUSES}
            for row in conn.execute("SELECT status, COUNT(*) AS n FROM applications GROUP BY status"):
                by_status[row["status"]] = row["n"]
            avg = conn.execute("SELECT AVG(match_score) FROM applications WHERE match_score IS NOT NULL").fetchone()[0]
            missing_rows = conn.execute("SELECT missing_skills FROM applications").fetchall()

        # Skills missing across many postings are the highest-leverage things to learn next.
        counts: dict[str, int] = {}
        for row in missing_rows:
            for skill in json.loads(row["missing_skills"] or "[]"):
                counts[skill] = counts.get(skill, 0) + 1
        top_missing = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[:10]

        total = sum(by_status.values())
        applied_total = total - by_status["saved"]
        responded = by_status["interviewing"] + by_status["offer"]
        return {
            "total": total,
            "by_status": by_status,
            "avg_match_score": round(avg, 1) if avg is not None else None,
            "response_rate_pct": round(responded / applied_total * 100, 1) if applied_total else None,
            "top_missing_skills": [{"skill": s, "count": n} for s, n in top_missing],
        }
