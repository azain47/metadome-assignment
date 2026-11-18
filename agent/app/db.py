"""Postgres tables owned by us: jobs, candidates, tool log.

ADK's DatabaseSessionService owns sessions, events and user/app state in the same database.
Rows hold object keys and small JSON only, never image bytes.
"""

import json
import logging
import uuid
from typing import Any

import asyncpg

log = logging.getLogger(__name__)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS ts_jobs (
    job_id      text PRIMARY KEY,
    session_id  text NOT NULL,
    location    text NOT NULL,
    status      text NOT NULL DEFAULT 'running',
    plan        jsonb NOT NULL DEFAULT '{}'::jsonb,
    state       jsonb NOT NULL DEFAULT '{}'::jsonb,
    attempts    int NOT NULL DEFAULT 0,
    final_key   text,
    verdict     text,
    best_score  double precision,
    created_at  timestamptz NOT NULL DEFAULT now(),
    updated_at  timestamptz NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS ts_jobs_session ON ts_jobs (session_id);

CREATE TABLE IF NOT EXISTS ts_candidates (
    id          text PRIMARY KEY,
    job_id      text NOT NULL REFERENCES ts_jobs(job_id),
    attempt     int NOT NULL,
    kind        text NOT NULL,
    key         text NOT NULL,
    scores      jsonb NOT NULL DEFAULT '{}'::jsonb,
    verdict     text,
    reason      text,
    meta        jsonb NOT NULL DEFAULT '{}'::jsonb,
    created_at  timestamptz NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS ts_candidates_job ON ts_candidates (job_id, attempt);

CREATE TABLE IF NOT EXISTS ts_tool_log (
    id            bigserial PRIMARY KEY,
    ts            timestamptz NOT NULL DEFAULT now(),
    session_id    text,
    invocation_id text,
    node          text,
    tool          text NOT NULL,
    source        text NOT NULL,
    status        text NOT NULL,
    args          jsonb,
    result        jsonb,
    duration_ms   int
);
CREATE INDEX IF NOT EXISTS ts_tool_log_session ON ts_tool_log (session_id, ts);
"""

_JOB_COLUMNS = {"status", "plan", "state", "attempts", "final_key", "verdict", "best_score", "location"}
_JSON_COLUMNS = {"plan", "state"}


def _dump(value: Any) -> str:
    return json.dumps(value, default=str)


def _row(record: asyncpg.Record | None) -> dict[str, Any] | None:
    if record is None:
        return None
    row = dict(record)
    for name, value in row.items():
        if isinstance(value, str) and name in ("plan", "state", "scores", "meta", "args", "result"):
            row[name] = json.loads(value)
    return row


class Database:
    """Job, candidate and tool-log persistence. Call `start()` once before use."""

    def __init__(self, dsn: str) -> None:
        self._dsn = dsn
        self._pool: asyncpg.Pool | None = None

    async def start(self) -> None:
        """Connect, create tables, and mark jobs left running by a previous process as interrupted."""
        if self._pool is not None:
            return
        self._pool = await asyncpg.create_pool(self._dsn, min_size=1, max_size=5)
        async with self._pool.acquire() as conn:
            await conn.execute(_SCHEMA)
            await conn.execute(
                "UPDATE ts_jobs SET status = 'interrupted', updated_at = now() WHERE status = 'running'")

    async def stop(self) -> None:
        if self._pool is not None:
            await self._pool.close()
            self._pool = None

    @property
    def pool(self) -> asyncpg.Pool:
        if self._pool is None:
            raise RuntimeError("Database.start() was not called")
        return self._pool

    async def health(self) -> dict[str, Any]:
        try:
            await self.pool.fetchval("SELECT 1")
            return {"ok": True, "detail": "postgres"}
        except Exception as ex:  # noqa: BLE001
            return {"ok": False, "detail": str(ex)[:200]}

    # --- jobs -----------------------------------------------------------------

    async def create_job(self, session_id: str, location: str, plan: dict[str, Any]) -> str:
        slug = "".join(c if c.isalnum() else "-" for c in location.lower()).strip("-")[:24] or "shot"
        job_id = f"{slug}-{uuid.uuid4().hex[:6]}"
        await self.pool.execute(
            "INSERT INTO ts_jobs (job_id, session_id, location, plan) VALUES ($1, $2, $3, $4::jsonb)",
            job_id, session_id, location, _dump(plan))
        return job_id

    async def update_job(self, job_id: str, **fields: Any) -> None:
        unknown = set(fields) - _JOB_COLUMNS
        if unknown:
            raise ValueError(f"unknown job columns: {sorted(unknown)}")
        if not fields:
            return
        sets, values = [], []
        for i, (name, value) in enumerate(fields.items(), start=2):
            is_json = name in _JSON_COLUMNS
            sets.append(f"{name} = ${i}{'::jsonb' if is_json else ''}")
            values.append(_dump(value) if is_json else value)
        await self.pool.execute(
            f"UPDATE ts_jobs SET {', '.join(sets)}, updated_at = now() WHERE job_id = $1", job_id, *values)

    async def get_job(self, job_id: str) -> dict[str, Any] | None:
        return _row(await self.pool.fetchrow("SELECT * FROM ts_jobs WHERE job_id = $1", job_id))

    # --- candidates -----------------------------------------------------------

    async def add_candidate(self, job_id: str, attempt: int, kind: str, key: str, *,
                            scores: dict[str, Any] | None = None, verdict: str | None = None,
                            reason: str | None = None, meta: dict[str, Any] | None = None) -> str:
        candidate_id = uuid.uuid4().hex[:10]
        await self.pool.execute(
            "INSERT INTO ts_candidates (id, job_id, attempt, kind, key, scores, verdict, reason, meta) "
            "VALUES ($1, $2, $3, $4, $5, $6::jsonb, $7, $8, $9::jsonb)",
            candidate_id, job_id, attempt, kind, key, _dump(scores or {}), verdict, reason, _dump(meta or {}))
        return candidate_id

    async def candidates(self, job_id: str) -> list[dict[str, Any]]:
        rows = await self.pool.fetch(
            "SELECT * FROM ts_candidates WHERE job_id = $1 ORDER BY attempt, created_at", job_id)
        return [_row(r) for r in rows]  # type: ignore[misc]

    # --- tool log ---------------------------------------------------------------

    async def log_tool(self, *, tool: str, source: str, status: str, args: Any = None, result: Any = None,
                       duration_ms: int | None = None, session_id: str | None = None,
                       invocation_id: str | None = None, node: str | None = None) -> None:
        """Append one tool-call row. Never raises: logging must not break a run."""
        try:
            await self.pool.execute(
                "INSERT INTO ts_tool_log (session_id, invocation_id, node, tool, source, status, args, result, "
                "duration_ms) VALUES ($1, $2, $3, $4, $5, $6, $7::jsonb, $8::jsonb, $9)",
                session_id, invocation_id, node, tool, source, status, _dump(args), _dump(result), duration_ms)
        except Exception as ex:  # noqa: BLE001
            log.warning("tool log insert failed: %s", ex)
