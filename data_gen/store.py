"""SQLite state of a generation run: the topic taxonomy, one row per case, and every LLM call.

A case row holds its plan (fixed at planning time), its data (grown stage by stage) and the last
stage it completed; the runner commits after every stage of every case. Every LLM call is stored
under a hash of its full request, so a re-run after a crash replays finished calls from the table
instead of asking the model again. Each thread gets its own connection (WAL mode).
"""

from __future__ import annotations

import json
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any, Iterator

SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT);
CREATE TABLE IF NOT EXISTS topics (
    domain TEXT, topic_id TEXT PRIMARY KEY, data TEXT);
CREATE TABLE IF NOT EXISTS cases (
    case_id TEXT PRIMARY KEY, idx INTEGER, split TEXT, plan TEXT, data TEXT,
    stage TEXT, status TEXT, error TEXT, updated REAL);
CREATE TABLE IF NOT EXISTS calls (
    key TEXT PRIMARY KEY, case_id TEXT, stage TEXT, prompt TEXT, prompt_version TEXT,
    request TEXT, response TEXT, finish_reason TEXT,
    prompt_tokens INTEGER, completion_tokens INTEGER, seconds REAL, created REAL);
CREATE TABLE IF NOT EXISTS stage_runs (
    stage TEXT, n_cases INTEGER, seconds REAL, finished REAL);
CREATE INDEX IF NOT EXISTS cases_stage ON cases (status, stage);
CREATE INDEX IF NOT EXISTS calls_case ON calls (case_id);
"""

# Case statuses. ``rejected`` is set by the filter (a check failed), ``failed`` by the runner (the
# model never produced a valid reply for a stage); both stop the case.
ACTIVE, REJECTED, FAILED = "active", "rejected", "failed"


class Store:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._local = threading.local()
        with self.conn() as c:
            c.executescript(SCHEMA)

    def conn(self) -> sqlite3.Connection:
        c = getattr(self._local, "conn", None)
        if c is None:
            c = sqlite3.connect(self.path, timeout=60)
            c.execute("PRAGMA journal_mode=WAL")
            c.execute("PRAGMA busy_timeout=60000")
            self._local.conn = c
        return c

    # meta
    def set_meta(self, key: str, value: Any) -> None:
        with self.conn() as c:
            c.execute("INSERT OR REPLACE INTO meta VALUES (?, ?)", (key, json.dumps(value)))

    def get_meta(self, key: str, default: Any = None) -> Any:
        row = self.conn().execute("SELECT value FROM meta WHERE key = ?", (key,)).fetchone()
        return json.loads(row[0]) if row else default

    # topics
    def add_topics(self, domain: str, topics: list[dict]) -> None:
        with self.conn() as c:
            c.executemany("INSERT OR REPLACE INTO topics VALUES (?, ?, ?)",
                          [(domain, t["topic_id"], json.dumps(t)) for t in topics])

    def topics(self, domain: str | None = None) -> list[dict]:
        sql, args = "SELECT data FROM topics", ()
        if domain is not None:
            sql, args = sql + " WHERE domain = ?", (domain,)
        return [json.loads(r[0]) for r in self.conn().execute(sql + " ORDER BY topic_id", args)]

    # cases
    def add_cases(self, cases: list[dict]) -> None:
        now = time.time()
        with self.conn() as c:
            c.executemany(
                "INSERT INTO cases VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                [(p["case_id"], p["idx"], p["split"], json.dumps(p), "{}", "planned", ACTIVE, None, now) for p in cases])

    def n_cases(self) -> int:
        return self.conn().execute("SELECT COUNT(*) FROM cases").fetchone()[0]

    def cases(self, status: str | None = None, stage: str | None = None) -> Iterator[dict]:
        sql, cond, args = "SELECT case_id, plan, data, stage, status, error FROM cases", [], []
        if status is not None:
            cond.append("status = ?"); args.append(status)
        if stage is not None:
            cond.append("stage = ?"); args.append(stage)
        if cond:
            sql += " WHERE " + " AND ".join(cond)
        for case_id, plan, data, st, status_, error in self.conn().execute(sql + " ORDER BY idx", args).fetchall():
            yield {"case_id": case_id, "plan": json.loads(plan), "data": json.loads(data),
                   "stage": st, "status": status_, "error": error}

    def save_case(self, case: dict) -> None:
        with self.conn() as c:
            c.execute("UPDATE cases SET data = ?, stage = ?, status = ?, error = ?, updated = ? WHERE case_id = ?",
                      (json.dumps(case["data"]), case["stage"], case["status"], case["error"], time.time(),
                       case["case_id"]))

    def counts(self) -> list[tuple[str, str, int]]:
        return self.conn().execute(
            "SELECT status, stage, COUNT(*) FROM cases GROUP BY status, stage ORDER BY status, stage").fetchall()

    # calls
    def cached_call(self, key: str) -> dict | None:
        row = self.conn().execute("SELECT response, finish_reason FROM calls WHERE key = ?", (key,)).fetchone()
        return {"content": row[0], "finish_reason": row[1]} if row else None

    def add_call(self, key: str, case_id: str, stage: str, prompt: str, prompt_version: str, request: dict,
                 reply) -> None:
        with self.conn() as c:
            c.execute("INSERT OR REPLACE INTO calls VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                      (key, case_id, stage, prompt, prompt_version, json.dumps(request), reply.content,
                       reply.finish_reason, reply.prompt_tokens, reply.completion_tokens, reply.seconds, time.time()))

    def call_usage(self) -> list[tuple]:
        return self.conn().execute(
            "SELECT stage, prompt, COUNT(*), SUM(prompt_tokens), SUM(completion_tokens), SUM(seconds) "
            "FROM calls GROUP BY stage, prompt ORDER BY stage, prompt").fetchall()

    # stage timing (wall clock, for the throughput estimate)
    def add_stage_run(self, stage: str, n_cases: int, seconds: float) -> None:
        with self.conn() as c:
            c.execute("INSERT INTO stage_runs VALUES (?, ?, ?, ?)", (stage, n_cases, seconds, time.time()))

    def stage_runs(self) -> list[tuple]:
        return self.conn().execute(
            "SELECT stage, SUM(n_cases), SUM(seconds) FROM stage_runs GROUP BY stage").fetchall()
