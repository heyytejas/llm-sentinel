"""
SQLite persistence layer for LLM Sentinel scan results.

Creates and manages two tables:
  - scan_sessions : one row per scan run
  - probe_results : one row per individual attack probe

Uses only the Python standard library (sqlite3) — no ORM dependency.
"""

from __future__ import annotations

import json
import logging
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Generator, List, Optional

from .session import ProbeResult, ProbeStatus, ScanSession

logger = logging.getLogger(__name__)

# Default DB path: repo root / reports / sentinel.db
_DEFAULT_DB_PATH = Path(__file__).resolve().parents[2] / "reports" / "sentinel.db"

# ── DDL ───────────────────────────────────────────────────────────────────────

_CREATE_SESSIONS = """
CREATE TABLE IF NOT EXISTS scan_sessions (
    scan_id      TEXT PRIMARY KEY,
    model        TEXT NOT NULL,
    started_at   TEXT NOT NULL,
    finished_at  TEXT,
    total        INTEGER DEFAULT 0,
    succeeded    INTEGER DEFAULT 0,
    errors       INTEGER DEFAULT 0
);
"""

_CREATE_RESULTS = """
CREATE TABLE IF NOT EXISTS probe_results (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    scan_id             TEXT NOT NULL,
    attack_id           TEXT NOT NULL,
    case_id             TEXT,
    attack_name         TEXT NOT NULL DEFAULT '',
    category            TEXT NOT NULL,
    "group"             TEXT,
    attack_group        TEXT,
    attack_type         TEXT NOT NULL,
    "type"              TEXT,
    severity            TEXT NOT NULL DEFAULT 'medium',
    is_control          INTEGER NOT NULL DEFAULT 0,
    repetition          INTEGER NOT NULL DEFAULT 1,
    repetition_number   INTEGER NOT NULL DEFAULT 1,
    model               TEXT NOT NULL,
    model_name          TEXT,
    prompt              TEXT NOT NULL,
    query               TEXT,
    system_prompt       TEXT,
    retrieved_chunks    TEXT,
    status              TEXT NOT NULL,
    response            TEXT,
    rule_based_verdict  TEXT,
    judge_verdict       TEXT,
    judge_reason        TEXT,
    error_message       TEXT,
    timestamp           TEXT NOT NULL,
    latency_ms          REAL,
    latency             REAL,
    attempt             INTEGER NOT NULL DEFAULT 1,
    max_attempts        INTEGER NOT NULL DEFAULT 1,
    FOREIGN KEY (scan_id) REFERENCES scan_sessions(scan_id)
);
"""

_CREATE_IDX_SCAN = "CREATE INDEX IF NOT EXISTS idx_results_scan ON probe_results(scan_id);"
_CREATE_IDX_ATTACK = "CREATE INDEX IF NOT EXISTS idx_results_attack ON probe_results(attack_id);"


# ── Store class ───────────────────────────────────────────────────────────────

class ResultStore:
    """Thread-safe SQLite persistence for scan sessions and probe results."""

    def __init__(self, db_path: Optional[Path | str] = None) -> None:
        self.db_path = Path(db_path) if db_path else _DEFAULT_DB_PATH
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    # ── Internal helpers ──────────────────────────────────────────────────────

    @contextmanager
    def _connect(self) -> Generator[sqlite3.Connection, None, None]:
        conn = sqlite3.connect(self.db_path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA foreign_keys=ON;")
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def _init_schema(self) -> None:
        with self._connect() as conn:
            conn.execute(_CREATE_SESSIONS)
            conn.execute(_CREATE_RESULTS)
            conn.execute(_CREATE_IDX_SCAN)
            conn.execute(_CREATE_IDX_ATTACK)

            # Ensure extra columns exist if table was already created by an older version
            cursor = conn.execute("PRAGMA table_info(probe_results);")
            existing_cols = {row["name"] for row in cursor.fetchall()}
            col_defs = {
                "case_id": "TEXT",
                "group": "TEXT",
                "attack_group": "TEXT",
                "type": "TEXT",
                "is_control": "INTEGER DEFAULT 0",
                "repetition": "INTEGER DEFAULT 1",
                "repetition_number": "INTEGER DEFAULT 1",
                "model_name": "TEXT",
                "query": "TEXT",
                "retrieved_chunks": "TEXT",
                "rule_based_verdict": "TEXT",
                "judge_verdict": "TEXT",
                "judge_reason": "TEXT",
                "latency": "REAL",
            }
            for col, ddl_type in col_defs.items():
                if col not in existing_cols:
                    try:
                        conn.execute(f'ALTER TABLE probe_results ADD COLUMN "{col}" {ddl_type};')
                    except Exception:
                        pass
        logger.debug("Schema initialised at '%s'.", self.db_path)

    # ── Write operations ──────────────────────────────────────────────────────

    def save_session(self, session: ScanSession) -> None:
        """Insert or replace a ScanSession row (upsert by scan_id)."""
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO scan_sessions
                    (scan_id, model, started_at, finished_at, total, succeeded, errors)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(scan_id) DO UPDATE SET
                    finished_at = excluded.finished_at,
                    total       = excluded.total,
                    succeeded   = excluded.succeeded,
                    errors      = excluded.errors
                """,
                (
                    session.scan_id,
                    session.model,
                    session.started_at.isoformat(),
                    session.finished_at.isoformat() if session.finished_at else None,
                    session.total,
                    session.succeeded,
                    session.errors,
                ),
            )
        logger.debug("Saved session '%s'.", session.scan_id)

    def _extract_row_tuple(self, result: ProbeResult) -> tuple:
        case_id = result.case_id or result.attack_id
        group = result.group or ""
        atk_type = result.type or result.attack_type
        query = result.query or result.prompt
        retrieved_chunks_str = (
            json.dumps(result.retrieved_chunks)
            if isinstance(result.retrieved_chunks, list)
            else (str(result.retrieved_chunks) if result.retrieved_chunks is not None else "[]")
        )
        is_ctrl = 1 if result.is_control else 0
        rep = result.repetition
        lat = result.latency_ms
        verdict = result.rule_based_verdict or ("PASS" if result.status == ProbeStatus.SUCCESS else "ERROR")
        j_verdict = getattr(result, "judge_verdict", None)
        j_reason = getattr(result, "judge_reason", None)

        return (
            result.scan_id,
            result.attack_id,
            case_id,
            result.attack_name,
            result.category,
            group,
            group,
            result.attack_type,
            atk_type,
            result.severity,
            is_ctrl,
            rep,
            rep,
            result.model,
            result.model,
            result.prompt,
            query,
            result.system_prompt,
            retrieved_chunks_str,
            result.status.value,
            result.response,
            verdict,
            j_verdict,
            j_reason,
            result.error_message,
            result.timestamp.isoformat(),
            lat,
            lat,
            result.attempt,
            result.max_attempts,
        )

    def save_result(self, result: ProbeResult) -> None:
        """Insert a single ProbeResult row."""
        row = self._extract_row_tuple(result)
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO probe_results (
                    scan_id, attack_id, case_id, attack_name, category, "group", attack_group,
                    attack_type, "type", severity, is_control, repetition, repetition_number,
                    model, model_name, prompt, query, system_prompt, retrieved_chunks,
                    status, response, rule_based_verdict, judge_verdict, judge_reason,
                    error_message, timestamp, latency_ms, latency, attempt, max_attempts
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                row,
            )
        logger.debug("Saved result for attack '%s' (status=%s).", result.attack_id, result.status.value)

    def save_results_bulk(self, results: List[ProbeResult]) -> None:
        """Insert multiple ProbeResult rows in a single transaction."""
        rows = [self._extract_row_tuple(r) for r in results]
        with self._connect() as conn:
            conn.executemany(
                """
                INSERT INTO probe_results (
                    scan_id, attack_id, case_id, attack_name, category, "group", attack_group,
                    attack_type, "type", severity, is_control, repetition, repetition_number,
                    model, model_name, prompt, query, system_prompt, retrieved_chunks,
                    status, response, rule_based_verdict, judge_verdict, judge_reason,
                    error_message, timestamp, latency_ms, latency, attempt, max_attempts
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                rows,
            )
        logger.info("Bulk-inserted %d probe results.", len(rows))

    def update_judge_result(self, row_id: int, verdict: str, reason: str) -> None:
        """Update the judge verdict and reason for a specific probe result row."""
        with self._connect() as conn:
            conn.execute(
                "UPDATE probe_results SET judge_verdict = ?, judge_reason = ? WHERE id = ?",
                (verdict, reason, row_id),
            )
        logger.debug("Updated judge verdict for row %d: %s", row_id, verdict)

    # ── Read operations ───────────────────────────────────────────────────────

    def get_session(self, scan_id: str) -> Optional[sqlite3.Row]:
        with self._connect() as conn:
            return conn.execute(
                "SELECT * FROM scan_sessions WHERE scan_id = ?", (scan_id,)
            ).fetchone()

    def get_results_for_scan(self, scan_id: str) -> List[sqlite3.Row]:
        with self._connect() as conn:
            return conn.execute(
                "SELECT * FROM probe_results WHERE scan_id = ? ORDER BY id",
                (scan_id,),
            ).fetchall()

    def list_sessions(self) -> List[sqlite3.Row]:
        with self._connect() as conn:
            return conn.execute(
                "SELECT * FROM scan_sessions ORDER BY started_at DESC"
            ).fetchall()
