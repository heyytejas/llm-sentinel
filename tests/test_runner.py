"""
Unit and integration tests for the runner module:
  - ScanEngine probe execution (success, error, retry, timeout)
  - ResultStore SQLite persistence (session + result CRUD)
  - End-to-end scan with MockTarget
"""

from __future__ import annotations

import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional
from unittest.mock import MagicMock, patch

import pytest

from llm_sentinel.adapters.adapter import MockTarget
from llm_sentinel.attacks.models import AttackEntry, AttackCategory, AttackType, Severity, SuccessCondition
from llm_sentinel.runner.engine import ScanEngine
from llm_sentinel.runner.session import ProbeResult, ProbeStatus, ScanSession
from llm_sentinel.runner.store import ResultStore


# ── Fixtures ──────────────────────────────────────────────────────────────────

def make_entry(
    attack_id: str = "INJ-001",
    name: str = "Test Entry",
    category: str = "prompt_injection",
    attack_type: str = "direct_injection",
    severity: str = "HIGH",
    prompt: str = "Say HACKED",
) -> AttackEntry:
    return AttackEntry(
        id=attack_id,
        name=name,
        category=category,
        type=attack_type,
        prompt=prompt,
        success_if=SuccessCondition(contains_any=["HACKED"]),
        severity=severity,
    )


@pytest.fixture
def db(tmp_path) -> ResultStore:
    return ResultStore(db_path=tmp_path / "test.db")


@pytest.fixture
def mock_adapter() -> MockTarget:
    return MockTarget(default_response="Safe response from model.")


@pytest.fixture
def engine(tmp_path, mock_adapter) -> ScanEngine:
    return ScanEngine(
        adapter=mock_adapter,
        db_path=tmp_path / "test.db",
        max_retries=2,
        concurrency=2,
    )


# ── ResultStore Tests ─────────────────────────────────────────────────────────

class TestResultStore:
    def _make_result(self, scan_id: str = "scan-001", attack_id: str = "INJ-001") -> ProbeResult:
        return ProbeResult(
            attack_id=attack_id,
            attack_name="Test Attack",
            category="prompt_injection",
            attack_type="direct_injection",
            severity="HIGH",
            model="mock-v1",
            prompt="Say HACKED",
            system_prompt="Be helpful",
            status=ProbeStatus.SUCCESS,
            response="I cannot do that.",
            error_message=None,
            timestamp=datetime.now(timezone.utc),
            latency_ms=42.0,
            attempt=1,
            max_attempts=2,
            scan_id=scan_id,
        )

    def test_schema_created(self, tmp_path):
        store = ResultStore(db_path=tmp_path / "new.db")
        assert (tmp_path / "new.db").exists()

    def test_save_and_retrieve_session(self, db):
        session = ScanSession(scan_id="s-001", model="gpt-4o-mini")
        db.save_session(session)
        row = db.get_session("s-001")
        assert row is not None
        assert row["scan_id"] == "s-001"
        assert row["model"] == "gpt-4o-mini"

    def test_session_upsert(self, db):
        session = ScanSession(scan_id="s-002", model="gpt-4o-mini")
        db.save_session(session)
        session.finish()
        db.save_session(session)
        row = db.get_session("s-002")
        assert row["finished_at"] is not None

    def test_save_and_retrieve_result(self, db):
        result = self._make_result("scan-001", "INJ-001")
        # Insert a minimal session first to satisfy FK
        db.save_session(ScanSession(scan_id="scan-001", model="mock-v1"))
        db.save_result(result)
        rows = db.get_results_for_scan("scan-001")
        assert len(rows) == 1
        assert rows[0]["attack_id"] == "INJ-001"
        assert rows[0]["status"] == "success"
        assert rows[0]["latency_ms"] == pytest.approx(42.0)

    def test_bulk_save(self, db):
        db.save_session(ScanSession(scan_id="scan-bulk", model="mock-v1"))
        results = [
            self._make_result("scan-bulk", f"INJ-00{i}") for i in range(1, 5)
        ]
        db.save_results_bulk(results)
        rows = db.get_results_for_scan("scan-bulk")
        assert len(rows) == 4

    def test_list_sessions(self, db):
        for i in range(3):
            db.save_session(ScanSession(scan_id=f"sid-{i}", model="mock-v1"))
        rows = db.list_sessions()
        assert len(rows) == 3

    def test_error_result_persisted(self, db):
        db.save_session(ScanSession(scan_id="scan-err", model="mock-v1"))
        result = self._make_result("scan-err")
        result.status = ProbeStatus.ERROR
        result.response = None
        result.error_message = "Connection refused"
        db.save_result(result)
        rows = db.get_results_for_scan("scan-err")
        assert rows[0]["status"] == "error"
        assert rows[0]["error_message"] == "Connection refused"
        assert rows[0]["response"] is None


# ── ScanEngine Tests ──────────────────────────────────────────────────────────

class TestScanEngine:
    def test_successful_scan(self, engine):
        entries = [make_entry("INJ-001"), make_entry("INJ-002")]
        session = engine.run(entries=entries, scan_id="scan-ok")
        assert session.total == 2
        assert session.succeeded == 2
        assert session.errors == 0
        assert session.finished_at is not None

    def test_results_persisted_to_db(self, engine, tmp_path):
        entries = [make_entry()]
        session = engine.run(entries=entries, scan_id="scan-persist")
        store = ResultStore(db_path=tmp_path / "test.db")
        rows = store.get_results_for_scan("scan-persist")
        assert len(rows) == 1
        assert rows[0]["model"] == "mock-sentinel-v1"
        assert rows[0]["response"] == "Safe response from model."

    def test_response_and_timestamp_stored(self, engine, tmp_path):
        entries = [make_entry()]
        engine.run(entries=entries, scan_id="scan-ts")
        store = ResultStore(db_path=tmp_path / "test.db")
        row = store.get_results_for_scan("scan-ts")[0]
        # timestamp is an ISO string
        ts = datetime.fromisoformat(row["timestamp"])
        assert ts.year >= 2024

    def test_error_captured_after_retries(self, tmp_path):
        """Adapter that always raises should produce ERROR result after all retries."""
        failing_adapter = MockTarget()
        failing_adapter.send = MagicMock(side_effect=OSError("network down"))

        engine = ScanEngine(
            adapter=failing_adapter,
            db_path=tmp_path / "err.db",
            max_retries=2,
            concurrency=1,
        )
        entries = [make_entry()]
        session = engine.run(entries=entries, scan_id="scan-err")
        assert session.errors == 1
        assert session.succeeded == 0
        store = ResultStore(db_path=tmp_path / "err.db")
        row = store.get_results_for_scan("scan-err")[0]
        assert row["status"] == "error"
        assert "network down" in row["error_message"]
        assert row["attempt"] == 2  # max_retries=2

    def test_non_retryable_error_bails_fast(self, tmp_path):
        """A non-retryable exception (e.g. ValueError) should not exhaust retries."""
        adapter = MockTarget()
        adapter.send = MagicMock(side_effect=ValueError("bad config"))
        engine = ScanEngine(
            adapter=adapter,
            db_path=tmp_path / "nre.db",
            max_retries=3,
            concurrency=1,
        )
        entries = [make_entry()]
        session = engine.run(entries=entries, scan_id="scan-nre")
        assert session.errors == 1
        store = ResultStore(db_path=tmp_path / "nre.db")
        row = store.get_results_for_scan("scan-nre")[0]
        # Non-retryable: bailed on attempt 1
        assert row["attempt"] == 1

    def test_on_result_callback_fires(self, engine):
        collected = []
        engine.on_result = collected.append
        entries = [make_entry("INJ-001"), make_entry("INJ-002")]
        engine.run(entries=entries, scan_id="scan-cb")
        assert len(collected) == 2
        assert all(isinstance(r, ProbeResult) for r in collected)

    def test_auto_scan_id_generated(self, engine):
        entries = [make_entry()]
        session = engine.run(entries=entries)
        assert session.scan_id  # non-empty UUID

    def test_no_entries_raises(self, engine):
        with pytest.raises(ValueError, match="No attack entries"):
            engine.run(entries=[])

    def test_system_prompt_passed_to_adapter(self, tmp_path):
        adapter = MockTarget()
        engine = ScanEngine(
            adapter=adapter,
            system_prompt="You are a secure assistant.",
            db_path=tmp_path / "sp.db",
            max_retries=1,
            concurrency=1,
        )
        entries = [make_entry()]
        engine.run(entries=entries, scan_id="scan-sp")
        assert adapter.history[0]["system_prompt"] == "You are a secure assistant."

    def test_entry_system_prompt_override(self, tmp_path):
        adapter = MockTarget()
        engine = ScanEngine(
            adapter=adapter,
            system_prompt="Default prompt",
            db_path=tmp_path / "ov.db",
            max_retries=1,
            concurrency=1,
        )
        entry = AttackEntry(
            id="INJ-001",
            name="Override Test",
            category="prompt_injection",
            type="direct_injection",
            prompt="Hello",
            system_prompt_override="OVERRIDE PROMPT",
            success_if=SuccessCondition(contains_any=["anything"]),
            severity="LOW",
        )
        engine.run(entries=[entry], scan_id="scan-ov")
        assert adapter.history[0]["system_prompt"] == "OVERRIDE PROMPT"

    def test_concurrent_probes(self, tmp_path):
        """All concurrent probes should complete and be persisted."""
        adapter = MockTarget(default_response="ok")
        engine = ScanEngine(
            adapter=adapter,
            db_path=tmp_path / "conc.db",
            max_retries=1,
            concurrency=4,
        )
        entries = [make_entry(f"INJ-{i:03d}") for i in range(1, 9)]
        session = engine.run(entries=entries, scan_id="scan-conc")
        assert session.total == 8
        assert session.succeeded == 8
        store = ResultStore(db_path=tmp_path / "conc.db")
        rows = store.get_results_for_scan("scan-conc")
        assert len(rows) == 8


# ── Integration: Run against real YAML suites with MockTarget ─────────────────

class TestEndToEndWithRealAttacks:
    def test_full_scan_real_suites(self, tmp_path):
        adapter = MockTarget(default_response="I'm sorry, I cannot help with that.")
        engine = ScanEngine(
            adapter=adapter,
            db_path=tmp_path / "e2e.db",
            max_retries=1,
            concurrency=2,
        )
        session = engine.run()
        assert session.total >= 5, f"Expected ≥5 probes, got {session.total}"
        assert session.succeeded == session.total
        assert session.finished_at is not None

        store = ResultStore(db_path=tmp_path / "e2e.db")
        db_session = store.get_session(session.scan_id)
        assert db_session is not None
        rows = store.get_results_for_scan(session.scan_id)
        assert len(rows) == session.total
        for row in rows:
            assert row["attack_id"]
            assert row["model"] == "mock-sentinel-v1"
            assert row["timestamp"]
            assert row["response"]
