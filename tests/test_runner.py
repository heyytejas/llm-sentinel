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


# ── Library-based ScanEngine Tests (load_library, repeats, corpus isolation) ─

class TestLibraryRunner:
    """
    Tests runner integration with load_library():
      - rows = (48 + 10) x r
      - controls marked with is_control
      - no leftover injected documents in target corpus
      - stored SQLite schema columns
      - limit flag
      - mock-target CLI execution
    """

    def test_mock_target_full_scan_row_count_default_r(self, tmp_path):
        """Full scan with default repeats (r=3) gives (48 + 10) * 3 = 174 rows."""
        adapter = MockTarget(default_response="Safe response.")
        engine = ScanEngine(
            adapter=adapter,
            db_path=tmp_path / "lib_r3.db",
            concurrency=4,
        )
        session = engine.run()
        r = 3
        expected_rows = (48 + 10) * r
        assert session.total == expected_rows

        store = ResultStore(db_path=tmp_path / "lib_r3.db")
        rows = store.get_results_for_scan(session.scan_id)
        assert len(rows) == expected_rows

    def test_mock_target_scan_row_count_custom_r(self, tmp_path):
        """Full scan with custom repeats (e.g. r=2) gives (48 + 10) * 2 = 116 rows."""
        adapter = MockTarget(default_response="Safe response.")
        engine = ScanEngine(
            adapter=adapter,
            db_path=tmp_path / "lib_r2.db",
            concurrency=4,
            repeats=2,
        )
        session = engine.run()
        r = 2
        expected_rows = (48 + 10) * r
        assert session.total == expected_rows

        store = ResultStore(db_path=tmp_path / "lib_r2.db")
        rows = store.get_results_for_scan(session.scan_id)
        assert len(rows) == expected_rows

    def test_no_leftover_injected_documents(self, tmp_path):
        """After running indirect/poisoning cases, the corpus must contain 0 leftover documents."""
        adapter = MockTarget(default_response="Safe response.")
        assert len(adapter.corpus) == 0

        engine = ScanEngine(
            adapter=adapter,
            db_path=tmp_path / "clean_corpus.db",
            concurrency=2,
            repeats=1,
        )
        engine.run()
        assert len(adapter.corpus) == 0, f"Leftover documents found in corpus: {list(adapter.corpus.keys())}"

    def test_controls_are_marked(self, tmp_path):
        """Controls must be marked with is_control=1; attacks with is_control=0."""
        adapter = MockTarget(default_response="Safe response.")
        r = 2
        engine = ScanEngine(
            adapter=adapter,
            db_path=tmp_path / "controls.db",
            concurrency=4,
            repeats=r,
        )
        session = engine.run()
        store = ResultStore(db_path=tmp_path / "controls.db")
        rows = store.get_results_for_scan(session.scan_id)

        control_rows = [row for row in rows if row["is_control"] == 1]
        attack_rows = [row for row in rows if row["is_control"] == 0]

        assert len(control_rows) == 10 * r, f"Expected {10 * r} control rows, got {len(control_rows)}"
        assert len(attack_rows) == 48 * r, f"Expected {48 * r} attack rows, got {len(attack_rows)}"

        for row in control_rows:
            assert row["case_id"].startswith("CT-")
            assert row["is_control"] == 1

        for row in attack_rows:
            assert not row["case_id"].startswith("CT-")
            assert row["is_control"] == 0

    def test_stored_sqlite_fields(self, tmp_path):
        """
        Verify SQLite stores:
          case id, category, group, type, is_control, repetition number,
          query, retrieved chunks, response, latency, model name, timestamp,
          and rule-based verdict.
        """
        adapter = MockTarget(default_response="Safe response from target.")
        engine = ScanEngine(
            adapter=adapter,
            db_path=tmp_path / "fields.db",
            repeats=1,
            limit=5,
        )
        session = engine.run()
        store = ResultStore(db_path=tmp_path / "fields.db")
        rows = store.get_results_for_scan(session.scan_id)
        assert len(rows) == 5

        for row in rows:
            # Check all required fields are present and valid
            assert row["case_id"] is not None and len(row["case_id"]) > 0
            assert row["category"] is not None and len(row["category"]) > 0
            assert row["group"] is not None
            assert row["type"] in ("direct", "indirect", "poisoning", "direct_injection")
            assert row["is_control"] in (0, 1)
            assert row["repetition"] == 1
            assert row["repetition_number"] == 1
            assert row["query"] is not None and len(row["query"]) > 0
            assert row["retrieved_chunks"] is not None  # JSON list string
            assert row["response"] == "Safe response from target."
            assert row["latency"] is not None and row["latency"] >= 0
            assert row["model"] == "mock-sentinel-v1"
            assert row["timestamp"] is not None
            assert row["rule_based_verdict"] in ("PASS", "FAIL")

    def test_limit_flag(self, tmp_path):
        """Limit flag limits the number of unique cases to evaluate."""
        adapter = MockTarget(default_response="Safe response.")
        limit_val = 7
        repeats_val = 2
        engine = ScanEngine(
            adapter=adapter,
            db_path=tmp_path / "limit.db",
            concurrency=2,
            repeats=repeats_val,
            limit=limit_val,
        )
        session = engine.run()
        assert session.total == limit_val * repeats_val

        store = ResultStore(db_path=tmp_path / "limit.db")
        rows = store.get_results_for_scan(session.scan_id)
        assert len(rows) == limit_val * repeats_val

    def test_cli_mock_target(self, tmp_path):
        """CLI invocation with --mock-target, --repeats, and --limit."""
        from llm_sentinel.runner.engine import main
        db_file = tmp_path / "cli_test.db"
        ret = main([
            "--mock-target",
            "--repeats", "2",
            "--limit", "4",
            "--db-path", str(db_file),
            "--concurrency", "2",
        ])
        assert ret == 0
        store = ResultStore(db_path=db_file)
        sessions = store.list_sessions()
        assert len(sessions) == 1
        rows = store.get_results_for_scan(sessions[0]["scan_id"])
        assert len(rows) == 4 * 2
