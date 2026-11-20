"""
Tests for RagTarget and Runner RAG integration.

Validates:
  (a) Injected text appears in the prompt sent to the backend for an indirect case.
  (b) The canary is in the system prompt sent to the backend.
  (c) After a run the index is back to its baseline.
  (d) A poisoning case retrieves its injected chunk for its own query.
  (e) Target canaries extraction and verdict normalization (SUCCESS | FAILURE | ERROR).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional
import pytest

from llm_sentinel.adapters.adapter import BaseAdapter
from llm_sentinel.attacks.loader import load_library
from llm_sentinel.runner.engine import ScanEngine
from llm_sentinel.runner.store import ResultStore, migrate_verdicts
from llm_sentinel.target.rag_app import RagTarget, SYSTEM_CANARY, CORPUS_CANARY


class RecordingBackend(BaseAdapter):
    """Mock backend that records prompts and system prompts received."""

    def __init__(self, model: str = "mock-recording-model") -> None:
        super().__init__(model=model)
        self.recorded_calls: List[Dict[str, Optional[str]]] = []

    def send(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        self.recorded_calls.append({
            "prompt": prompt,
            "system_prompt": system_prompt,
        })
        return "I am a recorded response from the campus assistant."


@pytest.fixture
def recording_rag_target() -> tuple[RagTarget, RecordingBackend]:
    backend = RecordingBackend()
    rag = RagTarget(backend_adapter=backend)
    return rag, backend


def test_rag_target_indirect_injection_and_canary(recording_rag_target, tmp_path):
    """
    Validates:
      (a) the injected text appears in the prompt sent to the backend for an indirect case;
      (b) the canary is in the system prompt.
    """
    rag, backend = recording_rag_target

    # Load attack cases
    attacks, _ = load_library()
    indirect_cases = [c for c in attacks if c.type == "indirect"]
    assert len(indirect_cases) > 0, "No indirect cases found in library"
    case = indirect_cases[0]

    engine = ScanEngine(
        adapter=rag,
        db_path=tmp_path / "test_rag.db",
        concurrency=1,
        repeats=1,
    )

    # Execute single indirect probe
    result = engine._execute_probe(case, scan_id="test-indirect-scan", repetition=1)

    assert len(backend.recorded_calls) >= 1
    last_call = backend.recorded_calls[-1]

    # (a) injected text appears in the prompt sent to backend
    assert case.injection in last_call["prompt"] or any(p in last_call["prompt"] for p in getattr(case, "patterns", []))

    # (b) the canary is in the system prompt
    assert last_call["system_prompt"] is not None
    assert rag.canary in last_call["system_prompt"]


def test_rag_target_index_baseline_restored_after_run(recording_rag_target, tmp_path):
    """
    Validates:
      (c) after a run the index is back to its baseline.
    """
    rag, backend = recording_rag_target

    baseline_docs = dict(rag.baseline_docs)
    baseline_corpus_keys = set(rag.corpus.keys())
    baseline_chunk_count = rag.collection.count()
    assert baseline_chunk_count > 0, "Baseline collection must contain chunks"

    attacks, _ = load_library()
    # Run 1 indirect, 1 poisoning, and 1 direct case
    selected_cases = [
        next(c for c in attacks if c.type == "indirect"),
        next(c for c in attacks if c.type == "poisoning"),
        next(c for c in attacks if c.type == "direct"),
    ]

    engine = ScanEngine(
        adapter=rag,
        db_path=tmp_path / "test_baseline.db",
        concurrency=1,
        repeats=1,
    )

    for case in selected_cases:
        engine._execute_probe(case, scan_id="test-baseline-scan", repetition=1)

    # (c) After run, active corpus matches baseline, chunk count matches baseline
    assert set(rag.corpus.keys()) == baseline_corpus_keys
    assert rag.corpus == baseline_docs
    assert rag.collection.count() == baseline_chunk_count


def test_rag_target_poisoning_case_retrieves_injected_chunk(recording_rag_target):
    """
    Validates:
      (d) a poisoning case retrieves its injected chunk for its own query.
    """
    rag, backend = recording_rag_target

    attacks, _ = load_library()
    poisoning_cases = [c for c in attacks if c.type == "poisoning"]
    assert len(poisoning_cases) > 0, "No poisoning cases found in library"
    case = poisoning_cases[0]

    doc_id = f"poisoning_{case.id}"
    poisoned_text = case.setup_document

    # Add document and reindex
    rag.add_document(doc_id, poisoned_text)
    rag.reindex()

    try:
        # Retrieve chunks for user_query
        retrieved_chunks = rag.get_retrieved_chunks(case.user_query)
        # Verify the injected setup_document or unique pattern was retrieved
        pattern = case.patterns[0]
        assert any(pattern in chunk or poisoned_text in chunk for chunk in retrieved_chunks), (
            f"Expected pattern '{pattern}' in retrieved chunks: {retrieved_chunks}"
        )
    finally:
        # Clean up
        rag.remove_document(doc_id)
        rag.reindex()


def test_verdict_normalisation_and_migration(tmp_path):
    """
    Test that ResultStore normalizes rule_based_verdict to SUCCESS | FAILURE | ERROR
    and migrates existing legacy rows (FAIL -> SUCCESS, PASS -> FAILURE).
    """
    db_path = tmp_path / "legacy.db"
    store = ResultStore(db_path=db_path)

    # Insert session and raw legacy verdicts
    with store._connect() as conn:
        conn.execute("INSERT INTO scan_sessions (scan_id, model, started_at) VALUES ('s1', 'm1', '2026-11-20T00:00:00');")
        conn.execute("""
            INSERT INTO probe_results
                (scan_id, attack_id, attack_name, category, attack_type, severity, model, prompt, status, timestamp, rule_based_verdict)
            VALUES
                ('s1', 'ATK-1', 'Name', 'LLM01', 'direct', 'high', 'm1', 'p1', 'SUCCESS', '2026-11-20T00:00:00', 'FAIL'),
                ('s1', 'ATK-2', 'Name', 'LLM01', 'direct', 'low', 'm1', 'p2', 'SUCCESS', '2026-11-20T00:00:00', 'PASS'),
                ('s1', 'ATK-3', 'Name', 'LLM01', 'direct', 'low', 'm1', 'p3', 'ERROR', '2026-11-20T00:00:00', 'ERROR')
        """)

    # Verify automatic migration during init / via migrate_verdicts
    with store._connect() as conn:
        conn.execute("UPDATE probe_results SET rule_based_verdict = 'FAIL' WHERE attack_id = 'ATK-1';")
        conn.execute("UPDATE probe_results SET rule_based_verdict = 'PASS' WHERE attack_id = 'ATK-2';")

    updated = store.migrate_legacy_verdicts()
    assert updated == 2

    # Verify rows in DB
    with store._connect() as conn:
        rows = conn.execute("SELECT attack_id, rule_based_verdict FROM probe_results ORDER BY id").fetchall()
        verdicts = {r["attack_id"]: r["rule_based_verdict"] for r in rows}

    assert verdicts["ATK-1"] == "SUCCESS"
    assert verdicts["ATK-2"] == "FAILURE"
    assert verdicts["ATK-3"] == "ERROR"
