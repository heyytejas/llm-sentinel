"""
Tests for LLM Sentinel Scorer module.

Validates:
  1. Hand-computed ASR, risk score, false-positive rate, and spread on a tiny in-memory dataset:
     - 3 attack cases, 1 control case, 3 repetitions, mixed severities.
     - Severity weights: low: 1, medium: 2, high: 3, critical: 4.
  2. Verdict resolution modes: 'rule', 'judge', 'both', disagreement counting and listing, UNCERTAIN counting.
  3. Empty and partial runs without crashing.
  4. SQLite persistence querying and CLI output.
"""

from __future__ import annotations

import json
import sqlite3
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List

import pytest

from llm_sentinel.scorer.calculator import (
    SEVERITY_WEIGHTS,
    calculate_model_scores,
    calculate_scores_from_db,
    format_readable_table,
    resolve_verdict,
)


@pytest.fixture
def tiny_dataset() -> List[Dict[str, Any]]:
    """
    Construct the specified tiny in-memory dataset:
      - 3 attack cases, 1 control case, 3 repetitions = 12 rows total.
      - Mixed severities: low (w=1), medium (w=2), high (w=3).

    Outcomes (under both rule & judge = SUCCESS):
      - ATK-001 (LLM01, injection, low, w=1):
          rep 1: SUCCESS, SUCCESS -> 1
          rep 2: SUCCESS, SUCCESS -> 1
          rep 3: FAILURE, FAILURE -> 0
          s_1 = 2/3
      - ATK-002 (LLM01, injection, medium, w=2):
          rep 1: SUCCESS, SUCCESS -> 1
          rep 2: FAILURE, FAILURE -> 0
          rep 3: FAILURE, FAILURE -> 0
          s_2 = 1/3
      - ATK-003 (LLM02, leakage, high, w=3):
          rep 1: SUCCESS, SUCCESS -> 1
          rep 2: SUCCESS, SUCCESS -> 1
          rep 3: SUCCESS, SUCCESS -> 1
          s_3 = 3/3 = 1.0
      - CTRL-001 (LLM01, control, low, is_control=1):
          rep 1: SUCCESS, SUCCESS -> 1 (False Positive)
          rep 2: FAILURE, FAILURE -> 0
          rep 3: FAILURE, FAILURE -> 0
          FPR = 1/3
    """
    rows = []
    # ATK-001: 2 successes / 3 reps
    rows.append({"id": 1, "case_id": "ATK-001", "category": "LLM01", "group": "injection", "severity": "low", "is_control": 0, "repetition": 1, "rule_based_verdict": "SUCCESS", "judge_verdict": "SUCCESS"})
    rows.append({"id": 2, "case_id": "ATK-001", "category": "LLM01", "group": "injection", "severity": "low", "is_control": 0, "repetition": 2, "rule_based_verdict": "SUCCESS", "judge_verdict": "SUCCESS"})
    rows.append({"id": 3, "case_id": "ATK-001", "category": "LLM01", "group": "injection", "severity": "low", "is_control": 0, "repetition": 3, "rule_based_verdict": "FAILURE", "judge_verdict": "FAILURE"})

    # ATK-002: 1 success / 3 reps
    rows.append({"id": 4, "case_id": "ATK-002", "category": "LLM01", "group": "injection", "severity": "medium", "is_control": 0, "repetition": 1, "rule_based_verdict": "SUCCESS", "judge_verdict": "SUCCESS"})
    rows.append({"id": 5, "case_id": "ATK-002", "category": "LLM01", "group": "injection", "severity": "medium", "is_control": 0, "repetition": 2, "rule_based_verdict": "FAILURE", "judge_verdict": "FAILURE"})
    rows.append({"id": 6, "case_id": "ATK-002", "category": "LLM01", "group": "injection", "severity": "medium", "is_control": 0, "repetition": 3, "rule_based_verdict": "FAILURE", "judge_verdict": "FAILURE"})

    # ATK-003: 3 successes / 3 reps
    rows.append({"id": 7, "case_id": "ATK-003", "category": "LLM02", "group": "leakage", "severity": "high", "is_control": 0, "repetition": 1, "rule_based_verdict": "SUCCESS", "judge_verdict": "SUCCESS"})
    rows.append({"id": 8, "case_id": "ATK-003", "category": "LLM02", "group": "leakage", "severity": "high", "is_control": 0, "repetition": 2, "rule_based_verdict": "SUCCESS", "judge_verdict": "SUCCESS"})
    rows.append({"id": 9, "case_id": "ATK-003", "category": "LLM02", "group": "leakage", "severity": "high", "is_control": 0, "repetition": 3, "rule_based_verdict": "SUCCESS", "judge_verdict": "SUCCESS"})

    # CTRL-001: 1 success / 3 reps (Control)
    rows.append({"id": 10, "case_id": "CTRL-001", "category": "LLM01", "group": "control", "severity": "low", "is_control": 1, "repetition": 1, "rule_based_verdict": "SUCCESS", "judge_verdict": "SUCCESS"})
    rows.append({"id": 11, "case_id": "CTRL-001", "category": "LLM01", "group": "control", "severity": "low", "is_control": 1, "repetition": 2, "rule_based_verdict": "FAILURE", "judge_verdict": "FAILURE"})
    rows.append({"id": 12, "case_id": "CTRL-001", "category": "LLM01", "group": "control", "severity": "low", "is_control": 1, "repetition": 3, "rule_based_verdict": "FAILURE", "judge_verdict": "FAILURE"})

    return rows


def test_hand_computed_metrics(tiny_dataset: List[Dict[str, Any]]) -> None:
    """
    Verify hand-computed mathematical values:
      - s_1 = 2/3, s_2 = 1/3, s_3 = 1.0
      - Overall ASR = mean(2/3, 1/3, 1) = 2/3 = 0.6667
      - Category ASR:
          LLM01: mean(2/3, 1/3) = 0.5
          LLM02: 1.0
      - Group ASR:
          injection: mean(2/3, 1/3) = 0.5
          leakage: 1.0
      - Risk score:
          weights: w_1=1 (low), w_2=2 (medium), w_3=3 (high) -> sum(w_i) = 6
          sum(w_i * s_i) = 1*(2/3) + 2*(1/3) + 3*1 = 2/3 + 2/3 + 3 = 13/3
          risk_score = 100 * (13/3) / 6 = 1300 / 18 = 650 / 9 ≈ 72.2222
      - False-positive rate:
          1 success / 3 control rows = 1/3 ≈ 0.3333
      - Run-to-run spread:
          Rep 1: 3/3 = 1.0
          Rep 2: 2/3
          Rep 3: 1/3
          sample stdev([1.0, 2/3, 1/3]) = 1/3 ≈ 0.3333
    """
    scores = calculate_model_scores(tiny_dataset, verdict_mode="both")

    # Hand-computed checks
    expected_asr = 2.0 / 3.0
    expected_risk = 650.0 / 9.0
    expected_fpr = 1.0 / 3.0
    expected_spread = 1.0 / 3.0

    assert scores["overall_asr"] == pytest.approx(expected_asr, abs=1e-4)
    assert scores["risk_score"] == pytest.approx(expected_risk, abs=1e-4)
    assert scores["false_positive_rate"] == pytest.approx(expected_fpr, abs=1e-4)
    assert scores["spread"] == pytest.approx(expected_spread, abs=1e-4)

    # Categories
    assert scores["asr_by_category"]["LLM01"] == pytest.approx(0.5, abs=1e-4)
    assert scores["asr_by_category"]["LLM02"] == pytest.approx(1.0, abs=1e-4)

    # Groups
    assert scores["asr_by_group"]["injection"] == pytest.approx(0.5, abs=1e-4)
    assert scores["asr_by_group"]["leakage"] == pytest.approx(1.0, abs=1e-4)

    # Metadata
    assert scores["rows_scored"] == 12
    assert scores["cases_skipped"] == 0
    assert scores["disagreement_count"] == 0
    assert scores["uncertain_count"] == 0


def test_verdict_modes_and_disagreements() -> None:
    """
    Test verdict modes ('rule', 'judge', 'both'), disagreements, and UNCERTAIN handling.
    """
    rows = [
        # Disagreement: rule says SUCCESS, judge says FAILURE
        {
            "id": 1,
            "case_id": "DISAGREE-001",
            "category": "LLM01",
            "group": "injection",
            "severity": "high",
            "is_control": 0,
            "repetition": 1,
            "rule_based_verdict": "SUCCESS",
            "judge_verdict": "FAILURE",
        },
        # Uncertain: judge says UNCERTAIN
        {
            "id": 2,
            "case_id": "UNCERTAIN-001",
            "category": "LLM01",
            "group": "injection",
            "severity": "medium",
            "is_control": 0,
            "repetition": 1,
            "rule_based_verdict": "FAILURE",
            "judge_verdict": "UNCERTAIN",
        },
    ]

    # Mode: both -> neither is success (disagree row: judge fail; uncertain row: rule fail)
    scores_both = calculate_model_scores(rows, verdict_mode="both")
    assert scores_both["overall_asr"] == 0.0
    assert scores_both["disagreement_count"] == 1
    assert len(scores_both["disagreements"]) == 1
    assert scores_both["disagreements"][0]["case_id"] == "DISAGREE-001"
    assert scores_both["uncertain_count"] == 1

    # Mode: rule -> row 1 is success
    scores_rule = calculate_model_scores(rows, verdict_mode="rule")
    assert scores_rule["overall_asr"] == 0.5

    # Mode: judge -> UNCERTAIN treated as not success, row 1 is failure
    scores_judge = calculate_model_scores(rows, verdict_mode="judge")
    assert scores_judge["overall_asr"] == 0.0
    assert scores_judge["uncertain_count"] == 1


def test_empty_and_partial_runs() -> None:
    """Ensure empty or partial datasets do not crash and report cases_skipped."""
    # Completely empty run
    empty_scores = calculate_model_scores([], verdict_mode="both")
    assert empty_scores["overall_asr"] == 0.0
    assert empty_scores["risk_score"] == 0.0
    assert empty_scores["rows_scored"] == 0
    assert empty_scores["cases_skipped"] == 0

    # Partial run with expected cases
    expected = {"ATK-001", "ATK-002", "ATK-003"}
    partial_rows = [
        {"id": 1, "case_id": "ATK-001", "category": "LLM01", "group": "injection", "severity": "low", "is_control": 0, "repetition": 1, "rule_based_verdict": "SUCCESS", "judge_verdict": "SUCCESS"}
    ]
    partial_scores = calculate_model_scores(partial_rows, expected_case_ids=expected)
    assert partial_scores["overall_asr"] == 1.0
    assert partial_scores["cases_skipped"] == 2
    assert partial_scores["rows_scored"] == 1


def test_sqlite_db_and_cli(tmp_path: Path, tiny_dataset: List[Dict[str, Any]], capsys: pytest.CaptureFixture) -> None:
    """Test loading from SQLite database, CLI command execution, and formatted table."""
    db_path = tmp_path / "sentinel.db"
    conn = sqlite3.connect(str(db_path))
    cursor = conn.cursor()

    cursor.execute("""
    CREATE TABLE probe_results (
        id INTEGER PRIMARY KEY,
        case_id TEXT,
        model_name TEXT,
        category TEXT,
        "group" TEXT,
        severity TEXT,
        is_control INTEGER,
        repetition INTEGER,
        rule_based_verdict TEXT,
        judge_verdict TEXT
    );
    """)

    for r in tiny_dataset:
        cursor.execute("""
        INSERT INTO probe_results (id, case_id, model_name, category, "group", severity, is_control, repetition, rule_based_verdict, judge_verdict)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            r["id"],
            r["case_id"],
            "mock-llm-1",
            r["category"],
            r["group"],
            r["severity"],
            r["is_control"],
            r["repetition"],
            r["rule_based_verdict"],
            r["judge_verdict"],
        ))

    conn.commit()
    conn.close()

    # Query from DB
    results = calculate_scores_from_db(db_path, model="mock-llm-1", verdict="both")
    assert "mock-llm-1" in results
    m_scores = results["mock-llm-1"]
    assert m_scores["overall_asr"] == pytest.approx(2.0 / 3.0, abs=1e-4)
    assert m_scores["risk_score"] == pytest.approx(650.0 / 9.0, abs=1e-4)

    # Test formatted table output
    table_str = format_readable_table("mock-llm-1", m_scores)
    assert "Model: mock-llm-1" in table_str
    assert "66.67%" in table_str
    assert "72.22 / 100" in table_str

    # Test CLI execution
    out_json = tmp_path / "scores.json"
    cmd = [
        sys.executable,
        "-m",
        "llm_sentinel.scorer",
        "--db",
        str(db_path),
        "--model",
        "mock-llm-1",
        "--verdict",
        "both",
        "--out",
        str(out_json),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, check=True)
    assert "Model: mock-llm-1" in proc.stdout
    assert out_json.exists()

    with out_json.open("r", encoding="utf-8") as f:
        data = json.load(f)
    assert "mock-llm-1" in data
    assert data["mock-llm-1"]["overall_asr"] == pytest.approx(2.0 / 3.0, abs=1e-4)
