"""
Tests for LLM Sentinel Dashboard data layer, components, and seeding.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any, Dict, List
import pandas as pd
import pytest

from llm_sentinel.dashboard.data import (
    calculate_dashboard_metrics,
    filter_dataframe,
    get_all_models,
    get_all_sessions,
    get_results_dataframe,
    load_attack_library_cached,
    seed_demo_runs,
)
from llm_sentinel.dashboard.components import (
    build_category_asr_chart,
    build_group_asr_chart,
    build_model_comparison_chart,
    build_position_analysis_chart,
    build_type_breakdown_chart,
    render_severity_badge,
    render_verdict_badge,
)
from llm_sentinel.runner.store import ResultStore


@pytest.fixture
def mock_db_with_runs(tmp_path: Path) -> Path:
    """Create a temporary SQLite database with 2 mock scan sessions and probe results."""
    db_path = tmp_path / "test_dash.db"
    store = ResultStore(db_path=db_path)

    with store._connect() as conn:
        conn.execute("""
            INSERT INTO scan_sessions (scan_id, model, started_at, finished_at, total, succeeded, errors)
            VALUES
                ('scan-1', 'model-baseline', '2026-11-20T10:00:00', '2026-11-20T10:01:00', 4, 3, 0),
                ('scan-2', 'model-defended', '2026-11-20T11:00:00', '2026-11-20T11:01:00', 4, 1, 0)
        """)

        conn.execute("""
            INSERT INTO probe_results
                (scan_id, attack_id, case_id, attack_name, category, "group", attack_group, attack_type, "type", severity, is_control, repetition, model, model_name, prompt, status, timestamp, rule_based_verdict, judge_verdict)
            VALUES
                ('scan-1', 'PI-001', 'PI-001', 'Direct Injection', 'LLM01', 'direct_injection', 'direct_injection', 'direct', 'direct', 'high', 0, 1, 'model-baseline', 'model-baseline', 'ignore all', 'SUCCESS', '2026-11-20T10:00:10', 'SUCCESS', 'SUCCESS'),
                ('scan-1', 'II-001', 'II-001', 'Indirect Injection', 'LLM01', 'indirect_injection', 'indirect_injection', 'indirect', 'indirect', 'high', 0, 1, 'model-baseline', 'model-baseline', 'summarize email', 'SUCCESS', '2026-11-20T10:00:20', 'SUCCESS', 'SUCCESS'),
                ('scan-1', 'CP-001', 'CP-001', 'Corpus Poisoning', 'LLM04', 'corpus_poisoning', 'corpus_poisoning', 'poisoning', 'poisoning', 'medium', 0, 1, 'model-baseline', 'model-baseline', 'when are exams', 'SUCCESS', '2026-11-20T10:00:30', 'SUCCESS', 'SUCCESS'),
                ('scan-1', 'CT-001', 'CT-001', 'Exam query', 'LLM01', 'control', 'control', 'direct', 'direct', 'low', 1, 1, 'model-baseline', 'model-baseline', 'when are exams', 'SUCCESS', '2026-11-20T10:00:40', 'FAILURE', 'FAILURE'),

                ('scan-2', 'PI-001', 'PI-001', 'Direct Injection', 'LLM01', 'direct_injection', 'direct_injection', 'direct', 'direct', 'high', 0, 1, 'model-defended', 'model-defended', 'ignore all', 'SUCCESS', '2026-11-20T11:00:10', 'FAILURE', 'FAILURE'),
                ('scan-2', 'II-001', 'II-001', 'Indirect Injection', 'LLM01', 'indirect_injection', 'indirect_injection', 'indirect', 'indirect', 'high', 0, 1, 'model-defended', 'model-defended', 'summarize email', 'SUCCESS', '2026-11-20T11:00:20', 'FAILURE', 'FAILURE'),
                ('scan-2', 'CP-001', 'CP-001', 'Corpus Poisoning', 'LLM04', 'corpus_poisoning', 'corpus_poisoning', 'poisoning', 'poisoning', 'medium', 0, 1, 'model-defended', 'model-defended', 'when are exams', 'SUCCESS', '2026-11-20T11:00:30', 'SUCCESS', 'SUCCESS'),
                ('scan-2', 'CT-001', 'CT-001', 'Exam query', 'LLM01', 'control', 'control', 'direct', 'direct', 'low', 1, 1, 'model-defended', 'model-defended', 'when are exams', 'SUCCESS', '2026-11-20T11:00:40', 'FAILURE', 'FAILURE')
        """)

    return db_path


def test_data_sessions_and_models(mock_db_with_runs: Path):
    """Test retrieving scan sessions and distinct models."""
    sessions = get_all_sessions(mock_db_with_runs)
    assert len(sessions) == 2
    models = get_all_models(mock_db_with_runs)
    assert set(models) == {"model-baseline", "model-defended"}


def test_get_results_dataframe_and_filtering(mock_db_with_runs: Path):
    """Test dataframe creation, type conversions, and filtering helpers."""
    df = get_results_dataframe(db_path=mock_db_with_runs)
    assert len(df) == 8
    assert "case_id" in df.columns
    assert "severity" in df.columns

    # Filter by model
    df_base = get_results_dataframe(model="model-baseline", db_path=mock_db_with_runs)
    assert len(df_base) == 4

    # Filter by category
    filtered = filter_dataframe(df, categories=["LLM04"])
    assert len(filtered) == 2
    assert set(filtered["case_id"]) == {"CP-001"}

    # Filter by outcome
    succeeded = filter_dataframe(df, outcome="Succeeded (Breached)", verdict_mode="both")
    assert len(succeeded) == 4  # 3 in baseline + 1 in defended


def test_calculate_dashboard_metrics(mock_db_with_runs: Path):
    """Test computing dashboard metrics reusing scorer calculation."""
    df_base = get_results_dataframe(model="model-baseline", db_path=mock_db_with_runs)
    scores = calculate_dashboard_metrics(df_base, verdict_mode="both")

    assert scores["overall_asr"] == pytest.approx(1.0, abs=1e-4)  # 3 attacks all breached
    assert scores["false_positive_rate"] == 0.0  # Control safe
    assert scores["total_attacks"] == 3
    assert scores["succeeded_attacks"] == 3
    assert scores["control_count"] == 1


def test_components_chart_builders(mock_db_with_runs: Path):
    """Test that all Plotly chart builders run and return valid figures without error."""
    df = get_results_dataframe(db_path=mock_db_with_runs)
    scores = calculate_dashboard_metrics(df, verdict_mode="both")

    fig1 = build_category_asr_chart(scores.get("asr_by_category", {}))
    assert fig1 is not None

    fig2 = build_group_asr_chart(scores.get("asr_by_group", {}))
    assert fig2 is not None

    fig3 = build_type_breakdown_chart(df, verdict_mode="both")
    assert fig3 is not None

    fig4 = build_position_analysis_chart(df, verdict_mode="both")
    assert fig4 is not None

    fig5 = build_model_comparison_chart(
        cat_a={"LLM01": 1.0, "LLM04": 1.0},
        cat_b={"LLM01": 0.0, "LLM04": 1.0},
        name_a="Baseline",
        name_b="Defended",
    )
    assert fig5 is not None

    # HTML badges
    b1 = render_severity_badge("critical")
    assert "CRITICAL" in b1
    b2 = render_verdict_badge("SUCCESS", is_attack=True)
    assert "BREACHED" in b2


def test_load_attack_library():
    """Verify attack library loading returns active attacks and controls."""
    attacks, controls = load_attack_library_cached()
    assert len(attacks) == 48
    assert len(controls) == 10


def test_seed_demo_runs_creates_data(tmp_path: Path):
    """Verify seed_demo_runs seeds 2 sessions with results in an empty database."""
    test_db = tmp_path / "seeded.db"
    scan_a, scan_b = seed_demo_runs(db_path=test_db)
    assert scan_a != scan_b

    sessions = get_all_sessions(test_db)
    assert len(sessions) == 2

    df = get_results_dataframe(db_path=test_db)
    assert not df.empty
    models = get_all_models(test_db)
    assert "demo-undefended-baseline" in models
    assert "demo-hardened-sentinel" in models
