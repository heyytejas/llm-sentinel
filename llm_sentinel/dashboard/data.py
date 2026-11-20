"""
Data access and persistence helpers for LLM Sentinel Dashboard.

Reuses ResultStore, calculate_model_scores, and load_library directly.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import pandas as pd

from ..attacks.loader import load_library
from ..attacks.models import SeedAttackCase
from ..runner.engine import ScanEngine
from ..runner.store import ResultStore
from ..scorer.calculator import calculate_model_scores

logger = logging.getLogger(__name__)

_DEFAULT_DB_PATH = Path(__file__).resolve().parents[2] / "reports" / "sentinel.db"


def get_store(db_path: Optional[Union[str, Path]] = None) -> ResultStore:
    """Return a ResultStore instance."""
    return ResultStore(db_path=db_path or _DEFAULT_DB_PATH)


def get_all_sessions(db_path: Optional[Union[str, Path]] = None) -> List[Dict[str, Any]]:
    """Retrieve all scan sessions sorted by started_at DESC."""
    store = get_store(db_path)
    rows = store.list_sessions()
    return [dict(r) for r in rows]


def get_all_models(db_path: Optional[Union[str, Path]] = None) -> List[str]:
    """Retrieve all distinct model identifiers recorded in probe_results."""
    store = get_store(db_path)
    with store._connect() as conn:
        cursor = conn.execute(
            """
            SELECT DISTINCT COALESCE(model_name, model) AS model_id
            FROM probe_results
            WHERE COALESCE(model_name, model) IS NOT NULL
            ORDER BY model_id
            """
        )
        return [row["model_id"] for row in cursor.fetchall() if row["model_id"]]


def get_results_dataframe(
    scan_id: Optional[str] = None,
    model: Optional[str] = None,
    db_path: Optional[Union[str, Path]] = None,
) -> pd.DataFrame:
    """
    Fetch probe results matching scan_id and/or model into a clean pandas DataFrame.
    """
    store = get_store(db_path)
    query = """
        SELECT
            id,
            scan_id,
            COALESCE(case_id, attack_id) AS case_id,
            attack_name,
            category,
            COALESCE("group", attack_group) AS "group",
            COALESCE("type", attack_type) AS "type",
            severity,
            is_control,
            COALESCE(repetition, repetition_number, 1) AS repetition,
            COALESCE(model_name, model) AS model,
            prompt,
            query,
            response,
            status,
            rule_based_verdict,
            judge_verdict,
            judge_reason,
            latency_ms,
            timestamp,
            retrieved_chunks
        FROM probe_results
        WHERE 1=1
    """
    params: List[Any] = []
    if scan_id and scan_id != "all":
        query += " AND scan_id = ?"
        params.append(scan_id)
    if model and model != "all":
        query += " AND COALESCE(model_name, model) = ?"
        params.append(model)

    query += " ORDER BY id ASC"

    with store._connect() as conn:
        cursor = conn.execute(query, params)
        rows = [dict(r) for r in cursor.fetchall()]

    if not rows:
        return pd.DataFrame(
            columns=[
                "id", "scan_id", "case_id", "attack_name", "category", "group",
                "type", "severity", "is_control", "repetition", "model",
                "prompt", "query", "response", "status", "rule_based_verdict",
                "judge_verdict", "judge_reason", "latency_ms", "timestamp", "retrieved_chunks"
            ]
        )

    df = pd.DataFrame(rows)
    # Ensure consistent data types
    df["is_control"] = df["is_control"].apply(lambda v: bool(int(v)) if pd.notnull(v) else False)
    df["severity"] = df["severity"].fillna("medium").str.lower()
    df["category"] = df["category"].fillna("UNKNOWN")
    df["group"] = df["group"].fillna("unknown")
    df["type"] = df["type"].fillna("direct")
    df["rule_based_verdict"] = df["rule_based_verdict"].fillna("ERROR").str.upper()
    df["judge_verdict"] = df["judge_verdict"].fillna("NONE").str.upper()

    return df


def calculate_dashboard_metrics(
    df: pd.DataFrame,
    verdict_mode: str = "both",
) -> Dict[str, Any]:
    """
    Compute overall and breakdown metrics reusing calculate_model_scores.
    """
    if df.empty:
        return {
            "overall_asr": 0.0,
            "risk_score": 0.0,
            "asr_by_category": {},
            "asr_by_group": {},
            "false_positive_rate": 0.0,
            "spread": 0.0,
            "disagreement_count": 0,
            "uncertain_count": 0,
            "rows_scored": 0,
            "total_attacks": 0,
            "succeeded_attacks": 0,
            "failed_attacks": 0,
            "control_count": 0,
            "control_fp_count": 0,
        }

    rows = df.to_dict(orient="records")
    scores = calculate_model_scores(rows, verdict_mode=verdict_mode)

    # Attack counts
    attack_df = df[~df["is_control"]]
    control_df = df[df["is_control"]]

    if verdict_mode == "rule":
        succeeded_series = attack_df["rule_based_verdict"].isin(["SUCCESS", "FAIL"])
        control_fp_series = control_df["rule_based_verdict"].isin(["SUCCESS", "FAIL"])
    elif verdict_mode == "judge":
        succeeded_series = attack_df["judge_verdict"] == "SUCCESS"
        control_fp_series = control_df["judge_verdict"] == "SUCCESS"
    else:  # both
        succeeded_series = (attack_df["rule_based_verdict"].isin(["SUCCESS", "FAIL"])) & (attack_df["judge_verdict"] == "SUCCESS")
        control_fp_series = (control_df["rule_based_verdict"].isin(["SUCCESS", "FAIL"])) & (control_df["judge_verdict"] == "SUCCESS")

    scores["total_attacks"] = len(attack_df)
    scores["succeeded_attacks"] = int(succeeded_series.sum())
    scores["failed_attacks"] = scores["total_attacks"] - scores["succeeded_attacks"]
    scores["control_count"] = len(control_df)
    scores["control_fp_count"] = int(control_fp_series.sum())

    return scores


def filter_dataframe(
    df: pd.DataFrame,
    categories: Optional[List[str]] = None,
    groups: Optional[List[str]] = None,
    types: Optional[List[str]] = None,
    severities: Optional[List[str]] = None,
    outcome: Optional[str] = "All",
    verdict_mode: str = "both",
) -> pd.DataFrame:
    """Filter DataFrame based on selected sidebar criteria."""
    if df.empty:
        return df

    filtered = df.copy()

    if categories:
        filtered = filtered[filtered["category"].isin(categories)]
    if groups:
        filtered = filtered[filtered["group"].isin(groups)]
    if types:
        filtered = filtered[filtered["type"].isin(types)]
    if severities:
        filtered = filtered[filtered["severity"].isin(severities)]

    if outcome and outcome != "All":
        if outcome == "Succeeded (Breached)":
            if verdict_mode == "rule":
                filtered = filtered[filtered["rule_based_verdict"].isin(["SUCCESS", "FAIL"])]
            elif verdict_mode == "judge":
                filtered = filtered[filtered["judge_verdict"] == "SUCCESS"]
            else:
                filtered = filtered[(filtered["rule_based_verdict"].isin(["SUCCESS", "FAIL"])) & (filtered["judge_verdict"] == "SUCCESS")]
        elif outcome == "Blocked (Safe)":
            if verdict_mode == "rule":
                filtered = filtered[filtered["rule_based_verdict"].isin(["FAILURE", "PASS"])]
            elif verdict_mode == "judge":
                filtered = filtered[filtered["judge_verdict"].isin(["FAILURE", "NONE"])]
            else:
                filtered = filtered[~((filtered["rule_based_verdict"].isin(["SUCCESS", "FAIL"])) & (filtered["judge_verdict"] == "SUCCESS"))]
        elif outcome == "Error":
            filtered = filtered[filtered["rule_based_verdict"] == "ERROR"]

    return filtered


def load_attack_library_cached() -> Tuple[List[SeedAttackCase], List[SeedAttackCase]]:
    """Load the 48 active attack cases and 10 control cases."""
    return load_library()


def seed_demo_runs(db_path: Optional[Union[str, Path]] = None) -> Tuple[str, str]:
    """
    Seed two realistic demo benchmark runs into sentinel.db:
      1. 'demo-undefended-baseline': High ASR, leaks canaries and system prompts.
      2. 'demo-hardened-sentinel': Low ASR, safely refuses adversarial injections.

    Returns the tuple of scan_ids (scan_id_baseline, scan_id_hardened).
    """
    from ..demo.chatbot import DemoChatbot

    target_db = db_path or _DEFAULT_DB_PATH

    # Run 1: Undefended baseline
    adapter_vulnerable = DemoChatbot(
        vulnerable=True,
        model="demo-undefended-baseline",
    )
    engine_vulnerable = ScanEngine(
        adapter=adapter_vulnerable,
        db_path=target_db,
        concurrency=4,
        repeats=1,
    )
    session_vuln = engine_vulnerable.run()

    # Run 2: Hardened model
    adapter_defended = DemoChatbot(
        vulnerable=False,
        model="demo-hardened-sentinel",
    )
    engine_defended = ScanEngine(
        adapter=adapter_defended,
        db_path=target_db,
        concurrency=4,
        repeats=1,
    )
    session_def = engine_defended.run()

    return session_vuln.scan_id, session_def.scan_id
