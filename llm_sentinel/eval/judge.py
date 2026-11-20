"""
Batch LLM judge execution module for LLM Sentinel.
"""

from __future__ import annotations

import csv
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from ..adapters.adapter import BaseAdapter, get_adapter
from ..attacks.loader import load_library
from ..detectors.judge import JudgeDetector
from ..runner.store import ResultStore

logger = logging.getLogger(__name__)


def run_judge(
    model: str,
    all_attacks: bool = False,
    sample_path: Union[str, Path] = "reports/label_sample.csv",
    db_path: Optional[Union[str, Path]] = None,
    adapter: Optional[BaseAdapter] = None,
) -> int:
    """
    Judge stored rows that have no judge_verdict.

    Parameters
    ----------
    model       : Judge model name.
    all_attacks : If True, judges every attack row missing judge_verdict.
                  If False, judges only rows listed in sample_path.
    sample_path : CSV file containing row_ids to judge (when all_attacks=False).
    db_path     : SQLite database path.
    adapter     : Optional target adapter to use. Defaults to mock adapter with model.
    """
    store = ResultStore(db_path=db_path)

    # Load attack cases to retrieve judge_goal metadata
    case_map: Dict[str, Any] = {}
    try:
        attacks, controls = load_library()
        for c in attacks + controls:
            case_map[c.id] = c
    except Exception as exc:
        logger.warning("Could not load attack cases for judge goals: %s", exc)

    if adapter is None:
        adapter = get_adapter("mock", model=model)

    judge_detector = JudgeDetector(adapter=adapter, model=model)

    # Determine candidate rows
    with store._connect() as conn:
        if all_attacks:
            cursor = conn.execute(
                """
                SELECT id, case_id, response, query
                FROM probe_results
                WHERE (is_control = 0 OR is_control IS NULL)
                  AND (judge_verdict IS NULL OR judge_verdict = '')
                ORDER BY id
                """
            )
            rows = [dict(r) for r in cursor.fetchall()]
        else:
            sample_file = Path(sample_path)
            if not sample_file.exists():
                raise FileNotFoundError(
                    f"Sample CSV not found at '{sample_file}'. Run export-sample first or use --all."
                )

            row_ids: List[int] = []
            with sample_file.open("r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for r in reader:
                    if r.get("row_id"):
                        row_ids.append(int(r["row_id"]))

            if not row_ids:
                logger.info("No row_ids found in sample CSV.")
                return 0

            placeholders = ",".join("?" for _ in row_ids)
            cursor = conn.execute(
                f"""
                SELECT id, case_id, response, query
                FROM probe_results
                WHERE id IN ({placeholders})
                  AND (judge_verdict IS NULL OR judge_verdict = '')
                ORDER BY id
                """,
                row_ids,
            )
            rows = [dict(r) for r in cursor.fetchall()]

    if not rows:
        print("No unjudged rows found matching criteria.")
        return 0

    print(f"Judging {len(rows)} row(s) using model '{model}'...")

    judged_count = 0
    for r in rows:
        case = case_map.get(r["case_id"])
        response_text = r.get("response") or ""

        verdict, reason = judge_detector.judge(case=case, response=response_text)
        store.update_judge_result(row_id=r["id"], verdict=verdict, reason=reason)
        judged_count += 1
        logger.debug("Judged row %d: %s (%s)", r["id"], verdict, reason)

    print(f"Successfully judged {judged_count} row(s).")
    return judged_count
