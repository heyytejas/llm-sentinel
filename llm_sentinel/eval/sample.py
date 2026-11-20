"""
Sample extraction module for human annotation and validation.
"""

from __future__ import annotations

import csv
import logging
import random
from pathlib import Path
from typing import Dict, List, Optional, Union

from ..runner.store import ResultStore

logger = logging.getLogger(__name__)


def export_sample(
    n: int = 30,
    seed: int = 42,
    db_path: Optional[Union[str, Path]] = None,
    output_path: Union[str, Path] = "reports/label_sample.csv",
) -> List[Dict[str, Any]]:
    """
    Export a stratified sample of rows for annotation.

    Criteria:
      - Repetition 1 only
      - Exactly 3 controls + (n - 3) attacks
      - At least a third of rows where rule_based_verdict is SUCCESS (if enough exist)
      - The rest spread across categories
      - CSV columns: row_id, case_id, category, response, label_a, label_b
    """
    store = ResultStore(db_path=db_path)
    rng = random.Random(seed)

    with store._connect() as conn:
        cursor = conn.execute(
            """
            SELECT id AS row_id, case_id, category, response, is_control, rule_based_verdict
            FROM probe_results
            WHERE repetition = 1 OR repetition_number = 1
            ORDER BY id
            """
        )
        all_rows = [dict(r) for r in cursor.fetchall()]

    if not all_rows:
        raise ValueError("No repetition=1 rows found in probe_results database.")

    control_rows = [r for r in all_rows if r.get("is_control") == 1]
    attack_rows = [r for r in all_rows if not r.get("is_control")]

    # Sample 3 controls
    n_controls = min(3, len(control_rows))
    selected_controls = rng.sample(control_rows, n_controls) if control_rows else []

    target_attacks = n - len(selected_controls)
    if target_attacks <= 0:
        selected_attacks = []
    else:
        # Separate attacks into SUCCESS (attack succeeded) and other
        success_attacks = [
            r for r in attack_rows
            if str(r.get("rule_based_verdict", "")).strip().upper() in ("SUCCESS", "FAIL")
        ]
        other_attacks = [r for r in attack_rows if r not in success_attacks]

        # At least a third of target rows where rule_based_verdict is SUCCESS (if enough exist)
        min_success = target_attacks // 3
        n_success_pick = min(min_success, len(success_attacks))

        rng.shuffle(success_attacks)
        selected_success = success_attacks[:n_success_pick]

        # Remaining attack slots
        needed_remaining = target_attacks - len(selected_success)
        remaining_pool = success_attacks[n_success_pick:] + other_attacks

        # Stratify remaining by category
        by_category: Dict[str, List[Dict[str, Any]]] = {}
        for r in remaining_pool:
            by_category.setdefault(r["category"], []).append(r)

        for cat_list in by_category.values():
            rng.shuffle(cat_list)

        selected_remaining: List[Dict[str, Any]] = []
        categories = sorted(list(by_category.keys()))

        # Round-robin across categories
        idx = 0
        while len(selected_remaining) < needed_remaining and any(by_category.values()):
            cat = categories[idx % len(categories)]
            if by_category[cat]:
                selected_remaining.append(by_category[cat].pop(0))
            idx += 1

        selected_attacks = selected_success + selected_remaining

    selected_rows = selected_attacks + selected_controls

    # Order rows deterministically
    selected_rows.sort(key=lambda r: r["row_id"])

    # Write CSV
    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    with out_file.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["row_id", "case_id", "category", "response", "label_a", "label_b"])
        for r in selected_rows:
            writer.writerow([r["row_id"], r["case_id"], r["category"], r.get("response") or "", "", ""])

    logger.info("Exported %d sample rows to %s", len(selected_rows), out_file)
    print(f"Exported {len(selected_rows)} sample rows to {out_file}")

    return selected_rows
