"""
Evaluation metrics calculation for LLM Sentinel detectors.
"""

from __future__ import annotations

import csv
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from ..runner.store import ResultStore

logger = logging.getLogger(__name__)


def calculate_binary_metrics(y_true: List[str], y_pred: List[str]) -> Dict[str, Any]:
    """
    Compute confusion matrix, accuracy, precision, recall, F1, and Cohen's kappa.
    Positive class is "SUCCESS", negative class is "FAILURE".
    """
    tp = sum(1 for yt, yp in zip(y_true, y_pred) if yt == "SUCCESS" and yp == "SUCCESS")
    fp = sum(1 for yt, yp in zip(y_true, y_pred) if yt == "FAILURE" and yp == "SUCCESS")
    tn = sum(1 for yt, yp in zip(y_true, y_pred) if yt == "FAILURE" and yp == "FAILURE")
    fn = sum(1 for yt, yp in zip(y_true, y_pred) if yt == "SUCCESS" and yp == "FAILURE")

    total = len(y_true)
    accuracy = (tp + tn) / total if total > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    # Cohen's Kappa
    if total > 0:
        po = (tp + tn) / total
        p_act_pos = (tp + fn) / total
        p_act_neg = (tn + fp) / total
        p_pred_pos = (tp + fp) / total
        p_pred_neg = (tn + fn) / total
        pe = (p_act_pos * p_pred_pos) + (p_act_neg * p_pred_neg)
        if (1.0 - pe) != 0:
            kappa = (po - pe) / (1.0 - pe)
        else:
            kappa = 1.0 if po == 1.0 else 0.0
    else:
        kappa = 0.0

    return {
        "confusion_matrix": {"TP": tp, "FP": fp, "TN": tn, "FN": fn},
        "accuracy": round(accuracy, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "cohens_kappa": round(kappa, 4),
    }


def calculate_cohens_kappa(rater_a: List[str], rater_b: List[str]) -> float:
    """Compute Cohen's kappa between two annotators."""
    if not rater_a or len(rater_a) != len(rater_b):
        return 0.0

    total = len(rater_a)
    categories = sorted(list(set(rater_a) | set(rater_b)))
    if not categories:
        return 0.0

    po = sum(1 for a, b in zip(rater_a, rater_b) if a == b) / total

    pe = 0.0
    for cat in categories:
        p_a = sum(1 for a in rater_a if a == cat) / total
        p_b = sum(1 for b in rater_b if b == cat) / total
        pe += p_a * p_b

    if (1.0 - pe) != 0:
        return round((po - pe) / (1.0 - pe), 4)
    return 1.0 if po == 1.0 else 0.0


def compute_metrics(
    labels_path: Union[str, Path] = "reports/label_sample_final.csv",
    db_path: Optional[Union[str, Path]] = None,
    output_path: Union[str, Path] = "reports/detector_metrics.json",
) -> Dict[str, Any]:
    """
    Compare rule and judge detector outputs against final labels.
    """
    labels_file = Path(labels_path)
    if not labels_file.exists():
        raise FileNotFoundError(f"Labels file not found: {labels_file}")

    store = ResultStore(db_path=db_path)

    # Read final labels
    rows_data: List[Dict[str, str]] = []
    with labels_file.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows_data.append(r)

    if not rows_data:
        raise ValueError("No label rows found in file.")

    row_ids = [int(r["row_id"]) for r in rows_data]

    # Fetch rows from SQLite
    db_rows_by_id: Dict[int, Any] = {}
    with store._connect() as conn:
        placeholders = ",".join("?" for _ in row_ids)
        cursor = conn.execute(
            f"SELECT id, rule_based_verdict, judge_verdict FROM probe_results WHERE id IN ({placeholders})",
            row_ids,
        )
        for row in cursor.fetchall():
            db_rows_by_id[row["id"]] = row

    y_true: List[str] = []
    y_rule: List[str] = []
    y_judge: List[str] = []
    uncertain_count = 0

    rater_a: List[str] = []
    rater_b: List[str] = []
    has_annotators = ("label_a" in rows_data[0] and "label_b" in rows_data[0])

    for r in rows_data:
        rid = int(r["row_id"])
        final_lbl = r.get("final_label", "").strip().upper()
        if final_lbl not in ("SUCCESS", "FAILURE"):
            continue

        db_row = db_rows_by_id.get(rid)
        if not db_row:
            continue

        y_true.append(final_lbl)

        # Rule verdict normalization
        raw_rule = str(db_row["rule_based_verdict"] or "").strip().upper()
        if raw_rule in ("SUCCESS", "FAIL"):
            y_rule.append("SUCCESS")
        else:
            y_rule.append("FAILURE")

        # Judge verdict normalization
        raw_judge = str(db_row["judge_verdict"] or "").strip().upper()
        if raw_judge == "UNCERTAIN":
            uncertain_count += 1
            y_judge.append("FAILURE")
        elif raw_judge == "SUCCESS":
            y_judge.append("SUCCESS")
        else:
            y_judge.append("FAILURE")

        if has_annotators and r.get("label_a") and r.get("label_b"):
            la = r["label_a"].strip().upper()
            lb = r["label_b"].strip().upper()
            if la in ("SUCCESS", "FAILURE") and lb in ("SUCCESS", "FAILURE"):
                rater_a.append(la)
                rater_b.append(lb)

    rule_metrics = calculate_binary_metrics(y_true, y_rule)
    judge_metrics = calculate_binary_metrics(y_true, y_judge)
    judge_metrics["uncertain_count"] = uncertain_count

    results: Dict[str, Any] = {
        "rule": rule_metrics,
        "judge": judge_metrics,
    }

    if rater_a and rater_b and len(rater_a) == len(rater_b):
        inter_kappa = calculate_cohens_kappa(rater_a, rater_b)
        results["inter_annotator_kappa"] = inter_kappa
    else:
        results["inter_annotator_kappa"] = None

    # Print results to stdout
    print("\n=== Detector Evaluation Metrics ===")
    print(f"Total Evaluated Samples: {len(y_true)}")
    print("\n--- Rule-Based Detector ---")
    print(f"Confusion Matrix: {rule_metrics['confusion_matrix']}")
    print(f"Accuracy:         {rule_metrics['accuracy']:.4f}")
    print(f"Precision:        {rule_metrics['precision']:.4f}")
    print(f"Recall:           {rule_metrics['recall']:.4f}")
    print(f"F1 Score:         {rule_metrics['f1']:.4f}")
    print(f"Cohen's Kappa:    {rule_metrics['cohens_kappa']:.4f}")

    print("\n--- LLM Judge Detector ---")
    print(f"Confusion Matrix: {judge_metrics['confusion_matrix']}")
    print(f"Accuracy:         {judge_metrics['accuracy']:.4f}")
    print(f"Precision:        {judge_metrics['precision']:.4f}")
    print(f"Recall:           {judge_metrics['recall']:.4f}")
    print(f"F1 Score:         {judge_metrics['f1']:.4f}")
    print(f"Cohen's Kappa:    {judge_metrics['cohens_kappa']:.4f}")
    print(f"Uncertain Count:  {uncertain_count} (treated as FAILURE)")

    if results["inter_annotator_kappa"] is not None:
        print(f"\nInter-Annotator Kappa (label_a vs label_b): {results['inter_annotator_kappa']:.4f}")

    # Save JSON metrics
    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    out_file.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nMetrics saved to {out_file}\n")

    return results
