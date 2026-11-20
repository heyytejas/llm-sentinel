"""
CLI entry point for LLM Sentinel Scorer.

Usage:
    python -m llm_sentinel.scorer --db reports/sentinel.db --model all --verdict both --out reports/scores.json
"""

import argparse
import json
import logging
from pathlib import Path

from .calculator import calculate_scores_from_db, format_readable_table

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Calculate security risk scores and attack success rates.")
    parser.add_argument(
        "--db",
        type=str,
        default="reports/sentinel.db",
        help="Path to SQLite database (default: reports/sentinel.db)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="all",
        help="Model name to evaluate or 'all' (default: all)",
    )
    parser.add_argument(
        "--verdict",
        type=str,
        choices=["rule", "judge", "both"],
        default="both",
        help="Verdict source: rule, judge, or both (default: both)",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="reports/scores.json",
        help="Output path for JSON report (default: reports/scores.json)",
    )

    args = parser.parse_args()

    results = calculate_scores_from_db(
        db_path=args.db,
        model=args.model,
        verdict=args.verdict,
    )

    # Print readable tables to terminal
    if not results:
        print(f"\nNo results found for model: {args.model} in database {args.db}\n")
    else:
        for model_name, metrics in results.items():
            print(format_readable_table(model_name, metrics))

    # Save JSON report
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nJSON report written to {out_path}\n")


if __name__ == "__main__":
    main()
