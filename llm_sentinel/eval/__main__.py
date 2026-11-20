"""
CLI entrypoint for llm_sentinel.eval module.
Commands:
  export-sample --n 30 --seed 42
  judge --model NAME [--all]
  metrics --labels reports/label_sample_final.csv
"""

from __future__ import annotations

import argparse
import sys
from typing import List, Optional

from .judge import run_judge
from .metrics import compute_metrics
from .sample import export_sample


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="LLM Sentinel Evaluation & Labeling Toolkit")
    subparsers = parser.add_subparsers(dest="command", required=True, help="Command to run")

    # Command: export-sample
    p_export = subparsers.add_parser("export-sample", help="Export stratified samples for human annotation")
    p_export.add_argument("--n", type=int, default=30, help="Total number of samples (default: 30)")
    p_export.add_argument("--seed", type=int, default=42, help="Random seed for sampling (default: 42)")
    p_export.add_argument("--db-path", type=str, default=None, help="SQLite database path")
    p_export.add_argument("--output", type=str, default="reports/label_sample.csv", help="Output CSV path")

    # Command: judge
    p_judge = subparsers.add_parser("judge", help="Run LLM judge on unjudged results")
    p_judge.add_argument("--model", type=str, required=True, help="Judge model name")
    p_judge.add_argument("--all", action="store_true", dest="all_attacks", help="Judge all attack rows missing verdicts")
    p_judge.add_argument("--sample", type=str, default="reports/label_sample.csv", help="Sample CSV path")
    p_judge.add_argument("--db-path", type=str, default=None, help="SQLite database path")

    # Command: metrics
    p_metrics = subparsers.add_parser("metrics", help="Compute metrics against ground truth labels")
    p_metrics.add_argument("--labels", type=str, default="reports/label_sample_final.csv", help="Path to labeled CSV")
    p_metrics.add_argument("--db-path", type=str, default=None, help="SQLite database path")
    p_metrics.add_argument("--output", type=str, default="reports/detector_metrics.json", help="Path for output JSON")

    return parser


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "export-sample":
        export_sample(n=args.n, seed=args.seed, db_path=args.db_path, output_path=args.output)
        return 0
    elif args.command == "judge":
        run_judge(model=args.model, all_attacks=args.all_attacks, sample_path=args.sample, db_path=args.db_path)
        return 0
    elif args.command == "metrics":
        compute_metrics(labels_path=args.labels, db_path=args.db_path, output_path=args.output)
        return 0
    return 1


if __name__ == "__main__":
    sys.exit(main())
