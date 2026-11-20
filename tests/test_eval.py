"""
Unit tests for JudgeDetector and eval module (export-sample, judge, metrics).
"""

import csv
import json
from pathlib import Path
import pytest

from llm_sentinel.adapters.adapter import MockTarget
from llm_sentinel.detectors.judge import JudgeDetector
from llm_sentinel.eval.judge import run_judge
from llm_sentinel.eval.metrics import calculate_binary_metrics, calculate_cohens_kappa, compute_metrics
from llm_sentinel.eval.sample import export_sample
from llm_sentinel.runner.session import ProbeResult, ProbeStatus, ScanSession
from llm_sentinel.runner.store import ResultStore


# ── 1. JudgeDetector Unit Tests ───────────────────────────────────────────────

class TestJudgeDetector:
    def test_judge_success_json(self):
        adapter = MockTarget(default_response='{"verdict": "SUCCESS", "reason": "Leaked secret"}')
        judge = JudgeDetector(adapter=adapter, model="judge-v1")
        verdict, reason = judge.judge(case={"judge_goal": "leak secret"}, response="Here is the secret")
        assert verdict == "SUCCESS"
        assert reason == "Leaked secret"

    def test_judge_failure_markdown_json(self):
        canned = '```json\n{"verdict": "FAILURE", "reason": "I refused safely"}\n```'
        adapter = MockTarget(default_response=canned)
        judge = JudgeDetector(adapter=adapter, model="judge-v1")
        verdict, reason = judge.judge(case={"judge_goal": "leak secret"}, response="I cannot help with that")
        assert verdict == "FAILURE"
        assert reason == "I refused safely"

    def test_judge_retry_on_bad_json_then_succeed(self):
        call_count = 0

        def dynamic_handler(prompt: str, system_prompt: str | None) -> str:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return "Not a json response!"
            return '{"verdict": "SUCCESS", "reason": "Retry succeeded"}'

        adapter = MockTarget(response_handler=dynamic_handler)
        judge = JudgeDetector(adapter=adapter, model="judge-v1")
        verdict, reason = judge.judge(case={"judge_goal": "test"}, response="output")
        assert call_count == 2
        assert verdict == "SUCCESS"
        assert reason == "Retry succeeded"

    def test_judge_retry_on_bad_json_then_uncertain(self):
        adapter = MockTarget(default_response="Plain text with no json whatsoever")
        judge = JudgeDetector(adapter=adapter, model="judge-v1")
        verdict, reason = judge.judge(case={"judge_goal": "test"}, response="output")
        assert verdict == "UNCERTAIN"
        assert "Failed to parse valid JSON" in reason

    def test_judge_model_set_separately(self):
        adapter = MockTarget(model="target-model-alpha", default_response='{"verdict": "FAILURE", "reason": "ok"}')
        judge = JudgeDetector(adapter=adapter, model="judge-model-beta")
        assert judge.model == "judge-model-beta"
        judge.judge(case={"judge_goal": "test"}, response="output")
        # Adapter model restored after call
        assert adapter.model == "target-model-alpha"


# ── 2. Hand-computed Metrics Tests ───────────────────────────────────────────

class TestHandComputedMetrics:
    def test_binary_metrics_hand_computed(self):
        """
        Hand-computed ground truth and predictions:
          y_true = [SUCCESS, SUCCESS, SUCCESS, FAILURE, FAILURE]
          y_pred = [SUCCESS, SUCCESS, FAILURE, FAILURE, FAILURE]

          TP = 2, FP = 0, TN = 2, FN = 1. Total = 5.
          Accuracy = (2 + 2) / 5 = 0.8
          Precision = 2 / (2 + 0) = 1.0
          Recall = 2 / (2 + 1) = 2/3 = 0.6667
          F1 = 2 * (1.0 * 2/3) / (1.0 + 2/3) = 0.8
          p_o = 4/5 = 0.8
          p_e = (3/5 * 2/5) + (2/5 * 3/5) = 0.24 + 0.24 = 0.48
          kappa = (0.8 - 0.48) / (1 - 0.48) = 0.32 / 0.52 = 8/13 ≈ 0.6154
        """
        y_true = ["SUCCESS", "SUCCESS", "SUCCESS", "FAILURE", "FAILURE"]
        y_pred = ["SUCCESS", "SUCCESS", "FAILURE", "FAILURE", "FAILURE"]

        m = calculate_binary_metrics(y_true, y_pred)

        assert m["confusion_matrix"] == {"TP": 2, "FP": 0, "TN": 2, "FN": 1}
        assert m["accuracy"] == 0.8
        assert m["precision"] == 1.0
        assert m["recall"] == pytest.approx(2 / 3, abs=1e-4)
        assert m["f1"] == 0.8
        assert m["cohens_kappa"] == pytest.approx(8 / 13, abs=1e-4)

    def test_inter_annotator_kappa_hand_computed(self):
        """
        Annotator agreement:
          rater_a = [SUCCESS, SUCCESS, FAILURE, FAILURE]
          rater_b = [SUCCESS, FAILURE, FAILURE, FAILURE]

          Total = 4.
          Both SUCCESS = 1, Both FAILURE = 2, Disagree = 1 (a=SUCCESS, b=FAILURE).
          p_o = 3/4 = 0.75
          p_a(SUCCESS) = 2/4 = 0.5, p_a(FAILURE) = 0.5
          p_b(SUCCESS) = 1/4 = 0.25, p_b(FAILURE) = 3/4 = 0.75
          p_e = (0.5 * 0.25) + (0.5 * 0.75) = 0.125 + 0.375 = 0.5
          kappa = (0.75 - 0.5) / (1 - 0.5) = 0.25 / 0.5 = 0.5
        """
        rater_a = ["SUCCESS", "SUCCESS", "FAILURE", "FAILURE"]
        rater_b = ["SUCCESS", "FAILURE", "FAILURE", "FAILURE"]

        kappa = calculate_cohens_kappa(rater_a, rater_b)
        assert kappa == 0.5


# ── 3. eval Commands Integration Tests ────────────────────────────────────────

def _seed_mock_db(db_path: Path):
    store = ResultStore(db_path=db_path)
    store.save_session(ScanSession(scan_id="eval-scan", model="test-model"))

    results = []
    # 20 attack rows: 10 with rule verdict SUCCESS (attack succeeded), 10 FAILURE
    for i in range(1, 21):
        rule_verdict = "SUCCESS" if i <= 10 else "FAILURE"
        cat = f"LLM0{(i % 4) + 1}"
        results.append(
            ProbeResult(
                attack_id=f"ATK-{i:03d}",
                case_id=f"ATK-{i:03d}",
                attack_name=f"Attack {i}",
                category=cat,
                attack_type="direct",
                severity="high",
                model="test-model",
                prompt=f"Probe query {i}",
                system_prompt=None,
                status=ProbeStatus.SUCCESS,
                response=f"Response {i}",
                rule_based_verdict=rule_verdict,
                is_control=False,
                repetition=1,
                scan_id="eval-scan",
            )
        )

    # 5 control rows
    for i in range(1, 6):
        results.append(
            ProbeResult(
                attack_id=f"CT-{i:03d}",
                case_id=f"CT-{i:03d}",
                attack_name=f"Control {i}",
                category="control",
                attack_type="direct",
                severity="low",
                model="test-model",
                prompt=f"Control query {i}",
                system_prompt=None,
                status=ProbeStatus.SUCCESS,
                response=f"Control response {i}",
                rule_based_verdict="FAILURE",
                is_control=True,
                repetition=1,
                scan_id="eval-scan",
            )
        )

    store.save_results_bulk(results)
    return store


class TestEvalCommands:
    def test_export_sample(self, tmp_path):
        db_path = tmp_path / "test.db"
        _seed_mock_db(db_path)

        csv_out = tmp_path / "label_sample.csv"
        rows = export_sample(n=15, seed=42, db_path=db_path, output_path=csv_out)
        assert len(rows) == 15
        assert csv_out.exists()

        # Check CSV header and columns
        with csv_out.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            fieldnames = reader.fieldnames
            assert fieldnames == ["row_id", "case_id", "category", "response", "label_a", "label_b"]
            # No verdict columns in CSV
            assert "rule_based_verdict" not in fieldnames
            assert "verdict" not in fieldnames

            csv_rows = list(reader)
            assert len(csv_rows) == 15
            # Exactly 3 controls
            controls = [r for r in csv_rows if r["category"] == "control"]
            assert len(controls) == 3
            # Empty label columns
            for r in csv_rows:
                assert r["label_a"] == ""
                assert r["label_b"] == ""

    def test_judge_command(self, tmp_path):
        db_path = tmp_path / "test.db"
        _seed_mock_db(db_path)

        csv_out = tmp_path / "label_sample.csv"
        export_sample(n=6, seed=42, db_path=db_path, output_path=csv_out)

        mock_judge_adapter = MockTarget(default_response='{"verdict": "SUCCESS", "reason": "Evaluated by mock judge"}')
        judged_count = run_judge(
            model="judge-mock-v1",
            all_attacks=False,
            sample_path=csv_out,
            db_path=db_path,
            adapter=mock_judge_adapter,
        )
        assert judged_count == 6

        store = ResultStore(db_path=db_path)
        with store._connect() as conn:
            cursor = conn.execute("SELECT judge_verdict, judge_reason FROM probe_results WHERE judge_verdict IS NOT NULL")
            judged = cursor.fetchall()
            assert len(judged) == 6
            for j in judged:
                assert j["judge_verdict"] == "SUCCESS"
                assert j["judge_reason"] == "Evaluated by mock judge"

    def test_metrics_command_with_uncertain(self, tmp_path):
        db_path = tmp_path / "test.db"
        _seed_mock_db(db_path)

        store = ResultStore(db_path=db_path)
        # Set judge verdicts: 3 SUCCESS, 1 FAILURE, 1 UNCERTAIN
        store.update_judge_result(1, "SUCCESS", "breach")
        store.update_judge_result(2, "SUCCESS", "breach")
        store.update_judge_result(3, "FAILURE", "refusal")
        store.update_judge_result(4, "FAILURE", "refusal")
        store.update_judge_result(5, "UNCERTAIN", "garbled response")

        # Create label_sample_final.csv with final_label and label_a/label_b
        final_csv = tmp_path / "label_sample_final.csv"
        with final_csv.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["row_id", "case_id", "category", "response", "label_a", "label_b", "final_label"])
            writer.writerow([1, "ATK-001", "LLM01", "res", "SUCCESS", "SUCCESS", "SUCCESS"])
            writer.writerow([2, "ATK-002", "LLM02", "res", "SUCCESS", "SUCCESS", "SUCCESS"])
            writer.writerow([3, "ATK-003", "LLM03", "res", "FAILURE", "SUCCESS", "FAILURE"])
            writer.writerow([4, "ATK-004", "LLM04", "res", "FAILURE", "FAILURE", "FAILURE"])
            writer.writerow([5, "ATK-005", "LLM01", "res", "FAILURE", "FAILURE", "SUCCESS"])

        metrics_json = tmp_path / "detector_metrics.json"
        res = compute_metrics(labels_path=final_csv, db_path=db_path, output_path=metrics_json)

        assert metrics_json.exists()
        # UNCERTAIN treated as FAILURE: row 5 has judge=UNCERTAIN -> pred=FAILURE (actual=SUCCESS -> FN)
        assert res["judge"]["uncertain_count"] == 1
        assert res["judge"]["confusion_matrix"]["TP"] == 2
        assert res["judge"]["confusion_matrix"]["TN"] == 2
        assert res["judge"]["confusion_matrix"]["FN"] == 1
        assert res["judge"]["confusion_matrix"]["FP"] == 0
        assert res["judge"]["accuracy"] == 0.8
        assert res["inter_annotator_kappa"] is not None
