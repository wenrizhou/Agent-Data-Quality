from __future__ import annotations

import importlib.util
import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory


REPO_ROOT = Path(__file__).resolve().parents[3]
MODULE_PATH = REPO_ROOT / "judge" / "tools" / "summarize_metric_score_distribution.py"


def load_module():
    spec = importlib.util.spec_from_file_location(
        "summarize_metric_score_distribution",
        MODULE_PATH,
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def result_row(
    sample_id: str,
    metric: str,
    score: object,
    *,
    task_id: str = "sample",
    error: object = None,
) -> dict:
    return {
        "sample_id": sample_id,
        "metric": metric,
        "metric_version": "0.1.0",
        "task_id": task_id,
        "score": score,
        "error": error,
    }


class SummarizeMetricScoreDistributionTest(unittest.TestCase):
    def test_groups_metrics_and_keeps_invalid_or_error_rows_out_of_score_buckets(
        self,
    ) -> None:
        module = load_module()
        with TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "run"
            write_jsonl(
                run_dir / "task_results.jsonl",
                [
                    result_row("sample-1", "specificity_en", 0),
                    result_row("sample-2", "specificity_en", 1.0),
                    result_row("sample-3", "specificity_en", "2"),
                    result_row("sample-4", "specificity_en", None),
                    result_row("sample-5", "specificity_en", 3),
                    result_row(
                        "sample-6",
                        "specificity_en",
                        2,
                        error={"type": "JSONDecodeError"},
                    ),
                    result_row("sample-1", "minimality_en", 2),
                    result_row("sample-2", "minimality_en", False),
                    result_row(
                        "sample-7",
                        "specificity_en",
                        0,
                        task_id="turn",
                    ),
                ],
            )

            summary = module.summarize_metric_score_distribution(
                run_dir,
                metric_filter=set(),
                task_id="sample",
            )

        specificity = summary["metrics"]["specificity_en"]
        self.assertEqual(specificity["total_results"], 6)
        self.assertEqual(specificity["valid_scored_results"], 3)
        self.assertEqual(specificity["score_counts"], {"0": 1, "1": 1, "2": 1})
        self.assertEqual(
            specificity["score_percentages"],
            {"0": 33.33, "1": 33.33, "2": 33.33},
        )
        self.assertEqual(specificity["unscored_or_invalid"], 3)

        minimality = summary["metrics"]["minimality_en"]
        self.assertEqual(minimality["total_results"], 2)
        self.assertEqual(minimality["valid_scored_results"], 1)
        self.assertEqual(minimality["score_counts"], {"0": 0, "1": 0, "2": 1})
        self.assertEqual(minimality["unscored_or_invalid"], 1)

    def test_filters_requested_metrics_and_writes_json_output(self) -> None:
        module = load_module()
        with TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "run"
            output_path = Path(tmp) / "distribution.json"
            write_jsonl(
                run_dir / "task_results.jsonl",
                [
                    result_row("sample-1", "specificity_en", 0),
                    result_row("sample-1", "minimality_en", 2),
                ],
            )

            module.main([
                "--run-dir",
                str(run_dir),
                "--metric",
                "minimality_en",
                "--output",
                str(output_path),
            ])
            with output_path.open(encoding="utf-8") as f:
                summary = json.load(f)

        self.assertEqual(list(summary["metrics"]), ["minimality_en"])
        self.assertEqual(summary["metrics"]["minimality_en"]["score_counts"], {"0": 0, "1": 0, "2": 1})


if __name__ == "__main__":
    unittest.main()
