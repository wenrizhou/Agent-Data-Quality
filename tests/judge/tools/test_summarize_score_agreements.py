from __future__ import annotations

import importlib.util
import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory


REPO_ROOT = Path(__file__).resolve().parents[3]
MODULE_PATH = REPO_ROOT / "judge" / "tools" / "summarize_score_agreements.py"


def load_module():
    spec = importlib.util.spec_from_file_location("summarize_score_agreements", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def write_results(run_dir: Path, rows: list[dict]) -> None:
    run_dir.mkdir(parents=True)
    with (run_dir / "task_results.jsonl").open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


class SummarizeScoreAgreementsTest(unittest.TestCase):
    def test_counts_agreed_scores_by_metric(self) -> None:
        module = load_module()
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_a = root / "run_a"
            run_b = root / "run_b"
            write_results(
                run_a,
                [
                    {"sample_id": "s1", "metric": "tool_selection", "task_id": "sample", "score": 2.0},
                    {"sample_id": "s2", "metric": "tool_selection", "task_id": "sample", "score": 1.0},
                    {"sample_id": "s3", "metric": "tool_selection", "task_id": "sample", "score": 0.0},
                    {"sample_id": "s4", "metric": "tool_selection", "task_id": "sample", "score": 1},
                    {"sample_id": "s5", "metric": "tool_selection", "task_id": "sample", "score": None},
                    {"sample_id": "s6", "metric": "specificity", "task_id": "sample", "score": 2.0},
                    {"sample_id": "only-a", "metric": "tool_selection", "task_id": "sample", "score": 2.0},
                ],
            )
            write_results(
                run_b,
                [
                    {"sample_id": "s1", "metric": "tool_selection", "task_id": "sample", "score": 2},
                    {"sample_id": "s2", "metric": "tool_selection", "task_id": "sample", "score": 0},
                    {"sample_id": "s3", "metric": "tool_selection", "task_id": "sample", "score": 0},
                    {"sample_id": "s4", "metric": "tool_selection", "task_id": "sample", "score": 1.0},
                    {"sample_id": "s5", "metric": "tool_selection", "task_id": "sample", "score": None},
                    {"sample_id": "s6", "metric": "specificity", "task_id": "sample", "score": 2.0},
                    {"sample_id": "only-b", "metric": "tool_selection", "task_id": "sample", "score": 0.0},
                ],
            )

            summary = module.summarize_agreements(
                run_a,
                run_b,
                metric_filter={"tool_selection"},
            )

        self.assertEqual(summary["common_results"], 5)
        self.assertEqual(summary["agreements"], 3)
        self.assertEqual(summary["disagreements"], 1)
        self.assertEqual(summary["unscored_or_invalid"], 1)
        self.assertEqual(summary["missing_from_run_b"], 1)
        self.assertEqual(summary["missing_from_run_a"], 1)
        self.assertEqual(summary["agreed_score_counts"], {"0": 1, "1": 1, "2": 1})


if __name__ == "__main__":
    unittest.main()
