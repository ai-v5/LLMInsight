"""Public end-to-end smoke using generated data only.

Run ``python scripts/verify.py`` from the repository root. All numbers below
are synthetic test inputs; they are not model, device or training benchmarks.
"""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from llminsight import cache
from llminsight.config import SETTINGS
from llminsight.insight.summarizer import build_summary
from llminsight.metrics import compute_all
from llminsight.parser import load_profile
from llminsight.report import build_report_html
from llminsight.rules import run_rules


def _write_synthetic_profile(root: Path) -> None:
    (root / "step_trace_time.csv").write_text(
        "Device_id,Step,Computing,Communication(Not Overlapped),Overlapped,"
        "Communication,Free,Stage,Bubble,Preparing\n"
        "0,1,600,250,100,350,150,1000,0,0\n", encoding="utf-8")
    (root / "kernel_details.csv").write_text(
        "Name,Type,Accelerator Core,Duration(us),Input Shapes,Input Data Types,"
        "Output Shapes,Output Data Types\n"
        'synthetic_matmul,MatMulV3,AI_CORE,600,"2,3;3,5",BF16;BF16,"2,5",BF16\n',
        encoding="utf-8")
    (root / "op_statistic.csv").write_text(
        "Device_id,OP Type,Core Type,Count,Total Time(us),Min Time(us),Avg Time(us),Max Time(us),Ratio(%)\n"
        "0,MatMulV3,AI_CORE,1,600,600,600,600,100\n", encoding="utf-8")


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="_verify_", dir=ROOT / "scripts") as directory:
        root = Path(directory)
        _write_synthetic_profile(root)
        with patch.object(cache, "_CACHE_DIR", str(root / "cache")), patch.object(SETTINGS, "data_dir", str(root)):
            profile = load_profile(str(root))
            metrics = compute_all(profile)
            cards = run_rules(metrics)
            summary = build_summary(metrics, cards)
            report = build_report_html(metrics, cards)
        measured = metrics["overview"]
        expected = {"computing": 600.0, "comm_not_overlapped": 250.0, "stage": 1000.0}
        for key, value in expected.items():
            assert abs(measured["us"][key] - value) < 1e-6, key
        assert abs(measured["ratios"]["effective_compute_pct"] - 60.0) < 1e-6
        assert abs(measured["ratios"]["free_pct"] - 15.0) < 1e-6
        assert abs(measured["ratios"]["overlap_rate_pct"] - round(100 / 350 * 100, 2)) < 1e-6
        assert metrics["efficiency"]["available"]
        assert "<html" in report and "MatMulV3" in report
        assert "synthetic_matmul" not in json.dumps(summary)
        assert isinstance(cards, list)
        print("PASS: synthetic CSV -> parser -> metrics -> rules -> redacted summary -> HTML report")
        print("Synthetic step: 1000 us; compute 600 us; exposed communication 250 us; free 150 us.")
        print("No private capture, network inference or hardware benchmark was used.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
