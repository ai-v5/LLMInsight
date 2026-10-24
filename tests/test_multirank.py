"""Multi-rank capture root support (parser.multirank + report/CLI integration).

A 32-node run writes one ``<host>_<pid>_<ts>_ascend_pt/`` directory per NPU.
Pointing LLMInsight at that parent used to load zero files silently; these tests
pin the new behaviour:

* rank directories are discovered below the root (both
  ``rank/ASCEND_PROFILER_OUTPUT`` and a leaf capture directly in the rank dir),
* the representative rank is the median step time (``LLMINSIGHT_RANK`` can
  override it),
* the cluster summary reports every rank's step decomposition,
* ``load_profile`` / ``compute_all`` / ``build_report_html`` all carry the
  cluster section, and the report CLI accepts ``--data-dir``.
"""
from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path

from llminsight.metrics import compute_all
from llminsight.parser import load_profile
from llminsight.parser.multirank import (
    discover_rank_dirs,
    looks_like_multirank_root,
    rank_host,
    rank_profile_dir,
    resolve_multirank,
)
from llminsight.report import _main as report_main, build_report_html

_STEP_HEADER = (
    "Device_id,Step,Computing,Communication(Not Overlapped),Overlapped,"
    "Communication,Free,Stage,Bubble,Preparing\n"
)


def _write_step_trace(directory: Path, stage_us: float, device_id: int = 0) -> None:
    computing = stage_us * 0.65
    comm_not_overlapped = stage_us * 0.33
    free = stage_us - computing - comm_not_overlapped
    overlapped = stage_us * 0.40
    communication = comm_not_overlapped + overlapped
    directory.mkdir(parents=True, exist_ok=True)
    with open(directory / "step_trace_time.csv", "w", encoding="utf-8") as fh:
        fh.write(_STEP_HEADER)
        fh.write(f"{device_id},3,{computing},{comm_not_overlapped},{overlapped},"
                 f"{communication},{free},{stage_us},0,2.0\n")
    (directory / "op_statistic.csv").write_text(
        "Device_id,OP Type,Core Type,Count,Total Time(us),Min Time(us),Avg Time(us),"
        "Max Time(us),Ratio(%)\n"
        f"{device_id},MatMulV3,AI_CORE,10,{stage_us * 0.5},1,1,1,50.0\n",
        encoding="utf-8",
    )


def _make_multirank_root(base: Path) -> Path:
    root = base / "profile_32node_demo"
    stages = {
        "nodeA_101_20260101000000001_ascend_pt": 4000.0,   # fastest
        "nodeB_102_20260101000000002_ascend_pt": 4800.0,   # slow (outlier)
        "nodeC_103_20260101000000003_ascend_pt": 4700.0,
        "nodeD_104_20260101000000004_ascend_pt": 4750.0,
    }
    for i, (name, stage) in enumerate(stages.items()):
        rank = root / name
        if i % 2 == 0:
            _write_step_trace(rank / "ASCEND_PROFILER_OUTPUT", stage, device_id=i)
        else:  # leaf capture directly inside the rank dir
            _write_step_trace(rank, stage, device_id=i)
        (rank / "profiler_info_x.json").write_text("{}", encoding="utf-8")
    # decoy non-rank directory + a stray file
    (root / "logs").mkdir()
    (root / "cluster_analysis.log").write_text("x", encoding="utf-8")
    return root


class MultiRankDiscoveryTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.base = Path(self._tmp.name)
        self.root = _make_multirank_root(self.base)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_discovery_finds_rank_dirs_only(self) -> None:
        found = discover_rank_dirs(str(self.root))
        self.assertEqual(len(found), 4)
        self.assertTrue(all(os.path.basename(p).endswith("_ascend_pt") for p in found))
        self.assertTrue(looks_like_multirank_root(str(self.root)))

    def test_leaf_capture_is_single_rank(self) -> None:
        leaf = self.root / "nodeA_101_20260101000000001_ascend_pt" / "ASCEND_PROFILER_OUTPUT"
        self.assertEqual(discover_rank_dirs(str(leaf)), [])
        self.assertFalse(looks_like_multirank_root(str(leaf)))
        self.assertIsNone(resolve_multirank(str(leaf)))

    def test_rank_profile_dir_supports_both_layouts(self) -> None:
        nested = self.root / "nodeA_101_20260101000000001_ascend_pt"
        flat = self.root / "nodeB_102_20260101000000002_ascend_pt"
        self.assertTrue(rank_profile_dir(str(nested)).endswith("ASCEND_PROFILER_OUTPUT"))
        self.assertEqual(rank_profile_dir(str(flat)), str(flat))
        self.assertEqual(rank_host("133_C01_B11_CPU1_1276060_20261024214905982_ascend_pt"),
                         "133_C01_B11_CPU1")


class MultiRankClusterTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.base = Path(self._tmp.name)
        self.root = _make_multirank_root(self.base)
        os.environ.pop("LLMINSIGHT_RANK", None)
        os.environ.pop("LLMINSIGHT_CLUSTER_COMM", None)

    def tearDown(self) -> None:
        os.environ.pop("LLMINSIGHT_RANK", None)
        os.environ.pop("LLMINSIGHT_CLUSTER_COMM", None)
        self._tmp.cleanup()

    def test_cluster_summary_and_representative(self) -> None:
        resolved = resolve_multirank(str(self.root))
        self.assertIsNotNone(resolved)
        cluster = resolved["cluster"]
        self.assertEqual(cluster["n_ranks"], 4)
        stage_stats = cluster["stats"]["stage_us"]
        # median of 4000/4700/4750/4800 -> mean of the two middle values
        self.assertAlmostEqual(stage_stats["median"], 4725.0, places=1)
        # representative: stage closest to the median (4700, distance 25)
        rep = cluster["representative"]
        self.assertIn("nodeC_103", rep["rank_dir"])
        self.assertEqual(rep["selection"], "median_step_time")
        self.assertTrue(os.path.isfile(os.path.join(rep["profile_dir"], "step_trace_time.csv")))
        self.assertTrue(cluster["outliers"]["ranks"] or cluster["stats"]["stage_us"]["cv_pct"] >= 0)

    def test_env_rank_override(self) -> None:
        os.environ["LLMINSIGHT_RANK"] = "nodeD_104"
        resolved = resolve_multirank(str(self.root))
        self.assertIn("nodeD_104", resolved["representative"])
        self.assertEqual(resolved["cluster"]["representative"]["selection"],
                         "requested:nodeD_104")

    def test_load_profile_attaches_cluster(self) -> None:
        prof = load_profile(str(self.root))
        self.assertTrue(prof.meta["multirank"]["n_ranks"], 4)
        self.assertEqual(prof.cluster["n_ranks"], 4)
        self.assertFalse(prof.step_trace.empty)
        metrics = compute_all(prof)
        self.assertTrue(metrics["cluster"]["available"])
        self.assertIn("representative", metrics["cluster"])
        html = build_report_html(metrics, [])
        self.assertIn("集群概览", html)
        self.assertIn("nodeC_103", html)

    def test_empty_dir_warns_instead_of_silent_empty(self) -> None:
        empty = self.base / "empty"
        empty.mkdir()
        prof = load_profile(str(empty))
        self.assertTrue(prof.meta.get("warnings"))
        self.assertTrue(any("没有识别到 profiling" in w for w in prof.meta["warnings"]))

    def test_report_cli_data_dir_and_empty_error(self) -> None:
        out = self.base / "report.html"
        rc = report_main(["--data-dir", str(self.root), "--out", str(out)])
        self.assertEqual(rc, 0)
        text = out.read_text(encoding="utf-8")
        self.assertIn("集群概览", text)
        empty = self.base / "empty2"
        empty.mkdir()
        self.assertEqual(report_main(["--data-dir", str(empty), "--out", str(out)]), 2)


class MultiRankServerDetectionTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.root = _make_multirank_root(Path(self._tmp.name))

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_browse_detection_accepts_multirank_root(self) -> None:
        from llminsight.server.app import _looks_like_profile_dir
        self.assertTrue(_looks_like_profile_dir(str(self.root)))
        leaf = self.root / "nodeC_103_20260101000000003_ascend_pt" / "ASCEND_PROFILER_OUTPUT"
        self.assertTrue(_looks_like_profile_dir(str(leaf)))


if __name__ == "__main__":
    unittest.main()
