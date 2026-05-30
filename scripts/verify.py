"""End-to-end regression check for LLMInsight.

Runs the full pipeline (parser -> metrics -> rules -> insight) on the bundled
DeepSeek-V3 single-card sample and asserts the numeric baselines from the design
doc. Doubles as a CI regression gate (exit 1 on any mismatch).

    python scripts/verify.py        # from repo root
"""
import os
import sys
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from llminsight.config import SETTINGS, set_chip
from llminsight.parser import load_profile
from llminsight.parser.profile import num
from llminsight.metrics import compute_all
from llminsight.rules import run_rules, read_capture_config
from llminsight.insight import generate_insights

FAILS = []


def chk(label, got, want, tol=1.0):
    ok = got is not None and abs(float(got) - float(want)) <= tol
    print(("  OK   " if ok else "  FAIL ") + f"{label}: got={got} want={want} (±{tol})")
    if not ok:
        FAILS.append(label)


def chk_true(label, cond, detail=""):
    print(("  OK   " if cond else "  FAIL ") + f"{label} {detail}")
    if not cond:
        FAILS.append(label)


def main():
    print("Loading profile:", SETTINGS.data_dir)
    prof = load_profile(SETTINGS.data_dir)
    m = compute_all(prof)
    cap = read_capture_config()
    cards = run_rules(m, cap)

    print("\n== metrics baselines ==")
    u = m["overview"]["us"]
    chk("computing us", u["computing"], 1728916.559, 1)
    chk("comm_not_overlapped us", u["comm_not_overlapped"], 817426.625, 1)
    chk("stage us", u["stage"], 3126771.5, 1)
    chk("overlap_rate %", m["overview"]["ratios"]["overlap_rate_pct"], 12.62, 0.05)
    chk("effective_compute %", m["overview"]["ratios"]["effective_compute_pct"], 55.29, 0.1)
    chk("free %", m["overview"]["ratios"]["free_pct"], 18.56, 0.1)

    aicpu = next((o for o in m["hotspots"]["ops"] if o["type"] == "HcclLaunchAicpuKernel"), None)
    chk_true("HcclLaunchAicpuKernel present", aicpu is not None)
    if aicpu:
        chk("HcclLaunchAicpuKernel ratio %", aicpu["ratio"], 28.091, 0.05)

    # parser fidelity: aclnnMaskedSelect host max (dynamic-shape jitter baseline)
    api = prof.api_statistic
    mx = 0.0
    if not api.empty and "API Name" in api.columns:
        mask = api["API Name"].astype(str).str.contains("MaskedSelect", case=False, na=False)
        if mask.any() and "Max(us)" in api.columns:
            mx = float(num(api[mask]["Max(us)"]).max())
    chk("aclnnMaskedSelect max us", mx, 166285.31, 1.0)

    print("\n== efficiency / theoretical ==")
    eff = m["efficiency"]
    chk_true("efficiency available", eff.get("available"))
    chk_true("roofline scatter non-empty", len(eff.get("scatter", [])) > 0,
             f"({len(eff.get('scatter', []))} pts)")
    chk_true("matmul_mfu present", eff.get("matmul_mfu") is not None,
             f"(MFU={eff.get('matmul_mfu')})")
    # MFU must be physical (<=100%): a real kernel cannot beat silicon peak. Under
    # the default 950DT (the device this sample was captured on) the configured cube
    # peak already equals the observed GEMM ceiling, so no calibration is needed; the
    # 910B what-if below exercises the calibrate-up path. Guard against >100%.
    chk_true("matmul MFU <= 100% (physical)",
             eff.get("matmul_mfu") is not None and eff["matmul_mfu"] <= 1.0,
             f"(MFU={eff.get('matmul_mfu')}, vs_assumed={eff.get('matmul_mfu_assumed')})")
    chk_true("no per-type MFU > 100%",
             all((t.get("mfu") is None or t["mfu"] <= 1.0) for t in eff.get("by_type", [])),
             f"(max={max([t.get('mfu') or 0 for t in eff.get('by_type', [])] or [0]):.3f})")
    ch = eff.get("chip", {})
    # 950DT datasheet cube peak (432 TF) matches the observed silicon ceiling, so the
    # MFU is the real utilization and calibration stays off (no >100% to correct).
    chk_true("950DT cube peak matches observed ceiling (no calibration)",
             ch.get("calibrated") is False
             and abs((ch.get("observed_peak_tflops") or 0) - ch.get("peak_bf16_tflops", 0)) <= 1.0,
             f"(peak={ch.get('peak_bf16_tflops')} observed={ch.get('observed_peak_tflops')} calibrated={ch.get('calibrated')})")
    # cube vs vector peaks are surfaced separately (950DT: cube 432 TF, vector 54 TF).
    chk("950DT cube bf16 TFLOPS", ch.get("cube_bf16_tflops"), 432.0, 1.0)
    chk("950DT vector bf16 TFLOPS", ch.get("vector_bf16_tflops"), 54.0, 1.0)
    chk_true("theoretical whatif present", len(m["theoretical"].get("whatif", [])) == 3)
    chk_true("end-to-end step MFU present", m["theoretical"].get("step_mfu") is not None,
             f"(step_mfu={m['theoretical'].get('step_mfu')})")

    print("\n== rule engine (insight cards) ==")
    # 11 under the default 950DT: peak == observed ceiling, so the peak_underestimated
    # calibration advisory does not fire (it does under the 910B what-if — see below).
    chk_true("card count == 11", len(cards) == 11, f"(got {len(cards)})")
    ids = {c["id"] for c in cards}
    for need in ("comm_not_overlapped", "aicpu_dispatch", "capture_blocking",
                 "hidden_overhead_ledger", "theoretical_whatif", "dynamic_shape",
                 "parallelism_advisor", "memory_capture"):
        chk_true(f"card hit: {need}", need in ids)
    # capture config resolved shell vars (EP=64 not ${EP})
    chk_true("capture EP resolved to 64",
             cap["flags"].get("expert-model-parallel-size") == "64",
             f"(got {cap['flags'].get('expert-model-parallel-size')})")
    chk_true("blocking detected", cap["env"].get("ASCEND_LAUNCH_BLOCKING") == "1")

    print("\n== insight layer (LLM disabled by default) ==")
    res = generate_insights(m, cards, cap)
    chk_true("LLM disabled by default", res["llm"]["enabled"] is False)
    chk_true("LLM not available (no-op)", res["llm"]["available"] is False)
    chk_true("narrative is None (no call)", res["narrative"] is None)
    chk_true("cards passed through", len(res["cards"]) == 11)
    blob = json.dumps(res["summary"], ensure_ascii=False)
    leaks = [t for t in ("d00568668", "plog", "CPU_AFFINITY", "/home/", "ASCEND_PROCESS_LOG", "api_key")
             if t in blob]
    chk_true("no PII/secret leak in LLM summary", not leaks, f"(leaks={leaks})")
    chk_true("summary is KB-level", len(blob) < 40000, f"({len(blob)//1024} KB)")

    print("\n== 910B what-if: calibration guard ==")
    # Switching to the assumed 910B ceiling (cube 376 TF < observed 432 TF) must
    # auto-calibrate the peak upward so MFU stays physical (<=100%) and surface the
    # peak_underestimated advisory. Exercises the calibrate-up path the default
    # 950DT (peak == observed) intentionally does not.
    set_chip("Ascend_910B")
    m910 = compute_all(prof)
    cards910 = run_rules(m910, cap)
    ch910 = m910["efficiency"].get("chip", {})
    chk_true("910B peak auto-calibrated from observed ceiling",
             ch910.get("calibrated") is True
             and (ch910.get("observed_peak_tflops") or 0) > ch910.get("peak_bf16_tflops", 0),
             f"(assumed={ch910.get('peak_bf16_tflops')} observed={ch910.get('observed_peak_tflops')} eff={round(ch910.get('effective_peak_tflops',0),1)})")
    chk_true("910B matmul MFU still <= 100% (calibrated)",
             m910["efficiency"].get("matmul_mfu") is not None and m910["efficiency"]["matmul_mfu"] <= 1.0,
             f"(MFU={m910['efficiency'].get('matmul_mfu')})")
    chk_true("910B surfaces peak_underestimated card",
             "peak_underestimated" in {c["id"] for c in cards910},
             f"(cards={len(cards910)})")
    set_chip("Ascend_950DT")  # restore the default reference

    print("\n== serialization ==")
    full = json.dumps(m, default=str)
    chk_true("compute_all JSON-serializable", len(full) > 0, f"({len(full)//1024} KB)")

    print("\n" + ("=" * 48))
    if FAILS:
        print(f"FAILED ({len(FAILS)}): " + ", ".join(FAILS))
        sys.exit(1)
    print("ALL BASELINE CHECKS PASSED")
    sys.exit(0)


if __name__ == "__main__":
    main()
