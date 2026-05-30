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

from llminsight.config import SETTINGS
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
    # MFU must be physical (<=100%): a real kernel cannot beat silicon peak. The
    # assumed 376 TF is too low for this SKU, so efficiency.py calibrates up to the
    # observed ~432 TF ceiling. Guard against a regression that re-surfaces >100%.
    chk_true("matmul MFU <= 100% (physical)",
             eff.get("matmul_mfu") is not None and eff["matmul_mfu"] <= 1.0,
             f"(MFU={eff.get('matmul_mfu')}, vs_assumed={eff.get('matmul_mfu_assumed')})")
    chk_true("no per-type MFU > 100%",
             all((t.get("mfu") is None or t["mfu"] <= 1.0) for t in eff.get("by_type", [])),
             f"(max={max([t.get('mfu') or 0 for t in eff.get('by_type', [])] or [0]):.3f})")
    ch = eff.get("chip", {})
    chk_true("peak auto-calibrated from observed ceiling",
             ch.get("calibrated") is True and (ch.get("observed_peak_tflops") or 0) > ch.get("peak_bf16_tflops", 0),
             f"(assumed={ch.get('peak_bf16_tflops')} observed={ch.get('observed_peak_tflops')} eff={round(ch.get('effective_peak_tflops',0),1)})")
    chk_true("theoretical whatif present", len(m["theoretical"].get("whatif", [])) == 3)

    print("\n== rule engine (insight cards) ==")
    chk_true("card count == 12", len(cards) == 12, f"(got {len(cards)})")
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
    chk_true("cards passed through", len(res["cards"]) == 12)
    blob = json.dumps(res["summary"], ensure_ascii=False)
    leaks = [t for t in ("d00568668", "plog", "CPU_AFFINITY", "/home/", "ASCEND_PROCESS_LOG", "api_key")
             if t in blob]
    chk_true("no PII/secret leak in LLM summary", not leaks, f"(leaks={leaks})")
    chk_true("summary is KB-level", len(blob) < 40000, f"({len(blob)//1024} KB)")

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
