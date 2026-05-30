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
    # Vector-core ops now carry a VECTOR-peak MFU: elementwise/norm/optimizer kernels
    # get a flop model so their headroom floor is max(vector-compute, memory), not
    # memory alone. Mul (FLOAT, AI_VECTOR_CORE) gets a small but physical MFU routed
    # to the vector peak (≈2.5%; an 8x-higher cube peak would read ~0.3%).
    bt = {t["type"]: t for t in eff.get("by_type", [])}
    _mul_mfu = bt.get("Mul", {}).get("mfu")
    chk_true("vector op (Mul) has physical vector-peak MFU",
             _mul_mfu is not None and 0.0 < _mul_mfu <= 1.0, f"(Mul mfu={_mul_mfu})")
    # Pure data-movement ops (factor 0: ZerosLike/Cast) do ~0 arithmetic, so they stay
    # MFU-less and honestly memory-bound — no fabricated compute.
    chk_true("zero-arith vector op (ZerosLike) stays MFU-less",
             "ZerosLike" not in bt or bt["ZerosLike"].get("mfu") is None,
             f"(mfu={bt.get('ZerosLike', {}).get('mfu')})")
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
    # 算子极致优化: per-op-class MFU ceilings (matmul 95 / FA 85 / FAG 70 on 950DT)
    # drive a reclaim-to-ceiling aggregate. matmul mostly saturates the cube, so many
    # of its kernels are already at/above the ceiling (n_capped>0) and the honest gain
    # comes from FA/FAG headroom — not a naive "everything to 100%".
    oco = eff.get("op_ceiling_opt") or {}
    ceils = oco.get("ceilings") or {}
    chk("op-ceiling matmul ceiling", ceils.get("matmul"), 0.95, 1e-9)
    chk("op-ceiling FA ceiling", ceils.get("attention"), 0.85, 1e-9)
    chk("op-ceiling FAG ceiling", ceils.get("attention_grad"), 0.70, 1e-9)
    chk_true("op-ceiling reclaim > 0 (FA/FAG headroom)",
             (oco.get("total_reclaim_us") or 0) > 0, f"(reclaim={oco.get('total_reclaim_us')})")
    chk_true("op-ceiling: some matmul kernels already at ceiling (excluded)",
             oco.get("n_capped", 0) > 0, f"(n_capped={oco.get('n_capped')}/{oco.get('n_modeled')})")
    chk_true("optimization candidates all have reclaimable time (>0)",
             all(r.get("reclaim_us", 0) > 0 for r in eff.get("top_optimization", [])),
             f"(min={min([r.get('reclaim_us', 0) for r in eff.get('top_optimization', [])] or [0])})")
    theo_m = m["theoretical"]
    levers = theo_m.get("whatif", [])
    lever_ids = {w.get("id") for w in levers}
    chk_true("theoretical what-if atomic levers (3, incl. op_ceiling)",
             len(levers) == 3 and "op_ceiling" in lever_ids, f"(ids={lever_ids})")
    chk_true("each what-if lever carries new_mfu",
             all(w.get("new_mfu") is not None for w in levers))
    op_lever = next((w for w in levers if w.get("id") == "op_ceiling"), None)
    chk_true("op_ceiling lever save_us matches efficiency reclaim",
             op_lever is not None
             and abs((op_lever.get("save_us") or 0) - (oco.get("total_reclaim_us") or 0)) <= 1.0,
             f"(lever={op_lever.get('save_us') if op_lever else None} reclaim={oco.get('total_reclaim_us')})")
    comb = theo_m.get("whatif_combined") or {}
    chk_true("combined what-if present (with MFU)", comb.get("new_mfu") is not None,
             f"(combined={comb})")
    # Disjoint levers => combined save == sum of per-lever saves (additivity invariant).
    chk_true("combined save == Σ lever saves (disjoint slices add)",
             abs((comb.get("save_us") or 0) - sum(w.get("save_us") or 0 for w in levers)) <= 1.0,
             f"(combined={comb.get('save_us')} sum={sum(w.get('save_us') or 0 for w in levers)})")
    chk_true("end-to-end step MFU present", theo_m.get("step_mfu") is not None,
             f"(step_mfu={theo_m.get('step_mfu')})")

    print("\n== what-if 现实地板 (realistic floor, derived from LOADED profile) ==")
    # The 严谨性分析 ("can it reach 0? if not, how far?") must be computed from the
    # loaded profile's measured slices, NOT a static sample. Floors are formulas over
    # the measured values; 失真 caveats are conditioned on THIS capture's blocking /
    # single-card state. Web panel + shareable report both render this same payload.
    rz = theo_m.get("realistic") or {}
    rlevers = rz.get("levers") or []
    rl_by = {l.get("id"): l for l in rlevers}
    chk_true("realistic block present", bool(rz), f"(keys={list(rz.keys())})")
    chk_true("realistic has comm+free(+op) levers",
             "comm_overlap" in rl_by and "free_zero" in rl_by,
             f"(ids={set(rl_by)})")
    chk_true("no lever can reach 0 (every slice keeps an irreducible floor)",
             rlevers and all(l.get("can_reach_zero") is False for l in rlevers))
    # comm/free floors are formulas over the LOADED measured slices: 0 < floor < measured,
    # so the recoverable is a strict positive fraction, never the whole slice → 0.
    for lid in ("comm_overlap", "free_zero"):
        lv = rl_by.get(lid) or {}
        chk_true(f"{lid}: 0 < floor < measured (cannot zero out)",
                 lv.get("floor_us") is not None and lv.get("measured_us") is not None
                 and 0 < lv["floor_us"] < lv["measured_us"],
                 f"(floor={lv.get('floor_us')} measured={lv.get('measured_us')})")
        chk_true(f"{lid}: recoverable > 0 with lo<=mid<=hi band",
                 (lv.get("recoverable_us") or 0) > 0
                 and lv.get("recoverable_lo_us") <= lv.get("recoverable_us") <= lv.get("recoverable_hi_us"),
                 f"(lo={lv.get('recoverable_lo_us')} mid={lv.get('recoverable_us')} hi={lv.get('recoverable_hi_us')})")
    # op-余量 lever mirrors the efficiency ceiling reclaim (compute is useful work, not →0)
    op_rl = rl_by.get("op_ceiling")
    chk_true("op_ceiling realistic lever mirrors efficiency reclaim",
             op_rl is not None
             and abs((op_rl.get("recoverable_us") or 0) - (oco.get("total_reclaim_us") or 0)) <= 1.0,
             f"(lever={op_rl.get('recoverable_us') if op_rl else None} reclaim={oco.get('total_reclaim_us')})")
    # scenario labels carry the realistic floor UP into the upper What-if table (web +
    # report): comm/free read 实测%→地板% (e.g. "未掩盖通信 26.1%→4.5%"), op is descriptive.
    # This is what the user asked for: replace the unachievable "→0" labels with the floor.
    chk_true("every realistic lever has a non-empty scenario label",
             bool(rlevers) and all((l.get("scenario") or "").strip() for l in rlevers),
             f"(scenarios={[l.get('scenario') for l in rlevers]})")
    for lid in ("comm_overlap", "free_zero"):
        lv = rl_by.get(lid) or {}
        sc = lv.get("scenario") or ""
        chk_true(f"{lid}: scenario is 实测%→地板% (measured→floor, not →0)",
                 "%→" in sc
                 and f"{lv.get('measured_pct'):.1f}" in sc
                 and f"{lv.get('floor_pct'):.1f}" in sc,
                 f"(scenario={sc!r} measured_pct={lv.get('measured_pct')} floor_pct={lv.get('floor_pct')})")
    # 失真 caveats are CONDITIONED on this capture (single-card + blocking both on here).
    chk_true("realistic flags track capture (single_card & blocking on)",
             rz.get("single_card") is True and rz.get("blocking") is True,
             f"(single_card={rz.get('single_card')} blocking={rz.get('blocking')})")
    comm_cav = " ".join((rl_by.get("comm_overlap") or {}).get("caveats") or [])
    free_cav = " ".join((rl_by.get("free_zero") or {}).get("caveats") or [])
    chk_true("comm lever caveats note single-card + blocking distortion",
             "communication_matrix" in comm_cav and "ASCEND_LAUNCH_BLOCKING" in comm_cav,
             f"(caveats={(rl_by.get('comm_overlap') or {}).get('caveats')})")
    chk_true("free lever caveat notes blocking distortion",
             "ASCEND_LAUNCH_BLOCKING" in free_cav)
    # combined realistic floor: a genuine floor ABOVE the physical "→0" upper bound, and
    # strictly BELOW the base step (it does recover real time). Bands ordered.
    rcomb = rz.get("combined") or {}
    chk_true("realistic combined present (step+MFU+band)",
             rcomb.get("new_step_us") is not None and rcomb.get("new_mfu") is not None
             and rcomb.get("new_step_lo_us") is not None and rcomb.get("new_step_hi_us") is not None,
             f"(combined={rcomb})")
    chk_true("realistic floor sits between →0 upper bound and base step",
             comb.get("new_step_us") is not None
             and comb["new_step_us"] < (rcomb.get("new_step_us") or 0) < u["stage"],
             f"(upper={comb.get('new_step_us')} realistic={rcomb.get('new_step_us')} base={u['stage']})")
    chk_true("realistic band ordered (hi=fastest <= mid <= lo=slowest)",
             rcomb.get("new_step_hi_us") <= rcomb.get("new_step_us") <= rcomb.get("new_step_lo_us"),
             f"(hi={rcomb.get('new_step_hi_us')} mid={rcomb.get('new_step_us')} lo={rcomb.get('new_step_lo_us')})")
    chk_true("realistic combined recoverable == Σ lever recoverables (disjoint slices add)",
             abs((rcomb.get("recoverable_us") or 0) - sum(l.get("recoverable_us") or 0 for l in rlevers)) <= 1.0,
             f"(combined={rcomb.get('recoverable_us')} sum={sum(l.get('recoverable_us') or 0 for l in rlevers)})")
    chk("realistic combined new_step us (~1.90s floor)", rcomb.get("new_step_us"), 1900187.0, 5000.0)
    chk("realistic combined recoverable %", rcomb.get("recoverable_pct"), 39.23, 0.5)

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

    print("\n== shareable report (H4) ==")
    from llminsight.report import build_report_html
    rhtml = build_report_html(m, cards)
    chk_true("report is a self-contained HTML doc",
             rhtml.lstrip().lower().startswith("<!doctype html"),
             f"({len(rhtml)//1024} KB)")
    # self-contained: no external assets / network → emailable, offline, printable
    externals = [t for t in ("<script src", "<link", "@import", 'src="http',
                             "src='http", 'href="http', "url(http")
                 if t in rhtml.lower()]
    chk_true("report has no external assets", not externals, f"(found={externals})")
    # carries the same section views the UI shows (diagnostics-first ordering)
    for marker in ("诊断", "时间构成", "What-if", "算子热点", "通信", "隐性开销", "结构归因"):
        chk_true(f"report section: {marker}", marker in rhtml)
    # embeds every rule-engine card (titles rendered verbatim)
    chk_true("report embeds all card titles",
             all(c.get("title", "") in rhtml for c in cards if c.get("title")))
    # privacy: same whitelist as the LLM summary — no raw trace path / PII / key
    rleaks = [t for t in ("d00568668", "plog", "CPU_AFFINITY", "/home/",
                          "ASCEND_PROCESS_LOG", "api_key") if t in rhtml]
    chk_true("no PII/secret leak in report", not rleaks, f"(leaks={rleaks})")

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
