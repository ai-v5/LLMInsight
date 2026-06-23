"""Self-contained, shareable HTML report (H4 · 一键导出可分享报告).

Renders the rule-engine diagnostic cards + the key metrics from a computed
metrics dict into a SINGLE static HTML file — no server, no JavaScript framework,
no external assets or CDN — so it can be emailed, attached, archived, or
"另存为 PDF" and opened anywhere offline.

Privacy: the report carries the same aggregated numbers the UI shows. It never
embeds the raw trace, tensor data, the API key, or absolute paths (the data
directory is masked to its `secret/…` tail / basename). This keeps an exported
report safe to share, consistent with the LLM summary's privacy contract.

Usage
-----
    from llminsight.report import build_report_html
    html = build_report_html(state.metrics, state.cards)

    # headless (no server):
    python -m llminsight.report --out report.html
"""
from __future__ import annotations

import html as _html
from datetime import datetime
from typing import Any, Dict, List, Optional

# --------------------------------------------------------------------------- #
# Formatting helpers (mirror web/js/util.js fmt.* so numbers read identically).
# --------------------------------------------------------------------------- #
def _e(s: Any) -> str:
    return _html.escape("" if s is None else str(s))


def _us(v: Any) -> str:
    if v is None:
        return "—"
    try:
        v = float(v)
    except (TypeError, ValueError):
        return "—"
    if abs(v) >= 1e6:
        return f"{v / 1e6:.2f} s"
    if abs(v) >= 1e3:
        return f"{v / 1e3:.1f} ms"
    return f"{v:.0f} us"


def _pct(v: Any, d: int = 1) -> str:
    """v is already in percent units (e.g. 55.29 -> '55.3%')."""
    if v is None:
        return "—"
    try:
        return f"{float(v):.{d}f}%"
    except (TypeError, ValueError):
        return "—"


def _mfu(v: Any) -> str:
    """v is a fraction (0.4088 -> '40.9%')."""
    if v is None:
        return "—"
    try:
        return f"{float(v) * 100:.1f}%"
    except (TypeError, ValueError):
        return "—"


def _int(v: Any) -> str:
    if v is None:
        return "—"
    try:
        return f"{round(float(v)):,}"
    except (TypeError, ValueError):
        return "—"


def _tf(v: Any) -> str:
    if v is None:
        return "—"
    try:
        return f"{float(v):.0f} TFLOPS"
    except (TypeError, ValueError):
        return "—"


_SEV = {
    "high": ("高", "high"), "medium": ("中", "medium"),
    "low": ("低", "low"), "info": ("提示", "info"),
}


def _mask_path(p: Any) -> str:
    """Never leak home dirs / usernames: keep only the `secret/…` tail, else the
    basename."""
    if not p:
        return "—"
    s = str(p).replace("\\", "/")
    low = s.lower()
    i = low.rfind("/secret/")
    if i >= 0:
        return "secret/" + s[i + len("/secret/"):]
    if low.startswith("secret/"):
        return s
    return s.rsplit("/", 1)[-1]


def _table(headers: List[str], rows: List[List[str]],
           align: Optional[List[str]] = None,
           row_classes: Optional[List[str]] = None) -> str:
    """Build a <table>. Cells are pre-formatted HTML (callers escape text).

    `row_classes` (optional, aligned with `rows`) adds a class to a <tr> — used
    e.g. to dim a base row or rule-off a summary row."""
    if align is None:
        align = ["l"] + ["r"] * (len(headers) - 1)
    cls = {"l": "tl", "r": "tr", "c": "tc"}
    ths = "".join(f'<th class="{cls[a]}">{_e(h)}</th>'
                  for h, a in zip(headers, align))
    body = []
    for i, row in enumerate(rows):
        tds = "".join(f'<td class="{cls[a]}">{c}</td>'
                      for c, a in zip(row, align))
        rc = ""
        if row_classes and i < len(row_classes) and row_classes[i]:
            rc = f' class="{_e(row_classes[i])}"'
        body.append(f"<tr{rc}>{tds}</tr>")
    return (f'<table class="tbl"><thead><tr>{ths}</tr></thead>'
            f'<tbody>{"".join(body)}</tbody></table>')


def _metric(label: str, value: str, foot: str = "", tone: str = "") -> str:
    return (f'<div class="metric {tone}"><div class="m-label">{_e(label)}</div>'
            f'<div class="m-value">{value}</div>'
            + (f'<div class="m-foot">{_e(foot)}</div>' if foot else "")
            + "</div>")


def _pctf(frac: Any, d: int = 1) -> str:
    """frac is a 0–1 fraction (0.966 -> '96.6%')."""
    if frac is None:
        return "—"
    try:
        return f"{float(frac) * 100:.{d}f}%"
    except (TypeError, ValueError):
        return "—"


def _series_stats(series: Optional[List[Any]]):
    """(avg, p95, max) of a 0–1 fraction series, ignoring None. Empty -> all None."""
    xs = sorted(float(v) for v in (series or []) if v is not None)
    if not xs:
        return (None, None, None)
    n = len(xs)
    avg = sum(xs) / n
    p95 = xs[min(n - 1, int(round(0.95 * (n - 1))))]
    return (avg, p95, xs[-1])


def _ibar(frac: Any, color: str = "") -> str:
    """Inline mini-bar with a % label — the static stand-in for a curve/heat cell."""
    if frac is None:
        return "—"
    try:
        f = float(frac)
    except (TypeError, ValueError):
        return "—"
    w = max(0.0, min(1.0, f)) * 100
    return (f'<div class="ibar"><span style="width:{w:.0f}%;'
            f'background:{_e(color or "#0e7490")}"></span>'
            f'<em>{f * 100:.1f}%</em></div>')


# --------------------------------------------------------------------------- #
# Sections
# --------------------------------------------------------------------------- #
def _sec_header(meta: Dict, overview: Dict, theo: Dict,
                eff: Dict, generated_at: datetime) -> str:
    settings = (meta or {}).get("settings", {}) or {}
    cfg = (meta or {}).get("config", {}) or {}        # profiling-derived (parser.derive)
    dm = cfg.get("model", {}) or {}
    gs = cfg.get("guesses", {}) or {}
    cap = cfg.get("capture", {}) or {}
    schip = settings.get("chip", {}) or {}
    echip = (eff or {}).get("chip", {}) or {}

    def _glabel(key):                                 # 未知(猜X) label for underivable fields
        return (gs.get(key, {}) or {}).get("label", "未知")

    def _capval(key):
        return (cap.get(key, {}) or {}).get("value")

    arch_bits = [b for b in (
        f"hidden {dm['hidden_size']}" if dm.get("hidden_size") else None,
        f"{dm['num_attention_heads']} heads" if dm.get("num_attention_heads") else None,
    ) if b]
    arch = "MLA + MoE" + (" · " + " · ".join(arch_bits) if arch_bits else "")
    dtype = dm.get("dtype") or "—"
    par = f"EP={_glabel('ep_world_size')} · TP/PP/CP 未知（单卡不可得）"
    moe = (f"experts {_glabel('num_experts')} · top-{dm.get('moe_router_topk', '?')}"
           f" · 每卡 {dm.get('local_experts_per_rank', '?')} 专家 · 专家FFN {dm.get('moe_ffn_hidden_size', '?')}")
    rc = _capval("recompute")
    cap_txt = (f"recompute {rc if rc is not None else '未知'} · "
               f"{'单卡' if _capval('single_card') else '多卡'} · "
               f"blocking {'检出' if _capval('blocking') else '未检出（数据推断）'}")
    chip_name = echip.get("name") or schip.get("name") or "—"
    cube_peak = echip.get("cube_bf16_tflops") or echip.get("peak_bf16_tflops")
    eff_peak = echip.get("effective_peak_tflops")
    calibrated = echip.get("calibrated")
    peak_txt = _tf(cube_peak) + " cube bf16"
    if eff_peak and cube_peak and abs(float(eff_peak) - float(cube_peak)) > 1.0:
        peak_txt += f"（生效 {float(eff_peak):.0f}）"
    if calibrated:
        peak_txt += " · 已按实测校准"
    elif schip.get("assumed"):
        peak_txt += " · 参考峰值"

    rows = [
        ("模型结构", f"{_e(arch)} · {_e(dtype)}"),
        ("并行", _e(par)),
        ("MoE", _e(moe)),
        ("序列 / 批", f"seq {_int(dm.get('seq_length'))} · global-batch {_e(_glabel('global_batch_size'))}"),
        ("训练 / 采集", _e(cap_txt)),
        ("参考芯片", f"{_e(chip_name)} · {peak_txt}"),
        ("采集", f"step {overview.get('step','—')} · device {overview.get('device_id','—')} · "
                 f"{_e(_mask_path(settings.get('data_dir')))}"),
        ("生成时间", _e(generated_at.strftime("%Y-%m-%d %H:%M"))),
    ]
    kv = "".join(f'<div class="k">{_e(k)}</div><div class="v">{v}</div>'
                 for k, v in rows)

    # hero metrics
    r = (overview or {}).get("ratios", {}) or {}
    u = (overview or {}).get("us", {}) or {}
    step_mfu = (theo or {}).get("step_mfu")
    step_hfu = (theo or {}).get("step_hfu")
    rco_hero = (theo or {}).get("recompute") or {}
    comb = (theo or {}).get("whatif_combined") or {}
    cards = [
        _metric("Step 时长", _us(u.get("stage")),
                f"≈ {r.get('step_time_s','—')} s"),
        _metric("有效计算占比", _pct(r.get("effective_compute_pct")),
                "Computing / Stage",
                "good" if (r.get("effective_compute_pct") or 0) >= 60 else "warn"),
        _metric("未掩盖通信", _pct(r.get("comm_not_overlapped_pct")),
                "直接计入 step",
                "bad" if (r.get("comm_not_overlapped_pct") or 0) >= 20 else "warn"),
        _metric("空泡 Free", _pct(r.get("free_pct")), "device 空闲",
                "warn" if (r.get("free_pct") or 0) >= 10 else ""),
        _metric("通信掩盖率", _pct(r.get("overlap_rate_pct")),
                "Overlapped / Communication",
                "bad" if (r.get("overlap_rate_pct") or 0) < 40 else ""),
        _metric("端到端 MFU", _mfu(step_mfu),
                (f"HFU {_mfu(step_hfu)}（含重算）" if rco_hero.get("overhead_us")
                 else (f"全优化上界 {_mfu(comb.get('new_mfu'))}" if comb.get("new_mfu") else "达成算力 / 峰值"))),
    ]
    return (
        '<header class="rep-head">'
        '<div class="rh-title"><div class="rh-logo">LLM<span>Insight</span></div>'
        '<div class="rh-sub">昇腾 NPU 大模型训练 · 性能洞察报告</div></div>'
        f'<div class="rh-meta kv">{kv}</div>'
        f'<div class="hero grid">{"".join(cards)}</div>'
        '</header>'
    )


def _sec_insights(cards: List[Dict]) -> str:
    if not cards:
        return ""
    counts: Dict[str, int] = {}
    for c in cards:
        counts[c.get("severity", "info")] = counts.get(c.get("severity", "info"), 0) + 1
    chips = " ".join(
        f'<span class="sev {s}">{_SEV.get(s, (s, "info"))[0]} {counts[s]}</span>'
        for s in ("high", "medium", "low", "info") if counts.get(s))
    items = []
    for c in cards:
        sev = c.get("severity", "info")
        zh, scls = _SEV.get(sev, (sev, "info"))
        body_rows = [
            ("根因", c.get("root_cause")),
            ("建议", c.get("suggestion")),
            ("预计收益", c.get("expected_gain")),
        ]
        body = "".join(
            f'<div class="k">{_e(k)}</div>'
            f'<div class="v{" gain" if k == "预计收益" else ""}">{_e(v)}</div>'
            for k, v in body_rows if v)
        conf = c.get("confidence")
        conf_txt = f"置信度 {float(conf):.0%}" if isinstance(conf, (int, float)) else ""
        items.append(
            f'<div class="insight {scls}">'
            f'<div class="ic-head"><span class="sev {scls}">{_e(zh)}</span>'
            f'<span class="ic-cat">{_e(c.get("category",""))}</span>'
            f'<span class="ic-title">{_e(c.get("title",""))}</span>'
            f'<span class="ic-conf">{_e(conf_txt)}</span></div>'
            f'<div class="ic-body">{body}</div></div>')
    return _panel(
        "诊断卡片 · 规则引擎",
        f"按严重度排序，共 {len(cards)} 项 &nbsp; {chips}",
        "".join(items))


def _sec_overview(overview: Dict) -> str:
    if not overview or not overview.get("available"):
        return ""
    comp = overview.get("composition", []) or []
    colors = {"Computing": "#1a7f37",
              "Communication (Not Overlapped)": "#cf222e", "Free": "#bf8700"}
    segs, legend = [], []
    for c in comp:
        nm, pct = c.get("name", ""), c.get("pct", 0)
        col = colors.get(nm, "#8b949e")
        segs.append(f'<span style="width:{max(pct,0)}%;background:{col}" '
                    f'title="{_e(nm)} {pct}%"></span>')
        legend.append(f'<span class="lg"><i style="background:{col}"></i>'
                      f'{_e(nm)} · {_pct(pct)} · {_us(c.get("us"))}</span>')
    bar = (f'<div class="stack">{"".join(segs)}</div>'
           f'<div class="legend">{"".join(legend)}</div>')
    return _panel("Step 时间构成", "Stage = Computing + 未掩盖通信 + Free（三者合计 100%）", bar)


def _sec_theoretical(theo: Dict) -> str:
    """What-if section — mirrors the live 总览 page's 'What-if 收益模拟 · 现实可达地板'
    exactly: it renders the REALISTIC floor levers (theo['realistic']), each labelled
    实测%→地板%, with per-lever 单独节省(s)/(%) (= recoverable) shown as negatives and the
    端到端 MFU column as a gain (+x.x%). A ruled-off combined row sums the disjoint
    recoverables; its MFU is the absolute reached value plus the gain. The physical
    →0 upper bound (theo['whatif_combined']) is kept only as a one-line reference."""
    if not theo or not theo.get("available"):
        return ""
    smfu = theo.get("step_mfu")  # base end-to-end MFU, 0–1

    def saved_s(us):
        try:
            us = float(us)
        except (TypeError, ValueError):
            return "—"
        return f"-{us / 1e6:.2f} s" if us > 0 else "—"

    def saved_pct(p):
        try:
            return f"-{float(p):.1f}%" if float(p) > 0 else "—"
        except (TypeError, ValueError):
            return "—"

    def dmfu(nv):  # gain vs base, e.g. '+14.5%' / '-1.2%' / '—'
        if smfu is None or nv is None:
            return "—"
        d = (nv - smfu)
        return ("+" if d >= 0 else "-") + f"{abs(d) * 100:.1f}%"

    r = theo.get("realistic") or {}
    levers = r.get("levers") or []
    rows = [["当前（base）", "—", "—", _mfu(smfu)]]          # dim anchor row
    row_cls = ["dim-row"]
    for w in levers:
        rows.append([
            _e(w.get("scenario", "")),
            f'<span class="pos">{saved_s(w.get("recoverable_us"))}</span>',
            f'<span class="pos">{saved_pct(w.get("recoverable_pct"))}</span>',
            f'<span class="gain">{dmfu(w.get("new_mfu"))}</span>',
        ])
        row_cls.append("")
    comb = r.get("combined") or {}
    if comb:
        nv = comb.get("new_mfu")
        mfu_cell = "—" if nv is None else f'{_mfu(nv)}（{dmfu(nv)}）'
        rows.append([
            f"已启用组合 ({len(levers)}/{len(levers)})",
            f'<span class="pos">{saved_s(comb.get("recoverable_us"))}</span>',
            f'<span class="pos">{saved_pct(comb.get("recoverable_pct"))}</span>',
            f'<span class="gain">{mfu_cell}</span>',
        ])
        row_cls.append("sum-row")
    tbl = _table(["优化项", "单独节省(s)", "单独节省(%)", "端到端 MFU"],
                 rows, row_classes=row_cls)

    # compute-bound footnote — same wording/branching as the live page.
    cb = theo.get("compute_bound") or {}
    cb_note = ""
    if cb:
        if cb.get("peak_underestimated"):
            cb_note = ('<div class="banner warn">matmul 实测 MFU <strong>'
                       f'{_e(cb.get("matmul_mfu_pct"))}%</strong> &gt; 100% → '
                       '假设芯片峰值偏低，请在 config.ChipSpec 校正。</div>')
        else:
            ceil = "（≈ 天花板）" if cb.get("ceiling_based") else ""
            calib = ""
            if cb.get("calibrated"):
                ap = cb.get("assumed_peak_tflops")
                calib = (f'（芯片峰值按实测 {_e(cb.get("observed_peak_tflops"))} '
                         f'TFLOPS 校准，假设 {f"{ap:.0f}" if ap is not None else "—"}）')
            cb_note = (f'<div class="note">matmul MFU ≈ <strong>'
                       f'{_e(cb.get("matmul_mfu_pct"))}%</strong>{ceil}；'
                       f'算子达天花板后计算 {_us(cb.get("ideal_matmul_us"))}，'
                       f'可回收 {_us(cb.get("headroom_us"))}。{calib}</div>')

    ub = theo.get("whatif_combined") or {}   # physical →0 upper bound, reference only
    ub_ref = ""
    if ub.get("new_mfu") is not None:
        ub_ref = (f' 📐 物理上界（全部 →0，理论不可达）参考：step {_us(ub.get("new_step_us"))} / '
                  f'端到端 MFU {_mfu(ub.get("new_mfu"))} / 省 {_pct(ub.get("save_pct"))}。')
    tip = ('<div class="note">💡 此表为<strong>现实可达地板</strong>（非「→0」物理上界）：'
           '通信重叠至 80–90% 留残留、Free 留 step 2–5%、算子按各自 MFU 天花板'
           '（matmul 95% / FA 85% / FAG 70%）收口——单项收益小而精，而非冲到 100% 的虚高。'
           '优先级：先做计算-通信重叠（--moe-fb-overlap / 异步通信），再压同步空泡。'
           + ub_ref +
           '　⚠️ 「未掩盖通信」与算子页 <strong>HcclLaunchAicpuKernel</strong> '
           '是同一段集合通信（单卡几乎全是 Wait，非下发延迟），勿重复计入。</div>')

    note = (f'<div class="note">{_e(theo.get("note"))}</div>'
            if theo.get("note") else "")
    sub = "每项为达现实地板（业界可达上限）时单独可回收的收益（实测%→地板%）；底部「已启用组合」叠加各项（含重计算，导出快照）"
    return _panel("What-if 收益模拟 · 现实可达地板", sub,
                  tbl + cb_note + tip + note)


def _sec_whatif_floor(theo: Dict) -> str:
    """逐项「能否减到 0 / 现实地板」—— rendered from theo['realistic'], which is
    derived from the loaded profile (not a static sample)."""
    if not theo or not theo.get("available"):
        return ""
    r = theo.get("realistic") or {}
    levers = r.get("levers") or []
    if not levers:
        return ""
    items = []
    for lv in levers:
        zero = "可减到 0" if lv.get("can_reach_zero") else "不能减到 0（有不可消除下限）"
        methods = "".join(f"<li>{_e(x)}</li>" for x in lv.get("methods") or [])
        reasons = "".join(f"<li>{_e(x)}</li>" for x in lv.get("reasons") or [])
        caveats = "".join(
            f'<div class="banner" style="margin-top:8px">⚠️ {_e(c)}</div>'
            for c in lv.get("caveats") or [])
        line = (f'实测 <strong>{_us(lv.get("measured_us"))}</strong>'
                f'（step {_pct(lv.get("measured_pct"))}）'
                f' → 现实地板 ≈ <strong>{_us(lv.get("floor_us"))}</strong>'
                f'（step {_pct(lv.get("floor_pct"))}）'
                f' · 可回收 ≈ <strong>{_us(lv.get("recoverable_us"))}</strong>'
                f'（区间 {_us(lv.get("recoverable_lo_us"))}–{_us(lv.get("recoverable_hi_us"))}）'
                f' · 优化后 step {_us(lv.get("new_step_us"))} / 端到端 MFU {_mfu(lv.get("new_mfu"))}')
        items.append(
            '<div style="margin:13px 0;padding-top:11px;border-top:1px solid var(--line)">'
            f'<div class="mini-h">{_e(lv.get("title"))} · 能减到 0？{zero}</div>'
            f'<div class="p-sub" style="margin-bottom:7px">{_e(lv.get("floor_basis"))}</div>'
            f'<div style="margin-bottom:9px">{line}</div>'
            '<div class="two-col">'
            f'<div><div class="mini-h">优化方法</div><ul>{methods}</ul></div>'
            f'<div><div class="mini-h">为何不能到 0 / 下限来源</div><ul>{reasons}</ul></div>'
            f'</div>{caveats}</div>')
    comb = r.get("combined") or {}
    comb_html = ""
    if comb:
        comb_html = (
            '<div class="banner" style="margin-top:12px"><strong>综合现实地板</strong>：'
            f'优化后 step ≈ <strong>{_us(comb.get("new_step_us"))}</strong>'
            f'（区间 {_us(comb.get("new_step_hi_us"))}–{_us(comb.get("new_step_lo_us"))}），'
            f'端到端 MFU ≈ <strong>{_mfu(comb.get("new_mfu"))}</strong>'
            f'（{_mfu(comb.get("new_mfu_lo"))}–{_mfu(comb.get("new_mfu_hi"))}），'
            f'省 ≈ <strong>{_pct(comb.get("recoverable_pct"))}</strong>。'
            f'{_e(comb.get("basis"))}</div>')
    return _panel("What-if 严谨性分析 · 能否减到 0 / 现实地板",
                  _e(r.get("note", "")), "".join(items) + comb_html)


def _sec_hotspots(hot: Dict) -> str:
    if not hot or not hot.get("available"):
        return ""
    top = (hot.get("top") or [])[:15]
    rows = [[_e(o.get("type", "")), _e(o.get("core", "")),
             _int(o.get("count")), _us(o.get("total_us")),
             _pct(o.get("ratio"), 2)] for o in top]
    tbl = _table(["算子类型", "Core", "次数", "总耗时", "占比"], rows)
    core_rows = [[_e(b.get("core", "")), _int(b.get("count")),
                  _us(b.get("total_us")), _pct(b.get("pct"))]
                 for b in (hot.get("by_core") or [])]
    core_tbl = _table(["Core 类型", "次数", "总耗时", "占比"], core_rows)
    return _panel(
        "算子热点 Top 15",
        f"算子总耗时 {_us(hot.get('total_us'))}（含 AICPU 集合通信执行）",
        f'<div class="two-col"><div>{tbl}</div>'
        f'<div><div class="mini-h">按 Core 类型聚合</div>{core_tbl}</div></div>')


def _sec_efficiency(eff: Dict) -> str:
    if not eff or not eff.get("available"):
        return ""
    top = (eff.get("top_optimization") or [])[:8]
    rows = []
    for r in top:
        rows.append([
            _e(str(r.get("name", ""))[:46]),
            f'<span class="tag {_e(r.get("bound",""))}">{_e(r.get("bound",""))}</span>',
            _mfu(r.get("mfu")), _mfu(r.get("mbu")),
            _us(r.get("dur_us")), _us(r.get("wasted_us")),
        ])
    tbl = _table(["Kernel", "Bound", "MFU", "MBU", "实测耗时", "可优化余量"],
                 rows, align=["l", "c", "r", "r", "r", "r"])
    mm = eff.get("matmul_mfu")
    pu = eff.get("peak_underestimated")
    sub = (f"matmul（cube/GEMM）MFU {_mfu(mm)}"
           + ("（实测超假设峰值 → 已校准）" if pu else "")
           + f" · roofline ridge AI {eff.get('roofline_ridge_ai','—')}"
           + f" · 建模 kernel {_int(eff.get('kernels_with_flops'))} 个")
    return _panel("算子效率 · Top 优化候选（vs Roofline 理想）", sub, tbl)


def _sec_communication(comm: Dict) -> str:
    if not comm or not comm.get("available"):
        return ""
    def _bw(v):
        return f'{v:.1f} GB/s' if isinstance(v, (int, float)) else "—"
    rows = [[_e(t.get("type", "")), _int(t.get("count")),
             f'{t.get("elapse_ms","—")} ms', f'{t.get("wait_ms","—")} ms',
             _pct(t.get("wait_pct")), f'{t.get("transit_mb","—")} MB',
             _bw(t.get("bandwidth_with_wait_gbps")),
             _bw(t.get("bandwidth_gbps"))] for t in (comm.get("by_type") or [])]
    tbl = _table(["通信类型", "次数", "Elapse", "Wait", "等待占比", "流量",
                  "平均带宽(含等待)", "有效带宽(去等待)"], rows)
    obw = comm.get("overall_bandwidth_gbps")
    obw_wait = comm.get("overall_bandwidth_with_wait_gbps")
    sub = (f"{comm.get('count','—')} 次集合通信 · 总 Elapse {comm.get('total_elapse_ms','—')} ms · "
           f"平均等待占比 {_pct(comm.get('overall_wait_pct'))} · "
           f"Transit {comm.get('total_transit_mb','—')} MB · "
           f"平均带宽(含等待) {_bw(obw_wait)} · 有效带宽(去等待) {_bw(obw)}")
    bd = comm.get("breakdown") or {}
    wc = (bd.get("wall_clock") or {}).get("within_comm_not_overlapped") or {}
    bd_note = ""
    if bd:
        def _us_val(d, key):
            v = d.get(key, "—")
            return _e(f"{v} us")
        bd_note = (f'<div class="note">device 子任务分解：等待累加 {_us_val(bd, "wait_us")} '
                   f'({_pct(bd.get("wait_pct"))})，有效传输累加 {_us_val(bd, "transfer_us")} '
                   f'({_pct(bd.get("transfer_pct"))})'
                   + (f'；未掩盖通信窗口内等待墙钟 {_us_val(wc, "wait_wall_us")}，'
                      f'传输墙钟 {_us_val(wc, "transfer_wall_us")}' if wc else "")
                   + '</div>')
    note = f'<div class="note">{_e(comm.get("note", ""))}</div>'
    return _panel("通信分析", sub, tbl + bd_note + note)


def _sec_hidden(ho: Dict) -> str:
    if not ho or not ho.get("available"):
        return ""
    dom = {"device": "Device", "host": "Host", "config": "配置"}
    rows = []
    for b in (ho.get("buckets") or []):
        add = b.get("additive", True)
        label = _e(b.get("label", ""))
        rows.append([
            label,
            _e(dom.get(b.get("domain", ""), b.get("domain", ""))),
            (_us(b.get("us")) if not (b.get("additive", True) is False)
             else f'<span class="muted">{_us(b.get("us"))} *</span>')
            if b.get("us") is not None else "—",
            _e(str(b.get("suggestion", ""))[:64]),
        ])
    tbl = _table(["隐性开销项", "域", "耗时", "减负建议"],
                 rows, align=["l", "c", "r", "l"])
    sub = (f"Device 合计 {_us(ho.get('device_total_us'))}（与 step 同口径）· "
           f"Host 合计 {_us(ho.get('host_total_us'))}（反映下发/同步压力，不与 device 相加）")
    note = ('<div class="note">* AICPU 集合通信执行与「未掩盖通信」为同一段时间，'
            '仅作算子视角，不并入 Device 合计（避免重复计入）。</div>')
    return _panel("隐性开销总账", sub, tbl + note)


def _sec_attribution(attr: Dict) -> str:
    if not attr or not attr.get("available"):
        return ""
    rows = [[_e(m.get("module", "")), _us(m.get("us")), _pct(m.get("pct"))]
            for m in (attr.get("modules") or [])]
    tbl = _table(["模型结构模块", "device 耗时", "占比"], rows)
    moe = attr.get("moe_focus") or {}
    comm_rows = [[_e(c.get("module", "")), _us(c.get("us")), _pct(c.get("pct"))]
                 for c in (attr.get("comm_breakdown") or [])]
    comm_tbl = _table(["通信归因", "wall-clock", "占比"], comm_rows)
    moe_line = (f'<div class="note">MoE 专属：专家计算 (GroupedMatmul) '
                f'{_us(moe.get("expert_compute_us"))} vs 分发通信 (alltoallv) '
                f'{_us(moe.get("dispatch_comm_us"))}。</div>')
    return _panel(
        "模型结构归因",
        "按算子命名启发式归因到 MLA / MoE / Norm / Optimizer 等（基于 device 计算时间）",
        f'<div class="two-col"><div>{tbl}</div>'
        f'<div><div class="mini-h">通信（wall-clock，单列不入计算环）</div>{comm_tbl}'
        f'{moe_line}</div></div>')


def _sec_memory(mem: Dict) -> str:
    if not mem:
        return ""
    trade_rows = [[_e(t.get("feature", "")), _e(t.get("effect", "")), _e(t.get("advice", ""))]
                  for t in (mem.get("config_tradeoffs") or [])]
    trade_tbl = _table(["配置开关", "效果", "建议"], trade_rows, align=["l", "l", "l"])

    # ---- graceful fallback: capture lacked memory-level files ----
    if not (mem.get("available") and mem.get("hbm_timeline_available")):
        banner = ('<div class="banner info">本次采集无 memory-level 数据'
                  '（memory_record.csv / npu_module_mem.csv 缺失），无法绘制真实 HBM 峰值/构成；'
                  '以下为配置驱动的内存-时间权衡顾问。</div>')
        return _panel("显存洞察 · 内存-时间权衡", "待 memory-level 采集接入", banner + trade_tbl)

    # ---- real memory-level data ----
    s = mem.get("summary", {})

    def gib(mb):
        return "—" if mb is None else f"{mb / 1024:.1f} GiB"

    near = s.get("near_oom")
    snap = [
        _metric("HBM 峰值占用", gib(s.get("peak_reserved_mb")),
                f'{_pct(s.get("util_pct"))} of {s.get("capacity_gb")} GB', "warn" if near else "good"),
        _metric("活跃张量峰值", gib(s.get("peak_allocated_mb")), "Total Allocated"),
        _metric("保留未占用 (碎片)", gib(s.get("fragmentation_mb")),
                f'占峰值 {_pct(s.get("fragmentation_pct"))}', "warn" if (s.get("fragmentation_pct") or 0) >= 15 else ""),
        _metric("可用 headroom", gib(s.get("headroom_mb")), f'{s.get("capacity_gb")} GB', "bad" if near else "good"),
    ]
    banner = (f'<div class="banner {"warn" if near else "info"}">'
              + (f'<strong>显存逼近容量</strong> —— 峰值保留 {gib(s.get("peak_reserved_mb"))} ≈ '
                 f'<strong>{_pct(s.get("util_pct"))}</strong> of {s.get("capacity_gb")} GB，可用 headroom 仅 '
                 f'{gib(s.get("headroom_mb"))}，OOM 风险高。' if near
                 else f'HBM 峰值保留 {gib(s.get("peak_reserved_mb"))}（{_pct(s.get("util_pct"))} of '
                      f'{s.get("capacity_gb")} GB），尚有 {gib(s.get("headroom_mb"))} headroom。')
              + '</div>')

    decomp_rows = [
        ["峰值已保留 (进程 HBM 占用，计入容量)", gib(s.get("peak_reserved_mb"))],
        ["活跃张量峰值 (Total Allocated)", gib(s.get("peak_allocated_mb"))],
        ["分配器缓存池 (PTA/PTA+GE Reserved)", gib(s.get("pool_reserved_mb"))],
        ["└ 分配器缓存碎片 (池保留 − 活跃)", gib(s.get("alloc_slack_mb"))],
        ["通信/workspace/运行时保留", gib(s.get("nontensor_reserved_mb"))
         + (f"（HCCL {gib(s.get('hccl_reserved_mb'))}）" if s.get("hccl_reserved_mb") else "")],
    ]
    decomp_tbl = _table(["构成（峰值口径近似）", "大小"],
                        [[_e(r[0]), r[1]] for r in decomp_rows], align=["l", "r"])

    mod_rows = [[_e(o.get("module", "")), gib(o.get("mb")), _pct(o.get("pct"))]
                for o in (mem.get("modules") or [])]
    mod_tbl = _table(["驱动模块", "保留峰值", "占比"], mod_rows, align=["l", "r", "r"])

    alloc_rows = [[_e(a.get("name", "")), gib(a.get("total_mb")),
                   f'{a.get("count", "")}', gib(a.get("max_mb"))]
                  for a in (mem.get("top_allocators") or [])]
    alloc_tbl = _table(["框架算子", "累计分配", "次数", "单次峰值"], alloc_rows,
                       align=["l", "r", "r", "r"])

    live_rows = [[_e(l.get("name", "")), f'{l.get("life_s", 0):.2f} s', gib(l.get("mb"))]
                 for l in (mem.get("longest_lived") or [])]
    live_tbl = _table(["张量(算子)", "存活", "大小"],
                      live_rows or [["—", "—", "—"]], align=["l", "r", "r"])

    body = (
        f'<div class="grid g4">{"".join(snap)}</div>'
        + banner
        + '<div class="two-col"><div><div class="mini-h">显存构成（峰值口径近似）</div>'
        + decomp_tbl + '</div><div><div class="mini-h">驱动模块保留峰值（HCCL=集合通信缓冲）</div>'
        + mod_tbl + '</div></div>'
        + '<div class="two-col"><div><div class="mini-h">分配压力 Top 框架算子（累计分配）</div>'
        + alloc_tbl + '</div><div><div class="mini-h">最长存活张量（≥1 MB，长期占用 HBM）</div>'
        + live_tbl + '</div></div>'
        + '<div class="mini-h">内存 ↔ 时间权衡顾问（配置驱动）</div>' + trade_tbl
        + (f'<div class="note">{_e(mem.get("note", ""))}</div>' if mem.get("note") else ""))
    return _panel("显存洞察 · HBM 峰值 / 碎片 / 内存-时间权衡",
                  f'峰值 {gib(s.get("peak_reserved_mb"))} · {_pct(s.get("util_pct"))} of {s.get("capacity_gb")} GB · {s.get("span_s")}s · {s.get("samples")} 采样',
                  body)


_UTIL_SHORT = {"cube": "Cube", "vector": "Vector", "hbm_bw": "HBM", "comm": "通信"}
_STREAM_SHORT = {"cube": "Cube", "flash_attn": "FlashAttn", "vector": "Vector",
                 "mix": "MIX", "comm": "通信", "other": "其它"}


def _svg_smart_timeline(stl: Dict) -> str:
    """Self-contained inline-SVG of the Gantt × utilization lanes (no JS, no deps).

    Mirrors the live page's stacked layout on one shared time axis: utilization
    area lanes (Cube/Vector/HBM/通信, 0–100%) on top, operator stream lanes below.
    Vector output stays crisp in print / 另存为 PDF. Operator rects carry a native
    <title> (hover tooltip in browsers) without any script."""
    slices = stl.get("slices") or []
    streams = stl.get("streams") or []
    util = [u for u in (stl.get("utilization") or [])
            if u.get("available") and u.get("series")]
    span_us = float(stl.get("span_us") or 0)
    if not span_us or (not slices and not util):
        return ""
    span_ms = span_us / 1000.0
    span_s = float(stl.get("span_s") or (span_ms / 1000.0))

    W, PAD_L, PAD_R, PAD_T = 1040, 100, 14, 10
    H_U, GAP_U = 46, 7
    H_OP, GAP_OP = 20, 4
    SEP, AX = 16, 22
    plotW = W - PAD_L - PAD_R
    right_x = PAD_L + plotW

    by_stream: Dict[str, list] = {}
    for s in slices:
        by_stream.setdefault(s.get("stream"), []).append(s)
    scolor = {s.get("key"): s.get("color") for s in streams}
    present = [s.get("key") for s in streams if by_stream.get(s.get("key"))]
    for k in by_stream:                       # defensive: any stream not in streams[]
        if k not in present:
            present.append(k)

    nU, nOp = len(util), len(present)
    util_h = nU * H_U + max(0, nU - 1) * GAP_U
    gantt_h = nOp * H_OP + max(0, nOp - 1) * GAP_OP
    gantt_top = PAD_T + util_h + SEP
    grid_bot = gantt_top + gantt_h
    H = grid_bot + AX

    def xms(ms):
        return PAD_L + (float(ms) / span_ms) * plotW

    out = [f'<svg class="stl-svg" viewBox="0 0 {W} {H}" width="100%" '
           f'preserveAspectRatio="xMidYMid meet" '
           f'font-family="-apple-system,Segoe UI,Microsoft YaHei,sans-serif">']

    # vertical time grid + axis labels (behind content)
    ticks = 6
    for i in range(ticks + 1):
        fx = i / ticks
        gx = PAD_L + fx * plotW
        out.append(f'<line x1="{gx:.1f}" y1="{PAD_T}" x2="{gx:.1f}" '
                   f'y2="{grid_bot:.1f}" stroke="#eceff2"/>')
        out.append(f'<text x="{gx:.1f}" y="{H-7}" font-size="10" fill="#8b949e" '
                   f'text-anchor="middle">{fx*span_s:.2f}s</text>')

    cols = max(60, min(int(plotW), 930))

    def ds_max(series):
        n = len(series)
        if n <= cols:
            return [max(0.0, min(1.0, float(v or 0))) for v in series]
        res = []
        for c in range(cols):
            a, b = c * n // cols, max(c * n // cols + 1, (c + 1) * n // cols)
            mx = 0.0
            for v in series[a:b]:
                if v is not None and float(v) > mx:
                    mx = float(v)
            res.append(max(0.0, min(1.0, mx)))
        return res

    # utilization area lanes
    y = PAD_T
    for u in util:
        col = _e(u.get("color") or "#0e7490")
        ds = ds_max(u.get("series") or [])
        m = len(ds)
        band_bot = y + H_U
        out.append(f'<rect x="{PAD_L}" y="{y}" width="{plotW}" height="{H_U}" '
                   f'fill="#ffffff" stroke="#eceff2"/>')
        out.append(f'<line x1="{PAD_L}" y1="{y+H_U/2:.1f}" x2="{right_x}" '
                   f'y2="{y+H_U/2:.1f}" stroke="#f3f5f7" stroke-dasharray="3 3"/>')
        pts = [f"{PAD_L+(c+0.5)/m*plotW:.1f},{y+H_U*(1-v):.1f}"
               for c, v in enumerate(ds)]
        out.append(f'<path d="M{PAD_L},{band_bot:.1f} L{"L".join(pts)} '
                   f'L{right_x},{band_bot:.1f} Z" fill="{col}" fill-opacity="0.16"/>')
        out.append(f'<polyline points="{" ".join(pts)}" fill="none" '
                   f'stroke="{col}" stroke-width="1.1"/>')
        short = _e(_UTIL_SHORT.get(u.get("key"), str(u.get("key") or "")[:8]))
        out.append(f'<text x="{PAD_L-7}" y="{y+H_U/2+3.5:.1f}" font-size="11" '
                   f'fill="#57606a" text-anchor="end">{short}</text>')
        out.append(f'<text x="{right_x-3}" y="{y+12:.1f}" font-size="9.5" '
                   f'fill="{col}" text-anchor="end">峰值 {(max(ds) if ds else 0)*100:.0f}%</text>')
        y += H_U + GAP_U

    out.append(f'<line x1="{PAD_L}" y1="{gantt_top-SEP/2:.1f}" x2="{right_x}" '
               f'y2="{gantt_top-SEP/2:.1f}" stroke="#d8dee4"/>')

    # operator stream lanes (Gantt)
    for idx, key in enumerate(present):
        lane_top = gantt_top + idx * (H_OP + GAP_OP)
        col = _e(scolor.get(key) or "#8b949e")
        out.append(f'<rect x="{PAD_L}" y="{lane_top}" width="{plotW}" '
                   f'height="{H_OP}" fill="#fafbfc" stroke="#eceff2"/>')
        short = _e(_STREAM_SHORT.get(key, str(key or "")[:9]))
        out.append(f'<text x="{PAD_L-7}" y="{lane_top+H_OP/2+3.5:.1f}" '
                   f'font-size="10" fill="#57606a" text-anchor="end">{short}</text>')
        out.append(f'<g fill="{col}">')
        ry, rh = lane_top + 1.5, H_OP - 3
        for s in by_stream.get(key, []):
            x0 = xms(s.get("start_ms") or 0)
            w = (float(s.get("dur_ms") or 0) / span_ms) * plotW
            if x0 + max(w, 0.5) > right_x:
                w = right_x - x0
            w = max(0.5, w)
            if w >= 2.0:
                tip = f'{_e(str(s.get("name", ""))[:48])} · {_us((s.get("dur_ms") or 0)*1e3)}'
                if s.get("mfu") is not None:
                    tip += f' · MFU {_mfu(s.get("mfu"))}'
                out.append(f'<rect x="{x0:.1f}" y="{ry:.1f}" width="{w:.1f}" '
                           f'height="{rh}"><title>{tip}</title></rect>')
            else:
                out.append(f'<rect x="{x0:.1f}" y="{ry:.1f}" '
                           f'width="{w:.1f}" height="{rh}"/>')
        out.append('</g>')

    out.append('</svg>')
    return "".join(out)


def _sec_smart_timeline(stl: Dict) -> str:
    if not stl or not stl.get("available"):
        return ""
    util = [u for u in (stl.get("utilization") or [])
            if u.get("available") and u.get("series")]
    cards = [
        _metric("时间跨度", f"{stl.get('span_s', '—')} s",
                f"{_int(stl.get('bins'))} 桶 · {_us(stl.get('bin_us'))}/桶"),
        _metric("利用率泳道", _int(len(util)), "Cube / Vector / HBM / 通信"),
        _metric("算子切片", _int(stl.get("shown_slices")),
                f"共 {_int(stl.get('total_slices'))}（按时长下采样）"),
        _metric("已建模占比", _pct(stl.get("modeled_pct")),
                "matmul/attention 计算覆盖墙钟",
                "good" if (stl.get("modeled_pct") or 0) >= 40 else "warn"),
    ]
    kpi = f'<div class="grid g4">{"".join(cards)}</div>'

    # utilization lanes -> avg / p95 / peak%, plus peak absolute for rate lanes
    urows = []
    for u in util:
        avg, p95, mx = _series_stats(u.get("series"))
        peak_abs = "—"
        if u.get("kind") == "rate":
            mxa = max((float(x) for x in (u.get("abs") or []) if x is not None),
                      default=None)
            if mxa is not None:
                peak_abs = (f'{mxa:,.0f} {_e(u.get("abs_unit", ""))} / 峰值 '
                            f'{_int(u.get("peak"))} {_e(u.get("peak_unit", ""))}')
        urows.append([
            f'<span class="dot" style="background:{_e(u.get("color"))}"></span>'
            f'{_e(u.get("label", ""))}',
            _ibar(avg, u.get("color")), _pctf(p95), _pctf(mx), peak_abs,
        ])
    util_tbl = _table(["利用率泳道", "平均", "P95", "峰值", "峰值绝对值（rate 类）"],
                      urows, align=["l", "l", "r", "r", "r"])

    # longest slices -> mirror the Gantt hover (per-slice MFU/MBU/dtype)
    slabel = {s.get("key"): s.get("label") for s in (stl.get("streams") or [])}
    longest = sorted((stl.get("slices") or []),
                     key=lambda s: -(s.get("dur_ms") or 0))[:10]
    srows = [[
        _e(str(s.get("name", ""))[:44]),
        _e(slabel.get(s.get("stream"), s.get("stream", ""))),
        _us((s.get("dur_ms") or 0) * 1e3),
        _mfu(s.get("mfu")), _mfu(s.get("mbu")), _e(s.get("dtype") or "—"),
    ] for s in longest]
    slice_tbl = _table(["算子切片", "泳道", "耗时", "MFU", "MBU", "dtype"],
                       srows, align=["l", "l", "r", "r", "r", "c"])

    svg = _svg_smart_timeline(stl)
    util_leg = "".join(
        f'<span class="lg"><i style="background:{_e(u.get("color"))}"></i>'
        f'{_e(u.get("label", ""))}</span>' for u in util)
    op_leg = "".join(
        f'<span class="lg"><i style="background:{_e(s.get("color"))}"></i>'
        f'{_e(s.get("label", ""))}</span>' for s in (stl.get("streams") or []))
    legend = (f'<div class="legend"><span class="muted">利用率</span>{util_leg}</div>'
              f'<div class="legend" style="margin-top:5px">'
              f'<span class="muted">算子泳道</span>{op_leg}</div>')

    sub = ("算子按 stream 泳道铺成 Gantt，上叠 Cube/Vector/HBM/通信 利用率（0–100%），"
           "共享时间轴；矢量内联图，打印/转 PDF 清晰。下方表格补充利用率分布与最长切片 MFU/MBU。")
    body = (kpi + (legend + svg if svg else "")
            + '<div class="mini-h" style="margin-top:14px">利用率泳道统计</div>' + util_tbl
            + '<div class="mini-h" style="margin-top:13px">'
            '最长算子切片 · 悬停同款 MFU/MBU（Top 10）</div>' + slice_tbl)
    return _panel("智能时间线 · 算子泳道 Gantt × 利用率泳道", sub, body)


def _sec_timeline(tl: Dict) -> str:
    if not tl or not tl.get("available"):
        return ""
    comp = tl.get("computing_pct")
    notov = tl.get("not_overlapped_pct")
    free = tl.get("free_pct")
    cards = [
        _metric("时间跨度", f"{tl.get('span_s', '—')} s",
                f"{_int(tl.get('bins'))} 桶 · {len(tl.get('lanes') or [])} 泳道"),
        _metric("有效计算", _pct(comp), "Computing / 总步长",
                "good" if (comp or 0) >= 60 else "warn"),
        _metric("未掩盖通信", _pct(notov), "通信未被计算掩盖",
                "bad" if (notov or 0) >= 20 else "warn"),
        _metric("Free 空泡", _pct(free), "设备完全空闲（可优化）",
                "warn" if (free or 0) >= 10 else ""),
    ]
    kpi = f'<div class="grid g4">{"".join(cards)}</div>'

    lrows = []
    for l in (tl.get("lanes") or []):
        occ = l.get("occupancy") or []
        avg = (sum(occ) / len(occ)) if occ else None
        lrows.append([_e(l.get("label", "")), _ibar(avg, "#0e7490")])
    lane_tbl = _table(["泳道", "平均占用"], lrows, align=["l", "l"])

    freq = [f.get("mhz") for f in (tl.get("ai_core_freq") or [])
            if f.get("mhz") is not None]
    freq_note = ""
    if freq:
        freq_note = (f'<div class="note">AI Core 频率：min {min(freq):,} · '
                     f'平均 {sum(freq) / len(freq):,.0f} · max {max(freq):,} MHz'
                     f'（{len(freq)} 采样点）</div>')

    top = (tl.get("top_slices") or [])[:15]
    srows = [[_e(str(s.get("name", ""))[:46]),
              f'{(s.get("start_us", 0) / 1e6):.2f} s', _us(s.get("dur_us"))]
             for s in top]
    slice_tbl = _table(["最长计算 Kernel (>1.5ms)", "起始", "时长"],
                       srows, align=["l", "r", "r"])

    sub = ("泳道占用热力 · Overlap 三段（Computing / 未掩盖通信 / Free）· "
           "AI Core 频率 · 最长计算切片。静态报告以表格呈现热力图与曲线。")
    body = (kpi + '<div class="two-col"><div>' + lane_tbl + freq_note + '</div>'
            + '<div><div class="mini-h">最长真实计算 kernel（已剔除 WAIT/NOTIFY）'
            '</div>' + slice_tbl + '</div></div>')
    return _panel("时间线 · 泳道占用 / 频率 / 切片", sub, body)


def _sec_replay(overview: Dict) -> str:
    r = (overview or {}).get("ratios", {}) or {}
    step = (overview or {}).get("step", "—")
    snap = [
        _metric("有效计算", _pct(r.get("effective_compute_pct")), "", "good"),
        _metric("未掩盖通信", _pct(r.get("comm_not_overlapped_pct")), "", "bad"),
        _metric("空闲 Free", _pct(r.get("free_pct")), "", "warn"),
        _metric("Step 时间", f'{r.get("step_time_s", "—")} s'),
    ]
    bn = ('<div class="banner info"><strong>全训练回放（骨架）</strong> —— '
          f'当前数据为单 step（step {_e(step)}）单帧；回放轴与联动机制已搭好，'
          '多 step 全程指标（loss / 吞吐 / 显存 / 通信抖动）随后续多 step + '
          '训练日志采集接入。</div>')
    plan = ('<div class="note">回放将支持：全程趋势叠加（耗时构成 / 未掩盖通信占比 / '
            '显存峰值 / loss·吞吐·grad-norm 趋势线，定位「第几步开始变慢」）；'
            '任意两 step 对比（Δ 自动定位回归来源算子，轴上打标变慢 / 显存爬升 / '
            '通信抖动 / loss 突刺）。</div>')
    body = (bn + '<div class="mini-h">当前帧快照（联动总览）</div>'
            f'<div class="grid g4">{"".join(snap)}</div>' + plan)
    return _panel("全训练回放", "单 step 单帧 · 多 step 趋势与对比为设计预留", body)


# --------------------------------------------------------------------------- #
def _panel(title: str, sub: str, body: str) -> str:
    return (f'<section class="panel"><h2>{_e(title)}</h2>'
            + (f'<div class="p-sub">{sub}</div>' if sub else "")
            + f'<div class="p-body">{body}</div></section>')


_CSS = """
:root{--bg:#f4f6f8;--card:#fff;--ink:#1f2329;--dim:#57606a;--mut:#8b949e;
--line:#d8dee4;--line2:#c4ccd4;--accent:#0e7490;--accent2:#0f766e;
--good:#1a7f37;--warn:#9a6700;--bad:#cf222e;--low:#0969da;--info:#57606a;}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);
font-family:-apple-system,"Segoe UI","Microsoft YaHei",Roboto,Helvetica,Arial,sans-serif;
font-size:13px;line-height:1.5;-webkit-font-smoothing:antialiased;}
.wrap{max-width:1080px;margin:0 auto;padding:22px 22px 60px;}
.toolbar{display:flex;justify-content:flex-end;gap:10px;margin-bottom:12px;}
.toolbar button{background:var(--accent);color:#fff;border:none;border-radius:7px;
padding:7px 14px;font:inherit;font-weight:600;cursor:pointer;}
.toolbar button:hover{filter:brightness(1.08);}
.rep-head{background:var(--card);border:1px solid var(--line);border-radius:12px;
padding:20px 22px;margin-bottom:16px;box-shadow:0 1px 3px rgba(27,31,36,.06);}
.rh-logo{font-size:24px;font-weight:800;letter-spacing:.3px;}
.rh-logo span{color:var(--accent);}
.rh-sub{color:var(--mut);font-size:12.5px;margin-top:2px;}
.rh-meta{margin:16px 0 6px;}
.kv{display:grid;grid-template-columns:auto 1fr;gap:5px 16px;font-size:12.8px;}
.kv .k{color:var(--mut);white-space:nowrap;}
.kv .v{color:var(--ink);}
.hero{margin-top:16px;}
.grid{display:grid;grid-template-columns:repeat(6,1fr);gap:10px;}
@media(max-width:820px){.grid{grid-template-columns:repeat(3,1fr);}}
.grid.g4{grid-template-columns:repeat(4,1fr);}
@media(max-width:820px){.grid.g4{grid-template-columns:repeat(2,1fr);}}
.ibar{position:relative;background:#eef1f4;border:1px solid var(--line);
border-radius:5px;height:16px;min-width:122px;overflow:hidden;}
.ibar span{position:absolute;left:0;top:0;bottom:0;border-radius:5px 0 0 5px;opacity:.85;}
.ibar em{position:relative;font-style:normal;font-size:11px;padding-left:7px;
line-height:16px;color:var(--ink);}
.dot{display:inline-block;width:9px;height:9px;border-radius:50%;
margin-right:7px;vertical-align:middle;}
.stl-svg{display:block;width:100%;height:auto;background:#fff;
border:1px solid var(--line);border-radius:8px;margin:8px 0 2px;}
.metric{background:var(--bg);border:1px solid var(--line);border-radius:9px;padding:10px 12px;}
.metric .m-label{font-size:11.5px;color:var(--dim);}
.metric .m-value{font-size:20px;font-weight:750;margin-top:3px;}
.metric .m-foot{font-size:10.5px;color:var(--mut);margin-top:2px;}
.metric.good .m-value{color:var(--good);}
.metric.warn .m-value{color:var(--warn);}
.metric.bad .m-value{color:var(--bad);}
.panel{background:var(--card);border:1px solid var(--line);border-radius:12px;
padding:15px 18px;margin-bottom:14px;box-shadow:0 1px 3px rgba(27,31,36,.06);}
.panel h2{margin:0 0 3px;font-size:15px;font-weight:700;}
.panel .p-sub{font-size:11.8px;color:var(--mut);margin-bottom:11px;}
.two-col{display:grid;grid-template-columns:1.4fr 1fr;gap:18px;align-items:start;}
@media(max-width:820px){.two-col{grid-template-columns:1fr;}}
.mini-h{font-size:11.5px;color:var(--dim);font-weight:600;margin:0 0 6px;}
table.tbl{width:100%;border-collapse:collapse;font-size:12.3px;}
table.tbl th,table.tbl td{padding:5px 9px;border-bottom:1px solid var(--line);}
table.tbl th{color:var(--mut);font-weight:600;background:var(--bg);}
.tl{text-align:left;} .tr{text-align:right;} .tc{text-align:center;}
table.tbl tbody tr:nth-child(even) td{background:#fafbfc;}
.muted{color:var(--mut);}
.tag{display:inline-block;padding:1px 7px;border-radius:5px;font-size:10.5px;font-weight:600;}
.tag.compute{background:rgba(9,105,218,.12);color:#0969da;}
.tag.memory{background:rgba(154,103,0,.14);color:#9a6700;}
.tag.vector{background:rgba(15,118,110,.12);color:#0f766e;}
.tag.other{background:#eef1f4;color:var(--dim);}
.stack{display:flex;height:24px;border-radius:6px;overflow:hidden;border:1px solid var(--line);}
.stack span{display:block;height:100%;}
.legend{display:flex;flex-wrap:wrap;gap:8px 18px;margin-top:10px;font-size:12px;color:var(--dim);}
.legend .lg{display:inline-flex;align-items:center;gap:6px;}
.legend i{width:11px;height:11px;border-radius:3px;display:inline-block;}
.insight{border:1px solid var(--line);border-left:4px solid var(--info);
border-radius:9px;padding:10px 14px;margin-bottom:9px;background:#fff;}
.insight.high{border-left-color:var(--bad);}
.insight.medium{border-left-color:var(--warn);}
.insight.low{border-left-color:var(--low);}
.ic-head{display:flex;align-items:center;gap:9px;margin-bottom:6px;flex-wrap:wrap;}
.ic-title{font-size:13.6px;font-weight:650;}
.ic-cat{font-size:11px;color:var(--mut);border:1px solid var(--line2);padding:1px 7px;border-radius:5px;}
.ic-conf{margin-left:auto;font-size:11px;color:var(--mut);}
.sev{font-size:10.5px;font-weight:700;padding:2px 8px;border-radius:5px;}
.sev.high{background:rgba(207,34,46,.13);color:var(--bad);}
.sev.medium{background:rgba(154,103,0,.15);color:var(--warn);}
.sev.low{background:rgba(9,105,218,.13);color:var(--low);}
.sev.info{background:#eef1f4;color:var(--dim);}
.ic-body{display:grid;grid-template-columns:60px 1fr;gap:4px 12px;font-size:12.6px;}
.ic-body .k{color:var(--mut);}
.ic-body .v.gain{color:var(--accent2);font-weight:600;}
.banner{border:1px solid rgba(14,116,144,.3);background:rgba(14,116,144,.07);
color:#0b5e73;border-radius:8px;padding:9px 13px;font-size:12.3px;margin-bottom:11px;}
.banner.warn{border-color:rgba(154,103,0,.35);background:rgba(154,103,0,.08);color:#7a5200;}
.pos{color:var(--good);}
.gain{color:var(--accent2);font-weight:600;}
table.tbl tr.dim-row td{color:var(--mut);}
table.tbl tr.sum-row td{border-top:2px solid rgba(15,118,110,.45);font-weight:600;}
.note{font-size:11.5px;color:var(--mut);margin-top:9px;line-height:1.6;}
.rep-foot{color:var(--mut);font-size:11.5px;text-align:center;margin-top:8px;line-height:1.7;}
@media print{body{background:#fff;}.toolbar{display:none;}
.panel,.rep-head{box-shadow:none;break-inside:avoid;}}
"""


def build_report_html(metrics: Dict[str, Any],
                      cards: List[Dict[str, Any]],
                      generated_at: Optional[datetime] = None) -> str:
    """Render a complete, self-contained HTML report string.

    `metrics` is the dict from `compute_all` (or `AppState.metrics`); `cards`
    is the rule-engine output (`run_rules`). No I/O, no external assets — the
    returned string is the whole file.
    """
    metrics = metrics or {}
    cards = cards or []
    gen = generated_at or datetime.now()

    meta = metrics.get("meta", {}) or {}
    overview = metrics.get("overview", {}) or {}
    theo = metrics.get("theoretical", {}) or {}
    eff = metrics.get("efficiency", {}) or {}

    # Walk the left-nav top-to-bottom so the report mirrors the app page order:
    # 总览 → 智能时间线 → 算子效率 → 算子热点 → 通信 → 隐性开销 → 结构归因 →
    # 显存 → 时间线 → LLM 洞察 → 全训练回放. The header is the report masthead
    # (总览 hero); overview composition + What-if are the rest of 总览.
    sections = [
        _sec_header(meta, overview, theo, eff, gen),
        _sec_overview(overview),                              # 总览
        _sec_theoretical(theo),                              # 总览 · What-if
        _sec_whatif_floor(theo),                             # 总览 · What-if 现实地板
        _sec_smart_timeline(metrics.get("smart_timeline", {})),  # 智能时间线
        _sec_efficiency(eff),                                # 算子效率
        _sec_hotspots(metrics.get("hotspots", {})),          # 算子热点
        _sec_communication(metrics.get("communication", {})),  # 通信分析
        _sec_hidden(metrics.get("hidden_overhead", {})),     # 隐性开销
        _sec_attribution(metrics.get("attribution", {})),    # 结构归因
        _sec_memory(metrics.get("memory", {})),              # 显存洞察
        _sec_timeline(metrics.get("timeline", {})),          # 时间线
        _sec_insights(cards),                                # LLM 洞察
        _sec_replay(overview),                               # 全训练回放
    ]
    foot = (
        '<div class="rep-foot">本报告由 LLMInsight 规则引擎生成（LLM 叙述为可选增强，未内联）。'
        'MFU / MBU / Roofline / What-if 基于参考芯片峰值，为上界估算用于优化排序，非精确预测。'
        '数据为单卡单 step 演示。</div>'
    )
    body = "".join(s for s in sections if s)
    return (
        '<!DOCTYPE html>\n<html lang="zh-CN">\n<head>\n'
        '<meta charset="UTF-8">\n'
        '<meta name="viewport" content="width=device-width, initial-scale=1.0">\n'
        '<title>LLMInsight 性能洞察报告</title>\n'
        f'<style>{_CSS}</style>\n</head>\n<body>\n<div class="wrap">\n'
        '<div class="toolbar"><button onclick="window.print()">🖨 打印 / 另存为 PDF</button></div>\n'
        f'{body}\n{foot}\n</div>\n</body>\n</html>\n'
    )


# --------------------------------------------------------------------------- #
def _main(argv=None) -> int:
    import argparse
    from .config import SETTINGS
    from .parser import load_profile
    from .metrics import compute_all
    from .rules import run_rules

    ap = argparse.ArgumentParser(
        description="Export a self-contained LLMInsight HTML report (no server).")
    ap.add_argument("--out", "-o", default="llminsight_report.html",
                    help="output HTML path (default: llminsight_report.html)")
    args = ap.parse_args(argv)

    print(f"[report] loading profile from: {SETTINGS.data_dir}")
    prof = load_profile(SETTINGS.data_dir)
    m = compute_all(prof)
    # compute_all derived the config from the profiling and stashed it on m.meta;
    # reuse it so cards reflect the data (no launch-script dependency).
    cards = run_rules(m, m.get("meta", {}).get("config"))
    html = build_report_html(m, cards)
    with open(args.out, "w", encoding="utf-8") as fh:
        fh.write(html)
    print(f"[report] wrote {args.out}  ({len(html):,} bytes, {len(cards)} cards)")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
