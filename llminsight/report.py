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
           align: Optional[List[str]] = None) -> str:
    """Build a <table>. Cells are pre-formatted HTML (callers escape text)."""
    if align is None:
        align = ["l"] + ["r"] * (len(headers) - 1)
    cls = {"l": "tl", "r": "tr", "c": "tc"}
    ths = "".join(f'<th class="{cls[a]}">{_e(h)}</th>'
                  for h, a in zip(headers, align))
    body = []
    for row in rows:
        tds = "".join(f'<td class="{cls[a]}">{c}</td>'
                      for c, a in zip(row, align))
        body.append(f"<tr>{tds}</tr>")
    return (f'<table class="tbl"><thead><tr>{ths}</tr></thead>'
            f'<tbody>{"".join(body)}</tbody></table>')


def _metric(label: str, value: str, foot: str = "", tone: str = "") -> str:
    return (f'<div class="metric {tone}"><div class="m-label">{_e(label)}</div>'
            f'<div class="m-value">{value}</div>'
            + (f'<div class="m-foot">{_e(foot)}</div>' if foot else "")
            + "</div>")


# --------------------------------------------------------------------------- #
# Sections
# --------------------------------------------------------------------------- #
def _sec_header(meta: Dict, overview: Dict, theo: Dict,
                eff: Dict, generated_at: datetime) -> str:
    settings = (meta or {}).get("settings", {}) or {}
    model = settings.get("model", {}) or {}
    schip = settings.get("chip", {}) or {}
    echip = (eff or {}).get("chip", {}) or {}

    name = model.get("name", "—")
    par = (f"TP{model.get('tp','?')} / PP{model.get('pp','?')} / "
           f"EP{model.get('ep','?')} / CP{model.get('cp','?')}")
    moe = (f"{model.get('num_experts','?')} experts · top-{model.get('moe_router_topk','?')}")
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
        ("模型", f"{_e(name)} · {_e(model.get('dtype','bf16'))}"),
        ("并行", _e(par)),
        ("MoE", _e(moe)),
        ("序列 / 批", f"seq {_int(model.get('seq_length'))} · global-batch {_e(model.get('global_batch_size','?'))}"),
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
                (f"全优化上界 {_mfu(comb.get('new_mfu'))}" if comb.get("new_mfu") else "达成算力 / 峰值")),
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
    if not theo or not theo.get("available"):
        return ""
    cur = theo.get("current_step_us")
    rows = []
    for w in theo.get("whatif", []) or []:
        rows.append([
            _e(w.get("scenario", "")),
            _us(w.get("save_us")),
            f'<strong>{_pct(w.get("save_pct"))}</strong>',
            _us(w.get("new_step_us")),
            _mfu(w.get("new_mfu")),
        ])
    comb = theo.get("whatif_combined") or {}
    if comb:
        rows.append([
            '<strong>全部优化项叠加</strong>',
            f'<strong>{_us(comb.get("save_us"))}</strong>',
            f'<strong>{_pct(comb.get("save_pct"))}</strong>',
            f'<strong>{_us(comb.get("new_step_us"))}</strong>',
            f'<strong>{_mfu(comb.get("new_mfu"))}</strong>',
        ])
    tbl = _table(["What-if 优化项", "可省", "占 step", "优化后 step", "端到端 MFU"], rows)
    sub = (f"当前 step {_us(cur)}。各项为互不重叠的原子优化，收益可叠加；"
           "数值为基于 step 构成的上界估算，用于排优先级。")
    return _panel("理论上界 & What-if 收益", sub, tbl)


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
    rows = [[_e(t.get("type", "")), _int(t.get("count")),
             f'{t.get("elapse_ms","—")} ms', f'{t.get("wait_ms","—")} ms',
             _pct(t.get("wait_pct"))] for t in (comm.get("by_type") or [])]
    tbl = _table(["通信类型", "次数", "Elapse", "Wait", "等待占比"], rows)
    sub = (f"{comm.get('count','—')} 次集合通信 · 总 Elapse {comm.get('total_elapse_ms','—')} ms · "
           f"平均等待占比 {_pct(comm.get('overall_wait_pct'))} · "
           f"Transit {comm.get('total_transit_mb','—')} MB")
    note = ('<div class="note">单卡采集：集合通信几乎全为 Wait / Synchronization，'
            'Transit≈0 → 通信时间以「等待对端」为主，真实链路带宽需多卡数据。</div>')
    return _panel("通信分析", sub, tbl + note)


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
    rows = [[_e(t.get("feature", "")), _e(t.get("effect", "")), _e(t.get("advice", ""))]
            for t in (mem.get("config_tradeoffs") or [])]
    tbl = _table(["配置开关", "效果", "建议"], rows, align=["l", "l", "l"])
    banner = ('<div class="banner info">本次采集无 memory-level 数据'
              '（memory_record.csv / npu_module_mem.csv 缺失），无法绘制真实 HBM 峰值/构成；'
              '以下为配置驱动的内存-时间权衡顾问。</div>')
    return _panel("显存洞察 · 内存-时间权衡", "待 memory-level 采集接入", banner + tbl)


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

    sections = [
        _sec_header(meta, overview, theo, eff, gen),
        _sec_insights(cards),
        _sec_overview(overview),
        _sec_theoretical(theo),
        _sec_hotspots(metrics.get("hotspots", {})),
        _sec_efficiency(eff),
        _sec_communication(metrics.get("communication", {})),
        _sec_hidden(metrics.get("hidden_overhead", {})),
        _sec_attribution(metrics.get("attribution", {})),
        _sec_memory(metrics.get("memory", {})),
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
    from .rules import run_rules, read_capture_config

    ap = argparse.ArgumentParser(
        description="Export a self-contained LLMInsight HTML report (no server).")
    ap.add_argument("--out", "-o", default="llminsight_report.html",
                    help="output HTML path (default: llminsight_report.html)")
    args = ap.parse_args(argv)

    print(f"[report] loading profile from: {SETTINGS.data_dir}")
    prof = load_profile(SETTINGS.data_dir)
    m = compute_all(prof)
    cards = run_rules(m, read_capture_config())
    html = build_report_html(m, cards)
    with open(args.out, "w", encoding="utf-8") as fh:
        fh.write(html)
    print(f"[report] wrote {args.out}  ({len(html):,} bytes, {len(cards)} cards)")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
