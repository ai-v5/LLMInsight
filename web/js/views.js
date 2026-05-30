/* LLMInsight frontend — view renderers. Each async fn fills its root element. */
(function () {
  const { api, apiPost, fmt, esc, metric, panel, banner, chart, axis, tooltipBase, sevColor } = LI;
  const V = {};

  function charts(defs) { // defs: [{id, option}] — init after DOM exists
    requestAnimationFrame(() => defs.forEach(d => {
      const node = document.getElementById(d.id);
      if (node && d.option) chart(node, d.option);
    }));
  }

  // ============================ 总览 Overview ============================ //
  V.overview = async function (root) {
    const [ov, theo] = await Promise.all([api("/api/overview"), api("/api/theoretical")]);
    if (!ov.available) { root.innerHTML = `<div class="empty">无 step_trace 数据</div>`; return; }
    const r = ov.ratios, u = ov.us;
    const smfu = theo.available ? theo.step_mfu : null;  // end-to-end (step) MFU, 0–1
    const cards = [
      metric("有效计算占比", fmt.pct(r.effective_compute_pct), { tone: r.effective_compute_pct >= 60 ? "good" : "warn", foot: "Computing / Stage", barPct: r.effective_compute_pct }),
      metric("未掩盖通信", fmt.pct(r.comm_not_overlapped_pct), { tone: "bad", foot: fmt.us(u.comm_not_overlapped), barPct: r.comm_not_overlapped_pct }),
      metric("空闲 Free", fmt.pct(r.free_pct), { tone: "warn", foot: fmt.us(u.free), barPct: r.free_pct }),
      metric("通信掩盖率", fmt.pct(r.overlap_rate_pct), { tone: r.overlap_rate_pct < 40 ? "bad" : "good", foot: "Overlapped / Communication", barPct: r.overlap_rate_pct }),
      metric("端到端 MFU", fmt.mfu(smfu), { tone: smfu == null ? undefined : (smfu >= 0.5 ? "good" : smfu >= 0.3 ? "warn" : "bad"), foot: "有效FLOPs /(峰值×step)", barPct: smfu != null ? smfu * 100 : undefined }),
      metric("Step 时间", r.step_time_s + " s", { foot: "Stage = " + fmt.us(u.stage) }),
    ];
    const levers = theo.available ? theo.whatif : [];
    const baseStep = theo.available ? theo.current_step_us : 0;
    // MFU column shows the gain vs base (step_mfu); the base row stays the absolute anchor.
    const dmfu = (nv) => (smfu == null || nv == null) ? "—" : (nv - smfu >= 0 ? "+" : "-") + fmt.mfu(Math.abs(nv - smfu));
    // step column shows time saved vs base, in seconds (— at base, which saves nothing).
    const saved = (us) => us > 0 ? "-" + (us / 1e6).toFixed(2) + " s" : "—";
    const baseRow = theo.available
      ? `<tr style="color:var(--text-dim)"><td></td><td>当前（base）</td><td>—</td><td>—</td><td>${fmt.mfu(smfu)}</td></tr>`
      : "";
    const leverRows = levers.map(w =>
      `<tr><td style="text-align:center"><input type="checkbox" class="wi-lever" style="accent-color:#5ee0b8;cursor:pointer" data-save="${w.save_us}" checked></td>`
      + `<td>${esc(w.scenario)}</td><td style="color:var(--accent-2)">${saved(w.save_us)}</td>`
      + `<td style="color:var(--accent-2)">-${fmt.pct(w.save_pct)}</td>`
      + `<td style="color:var(--accent)">${dmfu(w.new_mfu)}</td></tr>`).join("");
    const cb = theo.available && theo.compute_bound;
    // "已启用组合" row recomputed from whichever levers are ticked. 未掩盖通信 and Free
    // are disjoint slices of Stage, so savings add; MFU = smfu · stage / new_step.
    const recomputeCombined = () => {
      const ticked = Array.from(document.querySelectorAll("input.wi-lever")).filter(el => el.checked);
      const saveUs = ticked.reduce((s, el) => s + (+el.dataset.save || 0), 0);
      const newStep = Math.max(baseStep - saveUs, 0);
      const savePct = baseStep ? (saveUs / baseStep * 100) : 0;
      const newMfu = (smfu && newStep > 0) ? smfu * baseStep / newStep : null;
      const row = document.getElementById("wi-combined");
      if (row) row.innerHTML =
        `<td style="text-align:center;color:var(--accent-2)">✓</td>`
        + `<td><strong>已启用组合 (${ticked.length}/${levers.length})</strong></td>`
        + `<td style="color:var(--accent-2)"><strong>${saved(saveUs)}</strong></td>`
        + `<td style="color:var(--accent-2)">${saveUs > 0 ? "-" + fmt.pct(savePct) : "—"}</td>`
        + `<td style="color:var(--accent)"><strong>${dmfu(newMfu)}</strong></td>`;
    };
    root.innerHTML = `
      <div class="grid cols-6">${cards.join("")}</div>
      <div class="grid cols-2" style="margin-top:16px">
        ${panel("Step 时间构成", "Computing / 未掩盖通信 / Free（单位 us）", `<div id="ov-donut" class="chart"></div>`)}
        ${panel("理论上界 & What-if 收益模拟", "每项为单独启用的收益；勾选后底部「已启用组合」实时显示叠加效果",
          `<table class="tbl"><thead><tr><th style="width:38px">启用</th><th>优化项</th><th>单独节省(s)</th><th>单独节省(%)</th><th>端到端 MFU</th></tr></thead>`
          + `<tbody>${theo.available ? (baseRow + leverRows + `<tr id="wi-combined" style="border-top:2px solid rgba(94,224,184,.35)"></tr>`) : `<tr><td colspan=5 class="empty">—</td></tr>`}</tbody></table>
           ${cb ? (cb.peak_underestimated
             ? banner("warn","⚠️", `matmul 实测 MFU <strong>${cb.matmul_mfu_pct}%</strong> &gt; 100% → 假设芯片峰值偏低，请在 config.ChipSpec 校正。`)
             : `<div class="note">matmul MFU ≈ <strong>${cb.matmul_mfu_pct}%</strong>${cb.ceiling_based ? "（≈ 天花板）" : ""}；算子达天花板后计算 ${fmt.us(cb.ideal_matmul_us)}，可回收 ${fmt.us(cb.headroom_us)}。${cb.calibrated ? `（芯片峰值按实测 ${cb.observed_peak_tflops} TFLOPS 校准，假设 ${cb.assumed_peak_tflops != null ? cb.assumed_peak_tflops.toFixed(0) : "—"}）` : ""}</div>`) : ""}
           <div class="note" style="font-size:11px;line-height:1.55;margin-top:8px;border-top:1px solid rgba(255,255,255,.06);padding-top:6px">💡 优化优先级：「通信完全掩盖」通常是最大单项 → 先做计算-通信重叠（--moe-fb-overlap / 异步通信），再压同步空泡。「算子极致优化」按各算子 MFU 天花板（matmul 95% / FA 85% / FAG 70%）收口，已达标算子不再投入——单项收益小而精，而非冲到 100% 的虚高。底部「已启用组合」随勾选实时计算叠加后的 step 与端到端 MFU。　⚠️ 「通信完全掩盖」与算子页 <strong>HcclLaunchAicpuKernel</strong> 是同一段集合通信（单卡几乎全是 Wait，非下发延迟），勿重复计入。</div>`)}
      </div>
      <div class="note">${esc(theo.available ? theo.note : "")}</div>`;
    charts([{ id: "ov-donut", option: donut(ov.composition.map(c => ({ name: c.name, value: c.us, pct: c.pct }))) }]);
    if (theo.available) {
      document.querySelectorAll("input.wi-lever").forEach(el => el.addEventListener("change", recomputeCombined));
      recomputeCombined();
    }
  };

  function donut(items) {
    return {
      tooltip: Object.assign({ trigger: "item", formatter: p => `${p.name}<br/><b>${fmt.us(p.value)}</b> (${p.percent}%)` }, tooltipBase),
      legend: { bottom: 0, textStyle: { color: "#9aa7b8" } },
      series: [{ type: "pie", radius: ["48%", "72%"], center: ["50%", "44%"], avoidLabelOverlap: true,
        itemStyle: { borderColor: "#161d29", borderWidth: 2 },
        label: { color: "#c9d4e0", formatter: p => `${p.percent}%` },
        data: items }]
    };
  }

  // ============================ 算子热点 Hotspots ============================ //
  V.hotspots = async function (root) {
    const hs = await api("/api/hotspots");
    if (!hs.available) { root.innerHTML = `<div class="empty">无 op_statistic 数据</div>`; return; }
    const top = hs.top.slice(0, 12);
    const rows = hs.ops.slice(0, 40).map(o =>
      `<tr><td>${esc(o.type)}</td><td><span class="tag ${coreClass(o.core)}">${esc(o.core || "—")}</span></td>
       <td>${fmt.int(o.count)}</td><td>${fmt.us(o.total_us)}</td><td>${fmt.us(o.avg_us)}</td><td>${fmt.us(o.max_us)}</td><td>${fmt.pct(o.ratio, 2)}</td></tr>`).join("");
    root.innerHTML = `
      <div class="grid cols-2">
        ${panel("Top 算子（按 device 总耗时占比）", "HcclLaunchAicpuKernel = AI_CPU 通信下发", `<div id="hs-bar" class="chart tall"></div>`)}
        ${panel("按 Core 类型聚合", "AI_CORE / AI_VECTOR / MIX / AI_CPU", `<div id="hs-core" class="chart tall"></div>`)}
      </div>
      ${panel("算子明细 (Top 40)", "", `<div class="tbl-wrap"><table class="tbl"><thead><tr><th>OP Type</th><th>Core</th><th>Count</th><th>总耗时</th><th>均值</th><th>最大</th><th>占比</th></tr></thead><tbody>${rows}</tbody></table></div>`, "span-2")}`;
    charts([
      { id: "hs-bar", option: hbar(top.map(o => o.type), top.map(o => o.ratio), "占比%", { fmt: v => fmt.pct(v, 1), color: top.map(o => o.core === "AI_CPU" ? "#f85149" : "#3fb6e0") }) },
      { id: "hs-core", option: donut(hs.by_core.map(c => ({ name: c.core || "UNKNOWN", value: c.total_us, pct: c.pct }))) },
    ]);
  };
  function coreClass(c) { c = (c || "").toUpperCase(); if (c.includes("CPU")) return "other"; if (c.includes("VECTOR")) return "vector"; if (c.includes("CUBE") || c.includes("AIC")) return "compute"; return "compute"; }

  function hbar(cats, vals, name, opts) {
    opts = opts || {};
    return {
      grid: { left: 8, right: 60, top: 10, bottom: 24, containLabel: true },
      tooltip: Object.assign({ trigger: "axis", axisPointer: { type: "shadow" }, formatter: p => `${p[0].name}<br/><b>${opts.fmt ? opts.fmt(p[0].value) : p[0].value}</b>` }, tooltipBase),
      xAxis: axis({ type: "value", name }),
      yAxis: axis({ type: "category", data: cats.slice().reverse(), axisLabel: { color: "#c9d4e0", width: 150, overflow: "truncate" } }),
      series: [{ type: "bar", data: vals.slice().reverse().map((v, i) => ({ value: v, itemStyle: opts.color ? { color: opts.color.slice().reverse()[i] } : undefined })),
        barWidth: "62%", label: { show: true, position: "right", color: "#9aa7b8", formatter: p => opts.fmt ? opts.fmt(p.value) : p.value } }]
    };
  }

  // ============================ 算子效率 Roofline ============================ //
  V.efficiency = async function (root) {
    const ef = await api("/api/efficiency");
    if (!ef.available) { root.innerHTML = `<div class="empty">无 kernel_details 数据</div>`; return; }
    const chip = ef.chip;
    const top = ef.top_optimization.slice(0, 15);
    // optimization gain = reclaim_us (time recoverable by tuning to the kernel's
    // MFU ceiling; matmul/FA/FAG capped below 100%, others to the roofline). The
    // bracket shows MFU/MBU at that ceiling floor (e.g. FAG "MFU 56→70"). Kernels
    // already at/above their ceiling are filtered out upstream, so every row here
    // has real headroom.
    const arrow = (b, a) => {
      if (b == null && a == null) return null;
      const f = x => (x == null ? "—" : Math.round(x * 100));
      return `${f(b)}→${f(a)}`;
    };
    const gainCell = t => {
      const segs = [];
      if (t.mfu != null && t.mfu_after != null) segs.push(`MFU ${arrow(t.mfu, t.mfu_after)}`);
      if (t.mbu != null && t.mbu_after != null) segs.push(`MBU ${arrow(t.mbu, t.mbu_after)}`);
      const tail = segs.length ? ` <span style="color:var(--text-dim)">（${segs.join(", ")}）</span>` : "";
      return `<strong style="color:var(--accent-2)">${fmt.us(t.reclaim_us)}</strong>${tail}`;
    };
    const optRows = top.map(t =>
      `<tr><td class="mono">${esc(t.name)}</td><td><span class="tag">${esc(t.bound || "—")}</span></td><td>${fmt.us(t.dur_us)}</td><td>${gainCell(t)}</td></tr>`).join("");
    const byType = ef.by_type.slice(0, 20).map(t =>
      `<tr><td>${esc(t.type)}</td><td class="mono">${t.dtype ? esc(t.dtype) : "—"}</td><td>${fmt.int(t.count)}</td><td>${fmt.us(t.dur_us)}</td><td>${t.mfu != null ? fmt.mfu(t.mfu) : "—"}</td><td>${t.mbu != null ? fmt.mfu(t.mbu) : "—"}</td><td>${fmt.us(t.reclaim_us != null ? t.reclaim_us : t.wasted_us)}</td></tr>`).join("");
    // 算子极致优化 summary: per-class ceilings + how many kernels are already
    // saturated (filtered out of the ranking) + total reclaimable-to-ceiling time.
    const oco = ef.op_ceiling_opt;
    const ceilPct = x => Math.round((x || 0) * 100);
    const ocNote = (oco && oco.ceilings)
      ? `<div class="note" style="margin-top:8px">🎯 算子 MFU 天花板：matmul <strong>${ceilPct(oco.ceilings.matmul)}%</strong> / FA <strong>${ceilPct(oco.ceilings.attention)}%</strong> / FAG <strong>${ceilPct(oco.ceilings.attention_grad)}%</strong>　·　已达天花板 <strong>${oco.n_capped}/${oco.n_modeled}</strong> 个算子（不再优化）　·　全部提升至天花板可回收 <strong>${fmt.us(oco.total_reclaim_us)}</strong>。下方「优化候选 / 排行」已据此筛选。</div>`
      : "";
    root.innerHTML = `
      ${banner(chip.calibrated ? "warn" : "info", "🎯",
        `芯片：<strong>${esc(chip.name)}</strong> · CUBE BF16 <strong>${(chip.cube_bf16_tflops||chip.peak_bf16_tflops).toFixed(0)}</strong> / VECTOR BF16 <strong>${(chip.vector_bf16_tflops||0).toFixed(0)}</strong> TFLOPS · HBM <strong>${chip.hbm_tbps.toFixed(2)} TB/s</strong>` +
        (chip.hbm_capacity_gb ? ` · 显存 <strong>${chip.hbm_capacity_gb.toFixed(0)} GB</strong>` : "") +
        (chip.calibrated
          ? `　|　⚠️ 实测峰值 <strong>${chip.observed_peak_tflops.toFixed(0)} TFLOPS</strong> &gt; 假设值 → 已按实测校准（设真实 SKU 峰值可覆盖）`
          : `（${chip.assumed ? "参考峰值；顶部可切换芯片，或在 config.ChipSpec 校正" : "实测"}）`) +
        `　|　matmul MFU ≈ <strong>${ef.matmul_mfu != null ? (ef.matmul_mfu*100).toFixed(0)+"%" : "—"}</strong>　|　建模 kernel ${fmt.int(ef.kernels_with_flops)}/${fmt.int(ef.kernels_total)}`)}
      ${ocNote}
      <div class="grid cols-2">
        ${panel("Roofline（dtype 归一化）", "点 = kernel；x = 归一化算术强度 (AI / 脊点，1 = 拐点)，y = MFU (占该算子 cube/vector 峰值 %)；单条屋顶线让不同 dtype 同台对照", `<div id="ef-roof" class="chart tall"></div>`)}
        ${panel("优化空间排行 (Top 15)", "按到各算子 MFU 天花板的可回收时间排序；颜色 = 瓶颈类型", `<div id="ef-waste" class="chart tall"></div>`)}
      </div>
      ${panel("优化候选明细 (Top 15)", "优化收益 = 到 MFU 天花板的可回收时间（matmul/FA/FAG 封顶各自天花板，其余到 Roofline）；已达天花板的算子不入表；括号为优化后的 MFU / MBU", `<div class="tbl-wrap"><table class="tbl"><thead><tr><th>算子</th><th>瓶颈</th><th>当前耗时</th><th>优化收益</th></tr></thead><tbody>${optRows}</tbody></table></div>`, "span-2")}
      ${panel("按算子类型 MFU / MBU / 可回收（MFU 按各算子 cube/vector 峰值）", "", `<div class="tbl-wrap"><table class="tbl"><thead><tr><th>Type</th><th>dtype</th><th>Count</th><th>耗时</th><th>MFU</th><th>MBU</th><th>可回收(vs天花板)</th></tr></thead><tbody>${byType}</tbody></table></div>`, "span-2")}`;
    charts([
      { id: "ef-roof", option: roofline(ef.scatter, chip) },
      { id: "ef-waste", option: hbar(top.map(t => t.name.slice(0, 26)), top.map(t => t.reclaim_us), "可回收 us",
          { fmt: v => fmt.us(v), color: top.map(t => boundColor(t.bound)) }) },
    ]);
  };
  function boundColor(b) { return { compute: "#79b8ff", memory: "#e3b341", vector: "#5ee0b8" }[b] || "#6b7888"; }

  // Double-normalized Roofline: each point sits at (AI/ridge, MFU%) against its
  // OWN routed cube/vector dtype peak, so a single universal roof y=min(x,1)·100%
  // serves every dtype — bf16 / fp8 / fp4 GEMMs compare on one efficiency axis.
  function roofline(scatter, chip) {
    scatter = (scatter || []).filter(s => s.x_norm > 0 && s.mfu != null);
    const xs = scatter.map(s => s.x_norm);
    const ys = scatter.map(s => s.mfu * 100).filter(y => y > 0);
    const xmin = xs.length ? Math.min(0.5, Math.min.apply(null, xs) / 2) : 0.1;
    const xmax = xs.length ? Math.max(2, Math.max.apply(null, xs) * 2) : 10;
    const ymin = ys.length ? Math.max(0.05, Math.min.apply(null, ys) / 2) : 1;
    const ymax = ys.length ? Math.max(120, Math.max.apply(null, ys) * 1.1) : 120;
    const roof = [[xmin, xmin * 100], [1, 100], [xmax, 100]];  // y = min(x,1)·100%
    const byBound = {};
    scatter.forEach(s => { (byBound[s.bound] = byBound[s.bound] || []).push(
      [s.x_norm, s.mfu * 100, s.name, s.dur_us, s.dtype, s.ai, s.tflops, s.peak_tflops]); });
    const series = Object.keys(byBound).map(b => ({
      name: b, type: "scatter", symbolSize: d => Math.min(22, 4 + Math.sqrt(d[3]) / 6),
      itemStyle: { color: boundColor(b), opacity: .65 }, data: byBound[b],
    }));
    series.push({ name: "屋顶线", type: "line", data: roof, showSymbol: false, lineStyle: { color: "#f85149", width: 1.5, type: "dashed" }, tooltip: { show: false }, z: 1 });
    return {
      tooltip: Object.assign({ trigger: "item", formatter: p => p.seriesName === "屋顶线" ? "" :
        `${esc(p.data[2])}<br/>dtype=${esc(p.data[4] || "—")}　峰值 ${p.data[7].toFixed(0)} TFLOPS<br/>MFU=${p.data[1].toFixed(1)}%　归一化强度=${p.data[0].toFixed(2)}<br/>AI=${p.data[5].toFixed(2)} FLOP/B　达成=${p.data[6].toFixed(1)} TFLOPS<br/>耗时=${fmt.us(p.data[3])}　瓶颈: ${p.seriesName}` }, tooltipBase),
      legend: { top: 0, textStyle: { color: "#9aa7b8" } },
      grid: { left: 10, right: 20, top: 34, bottom: 36, containLabel: true },
      xAxis: axis({ type: "log", name: "归一化算术强度 (AI / 脊点)", min: xmin, max: xmax }),
      yAxis: axis({ type: "log", name: "MFU %（占 dtype 峰值）", min: ymin, max: ymax }),
      series,
      markLine: undefined,
    };
  }

  // ============================ 通信 Communication ============================ //
  V.communication = async function (root) {
    const cm = await api("/api/communication");
    if (!cm.available) { root.innerHTML = `<div class="empty">无 communication.json 数据</div>`; return; }
    const rows = cm.by_type.map(t =>
      `<tr><td>${esc(t.type)}</td><td>${fmt.int(t.count)}</td><td>${fmt.ms(t.elapse_ms)}</td><td>${fmt.ms(t.wait_ms)}</td><td>${fmt.pct(t.wait_pct)}</td><td>${fmt.ms(t.transit_ms)}</td><td>${t.transit_mb.toFixed(1)} MB</td></tr>`).join("");
    const topRows = cm.top.slice(0, 12).map(t =>
      `<tr><td class="mono">${esc(t.name)}</td><td>${esc(t.type)}</td><td>${fmt.ms(t.elapse_ms)}</td><td>${fmt.pct(t.wait_ratio * 100)}</td></tr>`).join("");
    root.innerHTML = `
      <div class="grid cols-4">
        ${metric("集合通信次数", fmt.int(cm.count))}
        ${metric("总 Elapse", fmt.ms(cm.total_elapse_ms), { foot: "wall-clock" })}
        ${metric("平均等待占比", fmt.pct(cm.overall_wait_pct), { tone: "bad", barPct: cm.overall_wait_pct })}
        ${metric("总 Transit", cm.total_transit_mb.toFixed(1) + " MB", { foot: "单卡≈0" })}
      </div>
      ${banner("info", "ℹ️", esc(cm.note))}
      <div class="grid cols-2">
        ${panel("按通信类型耗时 (Elapse ms)", "", `<div id="cm-bar" class="chart"></div>`)}
        ${panel("按类型等待占比", "Wait / Elapse（每 op 均值）", `<div id="cm-wait" class="chart"></div>`)}
      </div>
      <div class="grid cols-2">
        ${panel("类型聚合明细", "", `<div class="tbl-wrap"><table class="tbl"><thead><tr><th>Type</th><th>Count</th><th>Elapse</th><th>Wait</th><th>Wait%</th><th>Transit</th><th>流量</th></tr></thead><tbody>${rows}</tbody></table></div>`)}
        ${panel("Top 集合通信 (Elapse)", "", `<div class="tbl-wrap"><table class="tbl"><thead><tr><th>Name</th><th>Type</th><th>Elapse</th><th>Wait%</th></tr></thead><tbody>${topRows}</tbody></table></div>`)}
      </div>`;
    charts([
      { id: "cm-bar", option: hbar(cm.by_type.map(t => t.type), cm.by_type.map(t => t.elapse_ms), "ms", { fmt: v => fmt.ms(v) }) },
      { id: "cm-wait", option: hbar(cm.by_type.map(t => t.type), cm.by_type.map(t => t.wait_pct), "wait%", { fmt: v => fmt.pct(v), color: cm.by_type.map(() => "#d29922") }) },
    ]);
  };

  // ============================ 隐性开销 Hidden Overhead ============================ //
  V.hidden_overhead = async function (root) {
    const ho = await api("/api/hidden_overhead");
    if (!ho.available) { root.innerHTML = `<div class="empty">无数据</div>`; return; }
    const dev = ho.buckets.filter(b => b.domain === "device" && b.us != null && b.additive !== false);
    const host = ho.buckets.filter(b => b.domain === "host" && b.us != null);
    const cards = ho.buckets.map(b => `
      <div class="panel">
        <h3>${esc(b.label)} <span class="ic-cat">${esc(b.domain)}</span></h3>
        <div class="value" style="font-size:22px;font-weight:700;margin:2px 0 6px;color:${b.domain==='device'?'var(--accent)':b.domain==='host'?'var(--warn)':'var(--text-mut)'}">${b.us != null ? fmt.us(b.us) : "—"}</div>
        <div class="note" style="margin-top:0">${esc(b.detail)}</div>
        <div class="note" style="color:var(--accent-2)">💡 ${esc(b.suggestion)}</div>
      </div>`).join("");
    root.innerHTML = `
      <div class="grid cols-3">
        ${metric("Device 侧合计", fmt.us(ho.device_total_us), { tone: "warn", foot: "与 step 同口径可比" })}
        ${metric("Host 侧合计", fmt.us(ho.host_total_us), { foot: "下发/同步压力（被 blocking 放大）" })}
        ${metric("Step 时间", fmt.us(ho.stage_us))}
      </div>
      <div class="grid cols-2" style="margin-top:16px">
        ${panel("Device 侧隐性开销 (计入 step)", "未掩盖通信 / 空泡 / 格式转换初始化（AICPU 通信执行为同段通信，单列、不入合计）", `<div id="ho-dev" class="chart"></div>`)}
        ${panel("Host 侧隐性开销", "Launch 下发 / 同步阻塞 / 动态 shape（采集干扰）", `<div id="ho-host" class="chart"></div>`)}
      </div>
      ${banner("info", "ℹ️", esc(ho.note))}
      <div class="section-title">分桶明细与减负建议</div>
      <div class="grid cols-3">${cards}</div>`;
    charts([
      { id: "ho-dev", option: hbar(dev.map(b => b.label.split(" ")[0].slice(0,16)), dev.map(b => b.us), "us", { fmt: v => fmt.us(v), color: dev.map(() => "#3fb6e0") }) },
      { id: "ho-host", option: hbar(host.map(b => b.label.split(" ")[0].slice(0,16)), host.map(b => b.us), "us", { fmt: v => fmt.us(v), color: host.map(() => "#d29922") }) },
    ]);
  };

  // ============================ 结构归因 Attribution ============================ //
  V.attribution = async function (root) {
    const at = await api("/api/attribution");
    if (!at.available) { root.innerHTML = `<div class="empty">无数据</div>`; return; }
    const modRows = at.modules.map(m => `<tr><td>${esc(m.module)}</td><td>${fmt.us(m.us)}</td><td>${fmt.pct(m.pct)}</td></tr>`).join("");
    const mf = at.moe_focus;
    root.innerHTML = `
      <div class="grid cols-2">
        ${panel("模型结构归因（device 计算时间）", "按算子命名启发式归因到训练语义模块", `<div id="at-ring" class="chart tall"></div>`)}
        <div>
          ${panel("MoE 专属：专家计算 vs 分发通信", "GroupedMatmul vs alltoallv",
            `<div class="grid cols-2">
              ${metric("专家计算 (GroupedMatmul)", fmt.us(mf.expert_compute_us), { tone: "good" })}
              ${metric("分发通信 (alltoallv)", fmt.us(mf.dispatch_comm_us), { tone: "warn" })}
            </div>`)}
          ${panel("通信构成（wall-clock，单独列出不并入计算环）", "",
            `<table class="tbl"><thead><tr><th>通信</th><th>耗时</th><th>占比</th></tr></thead><tbody>${
              at.comm_breakdown.map(c => `<tr><td>${esc(c.module)}</td><td>${fmt.us(c.us)}</td><td>${fmt.pct(c.pct)}</td></tr>`).join("")}</tbody></table>`)}
        </div>
      </div>
      ${panel("模块明细", "", `<table class="tbl"><thead><tr><th>模块</th><th>device 耗时</th><th>占比</th></tr></thead><tbody>${modRows}</tbody></table>`, "span-2")}
      <div class="note">${esc(at.note)}</div>`;
    charts([{ id: "at-ring", option: donut(at.modules.map(m => ({ name: m.module, value: m.us, pct: m.pct }))) }]);
  };

  // ============================ 显存 Memory ============================ //
  V.memory = async function (root) {
    const [me, meta] = await Promise.all([api("/api/memory"), api("/api/meta")]);
    const chip = ((meta.meta || {}).settings || {}).chip || {};
    const trade = (me.config_tradeoffs || []).map(t => `
      <div class="panel">
        <h3>${esc(t.feature)}</h3>
        <div class="note" style="margin-top:2px">${esc(t.effect)}</div>
        <div class="note" style="color:var(--accent-2)">💡 ${esc(t.advice)}</div>
      </div>`).join("");
    root.innerHTML = `
      ${banner("warn", "🧠", `<strong>显存采集未开启</strong> —— ${esc(me.reason || "缺 memory-level 数据")}`)}
      <div class="grid cols-3">
        ${metric("HBM 容量", chip.hbm_capacity_gb ? chip.hbm_capacity_gb.toFixed(0) + " GB" : "—", { foot: (chip.name || "") + " · 峰值时间线待 memory 采集" })}
        ${metric("初始化类开销", fmt.us(me.init_overhead_us), { foot: "ZerosLike/TensorMove/Fill" })}
        ${metric("AI Core 频率", (me.ai_core_freq_mhz || "—") + " MHz")}
      </div>
      <div class="section-title">内存 ↔ 时间权衡顾问（配置驱动）</div>
      <div class="grid cols-2">${trade}</div>
      <div class="note">${esc(me.note || "")}</div>`;
  };

  // ============================ 时间线 Timeline ============================ //
  V.timeline = async function (root) {
    const tl = await api("/api/timeline");
    if (!tl.available) { root.innerHTML = `<div class="empty">无 trace_view.json：${esc(tl.reason || "")}</div>`; return; }
    const comp = tl.computing_pct || 0, notov = tl.not_overlapped_pct || 0, free = tl.free_pct || 0;
    const maxDur = tl.top_slices.length ? tl.top_slices[0].dur_us : 1;
    const sliceRows = tl.top_slices.length
      ? tl.top_slices.slice(0, 40).map(s => {
          const w = Math.max(2, Math.round(s.dur_us / maxDur * 100));
          return `<tr><td class="mono">${esc(s.name)}</td><td>${(s.start_us / 1e6).toFixed(2)} s</td>` +
            `<td><div style="position:relative;min-width:84px">` +
            `<div style="position:absolute;top:2px;bottom:2px;right:0;width:${w}%;background:rgba(63,182,224,.16);border-radius:3px"></div>` +
            `<span style="position:relative">${fmt.us(s.dur_us)}</span></div></td></tr>`;
        }).join("")
      : `<tr><td colspan="3" class="empty">无 >1.5ms 的真实计算 kernel</td></tr>`;
    root.innerHTML = `
      <div class="grid cols-4">
        ${metric("时间跨度", tl.span_s + " s", { foot: fmt.int(tl.bins) + " 桶 · " + tl.lanes.length + " 泳道" })}
        ${metric("有效计算", comp.toFixed(1) + "%", { tone: comp >= 60 ? "good" : comp >= 40 ? "warn" : "bad", barPct: comp, foot: "Computing / 总步长" })}
        ${metric("未掩盖通信", notov.toFixed(1) + "%", { tone: notov >= 20 ? "bad" : notov >= 8 ? "warn" : "good", barPct: notov, foot: "通信未被计算掩盖" })}
        ${metric("Free 空泡", free.toFixed(1) + "%", { tone: free >= 20 ? "bad" : free >= 8 ? "warn" : "good", barPct: free, foot: "设备完全空闲（可优化）" })}
      </div>
      ${panel("泳道占用率热力图（回放式时间轴）", "绿=满载（主机流常驻属正常），暖→红=占用骤降的空泡；与下方两图共享时间轴", `<div id="tl-heat" class="chart" style="height:264px"></div>`, "span-2")}
      ${panel("Overlap 时间条（每桶按 有效计算 / 未掩盖通信 / Free 三段拆分，合计=步长 100%）", "红=未掩盖通信，琥珀=Free 空泡——两者越多越值得优化", `<div id="tl-overlap" class="chart" style="height:176px"></div>`, "span-2")}
      ${panel("AI Core 频率", "MHz over time（x 轴已钉死并与上方对齐）", `<div id="tl-freq" class="chart" style="height:176px"></div>`, "span-2")}
      ${panel("最长计算 kernel 切片 (>1.5ms)", "已过滤同步等待项（WAIT / NOTIFY 等），仅保留真实计算 kernel", `<div class="tbl-wrap" style="max-height:260px"><table class="tbl"><thead><tr><th>Kernel</th><th>起始</th><th>时长</th></tr></thead><tbody>${sliceRows}</tbody></table></div>`, "span-2")}
      <div class="note">${esc(tl.note)}</div>`;
    charts([
      { id: "tl-heat", option: heatmap(tl) },
      { id: "tl-overlap", option: overlapBand(tl) },
      { id: "tl-freq", option: freqLine(tl.ai_core_freq, tl.span_us) },
    ]);
  };

  // shared horizontal grid -> the 3 time charts line up vertically (only left/right
  // must match; each keeps its own top/bottom). left gutter holds the lane labels.
  const TL_GX = { left: 150, right: 22 };
  const LANE_COLOR = {
    "Device (Ascend Hardware)": "#5ee0b8",  // the lane we care about -> bright accent
    "Communication": "#58a6ff",
    "Host Runtime (CANN)": "#9aa7b8",        // host lanes recede (muted)
    "Framework (Python)": "#6b7888",
  };
  const LANE_SHORT = {
    "Device (Ascend Hardware)": "Device",
    "Communication": "Communication",
    "Host Runtime (CANN)": "Host · CANN",
    "Framework (Python)": "Framework · Py",
  };
  const OV_COLOR = {
    "Computing": "#2f6f4f",                  // useful compute -> dim green (recedes)
    "Communication": "#58a6ff",              // comm hidden behind compute -> neutral
    "Communication(Not Overlapped)": "#f85149",  // exposed comm stall -> red (loud)
    "Free": "#d29922",                       // device idle bubble -> amber (loud)
  };
  const OV_LABEL = {
    "Computing": "Computing",
    "Communication": "通信(已掩盖)",
    "Communication(Not Overlapped)": "未掩盖通信",
    "Free": "Free 空泡",
  };

  // idle-loud: occupancy 0 (空泡) -> red, 1 (满载) -> dim green. Inverts the usual
  // "bright = busy" so the always-on host lanes recede and Device/通信 dips pop.
  function heatmap(tl) {
    if (!tl.lanes || !tl.lanes.length) return { xAxis: {}, yAxis: {}, series: [] };
    const lanes = tl.lanes.map(l => l.label);
    const data = [];
    tl.lanes.forEach((l, li) => l.occupancy.forEach((v, bi) => data.push([bi, li, v])));
    return {
      animation: false,
      tooltip: Object.assign({ position: "top", formatter: p => {
        const occ = p.data[2];
        const state = occ >= 0.85 ? "满载" : occ >= 0.4 ? "部分空闲" : "空泡 / 空闲";
        return `${lanes[p.data[1]]}<br/>t=${(p.data[0] * tl.bin_us / 1e6).toFixed(2)}s<br/>占用 ${(occ * 100).toFixed(0)}% · ${state}`;
      } }, tooltipBase),
      grid: Object.assign({ top: 12, bottom: 44 }, TL_GX),
      xAxis: axis({ type: "category", data: tl.lanes[0].occupancy.map((_, i) => i),
        axisLabel: { show: true, interval: Math.floor(tl.bins / 10), formatter: i => (i * tl.bin_us / 1e6).toFixed(1) + "s" } }),
      yAxis: axis({ type: "category", data: lanes,
        axisLabel: { width: 130, overflow: "truncate", color: v => LANE_COLOR[v] || "#c9d4e0", formatter: v => LANE_SHORT[v] || v } }),
      visualMap: { min: 0, max: 1, calculable: true, orient: "horizontal", left: "center", bottom: 0,
        text: ["满载", "空泡"], textStyle: { color: "#9aa7b8", fontSize: 10 },
        inRange: { color: ["#f85149", "#e8833a", "#caa53d", "#2f6f4f", "#21402f"] } },
      series: [{ type: "heatmap", data, progressive: 2400, itemStyle: { borderColor: "#0d1117", borderWidth: 0.5 } }],
    };
  }
  // 100% stacked band over the 3 step-partitioning tracks (Computing + 未掩盖通信 +
  // Free sum to ~1). Total "Communication" is excluded — it overlaps Computing and
  // would push the stack past 100%; the comm lane in the heatmap already shows it.
  function overlapBand(tl) {
    const ob = tl.overlap_bins || [];
    if (!ob.length) return { xAxis: {}, yAxis: {}, series: [] };
    const byTrack = {}; ob.forEach(t => { byTrack[t.track] = t.occupancy; });
    const order = ["Computing", "Communication(Not Overlapped)", "Free"].filter(k => byTrack[k]);
    if (!order.length) return { xAxis: {}, yAxis: {}, series: [] };
    const n = byTrack[order[0]].length;
    const cats = byTrack[order[0]].map((_, i) => i);
    const series = order.map(k => ({
      name: OV_LABEL[k] || k, type: "bar", stack: "ov", barWidth: "100%",
      itemStyle: { color: OV_COLOR[k] || "#888", borderWidth: 0 }, data: byTrack[k],
    }));
    return {
      animation: false,
      tooltip: Object.assign({ trigger: "axis", axisPointer: { type: "shadow" }, formatter: ps => {
        const t = (ps[0].dataIndex * tl.bin_us / 1e6).toFixed(2);
        const rows = ps.filter(p => p.data > 0.005).map(p => `${p.marker}${p.seriesName} ${(p.data * 100).toFixed(0)}%`).join("<br/>");
        return `t=${t}s<br/>${rows || "—"}`;
      } }, tooltipBase),
      legend: { data: order.map(k => OV_LABEL[k] || k), top: 0, right: 10, itemWidth: 10, itemHeight: 10, itemGap: 12, textStyle: { color: "#9aa7b8", fontSize: 11 } },
      grid: Object.assign({ top: 30, bottom: 26 }, TL_GX),
      xAxis: axis({ type: "category", data: cats, boundaryGap: true,
        axisLabel: { interval: Math.floor(n / 10), formatter: i => (i * tl.bin_us / 1e6).toFixed(1) + "s" } }),
      yAxis: axis({ type: "value", min: 0, max: 1, axisLabel: { formatter: v => (v * 100).toFixed(0) + "%" } }),
      series,
    };
  }
  function freqLine(freq, span_us) {
    return {
      tooltip: Object.assign({ trigger: "axis", formatter: p => `${(p[0].data[0] / 1e6).toFixed(2)}s<br/><b>${p[0].data[1]} MHz</b>` }, tooltipBase),
      grid: Object.assign({ top: 16, bottom: 26 }, TL_GX),
      xAxis: axis({ type: "value", name: "s", min: 0, max: span_us, axisLabel: { formatter: v => (v / 1e6).toFixed(1) } }),
      yAxis: axis({ type: "value", name: "MHz", scale: true }),
      series: [{ type: "line", showSymbol: false, smooth: true, areaStyle: { opacity: .12 }, lineStyle: { color: "#5ee0b8" }, itemStyle: { color: "#5ee0b8" },
        data: (freq || []).map(f => [f.t_us, f.mhz]) }],
    };
  }

  // ============================ 洞察 Insights (LLM) ============================ //
  V.insights = async function (root) {
    const ins = await api("/api/insights");
    renderInsights(root, ins);
  };
  function renderInsights(root, ins) {
    const llm = ins.llm || {};
    const cardsHtml = (ins.cards || []).map(insightCard).join("");
    const llmBanner = llm.available
      ? banner("info", "🤖", `LLM 已启用：<strong>${esc(llm.provider)}/${esc(llm.model)}</strong>。点击下方按钮生成自然语言诊断。`)
      : banner("warn", "🤖", `LLM 自动分析<strong>默认关闭</strong>（仅传 KB 级指标摘要，绝不上传原始 trace）。${esc(llm.reason || "")}<br/>开启方式：设环境变量 <span class="mono">LLMINSIGHT_LLM_ENABLED=1</span>，并在 <span class="mono">secret/api_key.txt</span> 放置 key（默认 ${esc(llm.provider)}/${esc(llm.model)}）。规则引擎洞察始终可用 ↓`);
    root.innerHTML = `
      ${llmBanner}
      <div style="display:flex;gap:10px;align-items:center;margin-bottom:8px">
        <button class="btn" id="llm-run" ${llm.available ? "" : "disabled"}>运行 LLM 分析</button>
        <span class="note" id="llm-status" style="margin:0"></span>
      </div>
      <div id="llm-narrative"></div>
      <div class="section-title">规则引擎诊断卡片（${(ins.cards||[]).length}） · 现象 / 根因 / 建议 / 预计收益 / 置信度</div>
      <div id="cards">${cardsHtml || `<div class="empty">无命中</div>`}</div>
      <details class="raw"><summary>查看将发送给 LLM 的结构化摘要（隐私可审计 · 不含原始 trace）</summary>
        <pre>${esc(JSON.stringify(ins.summary, null, 1))}</pre></details>`;
    const btn = document.getElementById("llm-run");
    if (btn) btn.addEventListener("click", async () => {
      btn.disabled = true; const st = document.getElementById("llm-status"); st.textContent = "正在请求 LLM…";
      try {
        const res = await apiPost("/api/llm");
        if (res.narrative) { document.getElementById("llm-narrative").innerHTML = `<div class="narrative">${esc(res.narrative)}</div>`; st.textContent = ""; }
        else { st.textContent = "未返回（" + (res.error || res.llm && res.llm.reason || "已禁用") + "）"; }
      } catch (e) { st.textContent = "失败：" + e.message; }
      btn.disabled = false;
    });
  }
  function insightCard(c) {
    return `<div class="insight ${c.severity}">
      <div class="ic-head">
        <span class="sev ${c.severity}">${c.severity}</span>
        <span class="ic-title">${esc(c.title)}</span>
        <span class="ic-cat">${esc(c.category)}</span>
        <span class="ic-conf">置信度 ${(c.confidence*100).toFixed(0)}%</span>
      </div>
      <div class="ic-body">
        <span class="k">根因</span><span class="v">${esc(c.root_cause)}</span>
        <span class="k">建议</span><span class="v">${esc(c.suggestion)}</span>
        <span class="k">预计收益</span><span class="v gain">${esc(c.expected_gain)}</span>
      </div></div>`;
  }
  V._renderInsights = renderInsights;

  // ============================ 回放 Replay ============================ //
  V.replay = async function (root) {
    const [ov, meta] = await Promise.all([api("/api/overview"), api("/api/meta")]);
    const step = ov.available ? ov.step : "—";
    const r = ov.available ? ov.ratios : {};
    root.innerHTML = `
      ${banner("info", "🎬", `<strong>全训练回放（骨架）</strong> —— 当前数据为单 step（step ${step}）单帧。回放轴与联动机制已搭好；多 step 全程指标（loss / 吞吐 / 显存 / 通信抖动）随后续多 step + 训练日志采集接入。`)}
      ${panel("训练时间轴（回放）", "拖动选择 step（当前仅 step " + step + "）",
        `<div class="slider-row">
           <span class="note" style="margin:0">step</span>
           <input type="range" min="${step}" max="${step}" value="${step}" disabled>
           <span class="badge"><strong>step ${step}</strong></span>
         </div>`)}
      <div class="section-title">当前帧快照（联动总览）</div>
      <div class="grid cols-4">
        ${metric("有效计算", fmt.pct(r.effective_compute_pct), { tone: "good" })}
        ${metric("未掩盖通信", fmt.pct(r.comm_not_overlapped_pct), { tone: "bad" })}
        ${metric("空闲", fmt.pct(r.free_pct), { tone: "warn" })}
        ${metric("Step 时间", (r.step_time_s || "—") + " s")}
      </div>
      <div class="section-title">回放将支持（设计）</div>
      <div class="grid cols-2">
        ${panel("📈 全程趋势叠加", "", `<div class="note" style="margin:0">耗时构成 / 未掩盖通信占比 / 显存峰值 / loss·吞吐·grad-norm 多指标趋势线，可拖动定位「第几步开始变慢」。</div>`)}
        ${panel("🔍 任意两 step 对比 (Δ)", "", `<div class="note" style="margin:0">选中两帧自动算法差异，定位回归来源算子；自动在轴上打标异常点（变慢 / 显存爬升 / 通信抖动 / loss 突刺）。</div>`)}
      </div>`;
  };

  // ====================== 智能时间线 Smart Timeline ====================== //
  V.smart_timeline = async function (root) {
    const tl = await api("/api/smart_timeline");
    if (!tl.available) { root.innerHTML = `<div class="empty">无 trace_view.json：${esc(tl.reason || "")}</div>`; return; }
    const utilLeg = (tl.utilization || []).filter(u => u.available && u.series).map(u =>
      `<span class="tl-leg"><i style="background:${u.color}"></i>${esc(u.label)}</span>`).join("");
    const opLeg = (tl.streams || []).map(s =>
      `<span class="tl-leg"><i style="background:${s.color}"></i>${esc(s.label)}</span>`).join("");
    const placeholders = (tl.utilization || []).filter(u => !u.available).map(u =>
      banner("warn", "🧠", `<strong>${esc(u.label)}</strong> —— ${esc(u.reason || "待采集")}`)).join("");
    const lay = tlLanes(tl);
    const utilN = lay.lanes.filter(L => L.kind === "util").length;
    root.innerHTML = `
      <div class="grid cols-4">
        ${metric("时间跨度", tl.span_s + " s", { foot: fmt.int(tl.bins) + " 桶 · " + fmt.us(tl.bin_us) + "/桶" })}
        ${metric("利用率泳道", fmt.int(utilN), { foot: "Cube / Vector / HBM / 通信" })}
        ${metric("算子切片", fmt.int(tl.shown_slices), { foot: "共 " + fmt.int(tl.total_slices) + "（按时长下采样）" })}
        ${metric("已建模占比", fmt.pct(tl.modeled_pct), { tone: (tl.modeled_pct || 0) >= 40 ? "good" : "warn", foot: "matmul/attention 计算覆盖墙钟时间", barPct: tl.modeled_pct })}
      </div>
      <div class="tl-legend"><span style="color:var(--text-mut)">利用率</span>${utilLeg}</div>
      <div class="tl-legend"><span style="color:var(--text-mut)">算子泳道</span>${opLeg}</div>
      ${panel("算子泳道 Gantt × 利用率泳道（统一时间轴）",
        "上方 Cube/Vector/HBM/通信 利用率（0–100%）· 下方算子按 stream 泳道铺条 · 悬停算子看 类型/Core/MFU/MBU/耗时 · 滚轮或拖滑块缩放 · 已剔除 Notify_Wait",
        `<div id="stl-all" style="width:100%;height:${lay.height}px"></div>`, "span-2")}
      ${placeholders}
      <div class="note">${esc(tl.note || "")}</div>`;
    charts([{ id: "stl-all", option: mergedOption(tl) }]);
  };

  // Stacked lane layout: utilization lanes (Cube/Vector/HBM/通信) on top, then
  // operator stream lanes. Each lane gets its own grid; heights are in px so the
  // container height (set before init) and the grid tops stay in sync.
  function tlLanes(tl) {
    const TOP = 10, uH = 46, oH = 30, GAP = 8;
    const utils = (tl.utilization || []).filter(u => u.available && u.series).map(u =>
      ({ kind: "util", key: u.key, label: u.label, color: u.color, series: u.series,
         abs: u.abs, absUnit: u.abs_unit, peak: u.peak, peakUnit: u.peak_unit, metric: u.kind }));
    const streams = tl.streams || [];
    const present = streams.filter(s => (tl.slices || []).some(d => d.stream === s.key));
    const ops = (present.length ? present : streams).map(s =>
      ({ kind: "op", key: s.key, label: s.label, color: s.color }));
    const lanes = utils.concat(ops);
    let y = TOP;
    lanes.forEach(L => { L.top = y; L.h = (L.kind === "util" ? uH : oH); y += L.h + GAP; });
    const plotBottom = y - GAP;
    return { lanes, plotBottom, height: plotBottom + 46 };  // +46: x-labels + slider
  }

  // One ECharts instance, many vertically-stacked grids sharing a single time
  // axis (linked dataZoom + axisPointer). Util lanes are 0–100% area lines;
  // operator lanes are Gantt bars (custom series). Lane labels are graphic text
  // pinned to the left gutter.
  function mergedOption(tl) {
    const { lanes } = tlLanes(tl);
    const LEFT = 150, RIGHT = 18;
    const spanMs = +(tl.span_us / 1e3).toFixed(2);
    const binMs = tl.bin_us / 1e3;
    const streams = tl.streams || [];
    const colorOf = {}, labelOf = {};
    streams.forEach(s => { colorOf[s.key] = s.color; labelOf[s.key] = s.label; });
    const byStream = {};
    (tl.slices || []).forEach(d => { (byStream[d.stream] = byStream[d.stream] || []).push(d); });
    const laneMeta = {};
    lanes.filter(L => L.kind === "util").forEach(L => {
      laneMeta[L.label] = { absUnit: L.absUnit, peak: L.peak, peakUnit: L.peakUnit, metric: L.metric };
    });

    const grid = [], xAxis = [], yAxis = [], series = [], graphic = [];
    const last = lanes.length - 1;
    const allX = lanes.map((_, i) => i);
    lanes.forEach((L, i) => {
      const isLast = i === last;
      grid.push({ left: LEFT, right: RIGHT, top: L.top, height: L.h });
      xAxis.push(axis({
        type: "value", min: 0, max: spanMs, gridIndex: i,
        axisLine: { show: isLast, lineStyle: { color: "#2d3b52" } },
        axisTick: { show: isLast },
        axisLabel: isLast ? { color: "#6b7888", formatter: v => (+v).toFixed(0) } : { show: false },
        splitLine: { show: false },
        name: isLast ? "ms" : "", nameLocation: "end", nameGap: 4, nameTextStyle: { color: "#6b7888" },
      }));
      if (L.kind === "util") {
        yAxis.push(axis({
          type: "value", min: 0, max: 100, gridIndex: i, splitNumber: 1,
          axisLine: { show: false }, axisTick: { show: false },
          axisLabel: { show: false }, splitLine: { show: false },
        }));
        // visible area line (silent: lets the full-height catcher own the hover).
        // smooth:false — utilization is a per-bin step quantity; splining it rounds
        // off the real sawtooth bursts and can dip below the true value between peaks.
        series.push({
          name: L.label, type: "line", xAxisIndex: i, yAxisIndex: i, silent: true,
          showSymbol: false, smooth: false, lineStyle: { width: 1.4, color: L.color },
          itemStyle: { color: L.color }, areaStyle: { opacity: .18, color: L.color },
          data: L.series.map((v, k) => [+(k * binMs).toFixed(2), +(v * 100).toFixed(1)]),
        });
        // invisible full-lane-height bars: catch hovers anywhere in the lane (not just
        // on the thin line) and carry the % + absolute value for the tooltip
        series.push({
          name: L.label, type: "bar", xAxisIndex: i, yAxisIndex: i, barWidth: "100%",
          itemStyle: { opacity: 0 }, emphasis: { disabled: true }, z: 1,
          data: L.series.map((v, k) => ({
            value: [+(k * binMs).toFixed(2), 100],
            pct: +(v * 100).toFixed(1), abs: (L.abs || [])[k],
          })),
        });
      } else {
        yAxis.push(axis({
          type: "category", data: [""], gridIndex: i,
          axisLine: { show: false }, axisTick: { show: false },
          axisLabel: { show: false }, splitLine: { show: false },
        }));
        const data = (byStream[L.key] || []).map(d =>
          [d.start_ms, d.start_ms + d.dur_ms, 0, colorOf[d.stream] || "#757575", d]);
        series.push({
          name: L.label, type: "custom", xAxisIndex: i, yAxisIndex: i,
          progressive: 2000, progressiveThreshold: 2000, encode: { x: [0, 1], y: 2 },
          renderItem: (params, api) => {
            const s = api.coord([api.value(0), 0]);
            const e = api.coord([api.value(1), 0]);
            const cs = params.coordSys;
            const h = Math.max(6, cs.height * 0.6);
            const rect = echarts.graphic.clipRectByRect(
              { x: s[0], y: cs.y + (cs.height - h) / 2, width: Math.max(e[0] - s[0], 1), height: h },
              { x: cs.x, y: cs.y, width: cs.width, height: cs.height });
            return rect && { type: "rect", shape: rect, style: { fill: api.value(3) } };
          },
          data,
        });
      }
      graphic.push({
        type: "text", left: 8, top: L.top + L.h / 2, z: 30, silent: true,
        style: { text: L.label, fill: L.kind === "util" ? L.color : "#c9d4e0",
          font: '600 11px -apple-system,"Segoe UI",sans-serif', verticalAlign: "middle" },
      });
    });

    return {
      tooltip: Object.assign({
        trigger: "item", confine: true,
        formatter: p => {
          if (p.seriesType === "bar") {                       // utilization lane (catcher)
            const m = laneMeta[p.seriesName] || {};
            const ms = (+p.data.value[0]).toFixed(1);
            const pct = (+p.data.pct).toFixed(0);
            let line2 = "";
            if (p.data.abs != null) {
              line2 = (m.metric === "occupancy")
                ? `占用 ${p.data.abs} / ${m.peak} ${m.absUnit}`
                : `${p.data.abs} ${m.absUnit} / 峰值 ${m.peak} ${m.peakUnit}`;
            }
            return `<b>${esc(p.seriesName)}</b><br/>t ≈ ${ms} ms · 利用率 <b>${pct}%</b>`
              + (line2 ? `<br/>${line2}` : "");
          }
          if (p.seriesType === "line") {                       // silent fallback
            return `${esc(p.seriesName)}<br/>t = ${(+p.data[0]).toFixed(1)} ms · <b>${(+p.data[1]).toFixed(0)}%</b>`;
          }
          const d = p.data && p.data[4]; if (!d) return "";
          const rows = [
            `泳道 <b>${esc(labelOf[d.stream] || d.stream)}</b>`,
            d.type ? `类型 ${esc(d.type)}` : null,
            d.core ? `Core ${esc(d.core)}` : null,
            `start ${d.start_ms.toFixed(3)} ms · 耗时 ${fmt.us(d.dur_ms * 1e3)}`,
            (d.mfu != null || d.mbu != null)
              ? `MFU ${fmt.mfu(d.mfu)} · MBU ${fmt.mfu(d.mbu)}${d.dtype ? " · " + esc(d.dtype) : ""}`
              : "MFU/MBU —（该切片未建模）",
          ].filter(Boolean);
          return `<b>${esc(d.name)}</b><br/>` + rows.join("<br/>");
        },
      }, tooltipBase),
      axisPointer: { link: [{ xAxisIndex: "all" }], type: "line",
        lineStyle: { color: "#3fb6e0", opacity: .5, width: 1 } },
      grid, xAxis, yAxis, series,
      dataZoom: [
        { type: "inside", xAxisIndex: allX, filterMode: "weakFilter" },
        { type: "slider", xAxisIndex: allX, height: 16, bottom: 6, filterMode: "weakFilter",
          backgroundColor: "#0d1117", borderColor: "#2d3b52",
          fillerColor: "rgba(63,182,224,.15)", handleStyle: { color: "#3fb6e0" }, textStyle: { color: "#6b7888" } },
      ],
      graphic,
    };
  }

  LI.views = V;
})();
