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
    const whatifRows = (theo.available ? theo.whatif : []).map(w =>
      `<tr><td>${esc(w.scenario)}</td><td>${fmt.us(w.new_step_us)}</td><td style="color:var(--accent-2)">-${fmt.pct(w.save_pct)}</td></tr>`).join("");
    const cb = theo.available && theo.compute_bound;
    root.innerHTML = `
      <div class="grid cols-6">${cards.join("")}</div>
      <div class="grid cols-2" style="margin-top:16px">
        ${panel("Step 时间构成", "Computing / 未掩盖通信 / Free（单位 us）", `<div id="ov-donut" class="chart"></div>`)}
        ${panel("理论上界 & What-if 收益模拟", "基于 step 时间构成的优化上界（用于排优先级，非精确预测）",
          `<table class="tbl"><thead><tr><th>场景</th><th>预计 step</th><th>节省</th></tr></thead><tbody>${whatifRows || `<tr><td colspan=3 class="empty">—</td></tr>`}</tbody></table>
           ${cb ? (cb.peak_underestimated
             ? banner("warn","⚠️", `matmul 实测 MFU <strong>${cb.matmul_mfu_pct}%</strong> &gt; 100% → 假设芯片峰值偏低，请在 config.ChipSpec 校正。`)
             : `<div class="note">matmul MFU ≈ <strong>${cb.matmul_mfu_pct}%</strong>；计算理想 ${fmt.us(cb.ideal_matmul_us)}，余量 ${fmt.us(cb.headroom_us)}。${cb.calibrated ? `（芯片峰值按实测 ${cb.observed_peak_tflops} TFLOPS 校准，假设 ${cb.assumed_peak_tflops != null ? cb.assumed_peak_tflops.toFixed(0) : "—"}）` : ""}</div>`) : ""}`)}
      </div>
      <div class="note">${esc(theo.available ? theo.note : "")}</div>`;
    charts([{ id: "ov-donut", option: donut(ov.composition.map(c => ({ name: c.name, value: c.us, pct: c.pct }))) }]);
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
    const byType = ef.by_type.slice(0, 20).map(t =>
      `<tr><td>${esc(t.type)}</td><td>${fmt.int(t.count)}</td><td>${fmt.us(t.dur_us)}</td><td>${t.mfu != null ? fmt.mfu(t.mfu) : "—"}</td><td>${t.mbu != null ? fmt.mfu(t.mbu) : "—"}</td><td>${fmt.us(t.wasted_us)}</td></tr>`).join("");
    root.innerHTML = `
      ${banner(chip.calibrated ? "warn" : "info", "🎯",
        `芯片：<strong>${esc(chip.name)}</strong> · 假设峰值 BF16 <strong>${chip.peak_bf16_tflops.toFixed(0)} TFLOPS</strong> · HBM <strong>${chip.hbm_tbps.toFixed(2)} TB/s</strong>` +
        (chip.hbm_capacity_gb ? ` · 显存 <strong>${chip.hbm_capacity_gb.toFixed(0)} GB</strong>` : "") +
        (chip.calibrated
          ? `　|　⚠️ 实测峰值 <strong>${chip.observed_peak_tflops.toFixed(0)} TFLOPS</strong> &gt; 假设值 → 已按实测校准（设真实 SKU 峰值可覆盖）`
          : `（${chip.assumed ? "参考峰值；顶部可切换芯片，或在 config.ChipSpec 校正" : "实测"}）`) +
        `　|　matmul MFU ≈ <strong>${ef.matmul_mfu != null ? (ef.matmul_mfu*100).toFixed(0)+"%" : "—"}</strong>　|　建模 kernel ${fmt.int(ef.kernels_with_flops)}/${fmt.int(ef.kernels_total)}`)}
      <div class="grid cols-2">
        ${panel("Roofline", "点 = kernel；x = 算术强度 (FLOP/Byte)，y = 达成算力 (TFLOPS)，对照屋顶线", `<div id="ef-roof" class="chart tall"></div>`)}
        ${panel("优化空间排行 (Top 15)", "按「实测 − Roofline 理想」的浪费时间排序；颜色 = 瓶颈类型", `<div id="ef-waste" class="chart tall"></div>`)}
      </div>
      ${panel("按算子类型 MFU / MBU / 浪费", "", `<div class="tbl-wrap"><table class="tbl"><thead><tr><th>Type</th><th>Count</th><th>耗时</th><th>MFU</th><th>MBU</th><th>浪费(vs理想)</th></tr></thead><tbody>${byType}</tbody></table></div>`, "span-2")}`;
    charts([
      { id: "ef-roof", option: roofline(ef.scatter, chip, ef.roofline_ridge_ai) },
      { id: "ef-waste", option: hbar(top.map(t => t.name.slice(0, 26)), top.map(t => t.wasted_us), "浪费 us",
          { fmt: v => fmt.us(v), color: top.map(t => boundColor(t.bound)) }) },
    ]);
  };
  function boundColor(b) { return { compute: "#79b8ff", memory: "#e3b341", vector: "#5ee0b8" }[b] || "#6b7888"; }

  function roofline(scatter, chip, ridge) {
    const peak = chip.effective_peak_tflops || chip.peak_bf16_tflops, bw = chip.hbm_tbps; // TFLOPS, TB/s
    scatter = scatter || [];
    const xs = scatter.map(s => s.ai).filter(x => x > 0);
    const xmin = Math.max(0.1, Math.min.apply(null, xs.concat([ridge])) / 3);
    const xmax = Math.max(xmin * 10, Math.max.apply(null, xs.concat([ridge])) * 3);
    const memRoof = [[xmin, xmin * bw], [ridge, peak]];
    const compRoof = [[ridge, peak], [xmax, peak]];
    const byBound = {};
    scatter.forEach(s => { (byBound[s.bound] = byBound[s.bound] || []).push([s.ai, s.tflops, s.name, s.dur_us]); });
    const series = Object.keys(byBound).map(b => ({
      name: b, type: "scatter", symbolSize: d => Math.min(22, 4 + Math.sqrt(d[3]) / 6),
      itemStyle: { color: boundColor(b), opacity: .65 }, data: byBound[b],
    }));
    series.push({ name: "屋顶线", type: "line", data: memRoof.concat(compRoof.slice(1)), showSymbol: false, lineStyle: { color: "#f85149", width: 1.5, type: "dashed" }, tooltip: { show: false }, z: 1 });
    return {
      tooltip: Object.assign({ trigger: "item", formatter: p => p.seriesName === "屋顶线" ? "" :
        `${esc(p.data[2])}<br/>AI=${p.data[0].toFixed(2)} FLOP/B<br/>达成=${p.data[1].toFixed(1)} TFLOPS<br/>耗时=${fmt.us(p.data[3])}<br/>瓶颈: ${p.seriesName}` }, tooltipBase),
      legend: { top: 0, textStyle: { color: "#9aa7b8" } },
      grid: { left: 10, right: 20, top: 34, bottom: 36, containLabel: true },
      xAxis: axis({ type: "log", name: "算术强度 FLOP/Byte", min: xmin, max: xmax }),
      yAxis: axis({ type: "log", name: "TFLOPS", min: 0.1, max: peak * 1.5 }),
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
    const dev = ho.buckets.filter(b => b.domain === "device" && b.us != null);
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
        ${panel("Device 侧隐性开销 (计入 step)", "AICPU下发 / 未掩盖通信 / 空泡 / 格式转换初始化", `<div id="ho-dev" class="chart"></div>`)}
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
    const sliceRows = tl.top_slices.slice(0, 40).map(s =>
      `<tr><td class="mono">${esc(s.name)}</td><td>${(s.start_us/1e3).toFixed(1)} ms</td><td>${fmt.us(s.dur_us)}</td></tr>`).join("");
    root.innerHTML = `
      <div class="grid cols-4">
        ${metric("时间跨度", tl.span_s + " s")}
        ${metric("时间桶", fmt.int(tl.bins), { foot: fmt.us(tl.bin_us) + "/桶" })}
        ${metric("泳道", fmt.int(tl.lanes.length))}
        ${metric("Overlap 段", fmt.int(tl.overlap_segments.length))}
      </div>
      ${panel("泳道占用率热力图（回放式时间轴）", "每个时间桶内事件覆盖比例（0–1）；颜色越亮越忙", `<div id="tl-heat" class="chart" style="height:210px"></div>`, "span-2")}
      <div class="grid cols-2">
        ${panel("AI Core 频率", "MHz over time", `<div id="tl-freq" class="chart short"></div>`)}
        ${panel("Top Device Kernel 切片 (>1.5ms)", "", `<div class="tbl-wrap" style="max-height:240px"><table class="tbl"><thead><tr><th>Kernel</th><th>起始</th><th>时长</th></tr></thead><tbody>${sliceRows}</tbody></table></div>`)}
      </div>
      <div class="note">${esc(tl.note)}</div>`;
    charts([
      { id: "tl-heat", option: heatmap(tl) },
      { id: "tl-freq", option: freqLine(tl.ai_core_freq) },
    ]);
  };

  function heatmap(tl) {
    if (!tl.lanes || !tl.lanes.length) return { xAxis: {}, yAxis: {}, series: [] };
    const lanes = tl.lanes.map(l => l.label);
    const data = [];
    tl.lanes.forEach((l, li) => l.occupancy.forEach((v, bi) => { if (v > 0.001) data.push([bi, li, v]); }));
    return {
      tooltip: Object.assign({ position: "top", formatter: p =>
        `${lanes[p.data[1]]}<br/>t=${(p.data[0]*tl.bin_us/1e6).toFixed(2)}s<br/>占用 ${(p.data[2]*100).toFixed(0)}%` }, tooltipBase),
      grid: { left: 10, right: 14, top: 10, bottom: 28, containLabel: true },
      xAxis: axis({ type: "category", data: tl.lanes[0].occupancy.map((_, i) => i), name: "时间桶",
        axisLabel: { show: true, interval: Math.floor(tl.bins / 10), formatter: i => (i*tl.bin_us/1e6).toFixed(1)+"s" } }),
      yAxis: axis({ type: "category", data: lanes, axisLabel: { color: "#c9d4e0", width: 150, overflow: "truncate" } }),
      visualMap: { min: 0, max: 1, calculable: true, orient: "horizontal", left: "center", bottom: -4, show: false,
        inRange: { color: ["#0d1117", "#1d4e63", "#3fb6e0", "#5ee0b8"] } },
      series: [{ type: "heatmap", data, progressive: 2000, itemStyle: { borderWidth: 0 } }],
    };
  }
  function freqLine(freq) {
    return {
      tooltip: Object.assign({ trigger: "axis", formatter: p => `${(p[0].data[0]/1e6).toFixed(2)}s<br/><b>${p[0].data[1]} MHz</b>` }, tooltipBase),
      grid: { left: 10, right: 16, top: 14, bottom: 28, containLabel: true },
      xAxis: axis({ type: "value", name: "s", axisLabel: { formatter: v => (v/1e6).toFixed(1) } }),
      yAxis: axis({ type: "value", name: "MHz", scale: true }),
      series: [{ type: "line", showSymbol: false, smooth: true, areaStyle: { opacity: .12 }, lineStyle: { color: "#5ee0b8" }, itemStyle: { color: "#5ee0b8" },
        data: freq.map(f => [f.t_us, f.mhz]) }],
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

  LI.views = V;
})();
