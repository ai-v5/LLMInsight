/* LLMInsight frontend — bootstrap, nav, routing. */
(function () {
  const { api, apiPost, clearCache, fmt, esc } = LI;

  const NAV = [
    { id: "overview",        label: "总览",        ico: "📊", sub: "Step 时间构成 · 关键比率 · 理论上界" },
    { id: "replay",          label: "全训练回放",  ico: "🎬", star: true, sub: "回放式时间轴（骨架）" },
    { id: "hotspots",        label: "算子热点",    ico: "🔥", sub: "Top 算子 · 按 Core 类型聚合" },
    { id: "efficiency",      label: "算子效率",    ico: "🎯", sub: "MFU / MBU / Roofline · 优化余量" },
    { id: "communication",   label: "通信分析",    ico: "🔗", sub: "集合通信 · 等待占比 · 带宽" },
    { id: "hidden_overhead", label: "隐性开销",    ico: "🫥", sub: "下发 / 等待 / 空泡 / 动态shape 总账" },
    { id: "attribution",     label: "结构归因",    ico: "🧩", sub: "MLA / MoE / Norm / Optimizer" },
    { id: "memory",          label: "显存洞察",    ico: "🧠", sub: "内存-时间权衡（待采集）" },
    { id: "timeline",        label: "时间线",      ico: "📽️", sub: "泳道占用 · 频率 · 切片" },
    { id: "smart_timeline",  label: "智能时间线",  ico: "🎞️", star: true, sub: "算子泳道 Gantt · MFU/MBU 悬停 · 利用率泳道" },
    { id: "insights",        label: "LLM 洞察",    ico: "🤖", star: true, sub: "诊断卡片 + 可选 LLM 叙述" },
  ];

  const rendered = {};
  let current = null;

  function buildNav() {
    const nav = document.getElementById("nav");
    NAV.forEach(item => {
      const node = LI.el("div", { class: "nav-item", "data-id": item.id, on: { click: () => go(item.id) } }, [
        LI.el("span", { class: "ico" }, item.ico),
        LI.el("span", {}, item.label),
        item.star ? LI.el("span", { class: "star" }, "★") : null,
      ]);
      nav.appendChild(node);
    });
  }

  function viewContainer(id) {
    let v = document.getElementById("view-" + id);
    if (!v) {
      v = LI.el("div", { class: "view", id: "view-" + id });
      document.getElementById("views").appendChild(v);
    }
    return v;
  }

  async function go(id) {
    if (current === id) return;
    current = id;
    location.hash = id;
    const item = NAV.find(n => n.id === id) || NAV[0];
    document.querySelectorAll(".nav-item").forEach(n => n.classList.toggle("active", n.getAttribute("data-id") === id));
    document.getElementById("view-title").textContent = item.label;
    document.getElementById("view-sub").textContent = item.sub || "";
    document.querySelectorAll(".view").forEach(v => v.classList.remove("active"));
    const root = viewContainer(id);
    root.classList.add("active");
    if (!rendered[id]) {
      root.innerHTML = `<div class="loading"><div class="spinner"></div><div>加载中…</div></div>`;
      try {
        await LI.views[id](root);
        rendered[id] = true;
      } catch (e) {
        root.innerHTML = `<div class="banner bad"><span class="bi">⚠️</span><div>渲染失败：${esc(e.message)}</div></div>`;
        console.error(e);
      }
    }
    setTimeout(LI.resizeAll, 60);
  }

  function chipNote(chip) {
    chip = chip || {};
    return (chip.name || "") + (chip.assumed ? "（参考峰值）" : "");
  }

  // Views whose numbers depend on the chip peak (MFU/MBU/Roofline/理论上界/cards)
  // or capacity (memory view's 显存容量).
  const CHIP_VIEWS = ["overview", "efficiency", "insights", "memory", "smart_timeline"];
  const CHIP_CACHES = ["/api/efficiency", "/api/theoretical", "/api/overview",
                       "/api/insights", "/api/meta", "/api/all", "/api/smart_timeline"];
  async function onChipChange(key) {
    const sel = document.getElementById("chip-select");
    const note = document.getElementById("chip-note");
    if (sel) sel.disabled = true;
    if (note) note.textContent = "切换芯片中…";
    try {
      const res = await apiPost("/api/chip", { chip: key });
      if (!res || !res.ok) throw new Error((res && res.error) || "切换失败");
      CHIP_CACHES.forEach(p => clearCache(p));            // drop stale client cache
      CHIP_VIEWS.forEach(id => { delete rendered[id]; });  // force re-render on visit
      if (note) note.textContent = chipNote(res.chip || {});
      const cur = current;                                 // re-render the open view
      current = null;
      await go(cur);
    } catch (e) {
      if (note) note.textContent = "切换失败：" + e.message;
      console.error(e);
    } finally {
      const s = document.getElementById("chip-select");
      if (s) s.disabled = false;
    }
  }

  async function boot() {
    buildNav();
    let meta;
    try {
      meta = await waitReady();
    } catch (e) {
      document.getElementById("loading-text").textContent = "加载失败：" + e.message;
      return;
    }
    // badges
    const m = meta.meta || {}, settings = (m.settings || {});
    const model = settings.model || {}, chip = settings.chip || {};
    const chips = settings.chips || [];
    const chipKey = settings.chip_key || "";
    const badges = document.getElementById("badges");
    const chipOptions = (chips.length ? chips : [{ key: chipKey, name: chip.name }])
      .map(c => `<option value="${esc(c.key)}"${c.key === chipKey ? " selected" : ""}>${esc(c.name)}</option>`).join("");
    badges.innerHTML = [
      `<span class="badge">模型 <strong>${esc(model.name || "—")}</strong></span>`,
      `<span class="badge">并行 <strong>TP${model.tp||"?"}/PP${model.pp||"?"}/EP${model.ep||"?"}/CP${model.cp||"?"}</strong></span>`,
      `<span class="badge assume" title="切换参考芯片 → 重算 MFU / MBU / Roofline / 理论上界">芯片 <select id="chip-select" class="chip-select">${chipOptions}</select></span>`,
      `<span class="badge">加载 <strong>${meta.load_seconds}s</strong></span>`,
    ].join("");
    const chipSel = document.getElementById("chip-select");
    if (chipSel) chipSel.addEventListener("change", () => onChipChange(chipSel.value));
    // llm pill
    const pill = document.getElementById("llm-pill");
    const llm = meta.llm || {};
    pill.className = "pill " + (llm.available ? "pill-on" : "pill-off");
    pill.textContent = "LLM: " + (llm.available ? (llm.provider + " ✓") : "关闭（规则引擎可用）");
    pill.title = llm.reason || "";
    document.getElementById("foot-data").textContent = (meta.data_dir || "").replace(/^.*[\\/]secret[\\/]/, "secret/");
    document.getElementById("chip-note").textContent = chipNote(chip);

    document.getElementById("loading").style.display = "none";
    const start = (location.hash || "").replace("#", "");
    go(NAV.find(n => n.id === start) ? start : "overview");
  }

  async function waitReady() {
    for (let i = 0; i < 120; i++) {
      const meta = await api("/api/meta");
      LI.clearCache("/api/meta");
      if (meta.ready) return meta;
      if (meta.error) throw new Error(meta.error);
      document.getElementById("loading-text").textContent = "正在解析 profiling 数据…（" + (i + 1) + "）";
      await new Promise(r => setTimeout(r, 1000));
    }
    throw new Error("加载超时");
  }

  window.addEventListener("hashchange", () => {
    const id = (location.hash || "").replace("#", "");
    if (id && id !== current && NAV.find(n => n.id === id)) go(id);
  });

  boot();
})();
