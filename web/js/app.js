/* LLMInsight frontend — bootstrap, nav, routing. */
(function () {
  const { api, apiPost, clearCache, fmt, esc } = LI;

  const NAV = [
    { id: "overview",        label: "总览",        ico: "📊", star: true, sub: "Step 时间构成 · 关键比率 · 理论上界" },
    { id: "smart_timeline",  label: "智能时间线",  ico: "🎞️", star: true, sub: "算子泳道 Gantt · MFU/MBU 悬停 · 利用率泳道" },
    { id: "efficiency",      label: "算子效率",    ico: "🎯", star: true, sub: "MFU / MBU / Roofline · 优化余量" },
    { id: "hotspots",        label: "算子热点",    ico: "🔥", sub: "Top 算子 · 按 Core 类型聚合" },
    { id: "communication",   label: "通信分析",    ico: "🔗", sub: "集合通信 · 等待占比 · 带宽" },
    { id: "hidden_overhead", label: "隐性开销",    ico: "🫥", sub: "下发 / 等待 / 空泡 / 动态shape 总账" },
    { id: "attribution",     label: "结构归因",    ico: "🧩", sub: "MLA / MoE / Norm / Optimizer" },
    { id: "memory",          label: "显存洞察",    ico: "🧠", sub: "内存-时间权衡（待采集）" },
    { id: "timeline",        label: "时间线",      ico: "📽️", sub: "泳道占用 · 频率 · 切片" },
    { id: "insights",        label: "LLM 洞察",    ico: "🤖", sub: "诊断卡片 + 可选 LLM 叙述" },
    { id: "replay",          label: "全训练回放",  ico: "🎬", sub: "回放式时间轴（骨架）" },
  ];

  const rendered = {};
  let current = null;
  let currentMeta = null;                 // set once a profile is loaded (ready)
  // directory-picker scratch state (rebuilt each time the picker is shown)
  let pickerInput, pickerCrumbs, pickerList, pickerPath = "";

  const sleep = ms => new Promise(r => setTimeout(r, ms));

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
    if (!currentMeta) return;            // no profile loaded yet — picker is showing
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
      meta = await fetchMeta();
    } catch (e) {
      showFatal("无法连接服务：" + e.message);
      return;
    }
    route(meta);
  }

  // Decide what to show based on backend status. Lazy-by-default: an idle/error
  // server shows the directory picker; a loading server shows the spinner+poll;
  // a ready server boots straight into the views.
  function route(meta) {
    if (meta.status === "ready" || meta.ready) { finishBoot(meta); return; }
    if (meta.status === "loading") { pollUntilReady(); return; }
    showPicker({ error: meta.status === "error" ? meta.error : null,
                 start: meta.suggested_dir || meta.data_dir || "" });
  }

  async function fetchMeta() {
    LI.clearCache("/api/meta");
    const m = await api("/api/meta");
    LI.clearCache("/api/meta");           // /api/meta must never be cached
    return m;
  }

  // -- ready: wire badges/pill/footer/buttons, then open the first view -------
  function finishBoot(meta) {
    currentMeta = meta;
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

    // Topbar buttons. Use .onclick (not addEventListener) so re-loads don't stack
    // duplicate handlers on these static elements.
    const exportBtn = document.getElementById("export-report");
    if (exportBtn) { exportBtn.disabled = false; exportBtn.onclick = () => window.open("/report.html", "_blank", "noopener"); }
    const changeBtn = document.getElementById("change-data");
    if (changeBtn) { changeBtn.disabled = false; changeBtn.onclick = reopenPicker; }

    document.getElementById("views").style.display = "";
    document.getElementById("loading").style.display = "none";
    const start = (location.hash || "").replace("#", "");
    current = null;                       // ensure go() actually renders
    go(NAV.find(n => n.id === start) ? start : "overview");
  }

  // -- loading spinner + poll -------------------------------------------------
  function showLoading(text) {
    document.getElementById("views").style.display = "none";
    const l = document.getElementById("loading");
    l.style.display = "";
    l.innerHTML = `<div class="spinner"></div><div id="loading-text"></div>`;
    l.querySelector("#loading-text").textContent = text || "正在解析 profiling 数据…";
  }

  async function pollUntilReady() {
    showLoading();
    for (let i = 0; i < 300; i++) {       // up to 5 min (cold 104MB trace parse)
      let m;
      try { m = await fetchMeta(); } catch (e) { await sleep(1000); continue; }
      if (m.status === "ready" || m.ready) { finishBoot(m); return; }
      if (m.status === "error") {
        showPicker({ error: m.error || "加载失败", start: m.data_dir || m.suggested_dir || "" });
        return;
      }
      const lt = document.getElementById("loading-text");
      if (lt) lt.textContent = "正在解析 profiling 数据…（" + (i + 1) + "）";
      await sleep(1000);
    }
    showPicker({ error: "加载超时", start: pickerPath });
  }

  function showFatal(msg) {
    const l = document.getElementById("loading");
    l.style.display = "";
    l.innerHTML = `<div class="banner bad"><span class="bi">⚠️</span><div>${esc(msg)}</div></div>`;
  }

  // -- directory picker -------------------------------------------------------
  function showPicker(opts) {
    opts = opts || {};
    document.getElementById("views").style.display = "none";
    const l = document.getElementById("loading");
    l.style.display = "";
    l.innerHTML = "";
    const host = LI.el("div", { class: "picker" });
    l.appendChild(host);

    if (opts.error) host.appendChild(LI.el("div", { class: "picker-err", html: "⚠️ " + esc(opts.error) }));
    host.appendChild(LI.el("div", { class: "picker-head" }, [
      LI.el("div", { class: "picker-title" }, "选择 profiling 目录"),
      LI.el("div", { class: "picker-sub" },
        "进入包含 kernel_details.csv / trace_view.json / step_trace_time.csv 等文件的目录，点「加载此目录」开始分析。"),
    ]));

    // path input row (paste an absolute path, or browse below)
    pickerInput = LI.el("input", { class: "picker-input", type: "text",
      placeholder: "粘贴 profiling 目录的绝对路径，或在下方浏览…", value: opts.start || "" });
    pickerInput.addEventListener("keydown", e => { if (e.key === "Enter") browse(pickerInput.value); });
    const goBtn = LI.el("button", { class: "picker-btn ghost", on: { click: () => browse(pickerInput.value) } }, "前往");
    const loadBtn = LI.el("button", { class: "picker-btn primary", on: { click: () => loadDir(pickerInput.value) } }, "加载此目录");
    const cancel = opts.canCancel
      ? LI.el("button", { class: "picker-btn ghost", on: { click: cancelPicker } }, "取消")
      : null;
    host.appendChild(LI.el("div", { class: "picker-row" }, [pickerInput, goBtn, loadBtn, cancel]));

    pickerCrumbs = LI.el("div", { class: "picker-crumbs" });
    pickerList = LI.el("div", { class: "picker-list" });
    host.appendChild(pickerCrumbs);
    host.appendChild(pickerList);
    browse(opts.start || "");
  }

  async function browse(path) {
    if (pickerList) pickerList.innerHTML = `<div class="picker-empty">列目录中…</div>`;
    let data;
    try {
      const r = await fetch("/api/browse?path=" + encodeURIComponent(path || ""));
      data = await r.json();
    } catch (e) {
      if (pickerList) pickerList.innerHTML = `<div class="picker-err">⚠️ 请求失败：${esc(e.message)}</div>`;
      return;
    }
    if (!data.ok) {
      if (pickerList) pickerList.innerHTML = `<div class="picker-err">⚠️ ${esc(data.error || "读取失败")}</div>`;
      return;
    }
    pickerPath = data.path;
    if (pickerInput) pickerInput.value = data.path;
    renderCrumbs(data);
    renderList(data);
  }

  function renderCrumbs(data) {
    pickerCrumbs.innerHTML = "";
    const sep = data.sep || "/";
    (data.drives || []).forEach(d => {
      const on = data.path.toLowerCase().indexOf(d.toLowerCase()) === 0;
      pickerCrumbs.appendChild(LI.el("span", { class: "crumb-chip" + (on ? " on" : ""), on: { click: () => browse(d) } }, d));
    });
    const startsSep = data.path.startsWith(sep);
    const parts = data.path.split(sep).filter(Boolean);
    let acc = "";
    parts.forEach((p, i) => {
      acc = i === 0 ? (/^[A-Za-z]:$/.test(p) ? p + sep : (startsSep ? sep + p : p)) : acc + sep + p;
      const full = acc;
      pickerCrumbs.appendChild(LI.el("span", { class: "crumb-seg", on: { click: () => browse(full) } }, p));
      if (i < parts.length - 1) pickerCrumbs.appendChild(LI.el("span", { class: "crumb-sl" }, sep));
    });
  }

  function pickerRow(icon, name, onOpen, onLoad, isProfile) {
    const left = LI.el("div", { class: "picker-item-name", on: { click: onOpen } }, [
      LI.el("span", { class: "picker-ico" }, icon),
      LI.el("span", { class: "picker-item-label" }, name),
      isProfile ? LI.el("span", { class: "tag-profile" }, "profiling") : null,
    ]);
    const kids = [left];
    if (onLoad) kids.push(LI.el("button", { class: "picker-btn primary sm", on: { click: onLoad } }, "加载"));
    return LI.el("div", { class: "picker-item" + (isProfile ? " is-profile" : "") }, kids);
  }

  function renderList(data) {
    pickerList.innerHTML = "";
    if (data.is_profile_dir) {
      pickerList.appendChild(LI.el("div", { class: "picker-cur" }, [
        LI.el("span", {}, "✓ 当前目录就是一个 profiling 目录"),
        LI.el("button", { class: "picker-btn primary sm", on: { click: () => loadDir(data.path) } }, "加载此目录"),
      ]));
    }
    if (data.parent) {
      pickerList.appendChild(pickerRow("⬆️", "..", () => browse(data.parent), null, false));
    }
    if (!data.dirs.length && !data.is_profile_dir) {
      pickerList.appendChild(LI.el("div", { class: "picker-empty" }, "（无子目录）"));
    }
    data.dirs.forEach(d => {
      pickerList.appendChild(pickerRow(d.is_profile ? "📊" : "📁", d.name,
        () => browse(d.path), d.is_profile ? () => loadDir(d.path) : null, d.is_profile));
    });
  }

  async function loadDir(path) {
    // keep a 取消 button while a profile is still loaded, so a bad path can't trap
    // the user in the picker with no way back to the data they already had.
    const back = !!currentMeta;
    path = (path || "").trim();
    if (!path) { showPicker({ error: "请先选择或输入一个目录", start: pickerPath, canCancel: back }); return; }
    pickerPath = path;
    showLoading("正在请求加载：" + path);
    let res;
    try { res = await apiPost("/api/load", { dir: path }); }
    catch (e) { showPicker({ error: "请求失败：" + e.message, start: path, canCancel: back }); return; }
    if (!res || !res.ok) { showPicker({ error: (res && res.error) || "加载失败", start: path, canCancel: back }); return; }
    resetViews();                         // drop any previously-rendered views/cache
    await pollUntilReady();
  }

  function resetViews() {
    Object.keys(rendered).forEach(k => delete rendered[k]);
    current = null;
    currentMeta = null;
    LI.clearCache();                      // drop all client-side API cache
    document.getElementById("views").innerHTML = "";  // remove stale view containers
  }

  function reopenPicker() {
    showPicker({ canCancel: true, start: (currentMeta && currentMeta.data_dir) || pickerPath || "" });
  }

  function cancelPicker() {
    if (!currentMeta) return;             // nothing loaded to fall back to
    document.getElementById("loading").style.display = "none";
    document.getElementById("views").style.display = "";
  }

  window.addEventListener("hashchange", () => {
    const id = (location.hash || "").replace("#", "");
    if (currentMeta && id && id !== current && NAV.find(n => n.id === id)) go(id);
  });

  boot();
})();
