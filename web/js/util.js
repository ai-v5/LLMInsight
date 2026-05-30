/* LLMInsight frontend — shared utilities (build-less, global `LI` namespace). */
const LI = (function () {
  const cache = {};

  async function api(path) {
    if (cache[path]) return cache[path];
    const r = await fetch(path);
    if (!r.ok) {
      let msg = r.status;
      try { const j = await r.json(); msg = j.error || msg; } catch (e) {}
      throw new Error(path + " → " + msg);
    }
    const j = await r.json();
    cache[path] = j;
    return j;
  }
  function apiPost(path, body) {
    const opts = { method: "POST" };
    if (body !== undefined) {
      opts.headers = { "Content-Type": "application/json" };
      opts.body = JSON.stringify(body);
    }
    return fetch(path, opts).then(r => r.json());
  }
  function clearCache(path) { if (path) delete cache[path]; else Object.keys(cache).forEach(k => delete cache[k]); }

  // ---- formatting ----
  const fmt = {
    us(v) { if (v == null) return "—"; v = +v;
      if (v >= 1e6) return (v / 1e6).toFixed(2) + " s";
      if (v >= 1e3) return (v / 1e3).toFixed(1) + " ms";
      return v.toFixed(0) + " us"; },
    ms(v) { if (v == null) return "—"; return (+v).toFixed(v < 10 ? 2 : 1) + " ms"; },
    pct(v, d = 1) { return v == null ? "—" : (+v).toFixed(d) + "%"; },
    num(v) { if (v == null) return "—"; return (+v).toLocaleString("en-US"); },
    int(v) { return v == null ? "—" : Math.round(+v).toLocaleString("en-US"); },
    bytes(v) { if (v == null) return "—"; v = +v; const u = ["B","KB","MB","GB","TB"]; let i = 0;
      while (v >= 1024 && i < u.length - 1) { v /= 1024; i++; } return v.toFixed(1) + " " + u[i]; },
    flops(v) { if (v == null) return "—"; v = +v; if (v >= 1e12) return (v/1e12).toFixed(1)+" T"; if (v>=1e9) return (v/1e9).toFixed(1)+" G"; return v.toFixed(0); },
    tflops(v) { return v == null ? "—" : (+v).toFixed(1) + " TF"; },
    mfu(v) { return v == null ? "—" : (v * 100).toFixed(1) + "%"; },
  };

  // ---- DOM ----
  function el(tag, attrs, children) {
    const e = document.createElement(tag);
    if (attrs) for (const k in attrs) {
      if (k === "class") e.className = attrs[k];
      else if (k === "html") e.innerHTML = attrs[k];
      else if (k === "on") for (const ev in attrs.on) e.addEventListener(ev, attrs.on[ev]);
      else e.setAttribute(k, attrs[k]);
    }
    if (children != null) (Array.isArray(children) ? children : [children]).forEach(c => {
      if (c == null) return;
      e.appendChild(typeof c === "string" ? document.createTextNode(c) : c);
    });
    return e;
  }
  function h(html) { const d = el("div"); d.innerHTML = html; return d; }
  function esc(s) { return String(s == null ? "" : s).replace(/[&<>"]/g, c => ({ "&":"&amp;","<":"&lt;",">":"&gt;","\"":"&quot;" }[c])); }

  // metric card
  function metric(label, value, opts) {
    opts = opts || {};
    const cls = "metric" + (opts.tone ? " " + opts.tone : "");
    return `<div class="${cls}">
      <div class="label">${esc(label)}</div>
      <div class="value">${value}</div>
      ${opts.foot ? `<div class="foot">${esc(opts.foot)}</div>` : ""}
      ${opts.barPct != null ? `<div class="bar" style="width:${Math.min(100, opts.barPct)}%"></div>` : ""}
    </div>`;
  }
  function panel(title, sub, bodyHtml, cls) {
    return `<div class="panel ${cls || ""}">
      ${title ? `<h3>${esc(title)}</h3>` : ""}
      ${sub ? `<div class="sub">${esc(sub)}</div>` : ""}
      ${bodyHtml || ""}</div>`;
  }
  function banner(tone, icon, html) {
    return `<div class="banner ${tone}"><span class="bi">${icon}</span><div>${html}</div></div>`;
  }

  // ---- ECharts ----
  const palette = ["#3fb6e0","#5ee0b8","#d29922","#f85149","#a371f7","#79b8ff","#56d364","#ec6cb9","#e3b341","#76e3ea"];
  const charts = [];
  function chart(node, option) {
    if (!node) return null;
    const c = echarts.init(node, null, { renderer: "canvas" });
    c.setOption(Object.assign({ color: palette, textStyle: { color: "#c9d4e0", fontFamily: "inherit" } }, option));
    charts.push(c);
    return c;
  }
  function resizeAll() { charts.forEach(c => { try { c.resize(); } catch (e) {} }); }
  window.addEventListener("resize", () => { clearTimeout(window.__rz); window.__rz = setTimeout(resizeAll, 120); });

  const axis = (extra) => Object.assign({
    axisLine: { lineStyle: { color: "#2d3b52" } },
    axisLabel: { color: "#9aa7b8" },
    splitLine: { lineStyle: { color: "#1b2433" } },
    nameTextStyle: { color: "#6b7888" },
  }, extra || {});
  const tooltipBase = { backgroundColor: "#0d1117", borderColor: "#2d3b52", textStyle: { color: "#e6edf3" }, confine: true };

  function sevColor(s) { return { high:"#f85149", medium:"#d29922", low:"#58a6ff", info:"#8b949e" }[s] || "#8b949e"; }

  return { api, apiPost, clearCache, fmt, el, h, esc, metric, panel, banner,
           chart, resizeAll, palette, axis, tooltipBase, sevColor };
})();
