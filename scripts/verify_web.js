// Headless render harness: runs each view renderer with REAL server JSON and a
// stubbed DOM + ECharts. Catches data-shape bugs (undefined.map, wrong keys,
// bad ECharts options) without a browser. Requires the server on :8765.
const fs = require("fs");
const vm = require("vm");
const path = require("path");

// Default to :8765, but honor LLMINSIGHT_PORT/LLMINSIGHT_BASE so the harness can target
// a fresh server on another port when 8765 is held by a stale process.
const BASE = process.env.LLMINSIGHT_BASE
  || `http://127.0.0.1:${process.env.LLMINSIGHT_PORT || 8765}`;
const WEB = path.join(__dirname, "..", "web", "js");

// ---- stub browser globals ----
function fakeNode() {
  return {
    _html: "",
    style: {}, className: "", textContent: "",
    classList: { toggle() {}, add() {}, remove() {} },
    set innerHTML(v) { this._html = String(v); },
    get innerHTML() { return this._html; },
    appendChild() {}, setAttribute() {}, getAttribute() { return null; },
    addEventListener() {}, querySelectorAll() { return []; },
  };
}
let chartCount = 0;
global.window = { addEventListener() {}, setTimeout, __rz: 0 };
global.document = {
  createElement: () => fakeNode(),
  getElementById: () => fakeNode(),
  querySelectorAll: () => [],
  addEventListener() {},
  createTextNode: (t) => ({ t }),
};
global.echarts = { init: () => ({ setOption(o) { chartCount++; JSON.stringify(o); }, resize() {} }) };
global.requestAnimationFrame = (fn) => fn();
// resolve the views' relative `/api/...` calls against the running server
const _realFetch = global.fetch;
global.fetch = (url, opts) => _realFetch(url.startsWith("http") ? url : BASE + url, opts);

// ---- load util.js + views.js into this context ----
vm.runInThisContext(fs.readFileSync(path.join(WEB, "util.js"), "utf8") + "\nglobalThis.LI = LI;");
vm.runInThisContext(fs.readFileSync(path.join(WEB, "views.js"), "utf8"));

// The app is lazy-by-default now: the server starts idle and a profiling dir is
// chosen via POST /api/load. Drive that flow (also exercises /api/browse + load)
// so the views/report below render against real loaded data.
async function ensureLoaded() {
  const getMeta = async () => (await _realFetch(BASE + "/api/meta")).json();
  let meta = await getMeta();
  if (meta.status === "ready" || meta.ready) { console.log("  loaded (already ready)"); return; }

  const br = await (await _realFetch(BASE + "/api/browse")).json();
  console.log(`  browse           ok=${br.ok} dirs=${(br.dirs || []).length} path=${br.path}`);

  const dir = meta.suggested_dir || meta.data_dir;
  const res = await (await _realFetch(BASE + "/api/load", {
    method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ dir }),
  })).json();
  if (!res.ok) throw new Error("load failed: " + (res.error || JSON.stringify(res)));
  for (let i = 0; i < 300; i++) {
    meta = await getMeta();
    if (meta.status === "ready" || meta.ready) { console.log(`  loaded in ${meta.load_seconds}s: ${dir}`); return; }
    if (meta.status === "error") throw new Error("load error: " + meta.error);
    await new Promise(r => setTimeout(r, 1000));
  }
  throw new Error("load timed out");
}

(async () => {
  try {
    await ensureLoaded();
  } catch (e) {
    console.log("  FAIL ensureLoaded     " + e.message);
    process.exit(1);
  }
  const ids = Object.keys(LI.views).filter(k => !k.startsWith("_"));
  let fail = 0;
  for (const id of ids) {
    const root = fakeNode();
    chartCount = 0;
    try {
      await LI.views[id](root);
      const len = root._html.length;
      // surface accidental "undefined" leaking into rendered HTML
      const undefs = (root._html.match(/undefined/g) || []).length;
      console.log(`  OK  ${id.padEnd(16)} html=${String(len).padStart(6)}B charts=${chartCount}` +
                  (undefs ? `  ⚠ ${undefs}×"undefined"` : ""));
    } catch (e) {
      fail++;
      console.log(`  FAIL ${id.padEnd(16)} ${e.stack.split("\n").slice(0,3).join("\n        ")}`);
    }
  }

  // ---- What-if 现实地板 panel must render inside 总览 (from theo.realistic) ----
  try {
    const root = fakeNode();
    await LI.views.overview(root);
    const h = root._html;
    const okFloor = h.includes("现实地板");        // realistic-floor panel present
    const okZero = h.includes("能减到 0");         // per-lever 能否减到0 verdict
    const okComb = h.includes("综合现实地板");       // combined floor banner
    const okSync = h.includes("%→");               // upper What-if table synced to floor (实测%→地板%)
    const undefs = (h.match(/undefined/g) || []).length;
    const ok = okFloor && okZero && okComb && okSync && undefs === 0;
    if (!ok) fail++;
    console.log(`  ${ok ? "OK  " : "FAIL"} ${"whatif-floor".padEnd(16)} floor=${okFloor} zero=${okZero} combined=${okComb} sync=${okSync}` +
                (undefs ? `  ⚠ ${undefs}×"undefined"` : ""));
  } catch (e) {
    fail++;
    console.log(`  FAIL ${"whatif-floor".padEnd(16)} ${e.stack.split("\n").slice(0,3).join("\n        ")}`);
  }

  // ---- shareable report (H4): self-contained HTML served at /report.html ----
  try {
    const r = await global.fetch("/report.html");
    const html = await r.text();
    const okDoc = /^<!doctype html/i.test(html.trim());
    const okDiag = html.includes("诊断");          // rule cards (diagnostics-first)
    const okView = html.includes("时间构成");        // overview section present
    const undefs = (html.match(/undefined/g) || []).length;
    const ok = r.status === 200 && okDoc && okDiag && okView && undefs === 0;
    if (!ok) fail++;
    console.log(`  ${ok ? "OK  " : "FAIL"} ${"report.html".padEnd(16)} status=${r.status} ` +
                `html=${String(html.length).padStart(6)}B doc=${okDoc} diag=${okDiag} overview=${okView}` +
                (undefs ? `  ⚠ ${undefs}×"undefined"` : ""));
  } catch (e) {
    fail++;
    console.log(`  FAIL ${"report.html".padEnd(16)} ${e.message}`);
  }

  console.log(fail ? `\n${fail} check(s) FAILED` : `\nAll ${ids.length} views + report rendered OK`);
  process.exit(fail ? 1 : 0);
})();
