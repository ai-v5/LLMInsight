// Headless render harness: runs each view renderer with REAL server JSON and a
// stubbed DOM + ECharts. Catches data-shape bugs (undefined.map, wrong keys,
// bad ECharts options) without a browser. Requires the server on :8765.
const fs = require("fs");
const vm = require("vm");
const path = require("path");

const BASE = "http://127.0.0.1:8765";
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

(async () => {
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
  console.log(fail ? `\n${fail} view(s) FAILED` : `\nAll ${ids.length} views rendered OK`);
  process.exit(fail ? 1 : 0);
})();
