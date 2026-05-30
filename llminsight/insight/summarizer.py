"""Build a compact, privacy-safe summary of the metrics + rule cards, and the
prompt messages for the LLM.

PRIVACY CONTRACT: only aggregated numbers and the rule cards leave this module.
We never include raw trace events, tensor data, file paths, usernames, or the
full captured environment. Capture config is whitelisted to a handful of
performance-relevant flags. The summary is KB-level and human-auditable (the
server exposes it so the user can see exactly what would be sent).
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

# Only these capture keys are ever forwarded — keeps usernames / log paths /
# CPU-affinity strings out of the payload.
_ENV_WHITELIST = ("ASCEND_LAUNCH_BLOCKING", "HCCL_BUFFSIZE",
                  "HCCL_ALG_MULTIPLE_DIMENSION_SPLIT_RATIO", "TASK_QUEUE_ENABLE")
_FLAG_WHITELIST = ("recompute-granularity", "recompute-num-layers",
                   "expert-model-parallel-size", "tensor-model-parallel-size",
                   "pipeline-model-parallel-size", "context-parallel-size",
                   "num-experts", "moe-router-topk", "global-batch-size",
                   "num-layers", "seq-length", "swap-optimizer", "moe-fb-overlap",
                   "moe-permutation-async-comm", "use-flash-attn", "sequence-parallel")

SYSTEM_PROMPT = (
    "你是昇腾（Ascend）NPU 大模型训练性能调优专家，精通 MindSpeed-LLM / torch_npu "
    "profiling、MoE（EP/alltoall）、MLA、Roofline/MFU/MBU、HCCL 通信与计算-通信重叠。"
    "你会收到一份【结构化性能指标摘要】和【规则引擎已命中的诊断卡片】（事实依据，禁止编造数据）。"
    "请基于这些事实，产出一份面向算法/训练工程师的简洁中文诊断报告，包含："
    "(1) 一句话总体结论；(2) 按收益排序的 Top 3-5 优化项，每项给【现象→根因→可执行建议→预计收益→置信度】；"
    "(3) 必要的采集可信度提醒。要求：只用给定数据，不臆造数字；建议要落地到具体开关/参数；"
    "对被 ASCEND_LAUNCH_BLOCKING 放大的 host 指标要标注为采集干扰。"
)


def _round(x, n=2):
    return round(x, n) if isinstance(x, (int, float)) else x


def build_summary(m: Dict[str, Any],
                  cards: List[Dict[str, Any]],
                  capture: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Compact JSON-serializable summary. This is the EXACT payload the LLM sees."""
    ov = m.get("overview", {})
    hot = m.get("hotspots", {})
    eff = m.get("efficiency", {})
    comm = m.get("communication", {})
    ho = m.get("hidden_overhead", {})
    attr = m.get("attribution", {})
    theo = m.get("theoretical", {})
    meta = m.get("meta", {})
    capture = capture or {}

    cap_env = {k: capture.get("env", {}).get(k) for k in _ENV_WHITELIST
               if k in capture.get("env", {})}
    cap_flags = {k: capture.get("flags", {}).get(k) for k in _FLAG_WHITELIST
                 if k in capture.get("flags", {})}

    settings = meta.get("settings", {})
    summary: Dict[str, Any] = {
        "model": settings.get("model", {}).get("name"),
        "chip": eff.get("chip"),
        "capture": {"env": cap_env, "flags": cap_flags},
        "overview": {
            "step_time_s": ov.get("ratios", {}).get("step_time_s"),
            "ratios": ov.get("ratios"),
            "composition": ov.get("composition"),
        } if ov.get("available") else None,
        "hotspots_top": [
            {"type": o["type"], "core": o["core"], "ratio": o["ratio"],
             "total_us": o["total_us"], "count": o["count"]}
            for o in hot.get("top", [])[:8]
        ] if hot.get("available") else None,
        "by_core": hot.get("by_core") if hot.get("available") else None,
        "efficiency": {
            "matmul_mfu": eff.get("matmul_mfu"),
            "peak_underestimated": eff.get("peak_underestimated"),
            "roofline_ridge_ai": _round(eff.get("roofline_ridge_ai"), 1),
            "kernels_with_flops": eff.get("kernels_with_flops"),
            "top_optimization": [
                {"name": r["name"][:40], "bound": r["bound"], "wasted_us": r["wasted_us"],
                 "dur_us": r["dur_us"], "mfu": r.get("mfu"), "mbu": r.get("mbu")}
                for r in eff.get("top_optimization", [])[:6]
            ],
        } if eff.get("available") else None,
        "communication": {
            "count": comm.get("count"),
            "total_elapse_ms": comm.get("total_elapse_ms"),
            "overall_wait_pct": comm.get("overall_wait_pct"),
            "by_type": [
                {"type": t["type"], "count": t["count"], "elapse_ms": t["elapse_ms"],
                 "wait_pct": t["wait_pct"]}
                for t in comm.get("by_type", [])
            ],
        } if comm.get("available") else None,
        "hidden_overhead": {
            "device_total_us": ho.get("device_total_us"),
            "host_total_us": ho.get("host_total_us"),
            "buckets": [
                {"key": b["key"], "domain": b["domain"], "us": b["us"], "label": b["label"]}
                for b in ho.get("buckets", [])
            ],
        } if ho.get("available") else None,
        "attribution": {
            "modules": attr.get("modules"),
            "moe_focus": attr.get("moe_focus"),
            "comm_breakdown": attr.get("comm_breakdown"),
        } if attr.get("available") else None,
        "theoretical": {
            "current_step_us": theo.get("current_step_us"),
            "whatif": theo.get("whatif"),
            "compute_bound": theo.get("compute_bound"),
        } if theo.get("available") else None,
        "rule_cards": [
            {"id": c["id"], "severity": c["severity"], "category": c["category"],
             "title": c["title"], "root_cause": c["root_cause"],
             "suggestion": c["suggestion"], "expected_gain": c["expected_gain"],
             "confidence": c["confidence"]}
            for c in cards
        ],
    }
    return summary


def build_messages(summary: Dict[str, Any]) -> tuple[str, str]:
    """Return (system, user) prompt strings for the provider."""
    import json
    payload = json.dumps(summary, ensure_ascii=False, indent=1)
    user = (
        "以下是本次单卡单 step 训练 profiling 的结构化指标摘要与规则引擎诊断（JSON）。"
        "请据此生成中文诊断报告。\n\n```json\n" + payload + "\n```"
    )
    return SYSTEM_PROMPT, user
