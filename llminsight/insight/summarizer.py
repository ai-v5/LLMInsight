"""Build a compact, privacy-safe summary of the metrics + rule cards, and the
prompt messages for the LLM.

PRIVACY CONTRACT: only aggregated numbers and the rule cards leave this module.
Raw trace events, tensors and the full environment are excluded. Operator
instances use local aliases; recognizable paths, URLs and identifiers are
redacted throughout the summary. This is limited redaction, not a guarantee of
anonymity: model traits and performance metrics can still be confidential.
Users can inspect the exact payload before explicitly requesting inference.
"""
from __future__ import annotations

import re
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
    "(3) 必要的采集可信度提醒。要求：只用给定数据，不臆造数字；建议要落地到具体开关/参数。"
    "模型特征与训练/采集配置均由 profiling 反推（model.derived 为反推值，model.unknown_fields 为"
    "当前数据不可得的未知项；capture.state 给出每条采集判定的依据与置信度）。"
    "仅当 capture.state.blocking.value 为真时，才把 host 指标标注为 ASCEND_LAUNCH_BLOCKING 采集干扰；"
    "否则只能把动态 shape D2H 作为待 trace 关联验证的候选原因，不能写成已证实因果。"
)


def _round(x, n=2):
    return round(x, n) if isinstance(x, (int, float)) else x


_IDENTIFIERS = re.compile(
    r"https?://[^\s<>\"']+|file://[^\s<>\"']+"
    r"|[A-Za-z]:[\\/][^\s<>\"']+|\\\\[^\s<>\"']+"
    r"|(?<![\w/])/(?:[^\s/<>\"']+/)*[^\s/<>\"']+"
    r"|\b[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}\b|\b[a-zA-Z]\d{8}\b"
    r"|\b(?:\d{1,3}\.){3}\d{1,3}\b"
)


def _redact(value: Any) -> Any:
    """Redact recognizable identifiers without altering numeric metrics."""
    if isinstance(value, str):
        # Drop the whole field so paths with spaces cannot leave a tail behind.
        return "[redacted]" if _IDENTIFIERS.search(value) else value
    if isinstance(value, dict):
        result = {}
        for index, (key, item) in enumerate(value.items(), 1):
            safe_key = f"redacted_field_{index}" if isinstance(key, str) and _IDENTIFIERS.search(key) else key
            # Preserve every value even if a synthetic input already uses the
            # replacement key. This view must not silently lose metric rows.
            while safe_key in result or (safe_key != key and safe_key in value):
                safe_key = "_" + safe_key
            result[safe_key] = _redact(item)
        return result
    if isinstance(value, (list, tuple)):
        return [_redact(item) for item in value]
    return value


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

    # Model architecture + training/capture state are now reconstructed FROM the
    # profiling (parser.derive), never a launch script. They remain potentially
    # sensitive even after identifiers are redacted.
    cfg = capture if capture.get("source") == "profiling" else (meta.get("config") or {})
    derived_model = cfg.get("model") or {}
    guesses = cfg.get("guesses") or {}
    cap_state = {
        k: {"value": f.get("value"), "confidence": f.get("confidence"),
            "evidence": f.get("evidence")}
        for k, f in (cfg.get("capture") or {}).items()
    }
    # The effective recompute state is decided once in theoretical (identified
    # replay outranks a caller-count "off"); shipping the raw capture value next to
    # it made the LLM prompt carry both "off" and "selective" for the same run.
    _rco = (m.get("theoretical") or {}).get("recompute") or {}
    if _rco.get("granularity") in ("full", "selective") and "recompute" in cap_state:
        cap_state["recompute"] = dict(cap_state["recompute"],
                                      value=_rco["granularity"],
                                      evidence=("有效粒度：profiling 反推的调用数 + "
                                                "可识别重放算子（theoretical.recompute）"))
    model_block = {
        "arch": derived_model.get("architecture") or "架构未识别（仅展示 profiling 可验证特征）",
        "derived": derived_model,
        "unknown_fields": {k: g.get("label") for k, g in guesses.items()},
        # Backward-compatible alias for existing consumers; values now remain
        # unknown rather than carrying model-family numeric guesses.
        "unknown_guesses": {k: g.get("label") for k, g in guesses.items()},
    } if (derived_model or guesses) else None

    summary: Dict[str, Any] = {
        "model": model_block,
        "chip": eff.get("chip"),
        "capture": {"env": cap_env, "flags": cap_flags, "state": cap_state},
        "overview": {
            "steps": ov.get("steps"),
            "step_count": ov.get("step_count"),
            "step_time_s": ov.get("ratios", {}).get("step_time_s"),
            "window_time_s": ov.get("ratios", {}).get("window_time_s"),
            "ratios": ov.get("ratios"),
            "composition": ov.get("composition"),
        } if ov.get("available") else None,
        "hotspots_top": [
            # display label: a Level-1 capture names fused kernels with a 1000+ char
            # C++ symbol, which would dominate the LLM prompt for no extra signal.
            {"type": o.get("display") or o["type"], "core": o["core"], "ratio": o["ratio"],
             "total_us": o["total_us"], "count": o["count"]}
            for o in hot.get("top", [])[:8]
        ] if hot.get("available") else None,
        "by_core": hot.get("by_core") if hot.get("available") else None,
        "efficiency": {
            "matmul_mfu": eff.get("matmul_mfu"),
            "peak_underestimated": eff.get("peak_underestimated"),
            "peak_inconsistent": eff.get("peak_inconsistent"),
            "efficiency_reliable": eff.get("efficiency_reliable"),
            "flop_model_complete": eff.get("flop_model_complete"),
            "flop_coverage_pct": eff.get("flop_coverage_pct"),
            "mfu_estimate": eff.get("mfu_estimate"),
            "workload_accounting": eff.get("workload_accounting"),
            "fused_moe": eff.get("fused_moe"),
            "model_mfu_compute_estimated": eff.get("model_mfu_compute_estimated"),
            "unmodeled_flop_types": eff.get("unmodeled_flop_types"),
            "roofline_ridge_ai": _round(eff.get("roofline_ridge_ai"), 1),
            "kernels_with_flops": eff.get("kernels_with_flops"),
            "top_optimization": [
                {"name": f"operator_{index}", "bound": r["bound"], "wasted_us": r["wasted_us"],
                 "dur_us": r["dur_us"], "mfu": r.get("mfu"), "mbu": r.get("mbu"),
                 "mbu_raw": r.get("mbu_raw"),
                 "mbu_over_physical": r.get("mbu_over_physical")}
                for index, r in enumerate(eff.get("top_optimization", [])[:6], 1)
            ],
        } if eff.get("available") else None,
        "communication": {
            "count": comm.get("count"),
            "total_elapse_ms": comm.get("total_elapse_ms"),
            "total_wait_ms": comm.get("total_wait_ms"),
            "overall_wait_pct": comm.get("overall_wait_pct"),
            "total_transit_mb": comm.get("total_transit_mb"),
            "total_transit_ms": comm.get("total_transit_ms"),
            "overall_bandwidth_gbps_without_wait": comm.get("overall_bandwidth_gbps"),
            "overall_bandwidth_gbps_with_wait": comm.get("overall_bandwidth_with_wait_gbps"),
            "device_task_breakdown": comm.get("breakdown"),
            "by_type": [
                {"type": t["type"], "count": t["count"], "elapse_ms": t["elapse_ms"],
                 "wait_pct": t["wait_pct"],
                 "bandwidth_gbps_without_wait": t.get("bandwidth_gbps"),
                 "bandwidth_gbps_with_wait": t.get("bandwidth_with_wait_gbps")}
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
            "whatif_combined": theo.get("whatif_combined"),
            "compute_bound": theo.get("compute_bound"),
            "step_mfu": theo.get("step_mfu"),
            "step_mfu_lo": theo.get("step_mfu_lo"),
            "step_mfu_hi": theo.get("step_mfu_hi"),
            "step_hfu": theo.get("step_hfu"),
            "mfu_estimated": theo.get("mfu_estimated"),
            "mfu_opinion": theo.get("mfu_opinion"),
            "recompute": theo.get("recompute"),
        } if theo.get("available") else None,
        "rule_cards": [
            {"id": c["id"], "severity": c["severity"], "category": c["category"],
             "title": c["title"], "root_cause": c["root_cause"],
             "suggestion": c["suggestion"], "expected_gain": c["expected_gain"],
             "confidence": c["confidence"]}
            for c in cards
        ],
    }
    return _redact(summary)


def build_messages(summary: Dict[str, Any]) -> tuple[str, str]:
    """Return (system, user) prompt strings for the provider."""
    import json
    payload = json.dumps(summary, ensure_ascii=False, indent=1)
    user = (
        "以下是本次训练 profiling 的结构化指标摘要与规则引擎诊断（JSON）。"
        "请据此生成中文诊断报告。\n\n```json\n" + payload + "\n```"
    )
    return SYSTEM_PROMPT, user
