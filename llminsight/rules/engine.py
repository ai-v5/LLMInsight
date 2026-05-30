"""Rule engine. Each rule reads the metrics dict and (optionally) the training
script, then emits an insight card. Cards are the fact base the LLM layer later
turns into natural language — and they render fine on their own when the LLM
call is disabled (graceful degradation).
"""
from __future__ import annotations

import os
import re
from typing import Any, Dict, List, Optional

from ..config import SETTINGS

SEV_ORDER = {"high": 3, "medium": 2, "low": 1, "info": 0}


def _card(cid, severity, category, title, root_cause, suggestion,
          expected_gain, confidence, evidence) -> Dict[str, Any]:
    return {
        "id": cid,
        "severity": severity,
        "category": category,
        "title": title,
        "root_cause": root_cause,
        "suggestion": suggestion,
        "expected_gain": expected_gain,
        "confidence": confidence,
        "evidence": evidence,
    }


def read_capture_config(script_path: Optional[str] = None) -> Dict[str, Any]:
    """Parse `export VAR=...` and key training flags from the launch script."""
    path = script_path or SETTINGS.script_path
    out: Dict[str, Any] = {"found": False, "env": {}, "flags": {}, "path": path}
    if not path or not os.path.exists(path):
        return out
    out["found"] = True
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as fh:
            text = fh.read()
    except OSError:
        return out
    for m in re.finditer(r"export\s+([A-Z0-9_]+)=([^\s\"]+)", text):
        out["env"][m.group(1)] = m.group(2)

    # Collect every plain shell assignment (TP=1, EP=64, ...) so we can resolve
    # ${VAR} references that the training flags use instead of literal values.
    shell_vars: Dict[str, str] = dict(out["env"])
    for m in re.finditer(r"^\s*([A-Za-z_][A-Za-z0-9_]*)=([^\s\"'$]+)", text, re.M):
        shell_vars.setdefault(m.group(1), m.group(2))

    def resolve(val: str) -> str:
        return re.sub(r"\$\{?([A-Za-z_][A-Za-z0-9_]*)\}?",
                      lambda mm: shell_vars.get(mm.group(1), mm.group(0)), val)

    for flag in ("recompute-granularity", "recompute-num-layers",
                 "expert-model-parallel-size", "tensor-model-parallel-size",
                 "pipeline-model-parallel-size", "context-parallel-size",
                 "num-experts", "moe-router-topk", "global-batch-size",
                 "num-layers", "seq-length"):
        m = re.search(rf"--{flag}\s+(\S+)", text)
        if m:
            out["flags"][flag] = resolve(m.group(1))
    for sw in ("swap-optimizer", "moe-fb-overlap", "moe-permutation-async-comm",
               "use-flash-attn", "sequence-parallel"):
        out["flags"][sw] = bool(re.search(rf"--{sw}\b", text))
    return out


# --------------------------------------------------------------------------- #
def run_rules(m: Dict[str, Any], capture: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
    if capture is None:
        capture = read_capture_config()
    cards: List[Dict[str, Any]] = []
    ov = m.get("overview", {})
    ratios = ov.get("ratios", {}) if ov.get("available") else {}
    us = ov.get("us", {}) if ov.get("available") else {}
    hot = m.get("hotspots", {})
    eff = m.get("efficiency", {})
    comm = m.get("communication", {})
    ho = m.get("hidden_overhead", {})
    theo = m.get("theoretical", {})
    env = capture.get("env", {})
    blocking = env.get("ASCEND_LAUNCH_BLOCKING") == "1"

    # 1. communication not overlapped --------------------------------------
    if ratios:
        overlap = ratios.get("overlap_rate_pct", 0)
        cno = ratios.get("comm_not_overlapped_pct", 0)
        if cno >= 10 or overlap < 40:
            cards.append(_card(
                "comm_not_overlapped", "high", "通信",
                f"通信掩盖严重不足：未掩盖通信占 step {cno}%，计算-通信重叠率仅 {overlap}%",
                "EP64 下 MoE alltoall 通信量大，且计算与通信重叠窗口不足；moe-fb-overlap / "
                "异步通信未能覆盖大部分通信。",
                "核对 --moe-fb-overlap / --moe-permutation-async-comm 是否生效；增大重叠窗口、"
                "调整通信切分与下发顺序。",
                f"理想全掩盖可减少约 {cno}% step 时间（What-if 上界）。",
                0.9,
                {"overlap_rate_pct": overlap, "comm_not_overlapped_pct": cno},
            ))

    # 2. AICPU communication dispatch --------------------------------------
    aicpu = next((o for o in hot.get("ops", []) if o["type"] == "HcclLaunchAicpuKernel"), None)
    if aicpu and aicpu["ratio"] >= 10:
        cards.append(_card(
            "aicpu_dispatch", "high", "下发",
            f"AICPU 通信下发开销过高：HcclLaunchAicpuKernel 占 device {aicpu['ratio']}%（单次 max {aicpu['max_us']/1000:.0f}ms）",
            "EP64 alltoall 集合通信启动密集，通信由 AI_CPU 下发，单次启动开销巨大。",
            "增大 HCCL_BUFFSIZE、调整 HCCL 通信算法 / 切分比；减少 dispatch 次数（合并通信、"
            "增大 token 批次）；评估 EP 规模。",
            "降低下发占比，directly 压缩 device 关键路径（当前占比 ~28%）。",
            0.85,
            {"ratio_pct": aicpu["ratio"], "count": aicpu["count"], "max_us": aicpu["max_us"]},
        ))

    # 3 & 11. capture checkup / host sync blocking -------------------------
    sync = next((b for b in ho.get("buckets", []) if b["key"] == "host_sync"), None)
    if blocking:
        cards.append(_card(
            "capture_blocking", "high", "采集体检",
            "采集配置扭曲：检测到 ASCEND_LAUNCH_BLOCKING=1，host 同步类耗时被放大、不可直接采信",
            "ASCEND_LAUNCH_BLOCKING=1 让每次下发同步等待 device 完成，aclrtSynchronize* 被人为放大；"
            "host/device 时间口径失真。",
            "性能复采时关闭 ASCEND_LAUNCH_BLOCKING（设 0 或不设）；本报告中 host 同步/下发数字按"
            "「采集干扰」解读，关注 device 侧与 step 构成。",
            f"关闭 blocking 后，host 同步（约 {(sync['us']/1e6 if sync else 0):.1f}s）大部分可被异步隐藏。",
            1.0,
            {"ASCEND_LAUNCH_BLOCKING": "1",
             "aclrtSynchronize_us": (sync["us"] if sync else None)},
        ))

    # 4. dynamic shape jitter ----------------------------------------------
    dyn = next((b for b in ho.get("buckets", []) if b["key"] == "dynamic_shape"), None)
    if dyn and dyn["us"] and dyn["us"] > 50000:
        mx = re.search(r"max\s+([\d,]+)us", dyn["detail"])
        cards.append(_card(
            "dynamic_shape", "medium", "动态shape",
            f"动态 shape 抖动：MaskedSelect/NonZero 等 host 累计 {dyn['us']/1000:.0f}ms，单次方差极大",
            "MoE 路由 / 掩码产生动态 shape，触发 host 重编译或同步等待，时延抖动大。",
            "固定 expert capacity / 对路由结果 padding，避免动态 shape；或用静态 capacity 的 dispatch。",
            "消除重编译抖动，降低 host 关键路径与尾延迟。",
            0.7,
            {"host_us": dyn["us"]},
        ))

    # 5. low-efficiency / optimization room --------------------------------
    if eff.get("available"):
        top = [r for r in eff.get("top_optimization", []) if r.get("wasted_us", 0) > 0][:5]
        if top:
            names = "、".join(f"{r['name'][:24]}({r['bound']},省~{r['wasted_us']/1000:.1f}ms)" for r in top[:3])
            total_waste = sum(r["wasted_us"] for r in top)
            cards.append(_card(
                "low_efficiency_ops", "medium", "算子效率",
                f"低效算子优化余量：Top 候选 {names}",
                "候选既有 compute-bound 的融合算子（如 FlashAttention 反向，离 cube 峰值仍有余量），"
                "也有访存-bound（MBU 主导、mte2/mte3 高）与初始化类算子，实测均低于 Roofline 理想。",
                "compute-bound 融合算子调 tiling / 提升 cube 利用率；memory-bound 减少读写量；"
                "算子融合 / 内存复用（避免 ZerosLike 重复初始化）/ 换更优库实现。",
                f"Top 候选合计约 {total_waste/1000:.1f}ms 优化空间（vs Roofline 理想）。",
                0.6,
                {"top": [{"name": r["name"][:32], "bound": r["bound"], "wasted_us": r["wasted_us"]} for r in top]},
            ))
        if eff.get("peak_underestimated"):
            chipinfo = eff.get("chip", {})
            assumed = chipinfo.get("peak_bf16_tflops")
            observed = chipinfo.get("observed_peak_tflops")
            cards.append(_card(
                "peak_underestimated", "info", "算子效率",
                f"芯片峰值假设偏低：实测 matmul 峰值 ≈ {observed:.0f} TFLOPS > 假设 {assumed:.0f} TFLOPS",
                "存在 matmul 实测达成算力超过 ChipSpec 假设峰值（真实 kernel 不可能超过硅片峰值），"
                "说明假设的 BF16 峰值偏低；已临时按实测上界校准，MFU 才不会出现 >100% 的非物理值。",
                "在 configs/chips/<芯片>.yaml 按实际 NPU SKU 设置 fp16_tflops / memory_bandwidth，"
                "用准确峰值替代实测校准，效率指标更可信。",
                "校正后 MFU/MBU 可作真实优化排序基准。",
                0.8,
                {"assumed_peak_tflops": assumed, "observed_peak_tflops": observed,
                 "matmul_mfu_assumed": eff.get("matmul_mfu_assumed"), "matmul_mfu": eff.get("matmul_mfu")},
            ))

    # 6. Free / bubble ------------------------------------------------------
    if ratios.get("free_pct", 0) >= 10:
        cards.append(_card(
            "free_bubble", "medium", "空泡",
            f"空闲占比偏高：Free 占 step {ratios['free_pct']}%",
            "device 存在空泡，可能来自同步等待、下发不及时或 host-bound 阶段（被 blocking 放大）。",
            "结合等待/下发分析定位空泡来源；提高下发并行度、减少同步点。",
            f"消除空泡上界可减少约 {ratios['free_pct']}% step（What-if）。",
            0.65,
            {"free_pct": ratios["free_pct"], "free_us": us.get("free")},
        ))

    # 7. recompute ----------------------------------------------------------
    rg = capture.get("flags", {}).get("recompute-granularity")
    if rg == "full":
        cards.append(_card(
            "recompute_full", "low", "显存-时间",
            "全量重计算开启：--recompute-granularity full（反向重跑前向）",
            "为省激活显存，反向阶段重跑前向，增加计算耗时；本次采集未单列重计算耗时。",
            "若显存不紧张，改用选择性重计算 / 减少重计算层，换取吞吐；配合显存采集量化收益。",
            "选择性重计算通常可回收部分前向重算时间（需显存余量）。",
            0.6,
            {"recompute_granularity": rg, "recompute_num_layers": capture.get("flags", {}).get("recompute-num-layers")},
        ))

    # 9. theoretical bound + what-if ---------------------------------------
    if theo.get("available"):
        best = max(theo.get("whatif", []), key=lambda w: w.get("save_pct", 0), default=None)
        if best:
            cards.append(_card(
                "theoretical_whatif", "info", "理论上界",
                f"理论上界与 What-if：最优场景「{best['scenario']}」可省约 {best['save_pct']}% step",
                "由 step 时间构成推导优化上界：通信掩盖、消除空泡为最大两块收益来源。",
                "按 What-if 收益排序优化优先级；先攻通信掩盖（最大单项），再压空泡。",
                f"综合 What-if 上界最高约 {best['save_pct']}%（{best.get('basis','')}）。",
                0.7,
                {"whatif": theo.get("whatif")},
            ))

    # 10. parallelism advisor ----------------------------------------------
    ep = capture.get("flags", {}).get("expert-model-parallel-size")
    if ep and ratios.get("comm_not_overlapped_pct", 0) >= 15:
        cards.append(_card(
            "parallelism_advisor", "medium", "并行策略",
            f"并行策略提示：EP={ep} 下通信（alltoall）占比偏高、未掩盖 {ratios.get('comm_not_overlapped_pct')}%",
            "大 EP 带来密集 alltoall 与下发开销；当前 TP=1，通信未能与计算充分重叠。",
            "评估 EP↓ + TP↑ 的再平衡，或加强通信-计算重叠；权衡专家并行的通信代价 vs 负载均衡。",
            "降低 alltoall 占比与下发频次，缓解通信瓶颈（需结合多卡负载数据确认）。",
            0.55,
            {"ep": ep, "tp": capture.get("flags", {}).get("tensor-model-parallel-size"),
             "comm_not_overlapped_pct": ratios.get("comm_not_overlapped_pct")},
        ))

    # 8. hidden-overhead总账 ------------------------------------------------
    if ho.get("available"):
        cards.append(_card(
            "hidden_overhead_ledger", "medium", "隐性开销",
            "隐性开销总账：下发 / 等待·同步 / 未掩盖通信 / 格式转换·初始化 / 动态 shape / 空泡 合计可观",
            "多项分散开销单看不起眼，合计是吞吐杀手；其中 device 侧可直接计入 step，host 侧反映下发/同步压力。",
            "按总账逐项减负：通信掩盖→AICPU 下发→空泡→格式转换；host 侧关 blocking、降下发次数。",
            "总账用于排优先级，避免只盯单点热点而漏掉合计更大的隐性项。",
            0.7,
            {"device_total_us": ho.get("device_total_us"), "host_total_us": ho.get("host_total_us"),
             "buckets": [{"key": b["key"], "us": b["us"], "domain": b["domain"]} for b in ho.get("buckets", [])]},
        ))

    # 12. memory capture ----------------------------------------------------
    mem = m.get("memory", {})
    if not mem.get("hbm_timeline_available", False):
        cards.append(_card(
            "memory_capture", "info", "显存",
            "缺少显存采集：本次无 memory-level 数据，无法绘制 HBM 峰值/构成",
            "采集未开启 memory level（缺 memory_record.csv / npu_module_mem.csv），仅有 AI Core Freq counter。",
            "复采时开启 profiler memory level 以获得显存峰值与构成；当前给出 recompute/swap 配置驱动的内存-时间权衡。",
            "拿到显存数据后可做 OOM 定位与 recompute/swap 的量化权衡。",
            0.9,
            {"hbm_timeline_available": False, "swap_optimizer": capture.get("flags", {}).get("swap-optimizer")},
        ))

    cards.sort(key=lambda c: (SEV_ORDER.get(c["severity"], 0),
                              c["confidence"]), reverse=True)
    return cards
