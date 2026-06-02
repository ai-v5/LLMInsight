"""Derive model architecture + training/capture config from the profiling data
ITSELF — never from an external launch script.

Rationale (H4 Zero-Config): a launch script does not necessarily correspond
1:1 with a given profiler output, so trusting it can mis-report the run (we
verified two real cases where the script disagreed with the data: blocking and
recompute). Everything here is read back from kernel shapes, op counts and the
HCCL/communication structure, so pointing at an `ASCEND_PROFILER_OUTPUT` dir is
enough.

Three public entry points:
  * ``derive_model(prof)``   -> model architecture facts (each {value, evidence,
                               confidence}). Shapes are unambiguous → mostly
                               high confidence.
  * ``derive_capture(prof)`` -> training/capture state (blocking, recompute,
                               host-sync stall, single-card, swap).
  * ``derive_config(prof)``  -> assembles a back-compat dict shaped like the old
                               ``read_capture_config`` (``env`` / ``flags``) so
                               existing consumers keep working, PLUS rich
                               ``model`` / ``capture`` / ``guesses`` blocks.

Fields that single-card / single-step profiling genuinely cannot pin down
(EP world size, total experts, total layers, global batch) are returned as
``guesses`` with a "未知(猜X)" label and a best-effort numeric guess — never as a
fabricated definite value.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

import pandas as pd

from .profile import ProfileData
from .shapes import parse_dtypes, parse_shapes


# --------------------------------------------------------------------------- #
# small helpers
# --------------------------------------------------------------------------- #
def _fact(value: Any, evidence: str, confidence: str = "high") -> Dict[str, Any]:
    return {"value": value, "evidence": evidence, "confidence": confidence}


_UNKNOWN = _fact(None, "数据中无可靠信号", "unknown")


def _name_series(kd: pd.DataFrame) -> pd.Series:
    col = "Name" if "Name" in kd.columns else kd.columns[1]
    return kd[col].astype(str)


def _rows_named(kd: pd.DataFrame, sub: str, exclude: Optional[str] = None) -> pd.DataFrame:
    if kd is None or kd.empty:
        return kd
    nm = _name_series(kd)
    mask = nm.str.contains(sub, case=False, na=False)
    if exclude:
        mask &= ~nm.str.contains(exclude, case=False, na=False)
    return kd[mask]


def _count_named(kd: pd.DataFrame, sub: str, exclude: Optional[str] = None) -> int:
    if kd is None or kd.empty:
        return 0
    nm = _name_series(kd)
    mask = nm.str.contains(sub, case=False, na=False)
    if exclude:
        mask &= ~nm.str.contains(exclude, case=False, na=False)
    return int(mask.sum())


def _shape_col(kd: pd.DataFrame, want: str) -> Optional[str]:
    for c in kd.columns:
        cl = c.lower()
        if want == "in" and ("input shape" in cl or "input_shapes" in cl):
            return c
        if want == "out" and ("output shape" in cl or "output_shapes" in cl):
            return c
    return None


def _first_shapes(rows: pd.DataFrame, col: Optional[str]) -> List[List[int]]:
    """First row in ``rows`` whose ``col`` parses to a non-empty shape list."""
    if rows is None or rows.empty or not col or col not in rows.columns:
        return []
    for v in rows[col]:
        ops = parse_shapes(v)
        if ops:
            return ops
    return []


def _api_sum(api: pd.DataFrame, contains: str = None, equals: str = None) -> tuple:
    """(count, time_us) over api_statistic rows matching a name."""
    if api is None or api.empty or "API Name" not in api.columns:
        return 0, 0.0
    nm = api["API Name"].astype(str)
    if equals is not None:
        mask = nm.str.fullmatch(equals, na=False)
    else:
        mask = nm.str.contains(contains, case=False, na=False)
    cnt = pd.to_numeric(api.loc[mask, "Count"], errors="coerce").fillna(0).sum() \
        if "Count" in api.columns else 0
    tcol = next((c for c in api.columns if c.lower().startswith("time")), None)
    t = pd.to_numeric(api.loc[mask, tcol], errors="coerce").fillna(0).sum() if tcol else 0.0
    return int(cnt), float(t)


# --------------------------------------------------------------------------- #
# MODEL architecture (from kernel shapes)
# --------------------------------------------------------------------------- #
def derive_model(prof: ProfileData) -> Dict[str, Dict[str, Any]]:
    """Reconstruct DeepSeek-V3 / MLA + MoE architecture from kernel shapes."""
    kd = prof.kernel_details
    facts: Dict[str, Dict[str, Any]] = {}
    if kd is None or kd.empty:
        return facts
    sin = _shape_col(kd, "in")
    sout = _shape_col(kd, "out")

    # --- hidden_size + MLA low-rank dims, from every RmsNorm gamma -----------
    # RmsNorm input is "[S,1,H];[H]"; the trailing 1-D operand is gamma whose
    # length == the normalized width. The widths seen are exactly
    # {hidden, q_lora_rank, kv_lora_rank}.
    gammas: List[int] = []
    seq_from_rms: Optional[int] = None
    rms = _rows_named(kd, "RmsNorm", exclude="Grad")
    if rms is not None and not rms.empty and sin:
        for cell in rms[sin]:
            ops = parse_shapes(cell)
            if len(ops) >= 2 and len(ops[-1]) == 1:
                gammas.append(ops[-1][0])
        gammas = sorted(set(gammas), reverse=True)
    hidden = gammas[0] if gammas else None
    if hidden:
        facts["hidden_size"] = _fact(
            hidden, f"RmsNorm gamma 宽度集合 {gammas} 取最大", "high")
        # seq from the hidden-width RmsNorm's activation operand [S,1,H]
        for cell in rms[sin]:
            ops = parse_shapes(cell)
            if ops and ops[-1] == [hidden] and len(ops[0]) >= 1:
                seq_from_rms = ops[0][0]
                break
        lora = [g for g in gammas if g != hidden]
        if len(lora) >= 2:
            facts["q_lora_rank"] = _fact(
                lora[0], f"RmsNorm 非 hidden 的低秩宽度，取较大 {lora[0]}", "high")
            facts["kv_lora_rank"] = _fact(
                lora[1], f"RmsNorm 非 hidden 的低秩宽度，取较小 {lora[1]}", "high")

    # --- seq_length ----------------------------------------------------------
    seq = seq_from_rms
    if seq:
        facts["seq_length"] = _fact(seq, "hidden-width RmsNorm 激活首维 [S,1,H]", "high")

    # --- attention heads + per-head dims, from FlashAttentionScore -----------
    fa = _rows_named(kd, "FlashAttentionScore", exclude="Grad")
    fa_in = _first_shapes(fa, sin)
    fa_out = _first_shapes(fa, sout)
    heads = None
    if fa_out and len(fa_out[0]) >= 2:
        heads = fa_out[0][1]                       # [1, Nh, S, 8]
        facts["num_attention_heads"] = _fact(
            heads, f"FlashAttentionScore 输出 {fa_out[0]} 第2维", "high")
    if heads and fa_in and len(fa_in) >= 3 and fa_in[0] and fa_in[2]:
        qk = fa_in[0][-1]                          # q: [S,1, Nh*qk_head_dim]
        vv = fa_in[2][-1]                          # v: [S,1, Nh*v_head_dim]
        if qk % heads == 0:
            facts["qk_head_dim"] = _fact(
                qk // heads, f"FA q 末维 {qk} / heads {heads}", "high")
        if vv % heads == 0:
            v_head = vv // heads
            facts["v_head_dim"] = _fact(
                v_head, f"FA v 末维 {vv} / heads {heads}", "high")
            # DeepSeek convention: qk_nope == v_head_dim; rope = qk_head_dim - nope
            qkh = facts.get("qk_head_dim", {}).get("value")
            if qkh and qkh > v_head:
                facts["qk_rope_head_dim"] = _fact(
                    qkh - v_head, f"qk_head_dim {qkh} - v_head_dim {v_head}（约定 nope==v）", "medium")
                facts["qk_nope_head_dim"] = _fact(
                    v_head, f"约定 qk_nope == v_head_dim {v_head}", "medium")

    # --- MoE: local experts, expert FFN width, router top-k -----------------
    # GroupedMatmul weight is the 3-D operand [groups, in, out]; activation is
    # the 2-D operand [tokens, in].
    gmm = _rows_named(kd, "GroupedMatmul", exclude="GroupedMatmulAdd")
    groups: List[int] = []
    moe_ffn_cands: List[int] = []
    tokens: Optional[int] = None
    if gmm is not None and not gmm.empty and sin:
        for cell in gmm[sin]:
            ops = parse_shapes(cell)
            w = [o for o in ops if len(o) == 3]
            if w:
                g, a, b = w[0]
                groups.append(g)
                if hidden and b == hidden:           # down-proj [g, moe_ffn, hidden]
                    moe_ffn_cands.append(a)
                elif hidden and a == hidden:          # up-proj   [g, hidden, 2*moe_ffn]
                    moe_ffn_cands.append(b // 2)
            if tokens is None:
                act = [o for o in ops if len(o) == 2 and hidden and o[1] == hidden]
                if act:
                    tokens = act[0][0]
    if groups:
        g = max(set(groups), key=groups.count)
        facts["local_experts_per_rank"] = _fact(
            g, f"GroupedMatmul 权重分组维 = {g}", "high")
    if moe_ffn_cands:
        moe_ffn = max(set(moe_ffn_cands), key=moe_ffn_cands.count)
        facts["moe_ffn_hidden_size"] = _fact(
            moe_ffn, f"GroupedMatmul 专家权重内维 = {moe_ffn}", "high")
    if tokens and seq:
        topk = round(tokens / seq)
        if topk >= 1:
            facts["moe_router_topk"] = _fact(
                topk, f"GroupedMatmul 激活 tokens {tokens} / seq {seq}", "high")

    # --- dense FFN width, as the widest SwiGlu output ------------------------
    # The dense layer's MLP is far wider than a single MoE expert (here
    # 18432 vs 2048), and the [S,1,*]-vs-[S,*] operand rank is not reliable
    # across captures, so pick by magnitude: the widest forward SwiGlu output.
    sg = _rows_named(kd, "SwiGlu", exclude="Grad")
    if sg is not None and not sg.empty and sout:
        widths: List[int] = []
        for ocell in sg[sout]:
            oops = parse_shapes(ocell)
            if oops and oops[0]:
                widths.append(oops[0][-1])
        if widths:
            facts["ffn_hidden_size"] = _fact(
                max(widths),
                f"SwiGlu 输出宽度集合 {sorted(set(widths))} 取最大（稠密层 MLP）", "high")

    # --- compute dtype, from MatMul inputs ----------------------------------
    mm = _rows_named(kd, "MatMulV3")
    if mm is not None and not mm.empty:
        dcol = next((c for c in kd.columns if "input data type" in c.lower()), None)
        if dcol:
            for v in mm[dcol]:
                dts = parse_dtypes(v)
                if dts:
                    norm = dts[0].upper().replace("DT_", "")
                    facts["dtype"] = _fact(norm, f"MatMulV3 输入 dtype {dts[0]}", "high")
                    break

    return facts


# --------------------------------------------------------------------------- #
# CAPTURE / training state (from op counts, api timings, comm structure)
# --------------------------------------------------------------------------- #
def derive_capture(prof: ProfileData) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    kd = prof.kernel_details
    api = prof.api_statistic

    # --- blocking (ASCEND_LAUNCH_BLOCKING) ----------------------------------
    # A blocking capture serializes every launch behind a device sync, so the
    # Synchronize/launch COUNT ratio approaches 1 and host self-time tracks
    # device time. Real runs sync only at a few barriers (ratio « 1).
    launch_n, _ = _api_sum(api, equals="launch")
    if launch_n == 0:
        launch_n, _ = _api_sum(api, contains="LaunchKernel")
    sync_n, _ = _api_sum(api, contains="Synchronize")
    ratio = (sync_n / launch_n) if launch_n else 0.0
    blocking = ratio > 0.5
    out["blocking"] = _fact(
        blocking,
        f"Synchronize/launch 次数比 = {ratio:.3f}（{sync_n}/{launch_n}）；"
        f"{'接近 1 → 疑似 blocking' if blocking else '远小于 1 → 异步下发，未检出 blocking'}",
        "high" if launch_n else "unknown",
    )

    # --- host-sync stall (the real host bottleneck, blocking or not) --------
    ss_n, ss_t = _api_sum(api, equals="aclrtSynchronizeStream")
    _, total_api_t = _api_sum(api, contains="")  # everything
    share = (ss_t / total_api_t) if total_api_t else 0.0
    # likely root cause: the heaviest aclnn* call (dynamic-shape D2H sync)
    root = None
    if api is not None and not api.empty and "API Name" in api.columns:
        tcol = next((c for c in api.columns if c.lower().startswith("time")), None)
        aclnn = api[api["API Name"].astype(str).str.startswith("aclnn")]
        if tcol and not aclnn.empty:
            top = aclnn.loc[pd.to_numeric(aclnn[tcol], errors="coerce").idxmax()]
            root = str(top["API Name"])
    out["host_sync_stall"] = _fact(
        share >= 0.2,
        f"aclrtSynchronizeStream 累计 {ss_t/1e6:.2f}s（{ss_n} 次），占 host API 时间 {share*100:.0f}%"
        + (f"；最重 host 调用为 {root}（动态 shape D2H 同步）" if root else ""),
        "high" if total_api_t else "unknown",
    )

    # --- recompute granularity, from FlashAttention fwd/grad count ----------
    # Forward attention runs once per layer; full recompute reruns it in the
    # backward, so fwd≈2×grad. recompute off → fwd≈grad.
    fa_grad = _count_named(kd, "FlashAttentionScoreGrad")
    fa_all = _count_named(kd, "FlashAttentionScore")
    fa_fwd = fa_all - fa_grad
    if fa_grad > 0:
        r = fa_fwd / fa_grad
        if r >= 1.5:
            val, conf = "full", "high"
        elif r <= 1.2:
            val, conf = "off", "high"
        else:
            val, conf = "selective", "medium"
        out["recompute"] = _fact(
            val, f"FlashAttention 前向/反向次数比 = {r:.2f}（fwd {fa_fwd} / grad {fa_grad}）", conf)
    else:
        out["recompute"] = _fact(None, "无 FlashAttentionScoreGrad，无法判定", "unknown")

    # --- single card, from communication_matrix ----------------------------
    cm = prof.communication_matrix or {}
    has_peer = any(
        isinstance(st, dict) and st.get("collective")
        for st in cm.values()
    ) if isinstance(cm, dict) else False
    out["single_card"] = _fact(
        not has_peer,
        "communication_matrix 无 collective 跨卡条目 → 单卡" if not has_peer
        else "communication_matrix 含跨卡 collective → 多卡",
        "high",
    )

    # --- swap-optimizer (low confidence single-card signal) -----------------
    mc_n, mc_t = _api_sum(api, contains="Memcpy")
    swap = mc_t > 200_000  # >0.2s of memcpy would hint at optimizer swap traffic
    out["swap_optimizer"] = _fact(
        swap if swap else None,
        f"Memcpy 类 api 累计 {mc_t/1000:.0f}ms（{mc_n} 次）"
        + ("，量级偏低，未检出优化器换入换出" if not swap else "，量级偏大，疑似 swap"),
        "low",
    )

    return out


# --------------------------------------------------------------------------- #
# Underivable from single-card / single-step → 未知(猜X)
# --------------------------------------------------------------------------- #
def derive_guesses(prof: ProfileData,
                   model: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """Fields the data cannot pin down; show 未知 + a parenthetical best guess."""
    local = (model.get("local_experts_per_rank", {}) or {}).get("value")

    # EP world size: alltoallv presence proves expert parallelism exists, but a
    # single rank cannot reveal the world size. DeepSeek-V3 here is typically EP64.
    has_alltoall = any(
        c.get("type", "").lower().startswith("alltoall")
        for c in (prof.communication or [])
    )
    ep_guess = 64 if has_alltoall else None
    guesses: Dict[str, Dict[str, Any]] = {
        "ep_world_size": {
            "label": f"未知(猜{ep_guess})" if ep_guess else "未知",
            "guess": ep_guess,
            "basis": "存在 alltoallv（专家并行确实开启），但单卡无法得知 world size；DeepSeek-V3 此处常见 EP64",
        },
        "num_experts": {
            "label": f"未知(猜{local * ep_guess})" if (local and ep_guess) else "未知",
            "guess": (local * ep_guess) if (local and ep_guess) else None,
            "basis": f"= local_experts_per_rank({local}) × EP({ep_guess})（依赖 EP 猜测）"
                     if (local and ep_guess) else "依赖 EP world size（未知）",
        },
        "num_layers": {
            "label": "未知(猜10)",
            "guess": 10,
            "basis": "FlashAttention 次数被 microbatch / PP 切分混淆，无法反推总层数；样例目录名暗示 1 dense + 9 moe = 10",
        },
        "global_batch_size": {
            "label": "未知",
            "guess": None,
            "basis": "单卡单 step 无 DP × 梯度累积 信号，无法推断全局 batch",
        },
    }
    return guesses


# --------------------------------------------------------------------------- #
# Assemble a back-compat config dict (drop-in for read_capture_config)
# --------------------------------------------------------------------------- #
def derive_config(prof: ProfileData) -> Dict[str, Any]:
    """Return the old ``{found, env, flags, path}`` shape (so every existing
    consumer keeps working) augmented with rich ``model`` / ``capture`` /
    ``guesses`` blocks.

    Deliberately:
      * ``env`` stays EMPTY — we never assert ASCEND_LAUNCH_BLOCKING from data
        (not detectable), so the blocking card will not fire.
      * ``flags`` carries only what is DERIVED with confidence. EP world size is
        left out (underivable) and surfaced under ``guesses`` instead.
    """
    model = derive_model(prof)
    capture = derive_capture(prof)
    guesses = derive_guesses(prof, model)

    flags: Dict[str, Any] = {}
    if (capture.get("recompute", {}) or {}).get("value") == "full":
        flags["recompute-granularity"] = "full"
    flags["swap-optimizer"] = bool((capture.get("swap_optimizer", {}) or {}).get("value"))
    # Derived training dims that map onto the old flag names (display only):
    if (model.get("seq_length", {}) or {}).get("value"):
        flags["seq-length"] = str(model["seq_length"]["value"])
    if (model.get("moe_router_topk", {}) or {}).get("value"):
        flags["moe-router-topk"] = str(model["moe_router_topk"]["value"])

    return {
        "found": True,
        "source": "profiling",
        "path": None,                    # no launch script; keep absolute paths out of the payload
        "env": {},                       # nothing asserted from env
        "flags": flags,
        "model": {k: v["value"] for k, v in model.items()},
        "model_facts": model,            # value + evidence + confidence
        "capture": capture,
        "guesses": guesses,
    }


# --------------------------------------------------------------------------- #
if __name__ == "__main__":  # developer probe: python -m llminsight.parser.derive <dir>
    import json
    import sys

    from .profile import load_profile

    prof = load_profile(sys.argv[1])
    cfg = derive_config(prof)
    print("=== MODEL ===")
    for k, f in cfg["model_facts"].items():
        print(f"  {k:24s} = {str(f['value']):>8}   [{f['confidence']}] {f['evidence']}")
    print("=== CAPTURE ===")
    for k, f in cfg["capture"].items():
        print(f"  {k:18s} = {str(f['value']):>6}   [{f['confidence']}] {f['evidence']}")
    print("=== GUESSES (underivable) ===")
    for k, g in cfg["guesses"].items():
        print(f"  {k:18s} = {g['label']:<14} {g['basis']}")
    print("=== back-compat env/flags ===")
    print("  env  :", json.dumps(cfg["env"], ensure_ascii=False))
    print("  flags:", json.dumps(cfg["flags"], ensure_ascii=False))
