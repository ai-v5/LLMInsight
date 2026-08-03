"""Derive model traits + training/capture config from the profiling data
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

Fields that single-rank / single-step profiling genuinely cannot pin down
(EP world size, total experts, total layers, global batch) remain ``未知``.  We do
not inject model-family priors as numeric guesses.
"""
from __future__ import annotations

from collections import Counter
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

# Model-compute families whose FLOP/phase semantics are not implemented yet.
# Their presence is still important: attention-only phase counts must not be
# promoted to a whole-model recompute conclusion on a hybrid KDA model.
_CUSTOM_MODEL_MARKERS = (
    "chunk_kda", "kda_", "gated_delta", "chunk_gla", "causal_conv1d",
)


def _custom_model_types(kd: pd.DataFrame) -> List[str]:
    if kd is None or kd.empty:
        return []
    values = (kd["Type"].astype(str) if "Type" in kd.columns else _name_series(kd))
    return sorted({
        value for value in values
        if any(marker in value.lower() for marker in _CUSTOM_MODEL_MARKERS)
    })


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


def _count_types(kd: pd.DataFrame, types: tuple[str, ...], exclude: Optional[str] = None) -> int:
    """Count profiler op types, falling back to names on older exports."""
    if kd is None or kd.empty:
        return 0
    if "Type" in kd.columns:
        exact = int(kd["Type"].astype(str).isin(types).sum())
        if exact:
            return exact
    names = _name_series(kd)
    mask = pd.Series(False, index=names.index)
    for typ in types:
        mask |= names.str.contains(typ, case=False, regex=False, na=False)
    if exclude:
        mask &= ~names.str.contains(exclude, case=False, regex=False, na=False)
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
    """Reconstruct only operator/shape traits supported by the current profile."""
    kd = prof.kernel_details
    facts: Dict[str, Dict[str, Any]] = {}
    if kd is None or kd.empty:
        return facts
    sin = _shape_col(kd, "in")
    sout = _shape_col(kd, "out")

    # Architecture is a generic operator-family description, not a model-name
    # guess.  SparseFlashAttention is used by more than one model family; neither
    # its presence nor MoE collectives prove DeepSeek/GLM or a particular EP size.
    type_values = set(kd["Type"].astype(str)) if "Type" in kd.columns else set(_name_series(kd))
    has_moe = any(t.startswith("GroupedMatmul") for t in type_values)
    has_sparse_attn = any(t.startswith("SparseFlashAttention") for t in type_values)
    has_fused_attn = has_sparse_attn or any("FlashAttention" in t for t in type_values)
    # causal_conv1d is also treated as an unsupported model-compute family for
    # FLOP coverage, but its presence alone does not prove a KDA architecture.
    has_kda = any(
        any(
            marker in t.lower()
            for marker in ("chunk_kda", "kda_", "gated_delta", "chunk_gla")
        )
        for t in type_values
    )
    traits: List[str] = []
    if has_moe:
        traits.append("MoE")
    if has_sparse_attn:
        traits.append("Sparse Attention")
    elif has_fused_attn:
        traits.append("Fused Attention")
    if has_kda:
        traits.append("KDA")
    if traits:
        facts["architecture"] = _fact(
            " + ".join(traits),
            "由当前 profiling 中出现的算子族归纳，不推断具体模型名称",
            "high",
        )

    # --- hidden_size + MLA low-rank dims, from every RmsNorm gamma -----------
    # RmsNorm input is "[S,1,H];[H]"; the trailing 1-D operand is gamma whose
    # length == the normalized width. The widths seen are exactly
    # {hidden, q_lora_rank, kv_lora_rank}.
    gammas: List[int] = []
    seq_candidates: List[int] = []
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
        # Seq can be [S,1,H], [1,S,H], or [S,H].  The old first-axis rule
        # silently returned batch=1 for [1,S,H].  Use the dominant non-unit
        # dimension before H across every hidden-width RmsNorm instead.
        for cell in rms[sin]:
            ops = parse_shapes(cell)
            if ops and ops[-1] == [hidden] and len(ops[0]) >= 1:
                prefix = [int(x) for x in ops[0][:-1] if int(x) > 1]
                if prefix:
                    seq_candidates.append(max(prefix))
        lora = [g for g in gammas if g != hidden]
        if len(lora) >= 2:
            facts["q_lora_rank"] = _fact(
                lora[0], f"RmsNorm 非 hidden 的低秩宽度，取较大 {lora[0]}", "high")
            facts["kv_lora_rank"] = _fact(
                lora[1], f"RmsNorm 非 hidden 的低秩宽度，取较小 {lora[1]}", "high")

    # --- seq_length ----------------------------------------------------------
    seq: Optional[int] = None
    if seq_candidates:
        counts = Counter(seq_candidates)
        seq, count = counts.most_common(1)[0]
        share = count / len(seq_candidates)
        if share >= 0.8:
            facts["seq_length"] = _fact(
                seq,
                f"hidden-width RmsNorm 激活非 hidden 维众数 {seq}（{count}/{len(seq_candidates)}）",
                "high",
            )
        else:
            facts["seq_length"] = _fact(
                None,
                f"hidden-width RmsNorm 序列维候选冲突：{dict(counts.most_common(5))}",
                "unknown",
            )
            seq = None

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
    ss_n, ss_t = _api_sum(api, contains="Synchronize")
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
        f"Synchronize APIs 累计 {ss_t/1e6:.2f}s（{ss_n} 次），占 host API 时间 {share*100:.0f}%"
        + (f"；最重 host 调用为 {root}（动态 shape D2H 同步）" if root else ""),
        "high" if total_api_t else "unknown",
    )

    # --- recompute granularity, from fused-attention fwd/grad count ----------
    # Forward attention runs once per layer; full recompute reruns it in the
    # backward, so fwd≈2×grad. recompute off → fwd≈grad. The ratio also quantifies
    # HOW MUCH forward is recomputed.  Let m=fwd/grad−1 be the number of repeated
    # forward passes per model forward (full→1, off→0, selective between), and
    # ρ=m/(1+m)=(fwd−grad)/fwd.  Carry both forms downstream so MFU can use the
    # exact m/(1+R+m) recompute-FLOP share rather than a fixed 1:2:1 split.
    fwd_types = (
        "FlashAttentionScore", "SparseFlashAttention", "SparseFlashMla",
        "PromptFlashAttention", "FusedInferAttentionScore",
    )
    grad_types = (
        "FlashAttentionScoreGrad", "SparseFlashAttentionGrad", "SparseFlashMlaGrad",
    )
    fa_fwd = _count_types(kd, fwd_types, exclude="Grad")
    fa_grad = _count_types(kd, grad_types)
    if fa_grad > 0:
        r = fa_fwd / fa_grad
        if r >= 1.5:
            attention_val, conf = "full", "high"
        elif r <= 1.2:
            attention_val, conf = "off", "high"
        else:
            attention_val, conf = "selective", "medium"
        custom_types = _custom_model_types(kd)
        if custom_types:
            type_counts = Counter(kd["Type"].astype(str)) if "Type" in kd.columns else Counter()
            operator_candidates: List[Dict[str, Any]] = []
            explicit_types = [t for t in custom_types if "recompute" in t.lower()]
            for typ in explicit_types:
                operator_candidates.append({
                    "type": typ,
                    "method": "explicit_kernel_name",
                    "count": int(type_counts.get(typ, 0)),
                    "confidence": "high",
                })
            pair_specs = (
                ("causal_conv1d_fwd_kernel", "causal_conv1d_bwd_kernel"),
                ("chunk_gated_delta_rule_fwd_kernel_h_blockdim64",
                 "chunk_gated_delta_rule_bwd_kernel_dhu_blockdim64"),
            )
            for fwd_type, bwd_type in pair_specs:
                nf, nb = int(type_counts.get(fwd_type, 0)), int(type_counts.get(bwd_type, 0))
                if nb > 0 and nf > nb:
                    operator_candidates.append({
                        "type": fwd_type,
                        "paired_backward_type": bwd_type,
                        "method": "forward_backward_excess",
                        "forward_count": nf,
                        "backward_count": nb,
                        "excess_forward_count": nf - nb,
                        "confidence": "medium",
                    })
            if fa_fwd > fa_grad:
                operator_candidates.append({
                    "type": "Fused Attention forward",
                    "method": "forward_backward_excess",
                    "forward_count": int(fa_fwd),
                    "backward_count": int(fa_grad),
                    "excess_forward_count": int(fa_fwd - fa_grad),
                    "confidence": "medium",
                })
            # Multiple independent signals prove recompute is enabled even though
            # not every custom family has a semantic FLOP model.  Report a useful
            # model-wide opinion: selective is the defensible observed state;
            # ratios near 2 plus explicit KDA replay make full the likely launch
            # granularity, surfaced separately as an estimate.
            val = "selective" if operator_candidates else attention_val
            conf = "medium"
            evidence = (
                f"融合 Attention 子集前向/反向次数比 = {r:.2f}（fwd {fa_fwd} / grad {fa_grad}），"
                f"并检出 {len(operator_candidates)} 组显式/超额 forward 重计算证据；"
                "判定重计算已开启，观测范围为选择性，启动配置更可能接近 full"
            )
        else:
            val = attention_val
            evidence = f"融合 Attention 前向/反向次数比 = {r:.2f}（fwd {fa_fwd} / grad {fa_grad}）"
        fact = _fact(val, evidence, conf)
        fact["attention_value"] = attention_val
        fact["scope"] = "model_wide_estimated" if custom_types else "whole_model_supported_families"
        if custom_types:
            fact["unmodeled_model_types"] = custom_types
            fact["likely_granularity"] = (
                "full" if attention_val == "full" and bool(explicit_types) else "selective"
            )
            fact["operator_candidates"] = operator_candidates
        # Raw signals for the recompute-overhead quantifier (m=fwd/grad−1).
        fact["fwd_grad_ratio"] = round(float(r), 4)
        fact["recompute_forward_multiplier"] = round(max(float(r) - 1.0, 0.0), 4)
        fact["fa_fwd"] = int(fa_fwd)
        fact["fa_grad"] = int(fa_grad)
        out["recompute"] = fact
    else:
        out["recompute"] = _fact(None, "无可配对的融合 Attention 反向算子，无法判定", "unknown")

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

    has_hcom = any(c.get("type") != "Total" for c in (prof.communication or []))
    if has_peer:
        out["single_card"] = _fact(False, "communication_matrix contains cross-rank collective entries", "high")
    elif has_hcom:
        out["single_card"] = _fact(
            False,
            "communication.json contains HCCL collectives, proving distributed execution; the empty matrix means the capture is a single-rank shard / lacks peer topology",
            "medium",
        )
    else:
        out["single_card"] = _fact(True, "communication_matrix has no cross-rank entries and no HCCL collectives", "high")

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
# Underivable from single-rank / single-step → 未知
# --------------------------------------------------------------------------- #
def derive_guesses(prof: ProfileData,
                   model: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """Fields the data cannot pin down; keep them unknown without family priors."""
    local = (model.get("local_experts_per_rank", {}) or {}).get("value")

    # alltoallv proves expert parallelism exists, but a single rank cannot reveal
    # its world size.  A model-family default (for example EP64) is not evidence.
    has_alltoall = any(
        c.get("type", "").lower().startswith("alltoall")
        for c in (prof.communication or [])
    )
    ep_guess = None
    guesses: Dict[str, Dict[str, Any]] = {
        "ep_world_size": {
            "label": "未知",
            "guess": ep_guess,
            "basis": ("存在 alltoallv（专家并行已开启），但单 rank profiling 无 peer topology，"
                      "无法推出 EP world size" if has_alltoall else "无可靠 EP world-size 信号"),
        },
        "num_experts": {
            "label": "未知",
            "guess": None,
            "basis": (f"已知 local_experts_per_rank={local}，但 EP world size 未知"
                      if local else "依赖 local experts 与 EP world size（均无法完整推出）"),
        },
        "num_layers": {
            "label": "未知",
            "guess": None,
            "basis": "Attention 次数被 microbatch、重计算与 PP 切分共同影响，无法反推总层数",
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
    recompute_value = (capture.get("recompute", {}) or {}).get("value")
    if recompute_value in ("full", "selective"):
        flags["recompute-granularity"] = recompute_value
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
