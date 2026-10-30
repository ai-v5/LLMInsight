"""Per-kernel efficiency: MFU / MBU / Roofline + optimization-room ranking.

For every device kernel we estimate FLOPs (matmul family) and moved bytes (from
shapes+dtypes), then compare achieved vs the chip roofline to rank kernels by
"wasted time" (duration that an ideal roofline execution would not have spent).
Chip peaks come from config.ChipSpec and are ASSUMED — surfaced in the UI.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from ..config import SETTINGS, dtype_bytes
from ..parser.profile import num
from ..parser.shapes import parse_shapes, parse_dtypes, numel

MATMUL_TYPES = {
    "MatMulV3", "MatMul", "GroupedMatmul", "GroupedMatmulAdd", "GemmV3", "Gemm",
    "BatchMatMul", "BatchMatMulV2", "BatchMatMulV3",
    # fp8 quantized matmul (routed-expert GEMMs in fp8 training). Same 2·M·N·K
    # work; x and weight are the first two ≥2D operands, the trailing E8M0 scale
    # tensors ([*,*,2]) are ignored by _estimate_matmul_flops. dtype FLOAT8_E4M3
    # routes to the fp8 cube peak via peak_cube_flops().
    "QuantBatchMatmulV3", "QuantBatchMatmulInplaceAdd",
}

# Fused attention kernels are matmul-dominated (QK^T + softmax·V on the cube), so
# their useful FLOPs should count toward MFU even though the op isn't a plain GEMM.
LIGHTNING_INDEXER_FWD_TYPES = {"LightningIndexer", "SparseLightningIndexer"}
LIGHTNING_INDEXER_KL_GRAD_TYPES = {
    "SparseLightningIndexerGradKLLoss",
    "SparseLightningIndexerKllossGrad",
}
DSA_INDEXER_TYPES = LIGHTNING_INDEXER_FWD_TYPES | LIGHTNING_INDEXER_KL_GRAD_TYPES

ATTENTION_FWD_TYPES = {
    "FlashAttentionScore", "PromptFlashAttention",
    "FusedInferAttentionScore", "IncreFlashAttention",
    # DeepSeek-V3.2 sparse attention (DSA): sparse Flash-MLA, shared-KV sparse
    # attention, and the lightning indexer. Routed to the cube peak + attention
    # lane so they're not misfiled as generic compute / left out of attribution.
    "SparseFlashMla", "SparseAttnSharedkv",
    # Generic sparse attention used by current MindSpeed/GLM profiles.  Its
    # token-wise causal useful Cube FLOPs are modeled only when the recorded
    # shapes prove the supported 950DT schema; all other layouts fail closed.
    "SparseFlashAttention",
} | LIGHTNING_INDEXER_FWD_TYPES
ATTENTION_GRAD_TYPES = {
    "FlashAttentionScoreGrad",
    "SparseFlashMlaGrad", "SparseLightningIndexerGrad",
    "SparseFlashAttentionGrad",
} | LIGHTNING_INDEXER_KL_GRAD_TYPES
ATTENTION_TYPES = ATTENTION_FWD_TYPES | ATTENTION_GRAD_TYPES
SPARSE_ATTENTION_TYPES = {"SparseFlashAttention", "SparseFlashAttentionGrad"}
# Hybrid linear-attention / convolution families seen in current KDA models.
# They are model work, so they belong in the whole-model FLOP coverage
# denominator even though their semantics-specific FLOP models are not yet
# implemented.  Omitting them made a GEMM+FA subset look like 100% coverage.
KDA_MODEL_COMPUTE_MARKERS = (
    "chunk_kda", "chunkkda", "kda_", "gated_delta", "chunk_gla", "causal_conv1d",
)
CUSTOM_MODEL_COMPUTE_MARKERS = KDA_MODEL_COMPUTE_MARKERS + (
    # MC2 fused MoE dispatch+GroupedMatMul kernels.  The paired AI_CPU wrapper
    # is excluded from efficiency rows, while the MIX_AIC task carries the
    # model compute and must participate in coverage/counter-calibrated MFU.
    "alltoallvgroupedmatmul", "groupedmatmulalltoallv", "mega_moe",
    "mhcpre", "mhcpost", "mhcsinkhorn",
)
# Recorded matrix-bearing compression/indexer variants have different contracts
# from the older formula-backed indexers. Count them via counters, not aliases
# that would silently apply an unverified shape formula.
COUNTER_MODEL_COMPUTE_TYPES = {
    "Compressor", "CompressorGrad", "LightningIndexerV2", "SparseLightningIndexerKLLossGrad",
}
# MC2 fused MoE dispatch/combine + GroupedMatMul tasks.  Unlike the other custom
# families these DO record their GEMM operands, so their useful Cube FLOPs are
# formula-derived (2·M·N·K, see _estimate_grouped_gemm_flops) instead of only
# counter-calibrated.  The paired AI_CPU launch wrapper is a different Type and
# is excluded from the compute rows entirely.
# Matched as case-insensitive SUBSTRINGS (see _is_mc2_grouped_gemm_type): a capture
# with op attrs publishes "AlltoAllvGroupedMatMul", but a Level-1 capture publishes
# the raw kernel symbol (_ZN7Catccos…Ascend950AllToAllVGroupedMatmul…EPh), which
# exact set membership silently missed.  Keep in sync with
# metrics.core._MC2_FUSED_TYPE_MARKERS and matrix_power._FUSED_MC2_MARKERS.
MC2_GROUPED_GEMM_MARKERS = ("alltoallvgroupedmatmul", "groupedmatmulalltoallv")
# These fused kernels perform repeated sparse gather/scatter work that cannot be
# reconstructed from the one-time input/output tensor footprint in profiler CSV.
SPARSE_DISCRETE_TRAFFIC_TYPES = (SPARSE_ATTENTION_TYPES | DSA_INDEXER_TYPES | COUNTER_MODEL_COMPUTE_TYPES
                                | {"SparseFlashMla", "SparseFlashMlaGrad", "SparseAttnSharedkv"})
# DeepSeek-V3 attention is causal: FlashAttention skips the masked (upper-triangle)
# blocks, so it does ~half the dense QK^T+PV work. Without this 0.5 the achieved
# rate would exceed silicon peak — i.e. the factor is physically required, not a
# convenience. The backward pass recomputes ~2.5x the forward matmul FLOPs.
_ATTN_CAUSAL_FACTOR = 0.5
_ATTN_BWD_FWD_RATIO = 2.5
# Datasheet peaks and profiler durations can differ by a small amount because of
# clock/counter granularity.  Do not rewrite a confirmed peak for this noise, but
# still fail closed on a material contradiction.
_PEAK_MEASUREMENT_TOL = 1.02
# A modeled kernel whose MFU and MBU are BOTH below this is "overhead-bound": its time
# is launch/scalar/dispatch, not on the compute or memory roofline, so the roofline
# "reclaim to 100%" is physically meaningless. We zero its reclaim (keep it out of the
# optimization ranking) and surface it in a separate table instead.
_OVERHEAD_EFF = 0.02


def _is_custom_model_compute_type(op_type: str) -> bool:
    value = (op_type or "").lower()
    return op_type in COUNTER_MODEL_COMPUTE_TYPES or any(marker in value for marker in CUSTOM_MODEL_COMPUTE_MARKERS)


def _is_kda_model_compute_type(op_type: str) -> bool:
    value = (op_type or "").lower()
    return any(marker in value for marker in KDA_MODEL_COMPUTE_MARKERS)


def _is_mc2_grouped_gemm_type(op_type: str) -> bool:
    """MC2 fused AllToAllv + GroupedMatMul kernels (plain label or mangled symbol)."""
    value = (op_type or "").lower()
    return any(marker in value for marker in MC2_GROUPED_GEMM_MARKERS)


def _is_aicpu_dispatch_type(op_type: str) -> bool:
    """True for AI_CPU scheduling wrappers (`*AicpuKernel`), which run no compute.

    An MC2 fused expert kernel is recorded twice per call: the MIX_AIC task that
    really performs the AllToAllv + GroupedMatMul, and an AI_CPU wrapper
    (`AlltoAllvGroupedMatMulMc2AicpuKernel`) that only enqueues the collective.
    The wrapper has no shapes at all, and the capture logs its Accelerator Core
    as AI_CPU *or* COMMUNICATION depending on the export, so a core=="AI_CPU"
    test alone lets half of them through.  Because the wrapper name also carries
    the MC2 compute markers, those rows would land in the whole-model FLOP
    coverage denominator where they can never be modelled.  Match the naming
    convention instead: dispatch belongs to the communication / hidden-overhead
    views, never to the compute-efficiency denominator.
    """
    return (op_type or "").lower().endswith("aicpukernel")


def _fused_moe_summary(prof, rows, counter_scale, proxy_available):
    phases = []
    for label, backward in (("forward", False), ("backward", True)):
        group = [r for r in rows if "mega_moe" in r["type"].lower()
                 and ("backward" in r["type"].lower()) == backward]
        if not group:
            continue
        duration = sum(r["dur_us"] for r in group)
        covered = [r for r in group if (r.get("counter_proxy_flops_raw") or 0) > 0]
        flops = sum(r["counter_proxy_flops_raw"] for r in covered) * counter_scale
        peak_time = sum(r["peak_flops"] * r["dur_us"] * 1e-6 for r in group)
        phases.append({"phase": label, "count": len(group), "fused_us": round(duration, 1),
                       "mac_active_us": round(sum(r["mac_active_us"] for r in group), 1),
                       "estimated_flops": flops if proxy_available and len(covered) == len(group) else None,
                       "fused_compute_utilization_estimated": flops / peak_time if proxy_available and peak_time and len(covered) == len(group) else None,
                       "communication_us": None, "communication_bytes": None,
                       "compute_only_us": None})
    semantics = (getattr(prof, "meta", {}) or {}).get("framework_semantics") or {}
    external = [c for c in (getattr(prof, "communication", None) or []) if c.get("type") == "alltoallv"]
    return {"available": bool(phases), "evidence_level": "PROXY", "phases": phases,
            "external_alltoallv": {"count": len(external),
                "elapse_us": round(sum(c.get("elapse_ms", 0) for c in external) * 1000, 1),
                "wait_us": round(sum(c.get("wait_ms", 0) for c in external) * 1000, 1),
                "attribution": "same_capture_external_ops_not_internal_fused_phases"},
            "framework_input_shapes": semantics.get("mega_moe_input_shapes", []),
            "basis": "same_capture_counter_calibration" if proxy_available else "calibration_unavailable",
            "note": "融合窗口包含计算、通信及重叠；MAC 活跃时间是计数器视角。缺少实际专家 token 数/内部阶段时，不以融合时长减 MAC 时间推算通信，不生成纯计算 MFU。周期代理含 padding/指令效率不确定性，非语义 FLOPs 实测。"}


def _op_class(op_type: str) -> Optional[str]:
    """Map a kernel Type to its ceiling-governed op-class (matmul / attention /
    attention_grad), or None for everything else (vector / memory ops, which
    optimize toward the 100% roofline instead of a capped ceiling)."""
    if op_type in MATMUL_TYPES:
        return "matmul"
    if op_type in ATTENTION_FWD_TYPES:
        return "attention"
    if op_type in ATTENTION_GRAD_TYPES:
        return "attention_grad"
    return None


def _matmul_mnk(shapes: List[List[int]], shapes_out: Optional[List[List[int]]] = None):
    """(M, N, K, batch) of a matmul, using the output to resolve transposes.

    The first two >=2-D inputs are the multiplicands.  When an output matrix is
    available, enumerate the four transpose combinations and require its trailing
    [M,N] dimensions to match.  This is essential for weight-gradient/addmm forms
    such as [K,M] x [K,N] -> [M,N], which a shape-only forward-GEMM heuristic
    over-counts by up to 2x.

    Single source of truth for both the FLOP count (2*M*N*K*batch) and the result
    size M*N*batch — the latter caps oversized inplace-add accumulators in the
    bytes model (see compute_efficiency)."""
    mats = [s for s in shapes if len(s) >= 2]
    if len(mats) < 2:
        return None
    a, b = mats[0], mats[1]

    def mk(s):
        if len(s) == 2:
            return s[0], s[1], 1
        batch = 1
        for d in s[:-2]:
            batch *= d
        return s[-2], s[-1], batch

    ar, ac, abz = mk(a)
    br, bc, _ = mk(b)

    # A can be [M,K] or [K,M]; B can be [K,N] or [N,K].  Keep only
    # algebraically valid combinations, then use the output to disambiguate.
    candidates = set()
    for M, K_a in ((ar, ac), (ac, ar)):
        for K_b, N in ((br, bc), (bc, br)):
            if K_a == K_b and min(M, N, K_a) > 0:
                candidates.add((M, N, K_a, abz))

    out_matrix = next((s for s in (shapes_out or []) if len(s) >= 2), None)
    if out_matrix is not None:
        out_m, out_n = out_matrix[-2], out_matrix[-1]
        matched = [c for c in candidates if c[0] == out_m and c[1] == out_n]
        # Multiple transpose paths with the same M/N/K are equivalent.  Different
        # K values are genuinely ambiguous and must not set a peak denominator.
        unique = set(matched)
        if len(unique) == 1:
            return unique.pop()
        if matched and len({c[2] for c in matched}) == 1:
            return matched[0]
        if candidates:
            return None

    # Old exports can omit output shapes.  Preserve only an unambiguous standard
    # A[M,K] x B[K,N] interpretation; otherwise fail closed instead of guessing a
    # transpose from equal dimensions.
    standard = (ar, bc, ac, abz) if ac == br and min(ar, bc, ac) > 0 else None
    if standard:
        return standard
    if len(candidates) == 1:
        return candidates.pop()
    return None


def _estimate_matmul_flops(
    shapes: List[List[int]], shapes_out: Optional[List[List[int]]] = None,
) -> Optional[float]:
    mnk = _matmul_mnk(shapes, shapes_out)
    if not mnk:
        return None
    M, N, K, batch = mnk
    return 2.0 * M * N * K * batch


def _estimate_grouped_gemm_flops(
    shapes_in: List[List[int]], shapes_out: List[List[int]],
) -> Optional[float]:
    """Useful Cube FLOPs of an MC2 fused MoE grouped GEMM = 2·M·N·K.

    kernel_details.csv records the two MC2 fused tasks as

        AlltoAllvGroupedMatMul : in [M,K] ; [G,X,Y]   out [M,N] (+ [M,K] copy)
        GroupedMatMulAlltoAllv : in [M,K] ; [G,X,Y]   out [M,N]

    where M is the post-AllToAllv/pass-through token count, G the number of LOCAL
    experts, and the per-expert weights are the 3-D tensor.  The export does not
    say whether that tensor is [G,N,K] or [G,K,N] (both orders occur in one
    capture), so K is pinned to the activation's trailing dimension — the
    contraction length — and N is the remaining one, validated against the
    recorded 2-D output [M,N].  Anything ambiguous fails closed.

    G is a leading group axis only: every token row is multiplied by exactly one
    expert's weight, so G partitions M instead of extending the contraction and
    must NOT multiply 2·M·N·K.

    Recorded examples (rank-133 capture, bf16):
      in [32768,6144] ; [2,4096,6144] -> out [32768,4096]  (down_proj)
      in [32768,6144] ; [2,6144,2048] -> out [32768,2048]  (up_proj)
      in [32768,2048] ; [2,6144,2048] -> out [32768,6144]  (gate/up, combine)
    """
    mats = [s for s in shapes_in if len(s) >= 2]
    if len(mats) < 2:
        return None
    act = mats[0]
    weight = next((s for s in mats[1:] if len(s) == 3), None)
    if weight is None or len(act) != 2:
        return None
    M, k_in = act
    dim_a, dim_b = weight[-2], weight[-1]
    if min(M, k_in, dim_a, dim_b) <= 0:
        return None
    if k_in == dim_a and dim_b != k_in:
        N = dim_b
    elif k_in == dim_b and dim_a != k_in:
        N = dim_a
    else:
        return None                       # ambiguous weight layout -> fail closed
    if not any(len(s) == 2 and s[0] == M and s[1] == N for s in shapes_out):
        return None                       # output does not confirm N
    return 2.0 * M * N * k_in


def _right_down_causal_pairs(sq: int, sk: int, limit: Optional[int] = None) -> int:
    """Valid token pairs for right-down causal attention, optionally capped per row."""
    total = 0
    for q_idx in range(sq):
        causal_prefix = max(0, min(sk, sk - sq + q_idx + 1))
        total += min(limit, causal_prefix) if limit is not None else causal_prefix
    return total


def _sparse_attention_geometry(shapes_in, shapes_out, is_grad: bool):
    """(B, Sq, Hq, Sk, Hkv, Dq, Dv, rope_dim, topk) of a token-wise sparse
    attention kernel, or None when the recorded layout does not prove it.

    Two export layouts carry the same contract:

      q               [B, Sq, Hq, Dq]            query (MLA absorbed: Dq=512)
      k, v            [B, Sk, Hkv, Dk/Dv]        latent KV (MLA absorbed: Hkv=1)
      sparse_indices  [B, Sq, Hkv, topk]         topk selected keys per query
      softmax stats   [B, Hkv, Sq, Hq/Hkv]  x2   (fwd outputs / grad inputs)
      q_rope, k_rope  the two trailing operands, [..., rope_dim]

    The forward records `q; k; v; indices; ...; q_rope; k_rope`, while the
    backward inserts its gradient operands before the value tensor, so k/v are
    located by shape instead of by position: k is operand #2 in both layouts and
    v is pinned by the recorded value head dim (the forward's result tensor
    [B,Sq,Hq,Dv], or the backward's v-gradient output [B,Sk,Hkv,Dv]).  That
    output-based disambiguation is what keeps equal-length training (Sk == Sq,
    where the indices tensor has the same [B,Sq,Hkv,*] prefix) provable.  The
    RoPE pair, the group-shaped indices tensor and the two [B,Hkv,Sq,G] softmax
    tensors are what make the "topk selected tokens per query" pattern provable
    from shapes alone.
    """
    if len(shapes_in) < 3 or len(shapes_in[0]) != 4:
        return None
    q = shapes_in[0]
    B, Sq, Hq, Dq = q
    if min(B, Sq, Hq, Dq) <= 0:
        return None
    q_rope, k_rope = shapes_in[-2], shapes_in[-1]
    if len(q_rope) != 4 or len(k_rope) != 4:
        return None
    if q_rope[:3] != [B, Sq, Hq] or q_rope[3] <= 0 or q_rope[3] != k_rope[3]:
        return None
    k = shapes_in[1]
    if len(k) != 4:
        return None
    k_b, Sk, Hkv, Dk = k
    if min(Sk, Hkv, Dk) <= 0 or k_b != B or Dk != Dq or Hq % Hkv != 0:
        return None
    if k_rope[:3] != [B, Sk, Hkv]:
        return None
    # Dv is read from the recorded output contract, not guessed from operand order.
    if is_grad:
        if (len(shapes_out) < 3 or shapes_out[0] != q or shapes_out[1] != k
                or len(shapes_out[2]) != 4):
            return None
        v_shape = shapes_out[2]           # v-gradient: [B, Sk, Hkv, Dv]
    else:
        if not shapes_out or len(shapes_out[0]) != 4:
            return None
        v_shape = shapes_out[0]           # attention result: [B, Sq, Hq, Dv]
    if v_shape[:3] != ([B, Sk, Hkv] if is_grad else [B, Sq, Hq]):
        return None
    Dv = v_shape[3]
    if Dv <= 0:
        return None
    # v is the last [B,Sk,Hkv,Dv] operand that is not the RoPE key (excluded above).
    kv_idx = [i for i in range(1, len(shapes_in) - 1)
              if len(shapes_in[i]) == 4 and shapes_in[i][:3] == [B, Sk, Hkv]
              and shapes_in[i][3] == Dv]
    if len(kv_idx) != 2:
        return None
    v = shapes_in[kv_idx[-1]]
    if is_grad and v != v_shape:
        return None
    # The selected-key tensor is the only [B,Sq,Hkv,*] operand besides q/q_rope/v
    # (k_rope joins that prefix when Sk == Sq, so both trailing RoPE operands are
    # excluded by position).
    idx_cand = [i for i in range(len(shapes_in))
                if i not in (0, len(shapes_in) - 2, len(shapes_in) - 1)
                and i not in kv_idx
                and len(shapes_in[i]) == 4 and shapes_in[i][:3] == [B, Sq, Hkv]
                and shapes_in[i][3] > 0]
    if len(idx_cand) != 1:
        return None
    topk = shapes_in[idx_cand[0]][3]
    if topk > Sk:
        return None
    group = Hq // Hkv
    all_shapes = list(shapes_in) + list(shapes_out)
    if sum(1 for s in all_shapes if s == [B, Hkv, Sq, group]) < 2:
        return None
    return B, Sq, Hq, Sk, Hkv, Dq, Dv, q_rope[3], topk


def _estimate_sparse_flash_attention_flops(
    shapes_in: List[List[int]], shapes_out: List[List[int]], is_grad: bool,
    sparse_mode: Optional[int] = None, sparse_block_size: Optional[int] = None,
) -> Optional[float]:
    """Useful BF16 cube FLOPs for token-wise sparse SparseFlashAttention.

    CANN SparseFlashAttention selects `topk` keys per query
    (sparse_indices[B,Sq,Nkv,topk]) and, on 950DT, treats every index as one
    token (sparse_block_size=1, one selected token per index).  The returned
    numerator counts only the matmul-dominated useful work:

      fwd = 2 * P * (Dqk + Dv)
      bwd = 2 * P * (3*Dqk + 2*Dv)      (recompute QK^T, dQ, dK, dV, dP)
      P   = B * Hkv * (Hq/Hkv) * (valid selected tokens per query)

    where Dqk = Dq + rope_dim (absorbed MLA: 512 nope + 64 RoPE, matching the
    recorded q [B,S,64,512] / q_rope [B,S,64,64] pair) and Dv is the value
    head dim.  `-1` padding indices are excluded by the causal cap below.

    kernel_details.csv records shapes but NOT sparse_mode, so the valid-pair
    count is only accepted when it is provable:
      * sparse_mode == 3 (right-down causal, the MindSpeed training semantic):
        the exact causal prefix min(topk, max(0, Sk-Sq+q+1)) per query row.
      * mode not recorded: the causal cap CANNOT bind when even the first query
        already sees >= topk keys (Sk-Sq+1 >= topk).  Every query then selects
        exactly topk tokens under ANY causal mask, so Sq*topk is exact.
      * any other explicitly declared mode, or a binding cap, fails closed
        instead of reinterpreting a mask the caller did specify.

    Softmax, gather/scatter, invalid tiling lanes and other vector/memory work
    are not cube FLOPs.  The CSV cannot reconstruct their repeated/discrete
    memory traffic, so this op is deliberately excluded from MBU, roofline and
    reclaim estimates.
    """
    # Only the block size (one selected token per index) is a chip semantic we
    # can pin down; the mask is either supplied or proven non-binding below.
    if sparse_block_size != 1:
        return None
    geo = _sparse_attention_geometry(shapes_in, shapes_out, is_grad)
    if geo is None:
        return None
    B, Sq, Hq, Sk, Hkv, Dq, Dv, rope_dim, topk = geo
    if sparse_mode == 3:
        valid_per_pattern = _right_down_causal_pairs(Sq, Sk, topk)
    elif sparse_mode is None and Sk - Sq + 1 >= topk:
        valid_per_pattern = Sq * topk
    else:
        return None
    selected_head_pairs = B * Hkv * (Hq // Hkv) * valid_per_pattern
    d_qk = Dq + rope_dim
    if is_grad:
        return 2.0 * selected_head_pairs * (3 * d_qk + 2 * Dv)
    return 2.0 * selected_head_pairs * (d_qk + Dv)


def _estimate_lightning_indexer_flops(
    shapes_in: List[List[int]], shapes_out: List[List[int]],
    sparse_mode: Optional[int] = None,
) -> Optional[float]:
    """Useful Cube FLOPs for fused LightningIndexer.

    The fused op first evaluates an index-query/index-key dot product for every
    causally visible key, then performs ReLU, head weighting and TopK on the
    vector side.  Only the QK matmul belongs in the BF16 Cube numerator:

      2 * B * Hq * P * Dq,   P = causally visible (query, key) pairs

    kernel_details.csv records q [B,Sq,Hq,Dq], k [B,Sk,Hkv,Dq], the per-query
    head weights [B,Sq,Hq] and the selected indices [B,Sq,Hkv,topk] — but not the
    mask attribute.  sparse_mode == 3 (right-down causal) gives the exact count;
    without it the same right-down-causal prefix is used as a LOWER bound,
    because the only alternative (scoring all Sk keys) is larger by at most
    Sq/(2*Sk) (1.2% on the rank-133 capture).  Under-counting keeps the reported
    MFU conservative instead of guess-inflated.
    """
    # `sparse_mode` is kept for call-site compatibility (metrics/matrix_power.py
    # passes it); the right-down-causal pair count below IS the sparse_mode=3
    # exact figure and a lower bound for any other mask.
    if len(shapes_in) < 3 or not shapes_out:
        return None
    q, k, weights = shapes_in[:3]
    indices = shapes_out[0]
    if not (len(q) == len(k) == len(indices) == 4 and len(weights) == 3):
        return None
    B, Sq, Hq, Dq = q
    Bk, Sk, Hkv, Dk = k
    Bi, Si, Hi, topk = indices
    if (
        min(B, Sq, Hq, Dq, Sk, Hkv, topk) <= 0
        or (Bk, Bi) != (B, B)
        or (Si, Hi) != (Sq, Hkv)
        or Dq != Dk
        or weights != [B, Sq, Hq]
        or Hq % Hkv != 0
        or topk > Sk
        or any(out != indices for out in shapes_out[1:2])
    ):
        return None
    causal_pairs = _right_down_causal_pairs(Sq, Sk)
    return 2.0 * B * Hq * causal_pairs * Dq


def _estimate_sparse_lightning_indexer_grad_kl_loss_flops(
    shapes_in: List[List[int]], shapes_out: List[List[int]],
    sparse_mode: Optional[int] = None,
) -> Optional[float]:
    """Useful Cube FLOPs for fused SparseLightningIndexerGradKLLoss.

    CANN executes four Cube products: main-attention QK (including RoPE), index
    QK, dQ-index and dK-index.  KL/softmax/ReLU/dW and sparse gather/scatter are
    vector/memory work and deliberately excluded from this Cube numerator.
    """
    if sparse_mode != 3 or len(shapes_in) < 10 or len(shapes_out) < 4:
        return None
    q, k, q_index, k_index, weights, indices = shapes_in[:6]
    softmax_max, softmax_sum, q_rope, k_rope = shapes_in[6:10]
    if any(len(s) != 4 for s in (q, k, q_index, k_index, indices, q_rope, k_rope)):
        return None
    if len(weights) != 3:
        return None
    B, Sq, Hq, Dq = q
    Bk, Sk, Hkv, Dk = k
    Bqi, Sqi, Hqi, Dqi = q_index
    Bki, Ski, Hkvi, Dki = k_index
    Bi, Si, Hi, topk = indices
    if (
        min(B, Sq, Hq, Dq, Sk, Hkv, Dk, Hqi, Dqi, topk) <= 0
        or (Bk, Bqi, Bki, Bi) != (B, B, B, B)
        or (Sqi, Ski, Si) != (Sq, Sk, Sq)
        or (Hkvi, Hi) != (Hkv, Hkv)
        or Dq != Dk or Dqi != Dki
        or Hq % Hkv != 0 or Hqi % Hkv != 0
        or weights != [B, Sq, Hqi]
        or topk > Sk
        or q_rope[:3] != [B, Sq, Hq]
        or k_rope[:3] != [B, Sk, Hkv]
        or q_rope[3] <= 0 or q_rope[3] != k_rope[3]
        or softmax_max != [B, Hkv, Sq, Hq // Hkv]
        or softmax_sum != softmax_max
        or shapes_out[:3] != [q_index, k_index, weights]
        or numel(shapes_out[3]) != 1
    ):
        return None
    selected_pairs = B * _right_down_causal_pairs(Sq, Sk, topk)
    main_qk = Hq * (Dq + q_rope[3])
    index_qk_and_grads = 3 * Hqi * Dqi
    return 2.0 * selected_pairs * (main_qk + index_qk_and_grads)


def _estimate_attention_flops(shapes_in, shapes_out, is_grad: bool) -> Optional[float]:
    """FLOPs for a fused FlashAttention kernel = 2·B·N·S²·(d_qk + d_v)·causal.

    Layout-agnostic: B, N, S are read from the softmax max/sum tensor ([B,N,S,k]
    with a small trailing dim — present as an output of the fwd op and an input of
    the grad op). Head dims come from the leading Q/K/V tensors via
    head_dim = numel / (B·N·S), so MLA's split d_qk(192)/d_v(128) is captured.
    """
    shapes = list(shapes_in) + list(shapes_out)
    bns = None
    for s in shapes:
        if len(s) == 4 and 0 < s[3] <= 16 and s[1] >= 1 and s[2] >= 8:
            bns = (s[0], s[1], s[2])
            break
    if not bns:
        return _estimate_attention_flops_tnd(shapes_in, shapes_out, is_grad)
    B, N, S = bns
    base = B * N * S
    if base <= 0:
        return None
    head_dims: List[int] = []
    for s in shapes_in:
        if len(s) < 2:
            continue
        nl = numel(s)
        if nl <= 0 or nl % base != 0:
            continue
        hd = nl // base
        if 16 <= hd <= 4096:       # excludes the mask / softmax stats tensors
            head_dims.append(hd)
    if not head_dims:
        return None
    d_qk = head_dims[0]                                  # query is always first
    d_v = next((h for h in head_dims if h != d_qk), d_qk)  # MLA: differs; MHA: ==
    fwd = 2.0 * B * N * (S ** 2) * (d_qk + d_v) * _ATTN_CAUSAL_FACTOR
    return fwd * (_ATTN_BWD_FWD_RATIO if is_grad else 1.0)


def _estimate_attention_flops_tnd(shapes_in, shapes_out, is_grad: bool) -> Optional[float]:
    """TND (packed variable-length) FlashAttention: q/k/v are 3-D [T, N, D] with T =
    packed total tokens across all sequences, and the softmax stats are [T, N, k<=16] —
    there is no 4-D [B,N,S,D] to read S from. Per-sequence lengths live only in the
    actual_seq_qlen VALUES (not the recorded shapes), so Σs_i² is approximated by the
    equal-length T²/num_seq, where num_seq = the 1-D actual_seq tensor's length. causal
    iff a square [S,S] attention-mask tensor is present (ViT full-attention has none)."""
    if not shapes_in or len(shapes_in[0]) != 3:
        return None
    T, N, d_qk = shapes_in[0]
    if T <= 0 or N <= 0 or d_qk <= 0:
        return None
    allsh = list(shapes_in) + list(shapes_out)
    # confirm fused attention: a 3-D softmax-stats tensor [T, N, k<=16] must be present
    if not any(len(s) == 3 and s[0] == T and s[1] == N and 0 < s[2] <= 16 for s in allsh):
        return None
    # value head_dim (GQA: k/v carry fewer heads but the same head_dim as q)
    d_v = shapes_in[2][2] if len(shapes_in) > 2 and len(shapes_in[2]) == 3 else d_qk
    num_seq = next((s[0] for s in shapes_in if len(s) == 1 and s[0] >= 1), 1)
    sigma_s2 = (T * T) / num_seq                          # equal-length approximation
    causal = (_ATTN_CAUSAL_FACTOR if any(len(s) == 2 and s[0] == s[1] for s in shapes_in)
              else 1.0)                                   # square mask present => causal
    fwd = 2.0 * N * sigma_s2 * (d_qk + d_v) * causal
    return fwd * (_ATTN_BWD_FWD_RATIO if is_grad else 1.0)


# Per-element FLOP factors for VECTOR-core (AI Vector Core / MIX_AIV) kernels.
# Unlike matmul / fused-attention (cube, modeled exactly from shapes), elementwise /
# normalization / optimizer kernels have no closed-form FLOP count, so we approximate
# work = factor·N, where N is the dominant tensor's element count. Factors are
# engineering estimates of arithmetic ops per element and are meant to be tuned; they
# drive the VECTOR-peak MFU and the max(vector-compute, memory) headroom floor.
# Pure data-movement ops (Cast / ZerosLike / Copy / Reshape / …) do ~0 arithmetic →
# factor 0 → left unmodeled for FLOPs, so they stay honestly memory-bound (MBU only),
# exactly as before.
_VECTOR_FLOPS_PER_ELEM: Dict[str, float] = {
    # elementwise arithmetic — ~1 op/element
    "Mul": 1.0, "Muls": 1.0, "Add": 1.0, "Adds": 1.0, "Sub": 1.0, "Subs": 1.0,
    "Div": 1.0, "RealDiv": 1.0, "Neg": 1.0, "Reciprocal": 1.0, "Sqrt": 1.0,
    "Rsqrt": 1.0, "Square": 1.0, "Abs": 1.0, "Maximum": 1.0, "Minimum": 1.0,
    "Sign": 1.0, "AddcmulV2": 3.0, "Axpy": 2.0,
    # activations / transcendentals — a few ops/element
    "Exp": 2.0, "Log": 2.0, "Sigmoid": 4.0, "Tanh": 4.0, "Gelu": 8.0,
    "GeluGrad": 10.0, "Silu": 4.0, "SwiGlu": 8.0, "Swiglu": 8.0, "SwiGluGrad": 12.0,
    "FastGelu": 8.0, "Relu": 1.0,
    # softmax / reductions
    "Softmax": 5.0, "SoftmaxV2": 5.0, "LogSoftmaxV2": 6.0, "SoftmaxGrad": 6.0,
    "ReduceSum": 1.0, "ReduceMean": 1.0,
    # normalization
    "RmsNorm": 5.0, "RmsNormGrad": 10.0, "LayerNorm": 6.0, "LayerNormV2": 6.0,
    "LayerNormGrad": 12.0,
    # optimizer step (per-parameter update math)
    "ApplyAdamWV2": 11.0, "ApplyAdamW": 11.0, "ApplyAdam": 11.0,
    # pure data movement / format — no arithmetic (stay memory-only)
    "Cast": 0.0, "ZerosLike": 0.0, "Zero": 0.0, "Fill": 0.0, "Fills": 0.0,
    "Copy": 0.0, "BroadcastTo": 0.0, "Transpose": 0.0, "Slice": 0.0,
    "StridedSlice": 0.0, "Concat": 0.0, "Gather": 0.0, "GatherV2": 0.0,
    "Reshape": 0.0,
}
# An unknown fused vector kernel is not necessarily light elementwise work
# (causal_conv1d is a counterexample).  Fail closed on FLOPs instead of inventing
# 1 op/element; known memory-only kernels still retain their MBU model below.
_VECTOR_DEFAULT_FPE = 0.0


def _estimate_vector_flops(op_type: str, shapes_in, shapes_out) -> Optional[float]:
    """Approx FLOPs for a VECTOR-core kernel = factor(op_type)·N, where N is the
    largest tensor's element count (the elementwise working set). Returns None for
    zero-arithmetic ops (factor 0) so they stay memory-only — identical to before."""
    factor = _VECTOR_FLOPS_PER_ELEM.get(op_type, _VECTOR_DEFAULT_FPE)
    if factor <= 0:
        return None
    n = 0
    for s in list(shapes_in) + list(shapes_out):
        n = max(n, numel(s))
    if n <= 0:
        return None
    return factor * n


# --------------------------------------------------------------------------- #
# KDA (Kimi Delta Attention) family — semantic FLOP formulas.
#
# KDA is the chunked gated-delta-rule linear attention of Kimi K3 (fla-org
# `chunk_kda` / `gated_delta_rule`).  Kernel geometry in the profiler CSVs:
#   q,k,v,g ∈ [B, T, H, K] (head dim K; V=K when no GVA), chunk_size BT=64.
# Formulas below are order-of-magnitude semantic MACs derived from the recorded
# tensor shapes — NOT exact instruction counts.  They exist so these key kernels
# count toward whole-model FLOP coverage instead of being written off as
# "unmodeled"; the same-capture counter calibration remains the cross-check when
# shapes are missing (formula returns None and the row falls back to proxy).
#
#   chunk fwd:  per token/head ≈ 4·BT·K MAC intra-chunk (Aqk, Akk, w/u WY solve)
#               + 2·K·V MAC state propagation + output  ->  2·T·H·(4·BT·K + 2·K·V)
#   bwd split (fractions of the fwd chunk work, fused gradient kernels):
#       chunk_kda_bwd_kernel_wy_dqkg_fused   ≈ 2×fwd   (dq/dk/dg fused)
#       chunk_kda_bwd_kernel_intra           ≈ 1×fwd   (intra-chunk gradients)
#       chunk_kda_bwd_kernel_dAv             ≈ 0.5×fwd (dv only)
#       chunk_gated_delta_rule_bwd_*         ≈ 2×fwd   (fused dq/dk/dv/dh)
#       recompute_w_u_fwd_kda_kernel         ≈ 0.5×fwd (WY w/u re-solve)
#   causal_conv1d (depthwise, width W=4):     2·T·D·W fwd; bwd ≈ 2×fwd (dx+dw)
#   gate / chunk-cumsum vector kernels:       2·T·H·K   (elementwise + scan)
# --------------------------------------------------------------------------- #
_KDA_CHUNK_SIZE = 64
_KDA_BWD_MULTIPLIERS = {
    "chunk_kda_bwd_kernel_wy_dqkg_fused": 2.0,
    "chunk_kda_bwd_kernel_intra": 1.0,
    "chunk_kda_bwd_kernel_dav": 0.5,
    "recompute_w_u_fwd_kda_kernel": 0.5,
    "causal_conv1d_bwd_kernel": 2.0,
}
_KDA_GATE_TYPES = {
    "kda_gate_bwd_kernel",
    "kda_gate_chunk_cumsum_vector_kernel",
    "chunk_local_cumsum_vector_kernel",
}


def _kda_chunk_geometry(shapes_in):
    """(T, H, K, V) from the [B, T, H, K]-shaped chunk operands (q/k/v/g are
    4-D; intermediate tensors like [B, H, T] decays / scalar params are skipped).
    V falls back to K (no GVA) when every value tensor matches q/k."""
    four_d = [s for s in shapes_in if len(s) == 4]
    if not four_d:
        # shape-less / non-chunk layout: give up and let the counter proxy decide
        return None
    t, h, k = four_d[0][-3], four_d[0][-2], four_d[0][-1]
    v = four_d[-1][-1] if four_d[-1][-2] == h else k
    if not (t and h and k and v):
        return None
    return float(t), float(h), float(k), float(v)


def _kda_causal_conv_flops(shapes_in) -> Optional[float]:
    """Depthwise causal conv1d: 2·T·D·W.  Activation [1,T,D] (or [T,D]) and
    depthwise weight [D, W] (width 4 in KDA)."""
    act = next((s for s in shapes_in if len(s) >= 2), None)
    wgt = next((s for s in shapes_in if len(s) == 2), None)
    if act is None:
        return None
    t = act[-2]
    d = act[-1]
    w = (wgt[-1] if wgt else 4) or 4
    if not (t and d):
        return None
    return 2.0 * t * d * w


def _estimate_kda_flops(op_type: str, shapes_in, shapes_out) -> Optional[float]:
    """Semantic FLOP estimate for a KDA-family kernel (fwd/bwd/gate/conv)."""
    t = (op_type or "").lower()
    if t == "causal_conv1d_fwd_kernel":
        return _kda_causal_conv_flops(shapes_in)
    if t == "causal_conv1d_bwd_kernel":
        f = _kda_causal_conv_flops(shapes_in)
        return 2.0 * f if f else None
    if t in _KDA_GATE_TYPES:
        ch = _kda_chunk_geometry(shapes_in)
        return (2.0 * ch[0] * ch[1] * ch[2]) if ch else None
    ch = _kda_chunk_geometry(shapes_in)
    if not ch:
        return None
    tt, h, k, v = ch
    fwd = 2.0 * tt * h * (4.0 * _KDA_CHUNK_SIZE * k + 2.0 * k * v)
    mult = 1.0
    if t.startswith("chunk_gated_delta_rule_bwd"):
        mult = 2.0
    else:
        mult = _KDA_BWD_MULTIPLIERS.get(t, 1.0)
    return fwd * mult


def _bound_from_ratios(mac, mte2, vec) -> str:
    mac = mac or 0.0
    mte2 = mte2 or 0.0
    vec = vec or 0.0
    if mac >= 0.30 and mac >= mte2:
        return "compute"
    if mte2 >= 0.30 and mte2 > mac:
        return "memory"
    if vec >= 0.30:
        return "vector"
    return "other"


def _estimate_matmul_training_flops(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Split repeated 2-D linear forwards from their gradient permutations.

    For one linear projection the forward, dX and dW GEMMs share the same three
    dimensions in different M/N/K order.  Activation checkpointing makes exactly
    the forward orientation more frequent.  Equal orientation counts therefore
    prove no replay (for example a terminal LM head), while a unique excess count
    measures replay directly. Ambiguous/square/grouped families retain the R=2
    fallback instead of being guessed.
    """
    mm_rows = [r for r in rows if r["type"] in MATMUL_TYPES and r.get("flops")]
    executed = sum(float(r["flops"]) for r in mm_rows)
    if executed <= 0:
        return {"available": False, "reason": "no modeled matmul rows"}

    groups: Dict[Any, Dict[Any, List[float]]] = {}
    for row in mm_rows:
        mnk = row.get("matmul_mnk")
        if not mnk:
            continue
        M, N, K, batch = mnk
        if batch != 1:
            continue
        key = tuple(sorted((int(M), int(N), int(K))))
        groups.setdefault(key, {}).setdefault((int(M), int(N), int(K)), []).append(
            float(row["flops"])
        )

    matched_flops = 0.0
    exact_recompute = 0.0
    exact_groups = 0
    ambiguous_groups = 0
    for orientations in groups.values():
        if len(orientations) < 2:
            continue
        values = [v for rows_v in orientations.values() for v in rows_v]
        if max(values) > min(values) * (1.0 + 1e-9):
            ambiguous_groups += 1
            continue
        counts = sorted((len(v) for v in orientations.values()), reverse=True)
        if len(set(counts)) == 1:
            replay_count = 0
        elif counts[0] > counts[1] and len(set(counts[1:])) == 1:
            replay_count = counts[0] - counts[1]
        else:
            ambiguous_groups += 1
            continue
        per_call = values[0]
        matched_flops += sum(values)
        exact_recompute += replay_count * per_call
        exact_groups += 1

    unmatched_flops = max(executed - matched_flops, 0.0)
    # Unpaired shapes do not establish any replay. Keep a sensitivity upper
    # bound, but never manufacture a full-recompute point estimate from them.
    fallback_recompute = 0.0
    fallback_recompute_lo = 0.0
    return {
        "available": True,
        "source": "linear_orientation_counts_plus_training_fallback",
        "executed_flops": executed,
        "exact_group_count": exact_groups,
        "ambiguous_group_count": ambiguous_groups,
        "exact_group_flops": matched_flops,
        "exact_recompute_forward_flops": exact_recompute,
        "fallback_flops": unmatched_flops,
        "recompute_forward_flops": exact_recompute + fallback_recompute,
        "recompute_forward_flops_lo": exact_recompute + fallback_recompute_lo,
        "recompute_forward_flops_hi": exact_recompute + unmatched_flops / 4.0,
        "exact_coverage_pct": 100.0 * matched_flops / executed,
    }


def _median(values: List[float]) -> float:
    ordered = sorted(v for v in values if v is not None)
    if not ordered:
        return 0.0
    mid = len(ordered) // 2
    return float(ordered[mid]) if len(ordered) % 2 else 0.5 * (ordered[mid - 1] + ordered[mid])


def _estimate_recompute_from_rows(
    rows: List[Dict[str, Any]],
    matmul_training: Dict[str, Any],
    counter_scale: float,
    executed_flops: float,
    executed_flops_lo: float,
    executed_flops_hi: float,
) -> Dict[str, Any]:
    """Best-effort model-wide recompute estimate with named operator evidence.

    Formula-backed matmul/attention work remains primary.  For custom model
    kernels we reuse the capture-local counter scale derived from modeled rows.
    Explicit ``recompute`` names are high-confidence evidence; excess forward
    calls over the paired backward family are medium-confidence evidence.
    """
    candidates: List[Dict[str, Any]] = []

    def _row_estimate(row: Dict[str, Any]) -> float:
        if row.get("flops"):
            return float(row["flops"])
        return float(row.get("counter_proxy_flops_raw") or 0.0) * counter_scale

    mm_est = float(matmul_training.get("recompute_forward_flops") or 0.0)
    mm_lo = float(matmul_training.get("recompute_forward_flops_lo") or mm_est)
    mm_hi = float(matmul_training.get("recompute_forward_flops_hi") or mm_est)
    if mm_est > 0:
        mm_rows = [r for r in rows if r["type"] in MATMUL_TYPES and r.get("flops")]
        mm_exec = sum(float(r["flops"]) for r in mm_rows)
        mm_dur = sum(float(r["dur_us"]) for r in mm_rows)
        candidates.append({
            "type": "MatMul/GEMM forward groups",
            "method": "orientation_phase_split",
            "confidence": ("medium" if float(matmul_training.get("exact_coverage_pct") or 0) < 50
                           else "high"),
            "count": None,
            "duration_us": round(mm_dur * mm_est / mm_exec, 1) if mm_exec > 0 else None,
            "recompute_flops": mm_est,
            "recompute_flops_lo": mm_lo,
            "recompute_flops_hi": mm_hi,
            "evidence": (
                f"{matmul_training.get('exact_group_count', 0)} 个方向配对组；"
                f"精确覆盖 {float(matmul_training.get('exact_coverage_pct') or 0):.1f}%，"
                "未配对部分不设重算点估计，仅保留敏感性上界"
            ),
        })

    explicit = [r for r in rows if "recompute" in str(r["type"]).lower()]
    explicit_est = sum(_row_estimate(r) for r in explicit)
    if explicit:
        candidates.append({
            "type": ", ".join(sorted({str(r["type"]) for r in explicit})),
            "method": "explicit_kernel_name",
            "confidence": "high",
            "count": len(explicit),
            "duration_us": round(sum(float(r["dur_us"]) for r in explicit), 1),
            "recompute_flops": explicit_est,
            "recompute_flops_lo": explicit_est * 0.75,
            "recompute_flops_hi": explicit_est * 1.25,
            "evidence": "kernel Type 显式包含 recompute",
        })

    # Forward/backward call families.  Recompute replays a forward kernel inside
    # the backward, so the forward count exceeds the paired backward count.  Both
    # the Ascend kernel names (snake_case, e.g. causal_conv1d_*_kernel) and the
    # profiler Type labels (CamelCase, e.g. CausalConv1d) appear across exports, so
    # the list is explicit AND augmented with the generic `<Type>` / `<Type>Grad`
    # convention — a model family nobody listed still gets counted.
    pairs = (
        ("FlashAttentionScore", "FlashAttentionScoreGrad"),
        ("SparseFlashAttention", "SparseFlashAttentionGrad"),
        ("SparseFlashMla", "SparseFlashMlaGrad"),
        ("causal_conv1d_fwd_kernel", "causal_conv1d_bwd_kernel"),
        ("chunk_gated_delta_rule_fwd_kernel_h_blockdim64",
         "chunk_gated_delta_rule_bwd_kernel_dhu_blockdim64"),
        ("CausalConv1d", "CausalConv1dBwd"),
        ("ChunkKdaFwdPrepare", "ChunkKdaBwdPrepare"),
        ("ChunkKdaFwdFinalize", "ChunkKdaBwdFinalize"),
        ("ChunkFwdH", "ChunkGatedDeltaRuleBwdDhu"),
        ("_apply_attn_res_kernel", "_apply_attn_res_backward_kernel"),
    )
    present = {str(r["type"]) for r in rows}
    extra = sorted(t for t in present if t + "Grad" in present)
    seen_pairs = set()
    for fwd_type, bwd_type in tuple(pairs) + tuple((t, t + "Grad") for t in extra):
        if (fwd_type, bwd_type) in seen_pairs:
            continue
        seen_pairs.add((fwd_type, bwd_type))
        fwd = [r for r in rows if r["type"] == fwd_type]
        bwd = [r for r in rows if r["type"] == bwd_type]
        if not fwd or not bwd or len(fwd) <= len(bwd):
            continue
        # Replay count is count-based: every layer contributes one forward and one
        # backward; a replayed layer adds a second forward, so excess = fwd - bwd.
        # Charging it at the Type's MEAN duration overstated a real capture 43x
        # because one Type mixed a 6.7 ms real fused-attention call with a 48 us aux
        # variant (mean was dominated by the minority).  MEDIAN is the robust
        # per-call cost here and stays conservative when the replayed variant is a
        # minority — the replay window is not observable from per-call rows anyway.
        excess_count = len(fwd) - len(bwd)
        per_call_us = _median([float(r["dur_us"]) for r in fwd])
        excess_us = excess_count * per_call_us
        per_call_flops = _median([_row_estimate(r) for r in fwd])
        excess_flops = excess_count * per_call_flops
        ratio = len(fwd) / len(bwd)
        formula_backed = all(r.get("flops") for r in fwd)
        band = 0.0 if formula_backed else 0.25
        candidates.append({
            "type": fwd_type,
            "paired_backward_type": bwd_type,
            "method": "forward_backward_excess",
            "confidence": "medium",
            "count": excess_count,
            "forward_count": len(fwd),
            "backward_count": len(bwd),
            "duration_us": round(excess_us, 1),
            "recompute_flops": excess_flops,
            "recompute_flops_lo": excess_flops * (1.0 - band),
            "recompute_flops_hi": excess_flops * (1.0 + band),
            "evidence": (
                f"forward/backward 调用数 {len(fwd)}/{len(bwd)}={ratio:.2f}；"
                f"超额 forward {excess_count} 次 × 单次中位 {per_call_us:,.0f}us"
            ),
        })

    recompute = sum(float(c.get("recompute_flops") or 0.0) for c in candidates)
    recompute_lo = sum(float(c.get("recompute_flops_lo") or 0.0) for c in candidates)
    recompute_hi = sum(float(c.get("recompute_flops_hi") or 0.0) for c in candidates)
    recompute = min(recompute, executed_flops)
    recompute_lo = min(recompute_lo, executed_flops_lo)
    recompute_hi = min(recompute_hi, executed_flops_hi)
    share = recompute / executed_flops if executed_flops > 0 else 0.0
    share_lo = recompute_lo / executed_flops_hi if executed_flops_hi > 0 else 0.0
    share_hi = recompute_hi / executed_flops_lo if executed_flops_lo > 0 else 0.0
    named = [c for c in candidates if c["method"] != "orientation_phase_split"]
    # Granularity here can only prove "*some* forward is replayed" vs "none": the
    # full-vs-selective scope decision lives in parser.derive (caller counts per
    # family, incl. the custom KDA/conv families this module does not model).  The
    # old FLOP-share rule (`share >= 0.18 and len(named) >= 2`) was structurally
    # unreachable whenever the replayed families are unmodeled — it never fired.
    likely = "selective" if named else "off"
    return {
        "available": bool(candidates and executed_flops > 0),
        "assessment": "recompute_enabled" if candidates else "unknown",
        "likely_granularity": likely,
        "confidence": "medium" if candidates else "low",
        "executed_flops": executed_flops,
        "recompute_flops": recompute,
        "recompute_flops_lo": recompute_lo,
        "recompute_flops_hi": recompute_hi,
        "flops_share": share,
        "flops_share_lo": share_lo,
        "flops_share_hi": share_hi,
        "candidates": sorted(
            candidates,
            key=lambda c: float(c.get("recompute_flops") or 0.0),
            reverse=True,
        ),
        "basis": (
            "MatMul 方向/次数相位拆分 + forward/backward 超额调用（按全部计算算子的"
            "每类 Type 中位单次耗时计费）+ 显式 recompute kernel；"
            "未建模 KDA/causal-conv/MC2 融合 MoE 工作量使用同一采集内 "
            "GEMM/Attention 校准后的 MAC/Vector 周期代理。"
        ),
    }


def compute_efficiency(prof) -> Dict[str, Any]:
    kd = prof.kernel_details
    phase = kd.get("Workload Phase")
    optimizer = kd[phase == "optimizer"] if phase is not None else kd.iloc[:0]
    optimizer_us = float(num(optimizer["Duration(us)"]).sum()) if not optimizer.empty else 0.0
    if phase is not None:
        kd = kd[phase != "optimizer"]
    chip = SETTINGS.chip
    if kd.empty:
        return {"available": False, "reason": "kernel_details.csv missing"}

    cols = kd.columns
    dur = num(kd["Duration(us)"]).fillna(0.0).to_numpy()
    types = kd["Type"].astype(str).to_numpy()
    names = kd["Name"].astype(str).to_numpy()
    core = kd["Accelerator Core"].astype(str).to_numpy() if "Accelerator Core" in cols else [""] * len(kd)
    in_sh = kd["Input Shapes"].to_numpy() if "Input Shapes" in cols else [None] * len(kd)
    in_dt = kd["Input Data Types"].to_numpy() if "Input Data Types" in cols else [None] * len(kd)
    out_sh = kd["Output Shapes"].to_numpy() if "Output Shapes" in cols else [None] * len(kd)
    out_dt = kd["Output Data Types"].to_numpy() if "Output Data Types" in cols else [None] * len(kd)

    def colf(name):
        return num(kd[name]).fillna(0.0).to_numpy() if name in cols else [0.0] * len(kd)

    mac_series = num(kd["aic_mac_ratio"]) if "aic_mac_ratio" in cols else None
    mac_r = mac_series.fillna(0.0).to_numpy() if mac_series is not None else [0.0] * len(kd)
    mac_valid = mac_series.notna().to_numpy() if mac_series is not None else [False] * len(kd)
    mte2_r = colf("aic_mte2_ratio")
    mte3_r = colf("aic_mte3_ratio")
    vec_series = num(kd["aiv_vec_ratio"]) if "aiv_vec_ratio" in cols else None
    vec_r = vec_series.fillna(0.0).to_numpy() if vec_series is not None else [0.0] * len(kd)
    vec_valid = vec_series.notna().to_numpy() if vec_series is not None else [False] * len(kd)
    cube_u = colf("cube_utilization(%)")
    mac_time = num(kd["aic_mac_time(us)"]).to_numpy() if "aic_mac_time(us)" in cols else [None] * len(kd)
    vec_time = num(kd["aiv_vec_time(us)"]).to_numpy() if "aiv_vec_time(us)" in cols else [None] * len(kd)

    def active_us(recorded, ratio, duration):
        if recorded is not None and recorded == recorded and recorded >= 0:
            return float(recorded)
        return float(ratio) * duration

    # ---- pre-pass: observed bf16 ceiling -> calibrate an ASSUMED peak ---------
    # A real kernel cannot exceed silicon peak. If the best clean bf16 GEMM's
    # achieved throughput exceeds an *assumed* ChipSpec peak, the assumption is
    # simply too low for this SKU (910B bins vary), so reporting MFU > 100% would
    # be nonsense. Calibrate the effective peak up to the observed ceiling (+2%
    # headroom so the ceiling kernel reads ~98%, not a suspicious flat 100%),
    # bounded at 2x the assumption so a stray shape mis-parse can't masquerade as a
    # giant peak bump. Calibration only ever raises the peak, and only the CUBE
    # bf16 peak (GEMM runs on the cube); the vector peak is left as configured.
    #
    # Two guards stop this from inflating an ALREADY-CORRECT peak:
    #   (1) only bf16/fp16 GEMMs set the ceiling. An fp8/fp4 GEMM does the same
    #       2·M·N·K FLOPs in 1/2 (1/4) the time, so mixing it in would look like a
    #       bf16 peak 2x (4x) too low and scale EVERY cube op's MFU denominator up
    #       — that was the "950DT bf16 reads 864T (= 432x2)" bug.
    #   (2) only assumed chips calibrate. A chip with real datasheet peaks
    #       (assumed=false, e.g. 950DT) is trusted as-is — an fp8 run in the same
    #       step must not rewrite its known-correct bf16 ceiling.
    configured_peak = chip.peak_cube_flops("BF16")
    observed_peak = 0.0
    for i in range(len(kd)):
        if types[i] not in MATMUL_TYPES or float(dur[i]) <= 0:
            continue
        # guard (1): skip non-bf16 GEMMs — i.e. those whose routed cube peak
        # differs from the bf16 peak (fp8 -> 2x, fp4 -> 4x on a chip that configs
        # them). dtype-unknown GEMMs fall back to bf16 and are kept, as before.
        di = (parse_dtypes(in_dt[i])[:1] or [""])[0]
        if chip.peak_cube_flops(di) != configured_peak:
            continue
        f = _estimate_matmul_flops(parse_shapes(in_sh[i]), parse_shapes(out_sh[i]))
        if f:
            observed_peak = max(observed_peak, f / (float(dur[i]) * 1e-6))
    # guard (2): trust real datasheet peaks; only an assumed peak gets calibrated.
    calibrated = chip.assumed and observed_peak > configured_peak
    effective_peak = min(observed_peak * 1.02, configured_peak * 2.0) if calibrated else configured_peak
    peak_scale = effective_peak / configured_peak if configured_peak else 1.0  # >= 1.0 (cube only)
    # For a datasheet-backed (assumed=false) chip, observed > configured cannot be
    # silently accepted: the selected SKU/precision or the FLOP shape model is
    # inconsistent.  Keep raw diagnostics, but fail closed on headline MFU/What-if.
    peak_inconsistent = bool(
        not calibrated
        and effective_peak > 0
        and observed_peak > effective_peak * _PEAK_MEASUREMENT_TOL
    )

    def peak_for(dtype: str, use_cube: bool) -> float:
        # Cube peak gets the observed-ceiling calibration (scale >= 1.0); the
        # vector peak is used as configured — there's no vector flop model yet to
        # calibrate against, so scaling it would be unfounded.
        if use_cube:
            return chip.peak_cube_flops(dtype) * peak_scale
        return chip.peak_vector_flops(dtype)

    rows: List[Dict[str, Any]] = []
    scatter: List[Dict[str, Any]] = []

    skipped_comm = 0
    for i in range(len(kd)):
        d_us = float(dur[i])
        if d_us <= 0:
            continue
        # Communication-dispatch "kernels" (AI_CPU / HCCL / *AicpuKernel) have no
        # compute or memory footprint we can model — they belong to the
        # communication and hidden-overhead views, not the compute-efficiency
        # ranking (nor to the whole-model FLOP-coverage denominator).
        if (str(core[i]).upper() == "AI_CPU"
                or str(types[i]).lower().startswith(("hccl", "hcom"))
                or _is_aicpu_dispatch_type(types[i])):
            skipped_comm += 1
            continue
        d_s = d_us * 1e-6
        shapes_in = parse_shapes(in_sh[i])
        shapes_out = parse_shapes(out_sh[i])
        dt_in = parse_dtypes(in_dt[i])
        dt_out = parse_dtypes(out_dt[i])
        dtype = dt_in[0] if dt_in else (dt_out[0] if dt_out else "BF16")

        # bytes moved (read inputs + write outputs). For matmul kernels, cap any
        # operand equal to the M*N result buffer: an inplace-add GEMM reports its
        # whole accumulator (>> M*N) as both a residual input and the output, so a
        # naive shape sum over-counts HBM traffic and pushes MBU past 100%. Only the
        # M*N block is truly read-modified-written; the activation (M*K) and weight
        # (K*N) stay intact. A plain GEMM has output == M*N, so this is a no-op.
        mn_elems = 0
        matmul_mnk = None
        matmul_phase_eligible = False
        if types[i] in MATMUL_TYPES:
            matmul_mnk = _matmul_mnk(shapes_in, shapes_out)
            if matmul_mnk:
                mn_elems = matmul_mnk[0] * matmul_mnk[1] * matmul_mnk[3]
                mats_in = [s for s in shapes_in if len(s) >= 2]
                matmul_phase_eligible = bool(
                    matmul_mnk[3] == 1
                    and len(mats_in) >= 2
                    and len(mats_in[0]) == len(mats_in[1]) == 2
                    and any(len(s) == 2 for s in shapes_out)
                )
        out_numels = {numel(s) for s in shapes_out}
        b_bytes = 0
        for j, sh in enumerate(shapes_in):
            dt = dt_in[j] if j < len(dt_in) else (dt_in[-1] if dt_in else dtype)
            ne = numel(sh)
            # inplace residual: same size as the oversized output and larger than
            # the M*N result -> only M*N is actually touched by the accumulate.
            if mn_elems and ne > mn_elems and ne in out_numels:
                ne = mn_elems
            b_bytes += ne * dtype_bytes(dt)
        for j, sh in enumerate(shapes_out):
            dt = dt_out[j] if j < len(dt_out) else (dt_out[-1] if dt_out else dtype)
            ne = numel(sh)
            if mn_elems and ne > mn_elems:
                ne = mn_elems
            b_bytes += ne * dtype_bytes(dt)

        is_matmul = types[i] in MATMUL_TYPES
        is_attention = types[i] in ATTENTION_TYPES
        is_custom_model = _is_custom_model_compute_type(types[i])
        is_sparse_discrete = types[i] in SPARSE_DISCRETE_TRAFFIC_TYPES
        core_u = str(core[i]).upper()
        is_vector = "VECTOR" in core_u or "AIV" in core_u
        if is_matmul:
            flops = _estimate_matmul_flops(shapes_in, shapes_out)
        elif types[i] in SPARSE_ATTENTION_TYPES:
            flops = _estimate_sparse_flash_attention_flops(
                shapes_in,
                shapes_out,
                types[i] in ATTENTION_GRAD_TYPES,
                sparse_mode=SETTINGS.sparse_attention_mode,
                sparse_block_size=(1 if chip.name == "Ascend 950DT" else None),
            )
        elif types[i] in LIGHTNING_INDEXER_FWD_TYPES:
            flops = _estimate_lightning_indexer_flops(
                shapes_in, shapes_out, sparse_mode=SETTINGS.sparse_attention_mode,
            )
        elif types[i] in LIGHTNING_INDEXER_KL_GRAD_TYPES:
            flops = _estimate_sparse_lightning_indexer_grad_kl_loss_flops(
                shapes_in, shapes_out, sparse_mode=SETTINGS.sparse_attention_mode,
            )
        elif is_attention:
            flops = _estimate_attention_flops(
                shapes_in, shapes_out, types[i] in ATTENTION_GRAD_TYPES)
        elif _is_kda_model_compute_type(types[i]):
            # KDA family (chunked gated-delta-rule attention + causal conv1d +
            # gate/cumsum helpers): semantic formulas so these key kernels count
            # toward whole-model FLOP coverage.  Falls back to the counter proxy
            # when shapes are missing (formula returns None).
            flops = _estimate_kda_flops(types[i], shapes_in, shapes_out)
        elif _is_mc2_grouped_gemm_type(types[i]):
            # MC2 fused AllToAllv + GroupedMatMul: the recorded activation and
            # per-expert weight give the exact 2·M·N·K Cube work, so the model
            # compute is formula-backed instead of counter-only.  The paired
            # `*AicpuKernel` launch wrapper is a separate Type and never reaches
            # this branch (dispatch rows are filtered out above).
            flops = _estimate_grouped_gemm_flops(shapes_in, shapes_out)
        elif is_custom_model:
            # Other recognized model-compute families (KDA is handled above) have
            # no shape-safe semantic FLOP formula yet.  Keep them in the model
            # coverage denominator and use only the same-capture counter estimate.
            flops = None
        elif is_vector:
            # VECTOR-core elementwise / norm / optimizer kernels: approximate FLOPs so
            # their MFU reads against the VECTOR peak and the headroom floor respects
            # max(vector-compute, memory) — not the memory roofline alone, which
            # over-states reclaim for FLOP-dense vector ops (AdamW / RMSNorm / …).
            flops = _estimate_vector_flops(types[i], shapes_in, shapes_out)
        else:
            flops = None

        # Sparse attention's useful FLOP numerator is reconstructible, but the
        # tensor list is not its physical HBM traffic: selected K/V blocks are
        # gathered repeatedly and gradient scatter traffic is hidden inside the
        # fused kernel.  Treating the one-time tensor footprint as bytes moved
        # would invent an MBU and an unattainable roofline speedup.
        if is_sparse_discrete:
            b_bytes = 0

        # Peak routing: matmul / fused-attention kernels run on the CUBE unit;
        # pure vector-core ops (AI_VECTOR_CORE / MIX_AIV: RMSNorm/SwiGlu/Cast/...)
        # run on the VECTOR unit; any other AI_CORE / MIX_AIC op defaults to cube.
        # On the 950DT cube bf16 (432T) is ~8x vector bf16 (54T), so picking the
        # right unit's peak is what makes per-op MFU/waste physically meaningful.
        use_cube = is_matmul or is_attention or (not is_vector)
        peak_flops = peak_for(dtype, use_cube)
        # Counter-derived executed-work proxy.  aic_mac_ratio / aiv_vec_ratio are
        # active-cycle shares rather than semantic FLOP counts, so this is NOT a
        # formula replacement by itself.  Later we calibrate it against the
        # simultaneously captured GEMM/Attention rows whose FLOPs are known, then
        # apply that capture-local scale to unmodeled KDA/causal-conv kernels.
        mac_active = active_us(mac_time[i], mac_r[i], d_us)
        vec_active = active_us(vec_time[i], vec_r[i], d_us)
        counter_proxy_raw = (
            (mac_active * peak_for(dtype, True)
             + vec_active * peak_for(dtype, False)) * 1e-6
            if bool(mac_valid[i] or vec_valid[i]) else None
        )

        achieved_flops = (flops / d_s) if flops else None
        achieved_bw = (b_bytes / d_s) if b_bytes else 0.0
        mfu = (achieved_flops / peak_flops) if achieved_flops else None
        mbu_raw = (achieved_bw / chip.hbm_bandwidth) if achieved_bw else None
        mbu = min(mbu_raw, 1.0) if mbu_raw is not None else None
        mbu_over_physical = bool(
            mbu_raw is not None and mbu_raw > _PEAK_MEASUREMENT_TOL
        )

        # We can only credibly model "ideal time" (and thus wasted time) when we
        # have a FLOP estimate (matmul / fused attention) OR the kernel is a
        # vector-core memory op. Other cube/MIX ops without a FLOP model would
        # look 100% wasted under a memory-only roofline — so we leave them unscored.
        modeled = (
            ((flops is not None) or (is_vector and b_bytes > 0))
            and not is_sparse_discrete
            # A custom model kernel without a semantic FLOP formula must not be
            # ranked against a memory-only roofline as if its arithmetic were 0.
            and not (is_custom_model and flops is None)
        )
        t_compute = (flops / peak_flops) if flops else 0.0
        t_mem = (b_bytes / chip.hbm_bandwidth) if b_bytes else 0.0
        # Ceiling-aware reclaim: matmul / FA / FAG have a realistic MFU ceiling
        # (<100%) below which further tuning isn't worth it. reclaim_us is the time
        # recoverable by optimizing *to that ceiling* — 0 once a kernel is already
        # at/above it (running at MFU>=ceiling means dur <= compute_time/ceiling).
        # Non-ceiling modeled ops optimize toward the 100% roofline, so their
        # reclaim == wasted_us.
        op_class = _op_class(types[i]) if flops is not None and not is_sparse_discrete else None
        ceiling = chip.mfu_ceiling(op_class)
        if modeled:
            ideal_us = max(t_compute, t_mem) * 1e6
            efficiency = max(0.0, min(ideal_us / d_us, 1.0)) if d_us > 0 else 0.0
            wasted_us = max(0.0, d_us - ideal_us)
            if flops is not None and t_compute >= t_mem:
                bound = "vector" if is_vector else "compute"  # vector→green, cube→blue
            else:
                bound = "memory"
            if ceiling:
                floor_us = max(t_compute / ceiling, t_mem) * 1e6
                reclaim_us = max(0.0, d_us - floor_us)
            else:
                reclaim_us = wasted_us
        else:
            efficiency = None
            wasted_us = 0.0
            reclaim_us = 0.0
            bound = _bound_from_ratios(mac_r[i], mte2_r[i], vec_r[i])

        # Overhead-bound: MFU & MBU both ~0 → time is launch/scalar/dispatch, not on the
        # roofline, so its "reclaim to 100%" is bogus. Zero the reclaim (drops it from the
        # optimization ranking) and surface it in overhead_bound_ops instead.
        overhead_bound = bool(modeled and (mfu or 0.0) < _OVERHEAD_EFF
                              and (mbu or 0.0) < _OVERHEAD_EFF)
        if overhead_bound:
            reclaim_us = 0.0

        # Custom KDA formula rows: the semantic FLOP formula is an
        # order-of-magnitude floor (not an instruction-accurate count), so
        # duration−ideal reclaim is far too sensitive to formula error to rank
        # as an optimization candidate — a 2× formula error turns into a 2×
        # "reclaimable" lie.  Keep the MFU/label display (formula floor vs
        # counter estimate) but zero the reclaim/waste so these rows never
        # surface in the optimization ranking or Roofline收益排行.
        if is_custom_model and flops is not None:
            reclaim_us = 0.0
            wasted_us = 0.0

        rows.append(
            {
                "name": names[i],
                "type": types[i],
                "core": core[i],
                "dur_us": d_us,
                "dtype": dtype,
                "flops": flops,
                "bytes": b_bytes,
                "peak_flops": peak_flops,
                "mfu": mfu,
                "mbu": mbu,
                "mbu_raw": mbu_raw,
                "mbu_over_physical": mbu_over_physical,
                "matmul_mnk": matmul_mnk if matmul_phase_eligible else None,
                "efficiency": efficiency,
                "wasted_us": wasted_us,
                "reclaim_us": reclaim_us,
                "overhead_bound": overhead_bound,
                "roofline_eligible": modeled,
                "op_class": op_class,
                "ceiling": ceiling,
                "bound": bound,
                "ai": (flops / b_bytes) if (flops and b_bytes) else None,
                "achieved_tflops": (achieved_flops / 1e12) if achieved_flops else None,
                "mac_ratio": float(mac_r[i]),
                "mac_active_us": mac_active,
                "mac_ratio_available": bool(mac_valid[i]),
                "vec_ratio": float(vec_r[i]),
                "vec_ratio_available": bool(vec_valid[i]),
                "counter_proxy_flops_raw": counter_proxy_raw,
                "mte2_ratio": float(mte2_r[i]),
                "cube_util": float(cube_u[i]),
            }
        )
        if flops and b_bytes:
            ai = flops / b_bytes
            # Double-normalize each point against its OWN routed peak (cube vs
            # vector, dtype-aware): x_norm = AI / ridge = AI·BW / peak, y = MFU.
            # Every dtype/unit then collapses onto one universal roof y=min(x,1).
            scatter.append(
                {
                    "name": names[i],
                    "type": types[i],
                    "ai": ai,
                    "tflops": achieved_flops / 1e12,
                    "dur_us": d_us,
                    "bound": bound,
                    "dtype": dtype,
                    "peak_tflops": peak_flops / 1e12,
                    "mfu": mfu,
                    "x_norm": ai * chip.hbm_bandwidth / peak_flops,
                }
            )

    # ---- aggregates -------------------------------------------------------
    by_type: Dict[str, Dict[str, Any]] = {}
    for r in rows:
        t = by_type.setdefault(
            r["type"],
            {"type": r["type"], "count": 0, "dur_us": 0.0, "flops": 0.0,
             "bytes": 0.0, "wasted_us": 0.0, "reclaim_us": 0.0,
             "peak_time": 0.0, "mac_ratio_time": 0.0,
             "mac_ratio_dur_us": 0.0, "dt_dur": {}},
        )
        t["count"] += 1
        t["dur_us"] += r["dur_us"]
        t["flops"] += r["flops"] or 0.0
        t["bytes"] += r["bytes"] or 0.0
        t["wasted_us"] += r["wasted_us"]
        t["reclaim_us"] += r["reclaim_us"]
        # Aggregate-MFU denominator consistent with per-kernel routing: sum each
        # kernel's OWN routed-peak × time, so type MFU = Σflops / Σ(peak·t) rather
        # than dividing everything by the single cube-bf16 peak (which under-reads
        # vector-heavy types). dt_dur tracks the duration-dominant dtype.
        if r["flops"]:
            t["peak_time"] += r["peak_flops"] * (r["dur_us"] * 1e-6)
        if r["mac_ratio_available"]:
            t["mac_ratio_time"] += r["mac_ratio"] * r["dur_us"]
            t["mac_ratio_dur_us"] += r["dur_us"]
        t["dt_dur"][r["dtype"]] = t["dt_dur"].get(r["dtype"], 0.0) + r["dur_us"]

    type_rows = []
    for t in by_type.values():
        d_s = t["dur_us"] * 1e-6
        mfu = (t["flops"] / t["peak_time"]) if t["peak_time"] else None
        mbu_raw = (t["bytes"] / d_s / SETTINGS.chip.hbm_bandwidth) if (d_s and t["bytes"]) else None
        mbu = min(mbu_raw, 1.0) if mbu_raw is not None else None
        mac_ratio_weighted = (
            t["mac_ratio_time"] / t["mac_ratio_dur_us"]
            if t["mac_ratio_dur_us"] > 0 else None
        )
        mac_ratio_coverage_pct = (
            100.0 * t["mac_ratio_dur_us"] / t["dur_us"]
            if t["dur_us"] > 0 else 0.0
        )
        dom_dtype = max(t["dt_dur"].items(), key=lambda kv: kv[1])[0] if t["dt_dur"] else None
        type_rows.append(
            {
                "type": t["type"],
                "count": t["count"],
                "dur_us": round(t["dur_us"], 1),
                "dtype": dom_dtype,
                "mfu": round(mfu, 4) if mfu is not None else None,
                "mac_ratio_weighted": (
                    round(mac_ratio_weighted, 4)
                    if mac_ratio_weighted is not None else None
                ),
                "mac_ratio_coverage_pct": round(mac_ratio_coverage_pct, 1),
                "mbu": round(mbu, 4) if mbu is not None else None,
                "mbu_raw": round(mbu_raw, 4) if mbu_raw is not None else None,
                "mbu_over_physical": bool(
                    mbu_raw is not None and mbu_raw > _PEAK_MEASUREMENT_TOL
                ),
                "wasted_us": round(t["wasted_us"], 1),
                "reclaim_us": round(t["reclaim_us"], 1),
            }
        )
    type_rows.sort(key=lambda x: x["dur_us"], reverse=True)

    # Overhead-bound ops (MFU & MBU both ~0): launch/scalar/dispatch dominated, so the
    # roofline reclaim doesn't apply — they were given reclaim=0 above (kept out of the
    # ranking). Aggregate by type so the time isn't silently dropped; it needs kernel-
    # level work (fusion / larger tiles / fewer launches), not a roofline target.
    ob_acc: Dict[str, Dict[str, Any]] = {}
    for r in rows:
        if not r.get("overhead_bound"):
            continue
        a = ob_acc.get(r["type"])
        if a is None:
            a = ob_acc[r["type"]] = {"type": r["type"], "count": 0, "dur_us": 0.0,
                                     "dtype": r["dtype"], "mfu_w": 0.0, "mbu_w": 0.0}
        a["count"] += 1
        a["dur_us"] += r["dur_us"]
        a["mfu_w"] += (r["mfu"] or 0.0) * r["dur_us"]
        a["mbu_w"] += (r["mbu"] or 0.0) * r["dur_us"]
    overhead_bound_ops = []
    for a in sorted(ob_acc.values(), key=lambda x: x["dur_us"], reverse=True)[:30]:
        d = a["dur_us"]
        overhead_bound_ops.append({
            "type": a["type"], "count": a["count"], "dur_us": round(d, 1),
            "dtype": a["dtype"],
            "mfu": round(a["mfu_w"] / d, 4) if d else None,
            "mbu": round(a["mbu_w"] / d, 4) if d else None,
        })
    overhead_bound_us = round(sum(a["dur_us"] for a in ob_acc.values()), 1)

    # Rank optimization candidates by ceiling-aware reclaimable time and drop the
    # ones already at/above their MFU ceiling (reclaim ~ 0): there's no point
    # ranking a matmul that's already saturating the cube. Non-ceiling ops keep
    # their vs-roofline gap (reclaim == wasted_us), so memory/vector candidates
    # still surface.
    rankable = [r for r in rows if r.get("reclaim_us", 0.0) > 0.05]
    top_opt = sorted(rankable, key=lambda r: r["reclaim_us"], reverse=True)[:25]
    for r in top_opt:
        # Post-optimization MFU/MBU: closing the reclaimable gap means running at
        # the kernel's floor duration (dur - reclaim). MFU/MBU scale inversely with
        # duration (FLOPs/bytes fixed), so by dur/floor; matmul/FA/FAG cap at their
        # ceiling, others at 1.0. Lets a row read "MFU 65→70" toward the FAG cap.
        floor_us = max(r["dur_us"] - r["reclaim_us"], 1e-9)
        scale = r["dur_us"] / floor_us
        cap = r.get("ceiling") or 1.0
        r["mfu_after"] = min(r["mfu"] * scale, cap) if r.get("mfu") else None
        r["mbu_after"] = min(r["mbu"] * scale, 1.0) if r.get("mbu") else None
        for k in ("flops", "bytes", "mfu", "mbu", "mbu_raw", "ai", "achieved_tflops",
                  "mfu_after", "mbu_after", "reclaim_us", "wasted_us"):
            if isinstance(r.get(k), float):
                r[k] = round(r[k], 4) if r[k] and r[k] < 1 else (round(r[k], 1) if r[k] else r[k])

    # ---- "算子极致优化" aggregate -----------------------------------------
    # Total reclaimable time if every modeled compute kernel (matmul / FA / FAG)
    # is tuned up to its MFU ceiling, with a per-class breakdown and how many are
    # already at/above their ceiling (left untouched). This is the save_us for the
    # What-if lever in theoretical() — disjoint from comm/free (it's pure compute),
    # so it stacks. The honest figure: matmul usually sits at its 95% ceiling, so
    # the gain comes from FA/FAG headroom rather than a naive "everything to 100%".
    oc_classes = ("matmul", "attention", "attention_grad")
    oc_by = {c: {"reclaim_us": 0.0, "n": 0, "n_capped": 0,
                 "ceiling": chip.mfu_ceiling(c)} for c in oc_classes}
    for r in rows:
        oc = r.get("op_class")
        if not oc or oc not in oc_by:
            continue
        b = oc_by[oc]
        b["n"] += 1
        b["reclaim_us"] += r["reclaim_us"]
        if r["reclaim_us"] <= 0.05:
            b["n_capped"] += 1
    oc_total = sum(b["reclaim_us"] for b in oc_by.values())
    for b in oc_by.values():
        b["reclaim_us"] = round(b["reclaim_us"], 1)
    op_ceiling_opt = {
        "total_reclaim_us": round(oc_total, 1),
        "by_class": oc_by,
        "n_modeled": sum(b["n"] for b in oc_by.values()),
        "n_capped": sum(b["n_capped"] for b in oc_by.values()),
        "ceilings": {c: chip.mfu_ceiling(c) for c in oc_classes},
    }

    # Headline matmul MFU stays *pure GEMM* (the calibration anchor); fused
    # attention has its own per-type MFU row and is excluded here so the headline
    # keeps its meaning. `flops_rows` (incl. attention) drives the modeled-kernel
    # count and the Roofline scatter.
    flops_rows = [r for r in rows if r["flops"]]
    mm = [r for r in flops_rows if r["type"] in MATMUL_TYPES]
    mm_flops = sum(r["flops"] for r in mm)
    matmul_training_flops = _estimate_matmul_training_flops(rows)
    # Divide by each matmul's OWN routed peak (cube + dtype-aware), summed as
    # peak·time — the same convention as by_type. A single bf16 effective_peak
    # here would divide an fp8 GEMM's ~2x throughput by the bf16 peak and read a
    # non-physical >100% on any mixed bf16+fp8 step (which then misfires the
    # peak_underestimated advisory in metrics/core.py).
    mm_peak_time = sum(r["peak_flops"] * r["dur_us"] * 1e-6 for r in mm if r.get("peak_flops"))
    matmul_mfu_raw = (mm_flops / mm_peak_time) if mm_peak_time else None
    # "assumed" strips the calibration scale (peak_scale) so a genuinely
    # underestimated *assumed* peak still trips >1 (the calibration trigger) —
    # now per-kernel vs each one's un-calibrated peak, not a single bf16 peak.
    mm_assumed_peak_time = (mm_peak_time / peak_scale) if peak_scale else 0.0
    matmul_mfu_assumed = (mm_flops / mm_assumed_peak_time) if mm_assumed_peak_time else None
    # Model compute MFU = GEMM + fused attention combined, each divided by its OWN
    # routed peak·time. matmul_mfu stays pure-GEMM (the calibration anchor); this adds
    # attention so the headline reflects the whole model's compute, not just the GEMMs.
    # (End-to-end/step MFU — incl. comm/idle — is theoretical.step_mfu, a different cut.)
    mc = [r for r in flops_rows if (r["type"] in MATMUL_TYPES or r["type"] in ATTENTION_TYPES
                                  or _is_custom_model_compute_type(r["type"]))]
    mc_flops = sum(r["flops"] for r in mc)
    mc_peak_time = sum(r["peak_flops"] * r["dur_us"] * 1e-6 for r in mc if r.get("peak_flops"))
    model_mfu_compute_partial = (mc_flops / mc_peak_time) if mc_peak_time else None

    # Duration-weighted coverage over recognized model-compute families.  A known
    # but unsupported dominant op must make the model MFU unavailable instead of
    # letting a GEMM-only numerator masquerade as whole-model efficiency.
    supported_model_rows = [
        r for r in rows
        if r["type"] in MATMUL_TYPES or r["type"] in ATTENTION_TYPES
    ]
    model_rows = [
        r for r in rows
        if (r["type"] in MATMUL_TYPES or r["type"] in ATTENTION_TYPES
            or _is_custom_model_compute_type(r["type"]))
    ]
    modeled_model_rows = [r for r in model_rows if r.get("flops")]
    model_total_us = sum(r["dur_us"] for r in model_rows)
    model_modeled_us = sum(r["dur_us"] for r in modeled_model_rows)
    # Calibrate raw MAC/Vector active-cycle work against the formula-backed rows
    # from THIS capture.  This converts custom KDA/causal-conv/MC2 fused-MoE cycles into an
    # empirical FLOP estimate without importing model-family constants.
    counter_reference_rows = [
        r for r in supported_model_rows
        if r.get("flops") and float(r.get("counter_proxy_flops_raw") or 0.0) > 0
    ]
    counter_reference_flops = sum(float(r["flops"]) for r in counter_reference_rows)
    counter_reference_proxy = sum(
        float(r.get("counter_proxy_flops_raw") or 0.0) for r in counter_reference_rows
    )
    counter_scale = (
        counter_reference_flops / counter_reference_proxy
        if counter_reference_proxy > 0 else 1.0
    )
    # Every uncovered recognized family needs a proxy, including shape-less
    # attention/GEMM. Formula-backed custom families are already in mc_flops below.
    custom_model_rows = [r for r in model_rows if not r.get("flops")]
    custom_counter_rows = [
        r for r in custom_model_rows
        if float(r.get("counter_proxy_flops_raw") or 0.0) > 0
    ]
    custom_counter_raw = sum(
        float(r.get("counter_proxy_flops_raw") or 0.0) for r in custom_counter_rows
    )
    custom_counter_est = custom_counter_raw * counter_scale
    counter_coverage_pct = (
        100.0 * sum(float(r["dur_us"]) for r in custom_counter_rows)
        / sum(float(r["dur_us"]) for r in custom_model_rows)
        if custom_model_rows else 100.0
    )
    estimate_available = bool(
        mc_flops > 0 and (not custom_model_rows or len(custom_counter_rows) == len(custom_model_rows))
        and (not custom_model_rows or counter_reference_proxy > 0)
        and not peak_inconsistent
    )
    # The point estimate uses the observed same-capture scale.  ±25% on the
    # custom-family contribution is kept as an explicit epistemic band because
    # active-cycle ratios are not semantic FLOP counters.
    formula_model_flops = sum(float(r["flops"]) for r in modeled_model_rows)
    estimated_flops = formula_model_flops + custom_counter_est
    estimated_flops_lo = formula_model_flops + custom_counter_est * 0.75
    estimated_flops_hi = formula_model_flops + custom_counter_est * 1.25
    model_peak_time = sum(
        float(r.get("peak_flops") or 0.0) * float(r["dur_us"]) * 1e-6
        for r in model_rows
    )
    model_mfu_compute_estimated = (
        estimated_flops / model_peak_time
        if estimate_available and model_peak_time > 0 else None
    )
    model_mfu_compute_estimated_lo = (
        estimated_flops_lo / model_peak_time
        if estimate_available and model_peak_time > 0 else None
    )
    model_mfu_compute_estimated_hi = (
        estimated_flops_hi / model_peak_time
        if estimate_available and model_peak_time > 0 else None
    )
    for type_row in type_rows:
        family = [r for r in model_rows if r["type"] == type_row["type"]
                  and (_is_custom_model_compute_type(r["type"]) or not r.get("flops"))]
        if not family:
            continue
        family_proxy = sum(
            float(r.get("counter_proxy_flops_raw") or 0.0) for r in family
        )
        family_peak_time = sum(
            float(r.get("peak_flops") or 0.0) * float(r["dur_us"]) * 1e-6
            for r in family
        )
        estimated = (
            family_proxy * counter_scale / family_peak_time
            if family_proxy > 0 and family_peak_time > 0 and counter_reference_proxy > 0
            and not peak_inconsistent else None
        )
        type_row["mfu_estimated"] = round(estimated, 4) if estimated is not None else None
        type_row["mfu_estimate_basis"] = (
            "same_capture_counter_calibration" if estimated is not None else None
        )
    # The public MAC label is deliberately GEMM+Attn, so expanding whole-model
    # FLOP coverage must not silently change its numerator or counter coverage.
    model_mac_rows = [r for r in supported_model_rows if r["mac_ratio_available"]]
    model_mac_dur_us = sum(r["dur_us"] for r in model_mac_rows)
    model_mac_active_us = sum(
        r["mac_ratio"] * r["dur_us"] for r in model_mac_rows
    ) if model_mac_dur_us > 0 else None
    model_compute_mac_ratio_weighted = (
        model_mac_active_us / model_mac_dur_us
        if model_mac_dur_us > 0 else None
    )
    model_mac_coverage_pct = (
        100.0 * model_mac_dur_us / sum(r["dur_us"] for r in supported_model_rows)
        if supported_model_rows else 0.0
    )
    flop_coverage = (model_modeled_us / model_total_us) if model_total_us else 0.0
    # Duration coverage is informative, not FLOP completeness: a short unknown
    # kernel can still carry substantial work and must not vanish from MFU.
    flop_model_complete = bool(model_rows and len(modeled_model_rows) == len(model_rows))
    recompute_estimate = _estimate_recompute_from_rows(
        # ALL compute rows, not just the FLOP-modelled families: a replay executes
        # the same forward kernels, and the cheap non-model families (RMSNorm,
        # activations, MoE routing, conv) are exactly where real captures replay.
        # Restricting this to model_rows made the measured replay a small fraction
        # of what the kernels actually show.
        rows,
        matmul_training_flops,
        counter_scale,
        estimated_flops,
        estimated_flops_lo,
        estimated_flops_hi,
    ) if estimate_available else {"available": False, "reason": "counter estimate unavailable"}
    attention_rows = [r for r in rows if r["type"] in ATTENTION_TYPES]
    attention_total_us = sum(r["dur_us"] for r in attention_rows)
    attention_modeled_us = sum(r["dur_us"] for r in attention_rows if r.get("flops"))
    attention_coverage = (attention_modeled_us / attention_total_us) if attention_total_us else None

    # Exact sparse-training phase anchor.  SFA fwd/grad counts identify repeated
    # activation forwards.  LightningIndexer follows the same pattern, but its
    # fused GradKLLoss call is real auxiliary-loss/backward work and must not be
    # stripped merely because checkpointing executes it in a grad-enabled replay.
    attention_training_flops: Dict[str, Any] = {
        "available": False,
        "reason": "requires uniform, fully modeled sparse-attention phase rows",
    }
    sparse_fwd_rows = [r for r in attention_rows if r["type"] == "SparseFlashAttention"]
    sparse_grad_rows = [r for r in attention_rows if r["type"] == "SparseFlashAttentionGrad"]
    indexer_fwd_rows = [r for r in attention_rows if r["type"] in LIGHTNING_INDEXER_FWD_TYPES]
    indexer_grad_rows = [
        r for r in attention_rows if r["type"] in LIGHTNING_INDEXER_KL_GRAD_TYPES
    ]
    sparse_training_types = SPARSE_ATTENTION_TYPES | DSA_INDEXER_TYPES
    only_sparse_family = bool(attention_rows) and all(
        r["type"] in sparse_training_types for r in attention_rows
    )
    all_sparse_modeled = all(r.get("flops") for r in attention_rows)
    indexer_phase_complete = bool(indexer_fwd_rows) == bool(indexer_grad_rows)
    if (
        only_sparse_family and sparse_fwd_rows and sparse_grad_rows
        and all_sparse_modeled and indexer_phase_complete
    ):
        fwd_values = [float(r["flops"]) for r in sparse_fwd_rows]
        grad_values = [float(r["flops"]) for r in sparse_grad_rows]
        fwd_uniform = max(fwd_values) <= min(fwd_values) * (1.0 + 1e-9)
        grad_uniform = max(grad_values) <= min(grad_values) * (1.0 + 1e-9)
        fwd_count = len(fwd_values)
        grad_count = len(grad_values)
        pass_ratio = fwd_count / grad_count
        indexer_fwd_values = [float(r["flops"]) for r in indexer_fwd_rows]
        indexer_grad_values = [float(r["flops"]) for r in indexer_grad_rows]
        indexer_uniform = bool(
            not indexer_fwd_values
            or (
                max(indexer_fwd_values) <= min(indexer_fwd_values) * (1.0 + 1e-9)
                and max(indexer_grad_values) <= min(indexer_grad_values) * (1.0 + 1e-9)
            )
        )
        indexer_pass_ratio = (
            len(indexer_fwd_values) / len(indexer_grad_values)
            if indexer_grad_values else 1.0
        )
        if (
            fwd_uniform and grad_uniform and pass_ratio >= 1.0
            and indexer_uniform and indexer_pass_ratio >= 1.0
        ):
            fwd_per_call = fwd_values[0]
            grad_per_call = grad_values[0]
            recompute_multiplier = pass_ratio - 1.0
            bwd_fwd_ratio = grad_per_call / fwd_per_call
            executed_fwd_flops = sum(fwd_values)
            executed_bwd_flops = sum(grad_values)
            sfa_recompute_flops = max(fwd_count - grad_count, 0) * fwd_per_call
            indexer_fwd_flops = sum(indexer_fwd_values)
            indexer_grad_flops = sum(indexer_grad_values)
            indexer_recompute_flops = (
                max(len(indexer_fwd_values) - len(indexer_grad_values), 0)
                * indexer_fwd_values[0]
                if indexer_fwd_values else 0.0
            )
            recompute_fwd_flops = sfa_recompute_flops + indexer_recompute_flops
            executed_attention_flops = sum(float(r["flops"]) for r in attention_rows)
            indexer_model_flops = (
                indexer_fwd_flops + indexer_grad_flops - indexer_recompute_flops
            )
            recompute_share = (
                recompute_fwd_flops / executed_attention_flops
                if executed_attention_flops > 0 else 0.0
            )
            attention_training_flops = {
                "available": True,
                "scope": ("SparseFlashAttention+DSAIndexer"
                          if indexer_fwd_values else "SparseFlashAttention"),
                "source": "modeled_sparse_training_phase_counts",
                "forward_count": fwd_count,
                "backward_count": grad_count,
                "forward_pass_ratio": pass_ratio,
                "recompute_forward_multiplier": recompute_multiplier,
                "forward_flops_per_call": fwd_per_call,
                "backward_flops_per_call": grad_per_call,
                "backward_forward_flop_ratio": bwd_fwd_ratio,
                "executed_forward_flops": executed_fwd_flops,
                "executed_backward_flops": executed_bwd_flops,
                "executed_attention_flops": executed_attention_flops,
                "sfa_recompute_forward_flops": sfa_recompute_flops,
                "indexer_forward_count": len(indexer_fwd_values),
                "indexer_grad_loss_count": len(indexer_grad_values),
                "indexer_forward_flops": indexer_fwd_flops,
                "indexer_grad_loss_flops": indexer_grad_flops,
                "indexer_recompute_forward_flops": indexer_recompute_flops,
                "indexer_model_flops": indexer_model_flops,
                "recompute_forward_flops": recompute_fwd_flops,
                "attention_recompute_flops_share": recompute_share,
            }
    unmodeled_acc: Dict[str, Dict[str, Any]] = {}
    for r in model_rows:
        if r.get("flops"):
            continue
        a = unmodeled_acc.setdefault(r["type"], {"count": 0, "dur_us": 0.0})
        a["count"] += 1
        a["dur_us"] += r["dur_us"]
    unmodeled_flop_types = {
        typ: {"count": a["count"], "dur_us": round(a["dur_us"], 1)}
        for typ, a in sorted(unmodeled_acc.items(), key=lambda kv: kv[1]["dur_us"], reverse=True)
    }
    efficiency_reliable = bool(flop_model_complete and not peak_inconsistent)
    matmul_mfu = matmul_mfu_raw if not peak_inconsistent else None
    model_mfu_compute = model_mfu_compute_partial if efficiency_reliable else None
    sparse_present = any(r["type"] in SPARSE_ATTENTION_TYPES for r in model_rows)
    sparse_flops_modeled = any(
        r["type"] in SPARSE_ATTENTION_TYPES and r.get("flops") for r in model_rows
    )
    dsa_indexer_present = any(r["type"] in DSA_INDEXER_TYPES for r in model_rows)
    dsa_indexer_modeled = any(
        r["type"] in DSA_INDEXER_TYPES and r.get("flops") for r in model_rows
    )
    flop_model_notes = []
    if sparse_flops_modeled:
        flop_model_notes.append(
            "SparseFlashAttention 按 token-wise 稀疏注意力（sparse_indices 给出每个 query "
            "选中的 topk 个 key，950DT sparse_block_size=1）计算有效稀疏 Cube FLOPs："
            "有效 query/head-key 对 = min(topk, 因果可见 key 数)。profiler CSV 未记录 "
            "sparse_mode：显式给出 LLMINSIGHT_SPARSE_MODE=3 时按 right-down causal 精确计数；"
            "未给出时仅在形状可证因果上限不生效（Sk-Sq+1 ≥ topk，首个 query 已可看到 topk 个 key，"
            "任何 causal 掩码下都恰好选 topk 个）时接受 Sq·topk，否则 fail-closed。"
            "不含 Softmax/Gather/Scatter 与 tiling padding。因 CSV 无法重建离散访存，"
            "该算子不参与 MBU、roofline 与可回收时延估算。"
        )
    elif sparse_present:
        flop_model_notes.append(
            "检测到 SparseFlashAttention，但 shape 未闭合 token-wise 稀疏契约"
            "（q/k/v + topk indices + RoPE 对 + 双 softmax stats），或因果上限会生效"
            "而 sparse_mode 未知 → 不猜测；确认本次运行后可显式设置 "
            "LLMINSIGHT_SPARSE_MODE=3，仅在 950DT token-wise（sparse_block_size=1）语义下"
            "启用有效 FLOPs。"
        )
    if dsa_indexer_modeled:
        flop_model_notes.append(
            "LightningIndexer 计入 Index-QK：2·B·Hq·P·Dq（P 为因果可见 query/key 对）；"
            "sparse_mode 未记录时取 right-down causal 下界，与不限上限的 Sq·Sk 相差 ≤ Sq/(2·Sk)。"
            "SparseLightningIndexerGradKLLoss 计入主 Attention QK、Index-QK、dQ、dK。"
            "KL/Softmax/ReLU/dW 与离散 Gather/Scatter 不计入 Cube FLOPs，且因物理访存"
            "不可由 CSV 重建，这两类算子的 MBU、roofline 与可回收时延保持 unavailable。"
        )
    elif dsa_indexer_present:
        flop_model_notes.append(
            "检测到 LightningIndexer/GradKLLoss，但 shape 或 sparse_mode 不足以闭合"
            " useful Cube FLOPs；相关 MFU 按 fail-closed 保持 unavailable。"
        )
    if any(_is_mc2_grouped_gemm_type(str(r["type"])) and r.get("flops") for r in model_rows):
        flop_model_notes.append(
            "AlltoAllvGroupedMatMul / GroupedMatMulAlltoAllv 按 2·M·N·K 计入 Cube FLOPs："
            "M = 通信后 token 数，N/K = expert 权重末两维（K 锚定在激活末维），group 维只切分 M、"
            "不乘进 K；输出 [M,N] 与权重末两维须自洽，否则 fail-closed。"
            "其配对的 AI_CPU 下发包装算子（*AicpuKernel）不计入计算覆盖。"
        )
    if estimate_available and not flop_model_complete:
        flop_model_notes.append(
            "完整模型 MFU(est) = 公式 FLOPs + 同一采集内已建模算子校准后的 "
            f"MAC/Vector 周期代理；公式覆盖 {flop_coverage*100:.1f}%，"
            f"自定义算子计数器覆盖 {counter_coverage_pct:.1f}%，校准系数 "
            f"{counter_scale:.3f}。自定义算子贡献保留 ±25% 区间；该估值用于全局观点，"
            "不用于未知算子的 Roofline 收益。"
        )

    scatter.sort(key=lambda s: s["dur_us"], reverse=True)
    scatter = scatter[:1500]

    # ---- per-name kernel index (for the smart-timeline trace join) -----------
    # Trace slices carry only a name (no shapes), so the timeline looks up
    # type/core/dtype and a representative MFU/MBU by kernel name. flops/bytes are
    # per-call averages; mfu/mbu are duration-weighted over the per-row values —
    # those were already routed to the correct cube/vector peak, so the index stays
    # correct when the chip (and thus those peaks) changes.
    kidx_acc: Dict[str, Dict[str, Any]] = {}
    for r in rows:
        a = kidx_acc.get(r["name"])
        if a is None:
            a = kidx_acc[r["name"]] = {
                "type": r["type"], "core": r["core"], "dtype": r["dtype"],
                "flops": 0.0, "bytes": 0.0, "count": 0, "dur_us": 0.0,
                "mfu_w": 0.0, "mfu_dur": 0.0, "mbu_w": 0.0, "mbu_dur": 0.0,
            }
        a["flops"] += r["flops"] or 0.0
        a["bytes"] += r["bytes"] or 0.0
        a["dur_us"] += r["dur_us"]
        a["count"] += 1
        if r["mfu"] is not None:
            a["mfu_w"] += r["mfu"] * r["dur_us"]
            a["mfu_dur"] += r["dur_us"]
        if r["mbu"] is not None:
            a["mbu_w"] += r["mbu"] * r["dur_us"]
            a["mbu_dur"] += r["dur_us"]
    kernel_index: Dict[str, Dict[str, Any]] = {}
    for name, a in kidx_acc.items():
        kernel_index[name] = {
            "type": a["type"], "core": a["core"], "dtype": a["dtype"],
            "flops": round(a["flops"] / a["count"], 1) if a["flops"] else None,
            "bytes": round(a["bytes"] / a["count"], 1) if a["bytes"] else None,
            # name-average throughput (FLOP/μs, byte/μs): chip-independent, lets the
            # smart-timeline attribute work ∝ each slice's actual duration instead of
            # a fixed per-call average — so a bin's utilization matches the name's MFU
            # regardless of how this instance's duration compares to the average.
            "flops_per_us": round(a["flops"] / a["dur_us"], 3) if (a["flops"] and a["dur_us"]) else None,
            "bytes_per_us": round(a["bytes"] / a["dur_us"], 3) if (a["bytes"] and a["dur_us"]) else None,
            "mfu": round(a["mfu_w"] / a["mfu_dur"], 4) if a["mfu_dur"] else None,
            "mbu": round(a["mbu_w"] / a["mbu_dur"], 4) if a["mbu_dur"] else None,
            "count": a["count"],
        }

    # Do not expose a roofline ranking when the selected denominator is already
    # contradicted by the data.  Raw aggregate diagnostics stay available via
    # *_raw / observed_peak, while user-facing MFU and reclaim fields fail closed.
    public_type_rows = type_rows[:40]
    public_top_opt = top_opt
    public_scatter = scatter
    if peak_inconsistent:
        public_type_rows = [
            {**r, "mfu": None, "reclaim_us": 0.0} for r in public_type_rows
        ]
        public_top_opt = []
        public_scatter = []
        for item in kernel_index.values():
            item["mfu"] = None

    return {
        "available": True,
        "workload_accounting": {
            "optimizer_kernel_count": len(optimizer), "optimizer_us": round(optimizer_us, 1),
            "phase_basis": ((getattr(prof, "meta", {}) or {}).get("framework_semantics") or {}).get("source"),
            "note": "模型 FLOPs 与重计算排除优化器窗口内下发的 kernel；优化器仍计入完整 step 分母和矩阵算力视图。",
        },
        "fused_moe": _fused_moe_summary(prof, model_rows, counter_scale,
                                        counter_reference_proxy > 0 and not peak_inconsistent),
        "kernel_index": kernel_index,
        "chip": {
            "name": chip.name,
            "peak_bf16_tflops": chip.cube_fp16_flops / 1e12,          # CUBE bf16 (matmul-MFU denom)
            "cube_bf16_tflops": chip.cube_fp16_flops / 1e12,          # cube/vector split surfaced
            "vector_bf16_tflops": chip.vector_bf16_flops / 1e12,      # ~8x lower on 950DT
            "effective_peak_tflops": effective_peak / 1e12,           # calibrated cube, used for MFU
            "observed_peak_tflops": round(observed_peak / 1e12, 1),   # measured ceiling
            "calibrated": calibrated,
            "hbm_tbps": chip.hbm_bandwidth / 1e12,
            "hbm_capacity_gb": chip.hbm_capacity_gb,
            "assumed": chip.assumed,
        },
        "roofline_ridge_ai": effective_peak / SETTINGS.chip.hbm_bandwidth,
        "matmul_mfu": round(matmul_mfu, 4) if matmul_mfu else None,
        "matmul_mfu_raw": round(matmul_mfu_raw, 4) if matmul_mfu_raw else None,
        "matmul_mfu_assumed": round(matmul_mfu_assumed, 4) if matmul_mfu_assumed else None,
        # GEMM + fused-attention combined compute MFU — the "model 算力 MFU" headline.
        "model_mfu_compute": round(model_mfu_compute, 4) if model_mfu_compute else None,
        "model_mfu_compute_partial": (round(model_mfu_compute_partial, 4)
                                      if model_mfu_compute_partial else None),
        "model_mfu_compute_estimated": (
            round(model_mfu_compute_estimated, 4)
            if model_mfu_compute_estimated is not None else None
        ),
        "model_mfu_compute_estimated_lo": (
            round(model_mfu_compute_estimated_lo, 4)
            if model_mfu_compute_estimated_lo is not None else None
        ),
        "model_mfu_compute_estimated_hi": (
            round(model_mfu_compute_estimated_hi, 4)
            if model_mfu_compute_estimated_hi is not None else None
        ),
        # Duration-weighted hardware MAC counter over the same GEMM + attention
        # families as model_mfu_compute. Missing counters are excluded, with their
        # duration reflected separately in the coverage percentage.
        "model_compute_mac_ratio_weighted": (
            round(model_compute_mac_ratio_weighted, 4)
            if model_compute_mac_ratio_weighted is not None else None
        ),
        "model_mac_active_us": (
            round(model_mac_active_us, 1)
            if model_mac_active_us is not None else None
        ),
        "model_mac_counter_coverage_pct": round(model_mac_coverage_pct, 1),
        # Total executed useful FLOPs (matmul + fused attention) in the captured
        # step — numerator for the end-to-end (step) MFU computed in theoretical().
        "useful_flops_total": mc_flops if efficiency_reliable else None,
        "useful_flops_modeled_partial": mc_flops,
        "useful_flops_estimated_total": estimated_flops if estimate_available else None,
        "useful_flops_estimated_lo": estimated_flops_lo if estimate_available else None,
        "useful_flops_estimated_hi": estimated_flops_hi if estimate_available else None,
        "mfu_estimate": {
            "available": estimate_available,
            "basis": "formula_flops_plus_same_capture_counter_calibration",
            "confidence": ("medium" if estimate_available and counter_coverage_pct >= 90
                           and not any("mega_moe" in r["type"].lower() for r in model_rows) else "low"),
            "interval_kind": "heuristic_sensitivity_not_statistical_confidence",
            "formula_coverage_pct": round(flop_coverage * 100.0, 1),
            "custom_counter_coverage_pct": round(counter_coverage_pct, 1),
            "counter_scale": round(counter_scale, 6),
            "counter_reference_flops": counter_reference_flops,
            "counter_reference_proxy_flops": counter_reference_proxy,
            "custom_estimated_flops": custom_counter_est,
            "custom_estimated_flops_lo": custom_counter_est * 0.75,
            "custom_estimated_flops_hi": custom_counter_est * 1.25,
        },
        "matmul_flops_total": mm_flops if efficiency_reliable else None,
        "attention_flops_total": (
            sum(r["flops"] for r in attention_rows if r.get("flops"))
            if efficiency_reliable else None
        ),
        "peak_underestimated": calibrated,
        "peak_inconsistent": peak_inconsistent,
        "efficiency_reliable": efficiency_reliable,
        "flop_model_complete": flop_model_complete,
        "flop_coverage_pct": round(flop_coverage * 100.0, 1),
        "attention_flop_coverage_pct": (round(attention_coverage * 100.0, 1)
                                        if attention_coverage is not None else None),
        "attention_training_flops": attention_training_flops,
        "matmul_training_flops": matmul_training_flops,
        "recompute_estimate": recompute_estimate,
        "model_compute_total_us": round(model_total_us, 1),
        "model_compute_modeled_us": round(model_modeled_us, 1),
        "attention_total_us": round(attention_total_us, 1),
        "unmodeled_flop_types": unmodeled_flop_types,
        "flop_model_notes": flop_model_notes,
        "by_type": public_type_rows,
        "top_optimization": public_top_opt,
        "op_ceiling_opt": {**op_ceiling_opt, "complete": efficiency_reliable},
        # Overhead-bound ops (MFU & MBU both <2%): excluded from top_optimization
        # because the roofline reclaim is physically bogus; listed separately so their
        # time is visible and flagged for kernel-level work (not a roofline target).
        "overhead_bound_ops": overhead_bound_ops,
        "overhead_bound_us": overhead_bound_us,
        "scatter": public_scatter,
        "kernels_with_flops": len(flops_rows),
        "kernels_total": len(rows),
        "comm_kernels_excluded": skipped_comm,
        # False on a msprof lightweight capture (shapes are N/A) -> no FLOPs can be
        # modeled, so MFU/MBU/Roofline are unavailable; the UI shows a degradation
        # banner instead of empty tables. True whenever any kernel had shapes.
        "shapes_available": len(flops_rows) > 0,
    }
