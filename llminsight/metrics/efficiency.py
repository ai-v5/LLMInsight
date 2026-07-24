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
ATTENTION_FWD_TYPES = {
    "FlashAttentionScore", "PromptFlashAttention",
    "FusedInferAttentionScore", "IncreFlashAttention",
    # DeepSeek-V3.2 sparse attention (DSA): sparse Flash-MLA, shared-KV sparse
    # attention, and the lightning indexer. Routed to the cube peak + attention
    # lane so they're not misfiled as generic compute / left out of attribution.
    "SparseFlashMla", "SparseAttnSharedkv", "SparseLightningIndexer",
    # Generic sparse attention used by current MindSpeed/GLM profiles.  Its
    # token-wise causal useful Cube FLOPs are modeled only when the recorded
    # shapes prove the supported 950DT schema; all other layouts fail closed.
    "SparseFlashAttention",
}
ATTENTION_GRAD_TYPES = {
    "FlashAttentionScoreGrad",
    "SparseFlashMlaGrad", "SparseLightningIndexerGrad",
    "SparseLightningIndexerKllossGrad",
    "SparseFlashAttentionGrad",
}
ATTENTION_TYPES = ATTENTION_FWD_TYPES | ATTENTION_GRAD_TYPES
SPARSE_ATTENTION_TYPES = {"SparseFlashAttention", "SparseFlashAttentionGrad"}
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


def _estimate_sparse_flash_attention_flops(
    shapes_in: List[List[int]], shapes_out: List[List[int]], is_grad: bool,
    sparse_mode: Optional[int] = None, sparse_block_size: Optional[int] = None,
) -> Optional[float]:
    """Useful BF16 cube FLOPs for token-wise causal SparseFlashAttention.

    CANN SparseFlashAttention uses sparse_indices[B,Sq,Nkv,K] and, on 950DT,
    sparse_block_size=1 (one selected token per index).  MindSpeed training uses
    right-down causal mode.  For query row q, the number of valid selected tokens
    is min(K, max(0, Sk-Sq+q+1)); -1 padding is therefore excluded.

    The returned numerator counts only the matmul-dominated useful work:
      fwd = 2 * G * C * (Dqk + Dv)
      bwd = 2 * G * C * (3*Dqk + 2*Dv)
    where Dqk includes the separate RoPE dimension and G=Hq/Hkv.  Softmax,
    gather/scatter, invalid tiling lanes and other vector/memory work are not cube
    FLOPs.  The CSV cannot reconstruct their repeated/discrete memory traffic, so
    this op is deliberately excluded from MBU, roofline and reclaim estimates.
    """
    # Neither attribute is present in kernel_details.csv.  Only compute the
    # causal token count after the caller supplies verified semantics.
    if sparse_mode != 3 or sparse_block_size != 1:
        return None
    if len(shapes_in) < 6 or len(shapes_out) < (3 if is_grad else 1):
        return None
    q, k, v, sparse_indices = shapes_in[:4]
    if any(len(s) != 4 for s in (q, k, v, sparse_indices)):
        return None
    B, Sq, Hq, Dq = q
    Bk, Sk, Hkv, Dk = k
    Bv, Sv, Hv, Dv = v
    Bi, Si, Hi, topk = sparse_indices
    if (
        min(B, Sq, Hq, Dq, Sk, Hkv, Dk, Dv, topk) <= 0
        or (Bk, Bv, Bi) != (B, B, B)
        or (Sv, Si) != (Sk, Sq)
        or (Hv, Hi) != (Hkv, Hkv)
        or Dq != Dk
        or Hq % Hkv != 0
        or topk > Sk
    ):
        return None

    q_rope, k_rope = shapes_in[-2:]
    if (
        len(q_rope) != 4 or len(k_rope) != 4
        or q_rope[:3] != [B, Sq, Hq]
        or k_rope[:3] != [B, Sk, Hkv]
        or q_rope[3] <= 0 or q_rope[3] != k_rope[3]
    ):
        return None
    group = Hq // Hkv
    if not is_grad:
        if shapes_out[0] != [B, Sq, Hq, Dv]:
            return None
        # Softmax stats [B,Nkv,Sq,G] prove the sparse pattern is shared by the G
        # query heads associated with each KV head.
        stats = shapes_out[1:3]
        if len(stats) < 2 or any(s != [B, Hkv, Sq, group] for s in stats):
            return None
    else:
        if shapes_out[:3] != [q, k, v]:
            return None

    # right-down causal: the first query can see Sk-Sq+1 keys; equal-length
    # training reduces to min(topk, q+1).  Multiply by B and KV-head patterns.
    valid_per_pattern = 0
    for q_idx in range(Sq):
        causal_prefix = max(0, min(Sk, Sk - Sq + q_idx + 1))
        valid_per_pattern += min(topk, causal_prefix)
    selected_head_pairs = B * Hkv * group * valid_per_pattern
    d_qk = Dq + q_rope[3]
    if is_grad:
        return 2.0 * selected_head_pairs * (3 * d_qk + 2 * Dv)
    return 2.0 * selected_head_pairs * (d_qk + Dv)


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
# Any vector kernel not listed: assume light elementwise (1 op/element). Safe — low
# arithmetic intensity keeps it memory-bound, so this only ADDS a (small) MFU read
# and never raises the headroom floor above the memory roofline.
_VECTOR_DEFAULT_FPE = 1.0


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


def compute_efficiency(prof) -> Dict[str, Any]:
    kd = prof.kernel_details
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

    mac_r = colf("aic_mac_ratio")
    mte2_r = colf("aic_mte2_ratio")
    mte3_r = colf("aic_mte3_ratio")
    vec_r = colf("aiv_vec_ratio")
    cube_u = colf("cube_utilization(%)")

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
        # Communication-dispatch "kernels" (AI_CPU / HCCL) have no compute or
        # memory footprint we can model — they belong to the communication and
        # hidden-overhead views, not the compute-efficiency ranking.
        if str(core[i]).upper() == "AI_CPU" or str(types[i]).lower().startswith(("hccl", "hcom")):
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
        if types[i] in MATMUL_TYPES:
            _mnk = _matmul_mnk(shapes_in, shapes_out)
            if _mnk:
                mn_elems = _mnk[0] * _mnk[1] * _mnk[3]   # M * N * batch
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
        is_sparse_attention = types[i] in SPARSE_ATTENTION_TYPES
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
        elif is_attention:
            flops = _estimate_attention_flops(
                shapes_in, shapes_out, types[i] in ATTENTION_GRAD_TYPES)
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
        if is_sparse_attention:
            b_bytes = 0

        # Peak routing: matmul / fused-attention kernels run on the CUBE unit;
        # pure vector-core ops (AI_VECTOR_CORE / MIX_AIV: RMSNorm/SwiGlu/Cast/...)
        # run on the VECTOR unit; any other AI_CORE / MIX_AIC op defaults to cube.
        # On the 950DT cube bf16 (432T) is ~8x vector bf16 (54T), so picking the
        # right unit's peak is what makes per-op MFU/waste physically meaningful.
        use_cube = is_matmul or is_attention or (not is_vector)
        peak_flops = peak_for(dtype, use_cube)

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
        modeled = ((flops is not None) or (is_vector and b_bytes > 0)) and not is_sparse_attention
        t_compute = (flops / peak_flops) if flops else 0.0
        t_mem = (b_bytes / chip.hbm_bandwidth) if b_bytes else 0.0
        # Ceiling-aware reclaim: matmul / FA / FAG have a realistic MFU ceiling
        # (<100%) below which further tuning isn't worth it. reclaim_us is the time
        # recoverable by optimizing *to that ceiling* — 0 once a kernel is already
        # at/above it (running at MFU>=ceiling means dur <= compute_time/ceiling).
        # Non-ceiling modeled ops optimize toward the 100% roofline, so their
        # reclaim == wasted_us.
        op_class = _op_class(types[i]) if flops is not None and not is_sparse_attention else None
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
             "peak_time": 0.0, "dt_dur": {}},
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
        t["dt_dur"][r["dtype"]] = t["dt_dur"].get(r["dtype"], 0.0) + r["dur_us"]

    type_rows = []
    for t in by_type.values():
        d_s = t["dur_us"] * 1e-6
        mfu = (t["flops"] / t["peak_time"]) if t["peak_time"] else None
        mbu_raw = (t["bytes"] / d_s / SETTINGS.chip.hbm_bandwidth) if (d_s and t["bytes"]) else None
        mbu = min(mbu_raw, 1.0) if mbu_raw is not None else None
        dom_dtype = max(t["dt_dur"].items(), key=lambda kv: kv[1])[0] if t["dt_dur"] else None
        type_rows.append(
            {
                "type": t["type"],
                "count": t["count"],
                "dur_us": round(t["dur_us"], 1),
                "dtype": dom_dtype,
                "mfu": round(mfu, 4) if mfu is not None else None,
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
    mc = [r for r in flops_rows if r["type"] in MATMUL_TYPES or r["type"] in ATTENTION_TYPES]
    mc_flops = sum(r["flops"] for r in mc)
    mc_peak_time = sum(r["peak_flops"] * r["dur_us"] * 1e-6 for r in mc if r.get("peak_flops"))
    model_mfu_compute_partial = (mc_flops / mc_peak_time) if mc_peak_time else None

    # Duration-weighted coverage over recognized model-compute families.  A known
    # but unsupported dominant op must make the model MFU unavailable instead of
    # letting a GEMM-only numerator masquerade as whole-model efficiency.
    model_rows = [r for r in rows if r["type"] in MATMUL_TYPES or r["type"] in ATTENTION_TYPES]
    modeled_model_rows = [r for r in model_rows if r.get("flops")]
    model_total_us = sum(r["dur_us"] for r in model_rows)
    model_modeled_us = sum(r["dur_us"] for r in modeled_model_rows)
    flop_coverage = (model_modeled_us / model_total_us) if model_total_us else 0.0
    flop_model_complete = bool(model_total_us > 0 and flop_coverage >= 0.90)
    attention_rows = [r for r in rows if r["type"] in ATTENTION_TYPES]
    attention_total_us = sum(r["dur_us"] for r in attention_rows)
    attention_modeled_us = sum(r["dur_us"] for r in attention_rows if r.get("flops"))
    attention_coverage = (attention_modeled_us / attention_total_us) if attention_total_us else None
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
    flop_model_notes = []
    if sparse_flops_modeled:
        flop_model_notes.append(
            "SparseFlashAttention 按 950DT token-wise（sparse_block_size=1）、"
            "right-down causal（sparse_mode=3）计算有效稀疏 Cube FLOPs；"
            "不含 Softmax/Gather/Scatter 与 tiling padding。因 CSV 无法重建离散访存，"
            "该算子不参与 MBU、roofline 与可回收时延估算。"
        )
    elif sparse_present:
        flop_model_notes.append(
            "profiler CSV 未记录 SparseFlashAttention 的 sparse_mode；默认不猜测。"
            "确认本次运行后可显式设置 LLMINSIGHT_SPARSE_MODE=3；仅在 950DT "
            "token-wise（sparse_block_size=1）语义下启用有效 FLOPs。"
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
        # Total executed useful FLOPs (matmul + fused attention) in the captured
        # step — numerator for the end-to-end (step) MFU computed in theoretical().
        "useful_flops_total": mc_flops if efficiency_reliable else None,
        "useful_flops_modeled_partial": mc_flops,
        "peak_underestimated": calibrated,
        "peak_inconsistent": peak_inconsistent,
        "efficiency_reliable": efficiency_reliable,
        "flop_model_complete": flop_model_complete,
        "flop_coverage_pct": round(flop_coverage * 100.0, 1),
        "attention_flop_coverage_pct": (round(attention_coverage * 100.0, 1)
                                        if attention_coverage is not None else None),
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
