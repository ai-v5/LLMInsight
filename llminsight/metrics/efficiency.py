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
    "BatchMatMul", "BatchMatMulV2",
}

# Fused attention kernels are matmul-dominated (QK^T + softmax·V on the cube), so
# their useful FLOPs should count toward MFU even though the op isn't a plain GEMM.
ATTENTION_FWD_TYPES = {
    "FlashAttentionScore", "PromptFlashAttention",
    "FusedInferAttentionScore", "IncreFlashAttention",
}
ATTENTION_GRAD_TYPES = {"FlashAttentionScoreGrad"}
ATTENTION_TYPES = ATTENTION_FWD_TYPES | ATTENTION_GRAD_TYPES
# DeepSeek-V3 attention is causal: FlashAttention skips the masked (upper-triangle)
# blocks, so it does ~half the dense QK^T+PV work. Without this 0.5 the achieved
# rate would exceed silicon peak — i.e. the factor is physically required, not a
# convenience. The backward pass recomputes ~2.5x the forward matmul FLOPs.
_ATTN_CAUSAL_FACTOR = 0.5
_ATTN_BWD_FWD_RATIO = 2.5


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


def _estimate_matmul_flops(shapes: List[List[int]]) -> Optional[float]:
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

    M, K, abz = mk(a)
    br, bc, bbz = mk(b)
    if br == K:
        N = bc
    elif bc == K:
        N = br
    else:
        N = bc
    # Batch comes from the activation side only. For GroupedMatmul the weight is
    # a 3-D [E,K,N] tensor but each token visits exactly one expert, so the token
    # dimension already totals the work — multiplying by E would over-count.
    batch = abz
    if min(M, N, K) <= 0:
        return None
    return 2.0 * M * N * K * batch


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
        return None
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

    # ---- pre-pass: observed BF16 ceiling -> calibrate the assumed peak --------
    # A real kernel cannot exceed silicon peak. If the best clean matmul's achieved
    # throughput exceeds the assumed ChipSpec peak, the assumption is simply too low
    # for this SKU (910B bins vary), so reporting MFU > 100% would be nonsense.
    # Calibrate the effective peak up to the observed ceiling (+2% headroom so the
    # ceiling kernel reads ~98%, not a suspicious flat 100%), bounded at 2x the
    # assumption so a stray shape mis-parse can't masquerade as a giant peak bump.
    # Both assumed and observed peaks are surfaced in the UI; setting the real SKU
    # peak in ChipSpec overrides this. Calibration only ever raises the peak.
    observed_peak = 0.0
    for i in range(len(kd)):
        if types[i] not in MATMUL_TYPES or float(dur[i]) <= 0:
            continue
        f = _estimate_matmul_flops(parse_shapes(in_sh[i]))
        if f:
            observed_peak = max(observed_peak, f / (float(dur[i]) * 1e-6))
    # Calibrate against the CUBE bf16 peak — GEMM runs on the cube unit.
    configured_peak = chip.peak_cube_flops("BF16")
    calibrated = observed_peak > configured_peak
    effective_peak = min(observed_peak * 1.02, configured_peak * 2.0) if calibrated else configured_peak
    peak_scale = effective_peak / configured_peak if configured_peak else 1.0  # >= 1.0 (cube only)

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

        # bytes moved (read inputs + write outputs)
        b_bytes = 0
        for j, sh in enumerate(shapes_in):
            dt = dt_in[j] if j < len(dt_in) else (dt_in[-1] if dt_in else dtype)
            b_bytes += numel(sh) * dtype_bytes(dt)
        for j, sh in enumerate(shapes_out):
            dt = dt_out[j] if j < len(dt_out) else (dt_out[-1] if dt_out else dtype)
            b_bytes += numel(sh) * dtype_bytes(dt)

        is_matmul = types[i] in MATMUL_TYPES
        is_attention = types[i] in ATTENTION_TYPES
        core_u = str(core[i]).upper()
        is_vector = "VECTOR" in core_u or "AIV" in core_u
        if is_matmul:
            flops = _estimate_matmul_flops(shapes_in)
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
        mbu = (achieved_bw / chip.hbm_bandwidth) if achieved_bw else None

        # We can only credibly model "ideal time" (and thus wasted time) when we
        # have a FLOP estimate (matmul / fused attention) OR the kernel is a
        # vector-core memory op. Other cube/MIX ops without a FLOP model would
        # look 100% wasted under a memory-only roofline — so we leave them unscored.
        modeled = (flops is not None) or (is_vector and b_bytes > 0)
        t_compute = (flops / peak_flops) if flops else 0.0
        t_mem = (b_bytes / chip.hbm_bandwidth) if b_bytes else 0.0
        # Ceiling-aware reclaim: matmul / FA / FAG have a realistic MFU ceiling
        # (<100%) below which further tuning isn't worth it. reclaim_us is the time
        # recoverable by optimizing *to that ceiling* — 0 once a kernel is already
        # at/above it (running at MFU>=ceiling means dur <= compute_time/ceiling).
        # Non-ceiling modeled ops optimize toward the 100% roofline, so their
        # reclaim == wasted_us.
        op_class = _op_class(types[i]) if flops is not None else None
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
                "efficiency": efficiency,
                "wasted_us": wasted_us,
                "reclaim_us": reclaim_us,
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
        mbu = (t["bytes"] / d_s / SETTINGS.chip.hbm_bandwidth) if (d_s and t["bytes"]) else None
        dom_dtype = max(t["dt_dur"].items(), key=lambda kv: kv[1])[0] if t["dt_dur"] else None
        type_rows.append(
            {
                "type": t["type"],
                "count": t["count"],
                "dur_us": round(t["dur_us"], 1),
                "dtype": dom_dtype,
                "mfu": round(mfu, 4) if mfu is not None else None,
                "mbu": round(mbu, 4) if mbu is not None else None,
                "wasted_us": round(t["wasted_us"], 1),
                "reclaim_us": round(t["reclaim_us"], 1),
            }
        )
    type_rows.sort(key=lambda x: x["dur_us"], reverse=True)

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
        for k in ("flops", "bytes", "mfu", "mbu", "ai", "achieved_tflops",
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
    mm_time_s = sum(r["dur_us"] for r in mm) * 1e-6
    mm_flops = sum(r["flops"] for r in mm)
    matmul_mfu = (mm_flops / mm_time_s / effective_peak) if mm_time_s else None
    # the same number against the *assumed* peak (>1 is what triggered calibration)
    matmul_mfu_assumed = (mm_flops / mm_time_s / configured_peak) if mm_time_s else None

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
        "matmul_mfu_assumed": round(matmul_mfu_assumed, 4) if matmul_mfu_assumed else None,
        # Total executed useful FLOPs (matmul + fused attention) in the captured
        # step — numerator for the end-to-end (step) MFU computed in theoretical().
        "useful_flops_total": sum(r["flops"] for r in flops_rows),
        "peak_underestimated": calibrated,
        "by_type": type_rows[:40],
        "top_optimization": top_opt,
        "op_ceiling_opt": op_ceiling_opt,
        "scatter": scatter,
        "kernels_with_flops": len(flops_rows),
        "kernels_total": len(rows),
        "comm_kernels_excluded": skipped_comm,
    }
