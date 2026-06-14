"""Global configuration: chip peak specs, model config, dtype sizes, paths.

Chip peaks are loaded from YAML under configs/chips/<name>.yaml (llmperf field
schema), carrying SEPARATE cube vs vector peaks so op-level MFU can route matmul/
attention to the cube peak and vector ops to the vector peak. MFU/MBU/Roofline
scale directly off these numbers and are surfaced in the UI ("assumed — adjust
here" when ChipSpec.assumed). Pick the active chip via LLMINSIGHT_CHIP (YAML stem,
e.g. Ascend_910B) and override the chip dir with LLMINSIGHT_CHIP_DIR.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional

import yaml


# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
def _default_data_dir() -> str:
    env = os.environ.get("LLMINSIGHT_DATA_DIR")
    if env:
        return env
    # repo-root default: the bundled DeepSeek-V3 single-card sample
    here = os.path.dirname(os.path.abspath(__file__))
    repo = os.path.dirname(here)
    return os.path.join(
        repo, "secret", "ASCEND_PROFILER_OUTPUT", "ASCEND_PROFILER_OUTPUT"
    )


def _default_script_path() -> str:
    # Model + training config are derived from the PROFILING (parser.derive), not
    # a launch script — a script need not correspond 1:1 to a given profiler
    # output. We therefore never auto-point at any script; honour an explicit
    # LLMINSIGHT_SCRIPT only as an opt-in escape hatch (default: none).
    return os.environ.get("LLMINSIGHT_SCRIPT", "")


DATA_DIR = _default_data_dir()
SCRIPT_PATH = _default_script_path()


# --------------------------------------------------------------------------- #
# Chip spec. Peak numbers are loaded from configs/chips/<name>.yaml (llmperf
# field schema). Crucially the spec carries SEPARATE cube vs vector peaks:
#   - CUBE (matmul / tensor-core): drives MFU for MatMul / Gemm / FlashAttention.
#   - VECTOR (AI Vector Core): drives efficiency for RMSNorm / SwiGlu / Cast / ...
# On the Ascend 950DT cube bf16 (432 TFLOPS) is ~8x vector bf16 (54 TFLOPS), so a
# single per-dtype peak would mis-measure every vector op. efficiency.py routes
# each kernel to the right peak via peak_cube_flops() / peak_vector_flops().
# The dataclass defaults below mirror the assumed Ascend 910B values so the app
# still works if the YAML dir is missing.
# --------------------------------------------------------------------------- #
# Realistic per-op-class MFU ceilings. matmul saturates the cube near peak;
# FlashAttention fwd (attention) and FlashAttentionScoreGrad (attention_grad)
# leave structural headroom (softmax / masking / recompute), so their realistic
# "done" target sits well below 100%. Above these, further kernel tuning isn't
# worth it. Per-chip override via the YAML `mfu_ceilings:` block.
_DEFAULT_MFU_CEILINGS = {"matmul": 0.95, "attention": 0.85, "attention_grad": 0.70}


@dataclass
class ChipSpec:
    name: str = "Ascend_910B"
    # CUBE (matmul / tensor-core) peaks, FLOP/s.
    cube_fp16_flops: float = 376.0e12   # bf16 / fp16 cube peak
    cube_fp8_flops: float = 0.0         # 0 -> fall back to cube_fp16
    cube_fp4_flops: float = 0.0         # 0 -> fall back to cube_fp16
    # VECTOR (AI Vector Core) peaks, FLOP/s.
    vector_fp32_flops: float = 75.0e12
    vector_bf16_flops: float = 150.0e12  # bf16 vector peak (default 2x fp32)
    # Peak HBM bandwidth, byte/s. 910B ~= 1.6 TB/s.
    hbm_bandwidth: float = 1.6e12
    # HBM capacity, bytes (drives the memory view's "显存容量" / OOM headroom).
    hbm_capacity_bytes: float = 64.0e9
    # llmperf-parity fields (kept for fidelity; not yet consumed by metrics).
    launch_overhead: float = 0.0
    pcie_bandwidth: float = 0.0
    dies_per_package: int = 1
    intra_package_link: Optional[str] = None
    assumed: bool = False
    # Realistic per-op-class MFU ceilings (matmul / attention / attention_grad):
    # the achievable-MFU target above which a kernel is "done". Defaults apply to
    # any chip that omits the YAML block. Drives the "算子极致优化" What-if lever
    # and the ceiling-aware operator ranking (see metrics/efficiency.py).
    mfu_ceilings: Dict[str, float] = field(
        default_factory=lambda: dict(_DEFAULT_MFU_CEILINGS))

    @property
    def hbm_capacity_gb(self) -> float:
        """Decimal GB (bytes / 1e9), matching llmperf's memory_capacity convention."""
        return self.hbm_capacity_bytes / 1e9

    def peak_cube_flops(self, dtype: str) -> float:
        """CUBE peak for matmul-family / fused-attention kernels."""
        d = (dtype or "").upper()
        if "FP8" in d or "HIF8" in d or "FLOAT8" in d or "E4M3" in d or "E5M2" in d:
            return self.cube_fp8_flops or self.cube_fp16_flops
        if "FP4" in d or "FLOAT4" in d:
            return self.cube_fp4_flops or self.cube_fp16_flops
        # bf16 / fp16 / half / unknown -> cube bf16/fp16 peak
        return self.cube_fp16_flops

    def peak_vector_flops(self, dtype: str) -> float:
        """VECTOR peak for AI Vector Core / MIX_AIV elementwise kernels."""
        d = (dtype or "").upper()
        if "FLOAT32" in d or "FP32" in d or d == "FLOAT":
            return self.vector_fp32_flops
        # bf16 / fp16 / half / unknown -> vector bf16 peak
        return self.vector_bf16_flops

    def peak_flops_for(self, dtype: str) -> float:
        """Back-compat: cube peak (matmul). fp32 has no cube unit -> vector fp32."""
        d = (dtype or "").upper()
        if "FLOAT32" in d or "FP32" in d or d == "FLOAT":
            return self.vector_fp32_flops
        return self.peak_cube_flops(dtype)

    def mfu_ceiling(self, op_class: Optional[str]) -> Optional[float]:
        """Realistic MFU ceiling for a ceiling-governed op-class (matmul /
        attention / attention_grad). None for any other class — those optimize
        toward the 100% roofline rather than a capped ceiling."""
        if not op_class:
            return None
        return self.mfu_ceilings.get(op_class)


# --------------------------------------------------------------------------- #
# YAML chip loader (mirrors llmperf configs/_yaml_loader.py::load_gpu_spec).
# Specs live in <repo>/configs/chips/<name>.yaml; override the dir with
# LLMINSIGHT_CHIP_DIR. *_tflops fields are scaled x1e12 -> FLOP/s; memory_*
# stay in bytes(/s). bf16_vector_tflops=null defaults to 2x fp32.
# --------------------------------------------------------------------------- #
# Legacy key aliases (old LLMINSIGHT_CHIP / saved UI values -> YAML stems).
_CHIP_ALIASES = {"910B": "Ascend_910B", "950DT": "Ascend_950DT"}


def _chip_dir() -> Path:
    env = os.environ.get("LLMINSIGHT_CHIP_DIR")
    if env:
        return Path(env)
    # Installed wheel ships configs INSIDE the package (llminsight/configs/chips);
    # a dev checkout keeps them at the repo root (<repo>/configs/chips).
    pkg = Path(__file__).resolve().parent  # <...>/llminsight
    for cand in (pkg / "configs" / "chips",          # installed wheel
                 pkg.parent / "configs" / "chips"):  # dev checkout
        if cand.is_dir():
            return cand
    return pkg.parent / "configs" / "chips"


def _canonical_chip_key(name: str) -> str:
    n = (name or "").strip()
    return _CHIP_ALIASES.get(n, n)


def load_chip_spec(name: str) -> ChipSpec:
    """Load a ChipSpec from configs/chips/<name>.yaml. Raises FileNotFoundError
    if the YAML is absent (callers that need a fallback should catch it)."""
    key = _canonical_chip_key(name)
    path = _chip_dir() / f"{key}.yaml"
    with open(path, "r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}

    spec_name = str(data.get("name", key))

    def _tf(field_name: str) -> Optional[float]:
        v = data.get(field_name)
        return float(v) * 1e12 if v is not None else None

    fp16 = _tf("fp16_tflops") or 0.0
    fp32 = _tf("fp32_tflops") or 0.0
    bf16_vec = data.get("bf16_vector_tflops")
    vector_bf16 = (float(bf16_vec) * 1e12) if bf16_vec is not None else (2.0 * fp32)

    # Per-op-class MFU ceilings: start from the defaults, override with whatever
    # the YAML's `mfu_ceilings:` block specifies (so a chip can omit it entirely).
    ceilings = dict(_DEFAULT_MFU_CEILINGS)
    raw_ceilings = data.get("mfu_ceilings")
    if isinstance(raw_ceilings, dict):
        for ck, cv in raw_ceilings.items():
            if cv is not None:
                ceilings[str(ck)] = float(cv)

    return ChipSpec(
        name=spec_name,
        cube_fp16_flops=fp16,
        cube_fp8_flops=_tf("fp8_tflops") or 0.0,
        cube_fp4_flops=_tf("fp4_tflops") or 0.0,
        vector_fp32_flops=fp32,
        vector_bf16_flops=vector_bf16,
        hbm_bandwidth=float(data.get("memory_bandwidth", 0.0) or 0.0),
        hbm_capacity_bytes=float(data.get("memory_capacity", 0.0) or 0.0),
        launch_overhead=float(data.get("launch_overhead", 0.0) or 0.0),
        pcie_bandwidth=float(data.get("pcie_bandwidth", 0.0) or 0.0),
        dies_per_package=int(data.get("dies_per_package", 1) or 1),
        intra_package_link=data.get("intra_package_link"),
        assumed=bool(data.get("assumed", False)),
        mfu_ceilings=ceilings,
    )


def available_chips() -> List[str]:
    """YAML stems available in the chip dir, sorted (drives the UI dropdown)."""
    d = _chip_dir()
    if not d.is_dir():
        return []
    return sorted(p.stem for p in d.glob("*.yaml"))


# Default to the 950DT reference ceiling (overridable via LLMINSIGHT_CHIP).
DEFAULT_CHIP = _canonical_chip_key(os.environ.get("LLMINSIGHT_CHIP", "Ascend_950DT"))
if DEFAULT_CHIP not in available_chips():
    DEFAULT_CHIP = "Ascend_950DT" if "Ascend_950DT" in available_chips() else "Ascend_910B"


def _load_default_chip() -> ChipSpec:
    try:
        return load_chip_spec(DEFAULT_CHIP)
    except Exception:
        return ChipSpec()  # built-in assumed 910B-class defaults


def _chip_dropdown() -> List[dict]:
    """Lightweight per-chip summary for the UI dropdown, built from the YAML dir.
    A malformed YAML is skipped rather than breaking the whole list."""
    out: List[dict] = []
    for stem in available_chips():
        try:
            spec = load_chip_spec(stem)
        except Exception:
            continue
        out.append({
            "key": stem,
            "name": spec.name,
            "peak_bf16_tflops": spec.cube_fp16_flops / 1e12,  # CUBE bf16 peak
            "hbm_tbps": spec.hbm_bandwidth / 1e12,
            "hbm_gb": spec.hbm_capacity_gb,
        })
    return out


# --------------------------------------------------------------------------- #
# Model config (parsed from the DeepSeek-V3 training script 8k_bf16_sbh_64p.sh)
# --------------------------------------------------------------------------- #
@dataclass
class ModelConfig:
    name: str = "DeepSeek-V3 (MoE, MLA)"
    num_layers: int = 10
    hidden_size: int = 7168
    ffn_hidden_size: int = 18432          # dense FFN (first_k_dense layers)
    num_attention_heads: int = 128
    seq_length: int = 8192
    micro_batch_size: int = 1
    global_batch_size: int = 64
    vocab_size: int = 129280
    # MoE
    num_experts: int = 128
    moe_router_topk: int = 8
    moe_ffn_hidden_size: int = 2048
    moe_shared_expert_intermediate_size: int = 2048
    first_k_dense_replace: int = 1
    moe_layer_freq: int = 1
    # MLA
    q_lora_rank: int = 1536
    kv_lora_rank: int = 512
    qk_head_dim: int = 128
    qk_pos_emb_head_dim: int = 64
    v_head_dim: int = 128
    # Parallelism
    tp: int = 1
    pp: int = 1
    ep: int = 64
    cp: int = 1
    # Memory/compute tradeoffs
    recompute_granularity: str = "full"
    recompute_num_layers: int = 1
    swap_optimizer: bool = True
    dtype: str = "bf16"


@dataclass
class Settings:
    data_dir: str = field(default_factory=_default_data_dir)
    script_path: str = field(default_factory=_default_script_path)
    chip: ChipSpec = field(default_factory=_load_default_chip)
    chip_key: str = DEFAULT_CHIP
    model: ModelConfig = field(default_factory=ModelConfig)
    # Cap how many trace events the timeline endpoint streams to the browser.
    timeline_max_slices: int = 4000
    timeline_bins: int = 600
    # Smart-timeline utilization lanes sample 5x finer than the Timeline page so
    # the Cube/Vector/HBM/通信 curves reflect real per-burst structure, not a
    # coarse rolling average. Only affects the utilization series resolution; the
    # Gantt slice cap (timeline_max_slices) is independent.
    smart_timeline_bins: int = 3000

    def to_dict(self) -> dict:
        # NOTE: model + training config are NOT shipped here anymore — they are
        # derived from the profiling and live under metrics.meta["config"]
        # (parser.derive). script_path is intentionally omitted too (no launch
        # script dependency; also avoids leaking an absolute path into payloads).
        return {
            "data_dir": self.data_dir,
            # asdict() omits the hbm_capacity_gb @property, so add it explicitly.
            "chip": {**asdict(self.chip), "hbm_capacity_gb": self.chip.hbm_capacity_gb},
            "chip_key": self.chip_key,
            # YAML-backed presets for the UI dropdown (no secrets; peak summary).
            # peak_bf16_tflops is the CUBE bf16 peak (the matmul-MFU denominator).
            "chips": _chip_dropdown(),
        }


# dtype -> bytes per element
DTYPE_BYTES = {
    "FLOAT": 4, "FLOAT32": 4, "FP32": 4,
    "FLOAT16": 2, "FP16": 2, "HALF": 2,
    "BF16": 2, "BFLOAT16": 2, "DT_BF16": 2,
    "INT64": 8, "UINT64": 8, "DOUBLE": 8, "COMPLEX64": 8,
    "INT32": 4, "UINT32": 4,
    "INT16": 2, "UINT16": 2,
    "INT8": 1, "UINT8": 1, "BOOL": 1,
    # Low-precision 950-series modes. FP4 is sub-byte (two elements per byte).
    "FP8": 1, "HIF8": 1, "HIFLOAT8": 1, "FLOAT8": 1, "FLOAT8_E4M3": 1, "FLOAT8_E5M2": 1,
    "FP4": 0.5, "FLOAT4": 0.5, "FLOAT4_E2M1": 0.5,
    "": 2,
}


def dtype_bytes(dtype: str) -> float:
    if not dtype:
        return 2
    d = dtype.strip().upper()
    if d in DTYPE_BYTES:
        return DTYPE_BYTES[d]
    # msprof emits dtype strings the table doesn't list verbatim: a DT_ prefix
    # and/or a sub-format suffix (e.g. DT_FLOAT8_E4M3FN, DT_FLOAT8_E8M0). Match by
    # family so fp8 (1B) / fp4 (0.5B) aren't silently taken as the 2B default,
    # which would over-state bytes -> MBU for quantized matmul ops.
    if "FLOAT8" in d or "FP8" in d or "HIF8" in d:
        return 1
    if "FLOAT4" in d or "FP4" in d:
        return 0.5
    if d.startswith("DT_"):
        return DTYPE_BYTES.get(d[3:], 2)
    return 2


SETTINGS = Settings()


def set_chip(key: str) -> bool:
    """Switch the active chip by YAML stem (driven by the UI dropdown).

    Returns False for an unknown / unloadable chip. Reassigns SETTINGS.chip to a
    freshly loaded spec (never mutates in place) so metric recomputation is
    deterministic. Accepts legacy keys ('910B' / '950DT') via the alias map.
    """
    try:
        spec = load_chip_spec(key)
    except Exception:
        return False
    SETTINGS.chip = spec
    SETTINGS.chip_key = _canonical_chip_key(key)
    return True


# --------------------------------------------------------------------------- #
# Persisted UI state: remember the last successfully-loaded profiling dir so the
# directory picker (and optional autoload) default to it across server restarts.
# Stored under ~/.llminsight (override with LLMINSIGHT_STATE_DIR).
# --------------------------------------------------------------------------- #
def _state_dir() -> str:
    env = os.environ.get("LLMINSIGHT_STATE_DIR")
    if env:
        return env
    return os.path.join(os.path.expanduser("~"), ".llminsight")


def save_last_dir(data_dir: str) -> None:
    """Persist the last loaded profiling dir (best-effort; never raises)."""
    try:
        d = _state_dir()
        os.makedirs(d, exist_ok=True)
        with open(os.path.join(d, "last_profile_dir.txt"), "w", encoding="utf-8") as fh:
            fh.write(str(data_dir or "").strip())
    except OSError:
        pass


def load_last_dir() -> Optional[str]:
    """The last loaded dir if it still exists on disk, else None."""
    try:
        with open(os.path.join(_state_dir(), "last_profile_dir.txt"), "r", encoding="utf-8") as fh:
            d = fh.read().strip()
        return d if d and os.path.isdir(d) else None
    except OSError:
        return None
