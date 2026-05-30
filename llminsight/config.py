"""Global configuration: chip peak specs, model config, dtype sizes, paths.

IMPORTANT: the chip peak numbers below are *assumptions* for an Ascend 910B-class
device. MFU/MBU/Roofline results scale directly off them, so they are surfaced in
the UI as "assumed — adjust here". Override via environment variables or by editing
this file.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field, asdict
from typing import Dict, Optional


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
    env = os.environ.get("LLMINSIGHT_SCRIPT")
    if env:
        return env
    here = os.path.dirname(os.path.abspath(__file__))
    repo = os.path.dirname(here)
    return os.path.join(repo, "secret", "8k_bf16_sbh_64p.sh")


DATA_DIR = _default_data_dir()
SCRIPT_PATH = _default_script_path()


# --------------------------------------------------------------------------- #
# Chip spec (ASSUMED — Ascend 910B class). Adjust to your device.
# --------------------------------------------------------------------------- #
@dataclass
class ChipSpec:
    name: str = "Ascend 910B (assumed)"
    # Peak dense matmul throughput, FLOP/s. 376 TFLOPS is a conservative public
    # 910B figure; bins vary. The bundled DeepSeek-V3 sample sustains ~432 TFLOPS
    # on clean GEMMs, so efficiency.py auto-calibrates the effective peak up to the
    # observed ceiling (a real kernel can't beat silicon) and flags it in the UI.
    # Set this to your real SKU's peak to replace calibration with an exact value.
    peak_bf16_flops: float = 376.0e12
    peak_fp16_flops: float = 376.0e12
    peak_fp32_flops: float = 75.0e12
    peak_int8_ops: float = 752.0e12
    # Peak HBM bandwidth, byte/s. 910B ~= 1.6 TB/s.
    hbm_bandwidth: float = 1.6e12
    # HBM capacity, GiB (drives the memory view's "显存容量" / OOM headroom context).
    hbm_capacity_gb: float = 64.0
    assumed: bool = True

    def peak_flops_for(self, dtype: str) -> float:
        d = (dtype or "").upper()
        if "BF16" in d or "BFLOAT" in d:
            return self.peak_bf16_flops
        if "FLOAT16" in d or "FP16" in d or d == "HALF":
            return self.peak_fp16_flops
        if "INT8" in d:
            return self.peak_int8_ops
        if "FLOAT" in d or "FP32" in d or d == "FLOAT32":
            return self.peak_fp32_flops
        return self.peak_bf16_flops


# --------------------------------------------------------------------------- #
# Switchable chip presets (UI dropdown). 910B is the conservative default and
# matches the ChipSpec defaults above. 950DT numbers are Huawei's announced
# specs (roadmap, GA ~2026 Q4): 1 PFLOPS FP16/BF16, 500 TFLOPS FP32/HF32,
# 1 PFLOPS FP8(HiF8)/INT8, 2 PFLOPS FP4, and HiZQ 2.0 HBM at 4 TB/s, 96 GB.
# Marked assumed=True because this captured run was NOT executed on a 950DT —
# selecting it answers "where would this workload sit against a 950DT ceiling".
# --------------------------------------------------------------------------- #
CHIP_PRESETS: Dict[str, "ChipSpec"] = {
    "910B": ChipSpec(),  # conservative Ascend 910B defaults (see ChipSpec above)
    "950DT": ChipSpec(
        name="Ascend 950DT",
        peak_bf16_flops=1000.0e12,   # FP16/BF16 ≈ 1 PFLOPS
        peak_fp16_flops=1000.0e12,
        peak_fp32_flops=500.0e12,    # FP32/HF32 ≈ 500 TFLOPS
        peak_int8_ops=1000.0e12,     # INT8 ≈ 1 POPS (HiF8/FP8 同量级)
        hbm_bandwidth=4.0e12,        # HiZQ 2.0 HBM, 4 TB/s
        hbm_capacity_gb=96.0,        # 96 GB HBM
        assumed=True,
    ),
}
DEFAULT_CHIP = os.environ.get("LLMINSIGHT_CHIP", "910B").strip()
if DEFAULT_CHIP not in CHIP_PRESETS:
    DEFAULT_CHIP = "910B"


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
    chip: ChipSpec = field(default_factory=lambda: CHIP_PRESETS[DEFAULT_CHIP])
    chip_key: str = DEFAULT_CHIP
    model: ModelConfig = field(default_factory=ModelConfig)
    # Cap how many trace events the timeline endpoint streams to the browser.
    timeline_max_slices: int = 4000
    timeline_bins: int = 600

    def to_dict(self) -> dict:
        return {
            "data_dir": self.data_dir,
            "script_path": self.script_path,
            "chip": asdict(self.chip),
            "chip_key": self.chip_key,
            # presets for the UI dropdown (no secrets; lightweight peak summary)
            "chips": [
                {
                    "key": k,
                    "name": v.name,
                    "peak_bf16_tflops": v.peak_bf16_flops / 1e12,
                    "hbm_tbps": v.hbm_bandwidth / 1e12,
                    "hbm_gb": v.hbm_capacity_gb,
                }
                for k, v in CHIP_PRESETS.items()
            ],
            "model": asdict(self.model),
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
    "": 2,
}


def dtype_bytes(dtype: str) -> int:
    if not dtype:
        return 2
    return DTYPE_BYTES.get(dtype.strip().upper(), 2)


SETTINGS = Settings()


def set_chip(key: str) -> bool:
    """Switch the active chip preset (driven by the UI dropdown).

    Returns False for an unknown key. Only ever *reassigns* SETTINGS.chip to a
    shared preset instance — never mutates a preset's fields in place — so the
    presets stay pristine and metric recomputation is deterministic.
    """
    spec = CHIP_PRESETS.get(key)
    if spec is None:
        return False
    SETTINGS.chip = spec
    SETTINGS.chip_key = key
    return True
