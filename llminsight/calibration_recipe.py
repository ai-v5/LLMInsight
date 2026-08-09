"""Export fail-closed profiling-derived GEMM calibration recipes.

The producer deliberately emits portable, product-neutral recipe candidates.
It does not turn profiler observations into calibrated performance results.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import shutil
import stat
import sys
import tempfile
from collections import Counter
from collections.abc import Iterable, Mapping, MutableMapping, Sequence
from decimal import ROUND_FLOOR, ROUND_HALF_UP, Decimal, InvalidOperation
from pathlib import Path
from typing import Any

from .metrics.efficiency import _matmul_mnk
from .parser import load_profile
from .parser.shapes import parse_dtypes

SCHEMA_NAME = "llm.profiling-calibration-recipe"
SCHEMA_VERSION = "v1"
READINESS = "PROFILE_DERIVED_RECIPE"
EVIDENCE_ROLE = "DERIVED_FROM_PROFILING"

_HEX40_RE = re.compile(r"^[0-9a-f]{40}$")
_HEX64_RE = re.compile(r"^[0-9a-f]{64}$")
_CASE_ID_RE = re.compile(r"^case_[0-9a-f]{64}$")
_DRIVE_PATH_RE = re.compile(r"^[A-Za-z]:")
_POSITIVE_DIM_RE = re.compile(r"^[1-9][0-9]*$")
_MAX_DIMENSION = 2**63 - 1

_TORCH_NPU_AUXILIARY_FILES = (
    "step_trace_time.csv",
    "op_statistic.csv",
    "api_statistic.csv",
    "operator_details.csv",
    "communication.json",
    "communication_matrix.json",
    "memory_record.csv",
    "npu_module_mem.csv",
    "operator_memory.csv",
)

_PROFILE_LAYOUTS = {
    "TORCH_NPU_ASCEND_PROFILER_OUTPUT",
    "HYBRID_TORCH_NPU_MINDSTUDIO_DB",
    "MINDSTUDIO_DB",
    "MSPROF_OP_SUMMARY",
}
_SOURCE_ROLES = {
    "KERNEL_DETAILS_CSV",
    "MINDSTUDIO_DB",
    "MSPROF_OP_SUMMARY_CSV",
}
_CAPTURE_SCOPES = {
    "SINGLE_RANK",
    "SINGLE_RANK_OR_MATRIX_MISSING",
    "MULTI_RANK_MATRIX",
    "UNKNOWN",
}
_IMPLEMENTATION_HINTS = {"MATMUL", "MATMUL_V3", "GEMM", "GEMM_V3"}
_DTYPES = {"BF16", "FP16", "FP32", "FP8_E4M3"}
_UNMAPPED_REASONS = {
    "MISSING_SHAPE",
    "CONFLICTING_SHAPE",
    "AMBIGUOUS_TRANSPOSE",
    "MISSING_DTYPE",
    "CONFLICTING_DTYPE",
    "BATCH_MATMUL_SEMANTICS_INCOMPLETE",
    "GROUPED_MATMUL_SEMANTICS_INCOMPLETE",
    "ATTENTION_SEMANTICS_INCOMPLETE",
    "FUSED_SEMANTICS_INCOMPLETE",
}

_ORDINARY_TYPES = {
    "matmul": "MATMUL",
    "matmulv3": "MATMUL_V3",
    "gemm": "GEMM",
    "gemmv3": "GEMM_V3",
}
_DTYPE_ALIASES = {
    "BF16": "BF16",
    "BFLOAT16": "BF16",
    "DT_BF16": "BF16",
    "DT_BFLOAT16": "BF16",
    "FLOAT16": "FP16",
    "FP16": "FP16",
    "HALF": "FP16",
    "DT_FLOAT16": "FP16",
    "DT_FP16": "FP16",
    "FLOAT": "FP32",
    "FLOAT32": "FP32",
    "FP32": "FP32",
    "DT_FLOAT": "FP32",
    "DT_FLOAT32": "FP32",
    "FLOAT8_E4M3": "FP8_E4M3",
    "FP8_E4M3": "FP8_E4M3",
    "DT_FLOAT8_E4M3": "FP8_E4M3",
}


class RecipeBuildError(ValueError):
    """The profile cannot produce a trustworthy v1 recipe."""


class RecipeValidationError(ValueError):
    """A recipe violates the exact v1 contract or integrity chain."""


def canonical_json_bytes(value: Any) -> bytes:
    """Return the contract's deterministic canonical JSON representation."""
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _select_profile_sources(profile_dir: Path) -> tuple[str, list[tuple[str, Path]]]:
    kernel_csv = profile_dir / "kernel_details.csv"
    mindstudio_db = profile_dir / "mindstudio_insight_data.db"
    op_summaries = sorted(profile_dir.glob("op_summary_*.csv"), key=lambda item: item.name)

    selected: list[tuple[str, Path]]
    if kernel_csv.is_file():
        if mindstudio_db.is_file():
            layout = "HYBRID_TORCH_NPU_MINDSTUDIO_DB"
            selected = [
                ("KERNEL_DETAILS_CSV", kernel_csv),
                ("MINDSTUDIO_DB", mindstudio_db),
            ]
        else:
            layout = "TORCH_NPU_ASCEND_PROFILER_OUTPUT"
            selected = [("KERNEL_DETAILS_CSV", kernel_csv)]
    elif mindstudio_db.is_file():
        layout = "MINDSTUDIO_DB"
        selected = [("MINDSTUDIO_DB", mindstudio_db)]
    elif op_summaries:
        layout = "MSPROF_OP_SUMMARY"
        # Match parser.msprof._first(), which deliberately selects the final
        # lexically sorted export when a directory contains multiple snapshots.
        selected = [("MSPROF_OP_SUMMARY_CSV", op_summaries[-1])]
    else:
        raise RecipeBuildError("no supported profiler source")

    return layout, sorted(selected, key=lambda item: item[0])


def _source_manifest(selected: Sequence[tuple[str, Path]]) -> list[dict[str, str]]:
    return [
        {"role": role, "content_sha256": _sha256_file(path)}
        for role, path in selected
    ]


def _snapshot_profile(
    source_dir: Path,
    snapshot_dir: Path,
    selected: Sequence[tuple[str, Path]],
) -> list[tuple[str, Path]]:
    destinations = {
        "KERNEL_DETAILS_CSV": "kernel_details.csv",
        "MINDSTUDIO_DB": "mindstudio_insight_data.db",
    }
    snapshot_selected: list[tuple[str, Path]] = []
    try:
        for role, source in selected:
            destination = snapshot_dir / destinations.get(role, source.name)
            shutil.copyfile(source, destination)
            snapshot_selected.append((role, destination))

        if any(role == "KERNEL_DETAILS_CSV" for role, _ in selected):
            for name in _TORCH_NPU_AUXILIARY_FILES:
                source = source_dir / name
                if source.is_file():
                    shutil.copyfile(source, snapshot_dir / name)
        elif any(role == "MSPROF_OP_SUMMARY_CSV" for role, _ in selected):
            for source in sorted(source_dir.glob("task_time_slice_*.csv")):
                if source.is_file():
                    shutil.copyfile(source, snapshot_dir / source.name)
    except OSError as exc:
        raise RecipeBuildError("profile snapshot is unavailable") from exc
    for snapshot_file in snapshot_dir.iterdir():
        if snapshot_file.is_file():
            snapshot_file.chmod(stat.S_IREAD)
    return snapshot_selected


def _capture_scope(meta: Mapping[str, Any]) -> dict[str, str]:
    raw_scope = str(meta.get("profile_scope") or "").strip().lower()
    scope_map = {
        "single_rank": "SINGLE_RANK",
        "single_rank_or_matrix_missing": "SINGLE_RANK_OR_MATRIX_MISSING",
        "multi_rank_matrix": "MULTI_RANK_MATRIX",
    }
    scope = scope_map.get(raw_scope)
    if scope is None and "multi_card" in meta:
        scope = (
            "SINGLE_RANK_OR_MATRIX_MISSING"
            if bool(meta.get("multi_card"))
            else "SINGLE_RANK"
        )
    if scope is None:
        return {
            "scope": "UNKNOWN",
            "evidence": "UNAVAILABLE",
            "unavailable_reason": "PARSER_SCOPE_NOT_RECORDED",
        }
    return {
        "scope": scope,
        "evidence": "PARSER_METADATA",
        "unavailable_reason": "NONE",
    }


def _classify_candidate(raw_type: Any) -> tuple[str, str | None]:
    value = str(raw_type or "").strip()
    folded = value.casefold()
    hint = _ORDINARY_TYPES.get(folded)
    if hint:
        return "ordinary", hint
    if "groupedmatmul" in folded:
        return "unmapped", "GROUPED_MATMUL_SEMANTICS_INCOMPLETE"
    if "attention" in folded:
        return "unmapped", "ATTENTION_SEMANTICS_INCOMPLETE"
    if "batchmatmul" in folded:
        return "unmapped", "BATCH_MATMUL_SEMANTICS_INCOMPLETE"
    if "matmul" in folded or "gemm" in folded:
        return "unmapped", "FUSED_SEMANTICS_INCOMPLETE"
    return "non_candidate", None


def _canonical_dtype(value: str) -> str | None:
    return _DTYPE_ALIASES.get(str(value).strip().strip('"').upper())


def _row_value(row: Mapping[str, Any], key: str) -> Any:
    value = row.get(key)
    if value is None:
        return None
    try:
        if math.isnan(value):
            return None
    except (TypeError, ValueError):
        pass
    return value


def _parse_strict_matrix_shapes(raw: Any, tensor_count: int) -> list[list[int]] | None:
    """Parse an exact list of positive-decimal rank-2 matrices."""
    if raw is None:
        return None
    value = str(raw).strip()
    if not value or value.upper() in {"N/A", "NAN", "NONE", "NULL"}:
        return None
    tensors = value.split(";")
    if len(tensors) != tensor_count:
        return None
    parsed: list[list[int]] = []
    for tensor in tensors:
        dimensions = tensor.strip().split(",")
        if len(dimensions) != 2:
            return None
        tokens = [dimension.strip() for dimension in dimensions]
        if any(not _POSITIVE_DIM_RE.fullmatch(token) for token in tokens):
            return None
        values = [int(token, 10) for token in tokens]
        if any(value > _MAX_DIMENSION for value in values):
            return None
        parsed.append(values)
    return parsed


def _parse_ordinary_case(
    row: Mapping[str, Any], hint: str
) -> tuple[dict[str, Any] | None, str | None, Decimal | None, str | None]:
    raw_in_shapes = _row_value(row, "Input Shapes")
    raw_out_shapes = _row_value(row, "Output Shapes")
    if not raw_in_shapes or not raw_out_shapes:
        return None, "MISSING_SHAPE", None, None
    shapes_in = _parse_strict_matrix_shapes(raw_in_shapes, 2)
    shapes_out = _parse_strict_matrix_shapes(raw_out_shapes, 1)
    if shapes_in is None or shapes_out is None:
        raw_values = {str(raw_in_shapes).strip().upper(), str(raw_out_shapes).strip().upper()}
        reason = (
            "MISSING_SHAPE"
            if raw_values.intersection({"", "N/A", "NAN", "NONE", "NULL"})
            else "CONFLICTING_SHAPE"
        )
        return None, reason, None, None

    a, b = shapes_in
    out_m, out_n = shapes_out[0]
    orientations: list[tuple[int, int, int, bool, bool]] = []
    for transpose_a in (False, True):
        m, k_a = (a[0], a[1]) if not transpose_a else (a[1], a[0])
        for transpose_b in (False, True):
            k_b, n = (b[0], b[1]) if not transpose_b else (b[1], b[0])
            if min(m, n, k_a) > 0 and k_a == k_b and (m, n) == (out_m, out_n):
                orientations.append((m, n, k_a, transpose_a, transpose_b))
    if not orientations:
        return None, "CONFLICTING_SHAPE", None, None
    if len(set(orientations)) != 1:
        return None, "AMBIGUOUS_TRANSPOSE", None, None

    m, n, k, transpose_a, transpose_b = orientations[0]
    mnk = _matmul_mnk(shapes_in, shapes_out)
    if mnk is None or tuple(mnk) != (m, n, k, 1):
        return None, "CONFLICTING_SHAPE", None, None

    raw_in_dtypes = parse_dtypes(_row_value(row, "Input Data Types"))
    raw_out_dtypes = parse_dtypes(_row_value(row, "Output Data Types"))
    if len(raw_in_dtypes) != 2 or len(raw_out_dtypes) != 1:
        return None, "MISSING_DTYPE", None, None
    in_dtypes = [_canonical_dtype(value) for value in raw_in_dtypes]
    out_dtype = _canonical_dtype(raw_out_dtypes[0])
    if any(value is None for value in in_dtypes) or out_dtype is None:
        return None, "MISSING_DTYPE", None, None
    if in_dtypes[0] != in_dtypes[1] or in_dtypes[0] != out_dtype:
        return None, "CONFLICTING_DTYPE", None, None
    dtype = str(in_dtypes[0])

    semantic = {
        "canonical_op": "GEMM",
        "shape": {"m": m, "n": n, "k": k},
        "dtype": dtype,
        "transpose": {"a": transpose_a, "b": transpose_b},
        "implementation_hint": hint,
    }

    duration: Decimal | None
    try:
        duration = Decimal(str(_row_value(row, "Duration(us)")))
        if not duration.is_finite() or duration <= 0:
            duration = None
    except (InvalidOperation, TypeError, ValueError):
        duration = None

    evidence = {
        "implementation_hint": hint,
        "input_shapes": shapes_in,
        "output_shapes": shapes_out,
        "dtype": dtype,
        "transpose": semantic["transpose"],
        "duration_us": format(duration, "f") if duration is not None else "UNAVAILABLE",
    }
    evidence_digest = _sha256_bytes(canonical_json_bytes(evidence))
    return semantic, None, duration, evidence_digest


def _allocate_scores(weights: Mapping[str, Decimal | int]) -> dict[str, int]:
    if not weights:
        return {}
    total = sum(weights.values(), Decimal(0))
    if total <= 0:
        raise RecipeBuildError("priority denominator is not positive")
    exact = {
        case_id: (weight * Decimal(1_000_000) / total)
        for case_id, weight in weights.items()
    }
    scores = {
        case_id: int(value.to_integral_value(rounding=ROUND_FLOOR))
        for case_id, value in exact.items()
    }
    remainder = 1_000_000 - sum(scores.values())
    order = sorted(
        exact,
        key=lambda case_id: (-(exact[case_id] - Decimal(scores[case_id])), case_id),
    )
    for case_id in order[:remainder]:
        scores[case_id] += 1
    return scores


def _normalize_decimal_weights(weights: Mapping[str, Decimal]) -> dict[str, int]:
    """Return a lossless, dimensionless integer ratio for positive Decimals."""
    if not weights:
        return {}
    decimal_places = max(max(-weight.as_tuple().exponent, 0) for weight in weights.values())
    integers: dict[str, int] = {}
    for case_id, weight in weights.items():
        parts = weight.as_tuple()
        coefficient = 0
        for digit in parts.digits:
            coefficient = coefficient * 10 + digit
        integers[case_id] = coefficient * 10 ** (decimal_places + parts.exponent)
    divisor = 0
    for weight in integers.values():
        divisor = math.gcd(divisor, weight)
    if divisor <= 0:
        raise RecipeBuildError("priority weight is not positive")
    return {case_id: weight // divisor for case_id, weight in integers.items()}


def _mapped_ppm(mapped: int, candidates: int) -> int:
    return int(
        (Decimal(mapped) * Decimal(1_000_000) / Decimal(candidates)).to_integral_value(
            rounding=ROUND_HALF_UP
        )
    )


def build_recipe(profile_dir: str | Path, producer_revision: str) -> dict[str, Any]:
    """Build and validate a product-neutral recipe from an existing profile."""
    if not isinstance(producer_revision, str) or not _HEX40_RE.fullmatch(producer_revision):
        raise RecipeBuildError("producer revision must be a lowercase 40-hex commit")
    source_dir = Path(profile_dir)
    if not source_dir.is_dir():
        raise RecipeBuildError("profile directory is unavailable")

    layout, selected_paths = _select_profile_sources(source_dir)
    with tempfile.TemporaryDirectory(prefix="llminsight-recipe-") as temporary_dir:
        snapshot_dir = Path(temporary_dir)
        snapshot_selected = _snapshot_profile(source_dir, snapshot_dir, selected_paths)
        snapshot_layout, detected_snapshot_sources = _select_profile_sources(snapshot_dir)
        if (
            snapshot_layout != layout
            or [role for role, _ in detected_snapshot_sources]
            != [role for role, _ in snapshot_selected]
        ):
            raise RecipeBuildError("profile snapshot layout is inconsistent")
        selected_sources = _source_manifest(snapshot_selected)
        try:
            profile = load_profile(str(snapshot_dir))
        except OSError as exc:
            raise RecipeBuildError("profile snapshot could not be parsed") from exc
        layout_after_load, sources_after_load = _select_profile_sources(snapshot_dir)
        if (
            layout_after_load != layout
            or _source_manifest(sources_after_load) != selected_sources
        ):
            raise RecipeBuildError("profile snapshot changed while being parsed")
    kernel_details = profile.kernel_details
    if kernel_details is None:
        raise RecipeBuildError("profile parser did not return kernel details")

    groups: MutableMapping[bytes, dict[str, Any]] = {}
    unmapped_counts: Counter[str] = Counter()
    candidate_rows = 0
    mapped_rows = 0
    all_mapped_durations_valid = True

    for raw_row in kernel_details.to_dict(orient="records"):
        kind, classification = _classify_candidate(raw_row.get("Type"))
        if kind == "non_candidate":
            continue
        candidate_rows += 1
        if kind == "unmapped":
            unmapped_counts[str(classification)] += 1
            continue

        semantic, reason, duration, row_evidence_digest = _parse_ordinary_case(
            raw_row, str(classification)
        )
        if reason is not None or semantic is None or row_evidence_digest is None:
            unmapped_counts[str(reason)] += 1
            continue

        mapped_rows += 1
        if duration is None:
            all_mapped_durations_valid = False
        key = canonical_json_bytes(semantic)
        group = groups.setdefault(
            key,
            {
                "semantic": semantic,
                "count": 0,
                "duration_total": Decimal(0),
                "evidence_digests": [],
            },
        )
        group["count"] += 1
        if duration is not None:
            group["duration_total"] += duration
        group["evidence_digests"].append(row_evidence_digest)

    priority_basis = (
        "OBSERVED_TOTAL_DURATION"
        if mapped_rows > 0 and all_mapped_durations_valid
        else "FREQUENCY"
    )
    cases: list[dict[str, Any]] = []
    observed_weights: dict[str, Decimal] = {}
    for key in sorted(groups):
        group = groups[key]
        semantic = group["semantic"]
        case_id = "case_" + _sha256_bytes(canonical_json_bytes(semantic))
        if priority_basis == "OBSERVED_TOTAL_DURATION":
            observed_weights[case_id] = group["duration_total"]
        cases.append(
            {
                "case_id": case_id,
                **semantic,
                "frequency": {"count": group["count"]},
                "priority": {
                    "basis": priority_basis,
                    "rank": 0,
                    "score_ppm": 0,
                },
                "source_evidence": {
                    "digest_sha256": _sha256_bytes(
                        canonical_json_bytes(sorted(group["evidence_digests"]))
                    ),
                    "confidence": "HIGH",
                },
            }
        )

    if priority_basis == "OBSERVED_TOTAL_DURATION":
        ranking_weights: dict[str, int] = _normalize_decimal_weights(observed_weights)
        for case in cases:
            case["priority"]["weight"] = ranking_weights[case["case_id"]]
    else:
        ranking_weights = {
            case["case_id"]: case["frequency"]["count"]
            for case in cases
        }
    scores = _allocate_scores(ranking_weights)
    priority_order = sorted(
        ranking_weights,
        key=lambda case_id: (-ranking_weights[case_id], case_id),
    )
    ranks = {case_id: rank for rank, case_id in enumerate(priority_order, 1)}
    for case in cases:
        case_id = case["case_id"]
        case["priority"]["rank"] = ranks[case_id]
        case["priority"]["score_ppm"] = scores[case_id]
    cases.sort(key=lambda case: case["case_id"])

    if candidate_rows:
        mapping: dict[str, Any] = {
            "status": "AVAILABLE",
            "mapped_ppm": _mapped_ppm(mapped_rows, candidate_rows),
        }
    else:
        mapping = {"status": "UNAVAILABLE", "reason": "NO_CANDIDATE_ROWS"}

    lineage = {
        "producer_revision": producer_revision,
        "profile_layout": layout,
        "selected_sources": selected_sources,
        "source_manifest_sha256": _sha256_bytes(canonical_json_bytes(selected_sources)),
        "capture_scope": _capture_scope(profile.meta or {}),
    }
    recipe: dict[str, Any] = {
        "schema": SCHEMA_NAME,
        "version": SCHEMA_VERSION,
        "readiness": READINESS,
        "evidence_role": EVIDENCE_ROLE,
        "lineage": lineage,
        "coverage": {
            "profile_kernel_rows": len(kernel_details.index),
            "candidate_kernel_rows": candidate_rows,
            "mapped_kernel_rows": mapped_rows,
            "unmapped_candidate_kernel_rows": candidate_rows - mapped_rows,
            "mapped_case_count": len(cases),
            "mapping": mapping,
        },
        "cases": cases,
        "unmapped": [
            {"reason": reason, "count": count}
            for reason, count in sorted(unmapped_counts.items())
        ],
    }
    recipe["recipe_sha256"] = _sha256_bytes(canonical_json_bytes(recipe))
    validate_recipe(recipe)
    return recipe


def _exact_keys(value: Any, expected: Iterable[str], label: str) -> Mapping[str, Any]:
    if type(value) is not dict:
        raise RecipeValidationError(f"{label} must be an object")
    expected_set = set(expected)
    if set(value) != expected_set:
        raise RecipeValidationError(f"{label} has missing or extra fields")
    return value


def _require_string(value: Any, allowed: Iterable[str] | None, label: str) -> str:
    if type(value) is not str:
        raise RecipeValidationError(f"{label} must be a string")
    if allowed is not None and value not in set(allowed):
        raise RecipeValidationError(f"{label} has an invalid value")
    return value


def _require_int(value: Any, label: str, minimum: int = 0, maximum: int | None = None) -> int:
    if type(value) is not int:
        raise RecipeValidationError(f"{label} must be an integer")
    if value < minimum or (maximum is not None and value > maximum):
        raise RecipeValidationError(f"{label} is out of range")
    return value


def _reject_path_like_strings(value: Any) -> None:
    if type(value) is str:
        if "/" in value or "\\" in value or _DRIVE_PATH_RE.match(value):
            raise RecipeValidationError("path-like strings are forbidden")
        return
    if type(value) is list:
        for item in value:
            _reject_path_like_strings(item)
    elif type(value) is dict:
        for item in value.values():
            _reject_path_like_strings(item)


def validate_recipe(recipe: Mapping[str, Any]) -> None:
    """Validate exact fields, cross-field invariants, and all content digests."""
    top = _exact_keys(
        recipe,
        {
            "schema",
            "version",
            "readiness",
            "evidence_role",
            "lineage",
            "coverage",
            "cases",
            "unmapped",
            "recipe_sha256",
        },
        "recipe",
    )
    _require_string(top["schema"], {SCHEMA_NAME}, "schema")
    _require_string(top["version"], {SCHEMA_VERSION}, "version")
    _require_string(top["readiness"], {READINESS}, "readiness")
    _require_string(top["evidence_role"], {EVIDENCE_ROLE}, "evidence_role")
    _reject_path_like_strings(top)

    lineage = _exact_keys(
        top["lineage"],
        {
            "producer_revision",
            "profile_layout",
            "selected_sources",
            "source_manifest_sha256",
            "capture_scope",
        },
        "lineage",
    )
    producer_revision = _require_string(lineage["producer_revision"], None, "producer_revision")
    if not _HEX40_RE.fullmatch(producer_revision):
        raise RecipeValidationError("producer_revision must be lowercase 40-hex")
    layout = _require_string(lineage["profile_layout"], _PROFILE_LAYOUTS, "profile_layout")
    sources = lineage["selected_sources"]
    if type(sources) is not list or not sources:
        raise RecipeValidationError("selected_sources must be a non-empty array")
    source_roles: list[str] = []
    for index, source in enumerate(sources):
        source_obj = _exact_keys(source, {"role", "content_sha256"}, f"selected_sources[{index}]")
        role = _require_string(source_obj["role"], _SOURCE_ROLES, "source role")
        digest = _require_string(source_obj["content_sha256"], None, "source digest")
        if not _HEX64_RE.fullmatch(digest):
            raise RecipeValidationError("source digest must be lowercase sha256")
        source_roles.append(role)
    if source_roles != sorted(source_roles) or len(source_roles) != len(set(source_roles)):
        raise RecipeValidationError("selected_sources must be sorted and unique")
    expected_roles = {
        "TORCH_NPU_ASCEND_PROFILER_OUTPUT": ["KERNEL_DETAILS_CSV"],
        "HYBRID_TORCH_NPU_MINDSTUDIO_DB": ["KERNEL_DETAILS_CSV", "MINDSTUDIO_DB"],
        "MINDSTUDIO_DB": ["MINDSTUDIO_DB"],
        "MSPROF_OP_SUMMARY": ["MSPROF_OP_SUMMARY_CSV"],
    }[layout]
    if source_roles != sorted(expected_roles):
        raise RecipeValidationError("source roles do not match profile layout")
    manifest_digest = _require_string(
        lineage["source_manifest_sha256"], None, "source manifest digest"
    )
    if manifest_digest != _sha256_bytes(canonical_json_bytes(sources)):
        raise RecipeValidationError("source manifest digest mismatch")

    capture = _exact_keys(
        lineage["capture_scope"],
        {"scope", "evidence", "unavailable_reason"},
        "capture_scope",
    )
    scope = _require_string(capture["scope"], _CAPTURE_SCOPES, "capture scope")
    evidence = _require_string(
        capture["evidence"], {"PARSER_METADATA", "UNAVAILABLE"}, "capture evidence"
    )
    unavailable_reason = _require_string(
        capture["unavailable_reason"],
        {"NONE", "PARSER_SCOPE_NOT_RECORDED"},
        "capture unavailable reason",
    )
    if scope == "UNKNOWN":
        if (evidence, unavailable_reason) != ("UNAVAILABLE", "PARSER_SCOPE_NOT_RECORDED"):
            raise RecipeValidationError("unknown capture scope lacks unavailable reason")
    elif (evidence, unavailable_reason) != ("PARSER_METADATA", "NONE"):
        raise RecipeValidationError("known capture scope has inconsistent evidence")

    coverage = _exact_keys(
        top["coverage"],
        {
            "profile_kernel_rows",
            "candidate_kernel_rows",
            "mapped_kernel_rows",
            "unmapped_candidate_kernel_rows",
            "mapped_case_count",
            "mapping",
        },
        "coverage",
    )
    profile_rows = _require_int(coverage["profile_kernel_rows"], "profile_kernel_rows")
    candidate_rows = _require_int(coverage["candidate_kernel_rows"], "candidate_kernel_rows")
    mapped_rows = _require_int(coverage["mapped_kernel_rows"], "mapped_kernel_rows")
    unmapped_rows = _require_int(
        coverage["unmapped_candidate_kernel_rows"], "unmapped_candidate_kernel_rows"
    )
    mapped_case_count = _require_int(coverage["mapped_case_count"], "mapped_case_count")
    if candidate_rows > profile_rows or candidate_rows != mapped_rows + unmapped_rows:
        raise RecipeValidationError("coverage row counts are inconsistent")
    mapping = coverage["mapping"]
    if candidate_rows:
        mapping_obj = _exact_keys(mapping, {"status", "mapped_ppm"}, "coverage mapping")
        _require_string(mapping_obj["status"], {"AVAILABLE"}, "mapping status")
        mapped_ppm = _require_int(mapping_obj["mapped_ppm"], "mapped_ppm", 0, 1_000_000)
        if mapped_ppm != _mapped_ppm(mapped_rows, candidate_rows):
            raise RecipeValidationError("mapped_ppm is inconsistent")
    else:
        mapping_obj = _exact_keys(mapping, {"status", "reason"}, "coverage mapping")
        _require_string(mapping_obj["status"], {"UNAVAILABLE"}, "mapping status")
        _require_string(mapping_obj["reason"], {"NO_CANDIDATE_ROWS"}, "mapping reason")
        if mapped_rows or unmapped_rows:
            raise RecipeValidationError("zero-candidate coverage is inconsistent")

    cases = top["cases"]
    if type(cases) is not list:
        raise RecipeValidationError("cases must be an array")
    if mapped_case_count != len(cases):
        raise RecipeValidationError("mapped_case_count is inconsistent")
    case_ids: list[str] = []
    mapped_frequency = 0
    ranks: list[int] = []
    priority_bases: set[str] = set()
    observed_priority_weights: dict[str, int] = {}
    score_total = 0
    for index, case in enumerate(cases):
        case_obj = _exact_keys(
            case,
            {
                "case_id",
                "canonical_op",
                "shape",
                "dtype",
                "transpose",
                "implementation_hint",
                "frequency",
                "priority",
                "source_evidence",
            },
            f"cases[{index}]",
        )
        case_id = _require_string(case_obj["case_id"], None, "case_id")
        if not _CASE_ID_RE.fullmatch(case_id):
            raise RecipeValidationError("case_id has invalid format")
        _require_string(case_obj["canonical_op"], {"GEMM"}, "canonical_op")
        shape = _exact_keys(case_obj["shape"], {"m", "n", "k"}, "case shape")
        for dim in ("m", "n", "k"):
            _require_int(shape[dim], f"shape.{dim}", 1, _MAX_DIMENSION)
        dtype = _require_string(case_obj["dtype"], _DTYPES, "case dtype")
        transpose = _exact_keys(case_obj["transpose"], {"a", "b"}, "transpose")
        if type(transpose["a"]) is not bool or type(transpose["b"]) is not bool:
            raise RecipeValidationError("transpose flags must be booleans")
        hint = _require_string(
            case_obj["implementation_hint"], _IMPLEMENTATION_HINTS, "implementation_hint"
        )
        frequency = _exact_keys(case_obj["frequency"], {"count"}, "frequency")
        count = _require_int(frequency["count"], "frequency count", 1)
        mapped_frequency += count
        raw_priority = case_obj["priority"]
        if type(raw_priority) is not dict:
            raise RecipeValidationError("priority must be an object")
        basis = _require_string(
            raw_priority.get("basis"),
            {"OBSERVED_TOTAL_DURATION", "FREQUENCY"},
            "priority basis",
        )
        priority_keys = (
            {"basis", "rank", "score_ppm", "weight"}
            if basis == "OBSERVED_TOTAL_DURATION"
            else {"basis", "rank", "score_ppm"}
        )
        priority = _exact_keys(raw_priority, priority_keys, "priority")
        priority_bases.add(basis)
        if basis == "OBSERVED_TOTAL_DURATION":
            observed_priority_weights[case_id] = _require_int(
                priority["weight"], "priority weight", 1
            )
        ranks.append(_require_int(priority["rank"], "priority rank", 1))
        score_total += _require_int(priority["score_ppm"], "priority score", 0, 1_000_000)
        source_evidence = _exact_keys(
            case_obj["source_evidence"], {"digest_sha256", "confidence"}, "source_evidence"
        )
        source_digest = _require_string(
            source_evidence["digest_sha256"], None, "source evidence digest"
        )
        if not _HEX64_RE.fullmatch(source_digest):
            raise RecipeValidationError("source evidence digest has invalid format")
        _require_string(source_evidence["confidence"], {"HIGH"}, "source evidence confidence")
        semantic = {
            "canonical_op": "GEMM",
            "shape": dict(shape),
            "dtype": dtype,
            "transpose": dict(transpose),
            "implementation_hint": hint,
        }
        if case_id != "case_" + _sha256_bytes(canonical_json_bytes(semantic)):
            raise RecipeValidationError("case_id digest mismatch")
        case_ids.append(case_id)
    if case_ids != sorted(case_ids) or len(case_ids) != len(set(case_ids)):
        raise RecipeValidationError("cases must be sorted and unique")
    if mapped_frequency != mapped_rows:
        raise RecipeValidationError("case frequencies do not match mapped rows")
    if cases:
        if set(ranks) != set(range(1, len(cases) + 1)):
            raise RecipeValidationError("priority ranks are not contiguous")
        if len(priority_bases) != 1 or score_total != 1_000_000:
            raise RecipeValidationError("priority fields are inconsistent")
        priority_basis = next(iter(priority_bases))
        by_id = {case["case_id"]: case for case in cases}
        if priority_basis == "FREQUENCY":
            priority_weights: dict[str, Decimal | int] = {
                case_id: Decimal(case["frequency"]["count"])
                for case_id, case in by_id.items()
            }
        else:
            priority_weights = observed_priority_weights
            weight_divisor = 0
            for weight in observed_priority_weights.values():
                weight_divisor = math.gcd(weight_divisor, weight)
            if weight_divisor != 1:
                raise RecipeValidationError("observed priority weights are not normalized")
        expected_scores = _allocate_scores(priority_weights)
        expected_order = sorted(
            priority_weights,
            key=lambda case_id: (-priority_weights[case_id], case_id),
        )
        expected_ranks = {
            case_id: rank for rank, case_id in enumerate(expected_order, 1)
        }
        for case_id, case in by_id.items():
            if (
                case["priority"]["score_ppm"] != expected_scores[case_id]
                or case["priority"]["rank"] != expected_ranks[case_id]
            ):
                raise RecipeValidationError("priority is inconsistent with its weight basis")
    elif ranks or priority_bases or observed_priority_weights or score_total:
        raise RecipeValidationError("empty cases have priority data")

    unmapped = top["unmapped"]
    if type(unmapped) is not list:
        raise RecipeValidationError("unmapped must be an array")
    unmapped_reasons: list[str] = []
    unmapped_total = 0
    for index, item in enumerate(unmapped):
        item_obj = _exact_keys(item, {"reason", "count"}, f"unmapped[{index}]")
        unmapped_reasons.append(
            _require_string(item_obj["reason"], _UNMAPPED_REASONS, "unmapped reason")
        )
        unmapped_total += _require_int(item_obj["count"], "unmapped count", 1)
    if unmapped_reasons != sorted(unmapped_reasons) or len(unmapped_reasons) != len(set(unmapped_reasons)):
        raise RecipeValidationError("unmapped reasons must be sorted and unique")
    if unmapped_total != unmapped_rows:
        raise RecipeValidationError("unmapped counts do not match coverage")

    recipe_digest = _require_string(top["recipe_sha256"], None, "recipe_sha256")
    if not _HEX64_RE.fullmatch(recipe_digest):
        raise RecipeValidationError("recipe_sha256 has invalid format")
    unsigned = {key: value for key, value in top.items() if key != "recipe_sha256"}
    if recipe_digest != _sha256_bytes(canonical_json_bytes(unsigned)):
        raise RecipeValidationError("recipe_sha256 mismatch")


def recipe_json_bytes(recipe: Mapping[str, Any]) -> bytes:
    validate_recipe(recipe)
    return canonical_json_bytes(recipe) + b"\n"


def export_recipe(
    profile_dir: str | Path, output_path: str | Path, producer_revision: str
) -> dict[str, Any]:
    recipe = build_recipe(profile_dir, producer_revision)
    destination = Path(output_path)
    if not destination.parent.is_dir():
        raise RecipeBuildError("output directory is unavailable")
    destination.write_bytes(recipe_json_bytes(recipe))
    return recipe


def _reject_constant(value: str) -> None:
    raise RecipeValidationError("non-standard numeric constant is forbidden")


def _reject_duplicate_keys(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise RecipeValidationError("duplicate JSON key")
        result[key] = value
    return result


def load_and_validate_recipe(path: str | Path) -> dict[str, Any]:
    try:
        payload = Path(path).read_text(encoding="utf-8")
        value = json.loads(
            payload,
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_constant,
        )
    except RecipeValidationError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise RecipeValidationError("recipe JSON is unreadable or invalid") from exc
    validate_recipe(value)
    return value


class _SafeArgumentParser(argparse.ArgumentParser):
    def error(self, message: str) -> None:
        del message
        raise RecipeBuildError("invalid CLI arguments")


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = _SafeArgumentParser(
        prog="python -m llminsight.calibration_recipe",
        description="Export a product-neutral profiling-derived GEMM recipe.",
    )
    parser.add_argument("--profile", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--producer-revision", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    try:
        args = _build_arg_parser().parse_args(argv)
        recipe = export_recipe(args.profile, args.output, args.producer_revision)
    # Parser backends may raise pandas/sqlite exceptions that are intentionally not
    # part of this narrow module's dependency surface. Catch them at the CLI boundary
    # so their messages cannot echo a profile path or raw row.
    except Exception as exc:  # noqa: BLE001
        print(f"recipe_export_error type={type(exc).__name__}", file=sys.stderr)
        return 2
    coverage = recipe["coverage"]
    print(
        "recipe_export_ok "
        f"cases={coverage['mapped_case_count']} "
        f"mapped={coverage['mapped_kernel_rows']} "
        f"unmapped={coverage['unmapped_candidate_kernel_rows']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
