from __future__ import annotations

import copy
import csv
import hashlib
import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from llminsight.calibration_recipe import (
    RecipeBuildError,
    RecipeValidationError,
    build_recipe,
    canonical_json_bytes,
    export_recipe,
    load_and_validate_recipe,
    recipe_json_bytes,
    validate_recipe,
)
from llminsight.parser import load_profile as parser_load_profile

FIXTURE_PROFILE = Path(__file__).parent / "fixtures" / "profiling_recipe"
EXPECTED_RECIPE = FIXTURE_PROFILE / "expected_recipe.json"
SCHEMA_FILE = Path(__file__).parents[1] / "doc" / "contracts" / "llm.profiling-calibration-recipe.v1.schema.json"
PRODUCER_REVISION = "0123456789abcdef0123456789abcdef01234567"


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _resign(recipe: dict) -> None:
    unsigned = {key: value for key, value in recipe.items() if key != "recipe_sha256"}
    recipe["recipe_sha256"] = _sha256(canonical_json_bytes(unsigned))


def _write_kernel_rows(profile: Path, rows: list[dict[str, str]]) -> Path:
    source = profile / "kernel_details.csv"
    fieldnames = [
        "Name",
        "Type",
        "Accelerator Core",
        "Duration(us)",
        "Input Shapes",
        "Input Data Types",
        "Output Shapes",
        "Output Data Types",
    ]
    with source.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return source


class CalibrationRecipeBuildTests(unittest.TestCase):
    def test_builds_only_unambiguous_ordinary_gemm_cases(self) -> None:
        recipe = build_recipe(FIXTURE_PROFILE, PRODUCER_REVISION)

        self.assertEqual(recipe["schema"], "llm.profiling-calibration-recipe")
        self.assertEqual(recipe["version"], "v1")
        self.assertEqual(recipe["readiness"], "PROFILE_DERIVED_RECIPE")
        self.assertEqual(recipe["evidence_role"], "DERIVED_FROM_PROFILING")
        self.assertEqual(recipe["coverage"]["profile_kernel_rows"], 15)
        self.assertEqual(recipe["coverage"]["candidate_kernel_rows"], 14)
        self.assertEqual(recipe["coverage"]["mapped_kernel_rows"], 5)
        self.assertEqual(recipe["coverage"]["unmapped_candidate_kernel_rows"], 9)
        self.assertEqual(recipe["coverage"]["mapped_case_count"], 4)
        self.assertEqual(
            recipe["coverage"]["mapping"],
            {"status": "AVAILABLE", "mapped_ppm": 357143},
        )

        cases = recipe["cases"]
        self.assertEqual([case["case_id"] for case in cases], sorted(case["case_id"] for case in cases))
        by_hint = {case["implementation_hint"]: case for case in cases}
        self.assertEqual(set(by_hint), {"MATMUL", "MATMUL_V3", "GEMM", "GEMM_V3"})
        self.assertEqual(by_hint["MATMUL_V3"]["shape"], {"m": 2, "n": 5, "k": 3})
        self.assertEqual(by_hint["MATMUL_V3"]["transpose"], {"a": True, "b": False})
        self.assertEqual(by_hint["MATMUL_V3"]["frequency"], {"count": 2})
        self.assertEqual(by_hint["GEMM"]["transpose"], {"a": False, "b": True})
        self.assertEqual(by_hint["MATMUL"]["transpose"], {"a": True, "b": True})
        self.assertEqual(by_hint["GEMM_V3"]["transpose"], {"a": False, "b": False})
        self.assertEqual(by_hint["GEMM"]["dtype"], "FP16")
        self.assertEqual(by_hint["MATMUL"]["dtype"], "FP32")
        self.assertEqual(by_hint["GEMM_V3"]["dtype"], "FP8_E4M3")
        self.assertTrue(all(case["priority"]["basis"] == "OBSERVED_TOTAL_DURATION" for case in cases))
        self.assertEqual(
            {hint: case["priority"]["weight"] for hint, case in by_hint.items()},
            {"MATMUL": 2, "MATMUL_V3": 15, "GEMM": 20, "GEMM_V3": 3},
        )

    def test_rejects_incomplete_candidate_families_without_guessing(self) -> None:
        recipe = build_recipe(FIXTURE_PROFILE, PRODUCER_REVISION)
        counts = {item["reason"]: item["count"] for item in recipe["unmapped"]}

        self.assertEqual(
            counts,
            {
                "AMBIGUOUS_TRANSPOSE": 1,
                "ATTENTION_SEMANTICS_INCOMPLETE": 1,
                "BATCH_MATMUL_SEMANTICS_INCOMPLETE": 1,
                "CONFLICTING_DTYPE": 1,
                "CONFLICTING_SHAPE": 1,
                "FUSED_SEMANTICS_INCOMPLETE": 1,
                "GROUPED_MATMUL_SEMANTICS_INCOMPLETE": 1,
                "MISSING_DTYPE": 1,
                "MISSING_SHAPE": 1,
            },
        )

    def test_lineage_and_all_integrity_digests_are_bound(self) -> None:
        recipe = build_recipe(FIXTURE_PROFILE, PRODUCER_REVISION)
        selected = recipe["lineage"]["selected_sources"]
        source_bytes = (FIXTURE_PROFILE / "kernel_details.csv").read_bytes()

        self.assertEqual(recipe["lineage"]["producer_revision"], PRODUCER_REVISION)
        self.assertEqual(recipe["lineage"]["profile_layout"], "TORCH_NPU_ASCEND_PROFILER_OUTPUT")
        self.assertEqual(
            selected,
            [{"role": "KERNEL_DETAILS_CSV", "content_sha256": _sha256(source_bytes)}],
        )
        self.assertEqual(recipe["lineage"]["source_manifest_sha256"], _sha256(canonical_json_bytes(selected)))
        self.assertEqual(
            recipe["lineage"]["capture_scope"],
            {"scope": "SINGLE_RANK", "evidence": "PARSER_METADATA", "unavailable_reason": "NONE"},
        )

        for case in recipe["cases"]:
            semantic = {
                "canonical_op": case["canonical_op"],
                "shape": case["shape"],
                "dtype": case["dtype"],
                "transpose": case["transpose"],
                "implementation_hint": case["implementation_hint"],
            }
            self.assertEqual(case["case_id"], "case_" + _sha256(canonical_json_bytes(semantic)))
        unsigned = {key: value for key, value in recipe.items() if key != "recipe_sha256"}
        self.assertEqual(recipe["recipe_sha256"], _sha256(canonical_json_bytes(unsigned)))

    def test_hybrid_lineage_binds_both_selected_sources_without_paths(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            profile = Path(td)
            shutil.copyfile(FIXTURE_PROFILE / "kernel_details.csv", profile / "kernel_details.csv")
            (profile / "mindstudio_insight_data.db").write_bytes(b"synthetic-not-a-real-db")
            recipe = build_recipe(profile, PRODUCER_REVISION)

        self.assertEqual(recipe["lineage"]["profile_layout"], "HYBRID_TORCH_NPU_MINDSTUDIO_DB")
        self.assertEqual(
            [source["role"] for source in recipe["lineage"]["selected_sources"]],
            ["KERNEL_DETAILS_CSV", "MINDSTUDIO_DB"],
        )
        self.assertNotIn(str(profile), recipe_json_bytes(recipe).decode("utf-8"))

    def test_msprof_lineage_binds_the_same_last_op_summary_as_loader(self) -> None:
        columns = (
            "Model Name,Op Name,OP Type,Task Type,Task Start Time(us),Task Duration(us),"
            "Input Shapes,Input Data Types,Output Shapes,Output Data Types,Task ID\n"
        )
        with tempfile.TemporaryDirectory() as td:
            profile = Path(td)
            first = profile / "op_summary_001.csv"
            last = profile / "op_summary_999.csv"
            first.write_text(
                columns
                + 'synthetic,first,MatMul,AI_CORE,0,2,"2,3;3,5",BF16;BF16,"2,5",BF16,1\n',
                encoding="utf-8",
            )
            last.write_text(
                columns
                + 'synthetic,last,Gemm,AI_CORE,0,3,"4,3;3,7",BF16;BF16,"4,7",BF16,2\n',
                encoding="utf-8",
            )
            last_digest = _sha256(last.read_bytes())
            recipe = build_recipe(profile, PRODUCER_REVISION)

        self.assertEqual(recipe["cases"][0]["implementation_hint"], "GEMM")
        self.assertEqual(recipe["cases"][0]["shape"], {"m": 4, "n": 7, "k": 3})
        self.assertEqual(
            recipe["lineage"]["selected_sources"],
            [{"role": "MSPROF_OP_SUMMARY_CSV", "content_sha256": last_digest}],
        )

    def test_fractional_or_malformed_shape_segments_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            profile = Path(td)
            source = profile / "kernel_details.csv"
            source.write_text(
                "Name,Type,Accelerator Core,Duration(us),Input Shapes,Input Data Types,"
                "Output Shapes,Output Data Types\n"
                'fractional,MatMul,AI_CORE,1,"2.9,3;3,5",BF16;BF16,"2,5",BF16\n'
                'extra_bad,MatMul,AI_CORE,1,"2,3;3,5;garbage",BF16;BF16,"2,5",BF16\n',
                encoding="utf-8",
            )
            recipe = build_recipe(profile, PRODUCER_REVISION)

        self.assertEqual(recipe["coverage"]["mapped_kernel_rows"], 0)
        self.assertEqual(
            recipe["unmapped"],
            [{"reason": "CONFLICTING_SHAPE", "count": 2}],
        )

    def test_shape_lexer_preserves_large_decimal_and_rejects_invalid_forms(self) -> None:
        rows = []
        for name, input_shapes, output_shapes in (
            ("large_exact", "9007199254740993,3;3,5", "9007199254740993,5"),
            ("fractional", "2.9,3;3,5", "2,5"),
            ("scientific", "2e3,3;3,5", "2000,5"),
            ("overflow", "9223372036854775808,3;3,5", "9223372036854775808,5"),
            ("garbage_tensor", "2,3;3,5;garbage", "2,5"),
        ):
            rows.append(
                {
                    "Name": name,
                    "Type": "MatMul",
                    "Accelerator Core": "AI_CORE",
                    "Duration(us)": "1",
                    "Input Shapes": input_shapes,
                    "Input Data Types": "BF16;BF16",
                    "Output Shapes": output_shapes,
                    "Output Data Types": "BF16",
                }
            )
        with tempfile.TemporaryDirectory() as td:
            profile = Path(td)
            _write_kernel_rows(profile, rows)
            recipe = build_recipe(profile, PRODUCER_REVISION)

        self.assertEqual(recipe["coverage"]["mapped_kernel_rows"], 1)
        self.assertEqual(recipe["cases"][0]["shape"]["m"], 9007199254740993)
        self.assertEqual(
            recipe["unmapped"],
            [{"reason": "CONFLICTING_SHAPE", "count": 4}],
        )

    def test_shape_lexer_rejects_embedded_quote_garbage(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            profile = Path(td)
            _write_kernel_rows(
                profile,
                [
                    {
                        "Name": "quoted_garbage",
                        "Type": "MatMul",
                        "Accelerator Core": "AI_CORE",
                        "Duration(us)": "1",
                        "Input Shapes": '2", "3;3,5',
                        "Input Data Types": "BF16;BF16",
                        "Output Shapes": "2,5",
                        "Output Data Types": "BF16",
                    }
                ],
            )
            recipe = build_recipe(profile, PRODUCER_REVISION)

        self.assertEqual(recipe["coverage"]["mapped_kernel_rows"], 0)
        self.assertEqual(
            recipe["unmapped"],
            [{"reason": "CONFLICTING_SHAPE", "count": 1}],
        )

    def test_export_is_byte_deterministic_and_matches_consumer_fixture(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            first = Path(td) / "first.json"
            second = Path(td) / "second.json"
            export_recipe(FIXTURE_PROFILE, first, PRODUCER_REVISION)
            export_recipe(FIXTURE_PROFILE, second, PRODUCER_REVISION)
            first_bytes = first.read_bytes()
            second_bytes = second.read_bytes()

        self.assertEqual(first_bytes, second_bytes)
        self.assertEqual(first_bytes, EXPECTED_RECIPE.read_bytes())
        self.assertEqual(first_bytes, recipe_json_bytes(build_recipe(FIXTURE_PROFILE, PRODUCER_REVISION)))
        self.assertTrue(first_bytes.endswith(b"\n"))
        self.assertFalse(first_bytes.endswith(b"\n\n"))

    def test_output_does_not_leak_paths_raw_rows_latency_or_product_identity(self) -> None:
        payload = recipe_json_bytes(build_recipe(FIXTURE_PROFILE, PRODUCER_REVISION)).decode("utf-8")
        lowered = payload.lower()

        self.assertNotIn(str(FIXTURE_PROFILE).lower(), lowered)
        self.assertNotIn("administrator", lowered)
        self.assertNotIn("profiling_recipe", lowered)
        self.assertNotIn("synthetic_matmul", lowered)
        for forbidden in ("duration_us", "latency", "mean", "p50", "maximum", "target", "product_ref", "950pr", "950dt"):
            self.assertNotIn(forbidden, lowered)

    def test_invalid_duration_falls_back_all_priorities_to_frequency(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            profile = Path(td)
            shutil.copyfile(FIXTURE_PROFILE / "kernel_details.csv", profile / "kernel_details.csv")
            with (profile / "kernel_details.csv").open(encoding="utf-8", newline="") as fh:
                rows = list(csv.DictReader(fh))
            rows[0]["Duration(us)"] = "N/A"
            with (profile / "kernel_details.csv").open("w", encoding="utf-8", newline="") as fh:
                writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
            recipe = build_recipe(profile, PRODUCER_REVISION)

        self.assertTrue(all(case["priority"]["basis"] == "FREQUENCY" for case in recipe["cases"]))
        self.assertEqual(sum(case["priority"]["score_ppm"] for case in recipe["cases"]), 1000000)

    def test_zero_candidate_coverage_is_unavailable_not_zero(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            profile = Path(td)
            (profile / "kernel_details.csv").write_text(
                "Name,Type,Accelerator Core,Duration(us),Input Shapes,Input Data Types,Output Shapes,Output Data Types\n"
                "only_add,Add,AI_VECTOR_CORE,1.0,\"2,5;2,5\",BF16;BF16,\"2,5\",BF16\n",
                encoding="utf-8",
            )
            recipe = build_recipe(profile, PRODUCER_REVISION)

        self.assertEqual(recipe["coverage"]["candidate_kernel_rows"], 0)
        self.assertEqual(recipe["coverage"]["mapping"], {"status": "UNAVAILABLE", "reason": "NO_CANDIDATE_ROWS"})
        self.assertEqual(recipe["cases"], [])
        self.assertEqual(recipe["unmapped"], [])

    def test_parser_and_lineage_consume_the_same_snapshot_across_aba_change(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            profile = Path(td)
            source = _write_kernel_rows(
                profile,
                [
                    {
                        "Name": "content_a",
                        "Type": "MatMul",
                        "Accelerator Core": "AI_CORE",
                        "Duration(us)": "1",
                        "Input Shapes": "2,3;3,5",
                        "Input Data Types": "BF16;BF16",
                        "Output Shapes": "2,5",
                        "Output Data Types": "BF16",
                    }
                ],
            )
            content_a = source.read_bytes()
            content_b = content_a.replace(b"content_a", b"content_b").replace(
                b'2,3;3,5', b'4,3;3,7'
            ).replace(b'2,5', b'4,7')

            def load_during_aba(path: str):
                source.write_bytes(content_b)
                try:
                    return parser_load_profile(path)
                finally:
                    source.write_bytes(content_a)

            with mock.patch(
                "llminsight.calibration_recipe.load_profile",
                side_effect=load_during_aba,
            ):
                recipe = build_recipe(profile, PRODUCER_REVISION)

        self.assertEqual(recipe["cases"][0]["shape"], {"m": 2, "n": 5, "k": 3})
        self.assertEqual(
            recipe["lineage"]["selected_sources"],
            [{"role": "KERNEL_DETAILS_CSV", "content_sha256": _sha256(content_a)}],
        )

    def test_snapshot_change_during_parse_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            profile = Path(td)
            shutil.copyfile(FIXTURE_PROFILE / "kernel_details.csv", profile / "kernel_details.csv")

            def load_then_change_snapshot(path: str):
                parsed = parser_load_profile(path)
                snapshot_source = Path(path) / "kernel_details.csv"
                snapshot_source.write_bytes(snapshot_source.read_bytes() + b"\n")
                return parsed

            with (
                mock.patch(
                    "llminsight.calibration_recipe.load_profile",
                    side_effect=load_then_change_snapshot,
                ),
                self.assertRaises(RecipeBuildError),
            ):
                build_recipe(profile, PRODUCER_REVISION)


class CalibrationRecipeValidationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.recipe = build_recipe(FIXTURE_PROFILE, PRODUCER_REVISION)

    def assertRejected(self, recipe: dict) -> None:
        with self.assertRaises(RecipeValidationError):
            validate_recipe(recipe)

    def test_rejects_extra_missing_and_wrong_numeric_types(self) -> None:
        mutations = []
        extra = copy.deepcopy(self.recipe)
        extra["target"] = "forbidden"
        mutations.append(extra)
        missing = copy.deepcopy(self.recipe)
        del missing["coverage"]
        mutations.append(missing)
        boolean_count = copy.deepcopy(self.recipe)
        boolean_count["coverage"]["profile_kernel_rows"] = True
        mutations.append(boolean_count)
        float_count = copy.deepcopy(self.recipe)
        float_count["coverage"]["mapped_kernel_rows"] = 5.0
        mutations.append(float_count)
        negative = copy.deepcopy(self.recipe)
        negative["coverage"]["mapped_kernel_rows"] = -1
        mutations.append(negative)
        nan_value = copy.deepcopy(self.recipe)
        nan_value["coverage"]["mapped_kernel_rows"] = float("nan")
        mutations.append(nan_value)
        path_value = copy.deepcopy(self.recipe)
        path_value["lineage"]["selected_sources"][0]["content_sha256"] = "C:\\private\\profile.csv"
        mutations.append(path_value)

        for mutation in mutations:
            with self.subTest(mutation=len(mutations)):
                self.assertRejected(mutation)

    def test_rejects_manifest_case_and_recipe_digest_tampering(self) -> None:
        manifest = copy.deepcopy(self.recipe)
        manifest["lineage"]["source_manifest_sha256"] = "0" * 64
        _resign(manifest)
        self.assertRejected(manifest)

        case_id = copy.deepcopy(self.recipe)
        case_id["cases"][0]["case_id"] = "case_" + "0" * 64
        case_id["cases"].sort(key=lambda item: item["case_id"])
        _resign(case_id)
        self.assertRejected(case_id)

        recipe_digest = copy.deepcopy(self.recipe)
        recipe_digest["cases"][0]["source_evidence"]["digest_sha256"] = "0" * 64
        self.assertRejected(recipe_digest)

        fully_resigned_but_inconsistent = copy.deepcopy(self.recipe)
        fully_resigned_but_inconsistent["coverage"]["mapped_kernel_rows"] += 1
        _resign(fully_resigned_but_inconsistent)
        self.assertRejected(fully_resigned_but_inconsistent)

    def test_rejects_unsorted_duplicate_and_invalid_nested_fields(self) -> None:
        unsorted_cases = copy.deepcopy(self.recipe)
        unsorted_cases["cases"].reverse()
        _resign(unsorted_cases)
        self.assertRejected(unsorted_cases)

        duplicate_source = copy.deepcopy(self.recipe)
        duplicate_source["lineage"]["selected_sources"].append(
            copy.deepcopy(duplicate_source["lineage"]["selected_sources"][0])
        )
        duplicate_source["lineage"]["source_manifest_sha256"] = _sha256(
            canonical_json_bytes(duplicate_source["lineage"]["selected_sources"])
        )
        _resign(duplicate_source)
        self.assertRejected(duplicate_source)

        boolean_dimension = copy.deepcopy(self.recipe)
        boolean_dimension["cases"][0]["shape"]["m"] = True
        _resign(boolean_dimension)
        self.assertRejected(boolean_dimension)

        non_boolean_transpose = copy.deepcopy(self.recipe)
        non_boolean_transpose["cases"][0]["transpose"]["a"] = 1
        _resign(non_boolean_transpose)
        self.assertRejected(non_boolean_transpose)

    def test_rejects_resigned_priority_semantic_inconsistency(self) -> None:
        frequency_lie = copy.deepcopy(self.recipe)
        for case in frequency_lie["cases"]:
            case["priority"]["basis"] = "FREQUENCY"
        _resign(frequency_lie)
        self.assertRejected(frequency_lie)

        observed_rank_lie = copy.deepcopy(self.recipe)
        by_rank = sorted(observed_rank_lie["cases"], key=lambda item: item["priority"]["rank"])
        by_rank[0]["priority"]["rank"], by_rank[-1]["priority"]["rank"] = (
            by_rank[-1]["priority"]["rank"],
            by_rank[0]["priority"]["rank"],
        )
        _resign(observed_rank_lie)
        self.assertRejected(observed_rank_lie)

        observed_score_and_rank_lie = copy.deepcopy(self.recipe)
        self.assertTrue(
            all("weight" in case["priority"] for case in observed_score_and_rank_lie["cases"])
        )
        ordered = sorted(observed_score_and_rank_lie["cases"], key=lambda item: item["case_id"])
        for rank, case in enumerate(reversed(ordered), 1):
            case["priority"]["rank"] = rank
            case["priority"]["score_ppm"] = 250000
        _resign(observed_score_and_rank_lie)
        self.assertRejected(observed_score_and_rank_lie)

        observed_noncanonical_weights = copy.deepcopy(self.recipe)
        for case in observed_noncanonical_weights["cases"]:
            case["priority"]["weight"] *= 2
        _resign(observed_noncanonical_weights)
        self.assertRejected(observed_noncanonical_weights)

    def test_observed_priority_tie_break_is_recomputed_from_case_id(self) -> None:
        rows = []
        for name, op_type in (("matmul", "MatMul"), ("gemm", "Gemm")):
            rows.append(
                {
                    "Name": name,
                    "Type": op_type,
                    "Accelerator Core": "AI_CORE",
                    "Duration(us)": "1",
                    "Input Shapes": "2,3;3,5",
                    "Input Data Types": "BF16;BF16",
                    "Output Shapes": "2,5",
                    "Output Data Types": "BF16",
                }
            )
        with tempfile.TemporaryDirectory() as td:
            profile = Path(td)
            _write_kernel_rows(profile, rows)
            tied = build_recipe(profile, PRODUCER_REVISION)

        by_rank = sorted(tied["cases"], key=lambda item: item["priority"]["rank"])
        self.assertEqual(
            [case["case_id"] for case in by_rank],
            sorted(case["case_id"] for case in tied["cases"]),
        )
        by_rank[0]["priority"]["rank"], by_rank[1]["priority"]["rank"] = 2, 1
        _resign(tied)
        self.assertRejected(tied)

    def test_load_rejects_duplicate_keys_and_nonstandard_nan(self) -> None:
        payload = recipe_json_bytes(self.recipe).decode("utf-8")
        duplicate = payload.replace(
            '"schema":"llm.profiling-calibration-recipe"',
            '"schema":"llm.profiling-calibration-recipe","schema":"llm.profiling-calibration-recipe"',
            1,
        )
        nan_payload = payload.replace('"mapped_kernel_rows":5', '"mapped_kernel_rows":NaN', 1)

        with tempfile.TemporaryDirectory() as td:
            duplicate_path = Path(td) / "duplicate.json"
            duplicate_path.write_text(duplicate, encoding="utf-8")
            nan_path = Path(td) / "nan.json"
            nan_path.write_text(nan_payload, encoding="utf-8")
            with self.assertRaises(RecipeValidationError):
                load_and_validate_recipe(duplicate_path)
            with self.assertRaises(RecipeValidationError):
                load_and_validate_recipe(nan_path)

    def test_checked_in_json_schema_freezes_all_material_object_shapes(self) -> None:
        schema = json.loads(SCHEMA_FILE.read_text(encoding="utf-8"))

        self.assertEqual(schema["$schema"], "https://json-schema.org/draft/2020-12/schema")
        self.assertFalse(schema["additionalProperties"])
        self.assertEqual(schema["properties"]["readiness"]["const"], "PROFILE_DERIVED_RECIPE")
        self.assertEqual(schema["properties"]["evidence_role"]["const"], "DERIVED_FROM_PROFILING")
        for name, definition in schema["$defs"].items():
            if definition.get("type") == "object":
                with self.subTest(definition=name):
                    self.assertIs(definition.get("additionalProperties"), False)


class CalibrationRecipeCliTests(unittest.TestCase):
    def test_cli_exports_path_free_valid_recipe(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            output = Path(td) / "recipe.json"
            result = subprocess.run(
                [
                    sys.executable,
                    "-B",
                    "-m",
                    "llminsight.calibration_recipe",
                    "--profile",
                    str(FIXTURE_PROFILE.resolve()),
                    "--output",
                    str(output.resolve()),
                    "--producer-revision",
                    PRODUCER_REVISION,
                ],
                cwd=Path(__file__).parents[1],
                text=True,
                capture_output=True,
                encoding="utf-8",
                check=False,
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("cases=4", result.stdout)
            self.assertIn("mapped=5", result.stdout)
            self.assertIn("unmapped=9", result.stdout)
            self.assertNotIn(str(FIXTURE_PROFILE.resolve()), result.stdout)
            self.assertNotIn(str(output.resolve()), result.stdout)
            loaded = load_and_validate_recipe(output)
            self.assertEqual(loaded["recipe_sha256"], build_recipe(FIXTURE_PROFILE, PRODUCER_REVISION)["recipe_sha256"])

    def test_cli_failure_does_not_echo_profile_or_output_paths(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            missing_profile = Path(td) / "private-profile-name"
            output = Path(td) / "private-output-name.json"
            result = subprocess.run(
                [
                    sys.executable,
                    "-B",
                    "-m",
                    "llminsight.calibration_recipe",
                    "--profile",
                    str(missing_profile),
                    "--output",
                    str(output),
                    "--producer-revision",
                    PRODUCER_REVISION,
                ],
                cwd=Path(__file__).parents[1],
                text=True,
                capture_output=True,
                encoding="utf-8",
                check=False,
            )

        self.assertEqual(result.returncode, 2)
        self.assertNotIn(str(missing_profile), result.stderr)
        self.assertNotIn(str(output), result.stderr)
        self.assertIn("RecipeBuildError", result.stderr)

    def test_cli_argument_error_does_not_echo_unknown_argv_or_path(self) -> None:
        marker = "private-path-marker"
        result = subprocess.run(
            [
                sys.executable,
                "-B",
                "-m",
                "llminsight.calibration_recipe",
                "--profiel",
                marker,
                "--output",
                marker,
                "--producer-revision",
                PRODUCER_REVISION,
            ],
            cwd=Path(__file__).parents[1],
            text=True,
            capture_output=True,
            encoding="utf-8",
            check=False,
        )

        self.assertEqual(result.returncode, 2)
        self.assertNotIn(marker, result.stderr)
        self.assertNotIn("--profiel", result.stderr)
        self.assertEqual(result.stderr.strip(), "recipe_export_error type=RecipeBuildError")


if __name__ == "__main__":
    unittest.main()
