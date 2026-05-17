# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU-only tests for the Qwen3.6 cold-prefill benchmark harness."""

import argparse
import importlib.util
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


_REPO_ROOT = Path(__file__).resolve().parents[5]
_BENCHMARK_PATH = _REPO_ROOT / "validation_scripts" / "qwen36_cold_prefill_benchmark.py"


def _load_benchmark():
    spec = importlib.util.spec_from_file_location(
        "qwen36_cold_prefill_benchmark",
        _BENCHMARK_PATH,
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _args(prompt_dir: Path):
    args = argparse.Namespace(
        model_path="/tmp/model",
        compiled_artifacts="/tmp/artifacts",
        compiled_artifacts_by_len=None,
        prompt_lengths=[128],
        seq_len=2048,
        max_tokens_values=[1, 32],
        tensor_parallel_size=4,
        logical_nc_config=2,
        max_num_seqs=1,
        ctx_batch_size=1,
        compiled_max_prompt_length=None,
        block_size=128,
        kernel_q_tile_size=128,
        kernel_kv_tile_size=1024,
        num_gpu_blocks_override=None,
        prompt_dir=prompt_dir,
        include_128k=False,
        include_262k=False,
        include_dense_fallback=False,
        variants=None,
        repetitions=1,
        dry_run=True,
        fail_fast=False,
    )
    args._tokenizer = None
    args._prompt_cache = {}
    return args


class _WhitespaceTokenizer:
    def encode(self, prompt, add_special_tokens=False):
        return prompt.split()


class _EvenTokenCountTokenizer:
    def encode(self, prompt, add_special_tokens=False):
        return [0] * (len(prompt.split()) * 2)


class TestColdPrefillBenchmark(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.benchmark = _load_benchmark()

    def test_128k_candidate_uses_short_buckets_and_prompt_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            spec = next(
                item
                for item in self.benchmark._variant_specs()
                if item["name"] == "H_128k_candidate"
            )

            row = self.benchmark._run_case(
                _args(Path(tmpdir)),
                spec,
                131072,
                self.benchmark._tile_cases(spec, _args(Path(tmpdir)))[0],
                1,
            )

        self.assertIn("--prompt-file", row["command"])
        self.assertNotIn("--prompt", row["command"])
        self.assertEqual(row["benchmark_schema_version"], 1)
        self.assertTrue(row["benchmark_script"].endswith("qwen36_cold_prefill_benchmark.py"))
        self.assertTrue(row["runner"].endswith("run_offline_inference.py"))
        self.assertIn("--cte-bucket-profile", row["command"])
        self.assertIn("short", row["command"])
        self.assertIn("--compact-cte-attention-mask", row["command"])
        self.assertIn("--text-only-cte", row["command"])
        self.assertEqual(row["block_size"], 128)
        self.assertEqual(row["max_tokens"], 1)
        self.assertEqual(row["model_path"], "/tmp/model")
        self.assertEqual(row["compiled_artifacts"], "/tmp/artifacts")
        self.assertEqual(row["max_model_len"], 131072)
        self.assertEqual(row["seq_len"], 131072)
        self.assertEqual(row["tensor_parallel_size"], 4)
        self.assertEqual(row["logical_nc_config"], 2)
        self.assertEqual(row["max_num_seqs"], 1)
        self.assertEqual(row["ctx_batch_size"], 1)
        self.assertEqual(row["command"][row["command"].index("--max-num-seqs") + 1], "1")
        self.assertIsNone(row["prompt_token_count"])
        self.assertIn("prompt_sha256", row)
        self.assertEqual(len(row["prompt_sha256"]), 64)
        self.assertEqual(row["command"][row["command"].index("--max-tokens") + 1], "1")
        self.assertLess(len(" ".join(row["command"])), 2000)

    def test_compiled_max_prompt_length_is_passed_to_runner(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            args = _args(Path(tmpdir))
            args.compiled_max_prompt_length = 1024
            spec = next(
                item
                for item in self.benchmark._variant_specs()
                if item["name"] == "A_single512_old_chunked"
            )

            row = self.benchmark._run_case(
                args,
                spec,
                128,
                self.benchmark._tile_cases(spec, args)[0],
                1,
            )

        self.assertIn("--compiled-max-prompt-length", row["command"])
        self.assertEqual(
            row["command"][row["command"].index("--compiled-max-prompt-length") + 1],
            "1024",
        )

    def test_per_length_compiled_artifacts_override_default_path(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            args = _args(Path(tmpdir))
            args.compiled_artifacts_by_len = [
                "2048=/tmp/artifacts-2k",
                "131072=/tmp/artifacts-128k",
                "262144=/tmp/artifacts-262k",
            ]
            specs = {item["name"]: item for item in self.benchmark._variant_specs()}

            short = self.benchmark._run_case(
                args,
                specs["A_single512_old_chunked"],
                128,
                self.benchmark._tile_cases(specs["A_single512_old_chunked"], args)[0],
                1,
            )
            long_128k = self.benchmark._run_case(
                args,
                specs["H_128k_candidate"],
                131072,
                self.benchmark._tile_cases(specs["H_128k_candidate"], args)[0],
                1,
            )
            long_262k = self.benchmark._run_case(
                args,
                specs["I_262k_recovery_block256"],
                262144,
                self.benchmark._tile_cases(specs["I_262k_recovery_block256"], args)[0],
                1,
            )

        self.assertEqual(short["compiled_artifacts"], "/tmp/artifacts-2k")
        self.assertEqual(long_128k["compiled_artifacts"], "/tmp/artifacts-128k")
        self.assertEqual(long_262k["compiled_artifacts"], "/tmp/artifacts-262k")
        self.assertIn("/tmp/artifacts-128k", long_128k["command"])
        self.assertIn("/tmp/artifacts-262k", long_262k["command"])
        self.assertNotIn("/tmp/artifacts", long_128k["command"])

    def test_run_case_records_compiled_artifact_path_evidence(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            artifact_dir = Path(tmpdir) / "artifacts"
            artifact_dir.mkdir()
            (artifact_dir / "manifest.json").write_text("{}", encoding="utf-8")
            args = _args(Path(tmpdir))
            args.compiled_artifacts = str(artifact_dir)
            spec = next(
                item
                for item in self.benchmark._variant_specs()
                if item["name"] == "A_single512_old_chunked"
            )

            row = self.benchmark._run_case(
                args,
                spec,
                128,
                self.benchmark._tile_cases(spec, args)[0],
                1,
            )

        self.assertEqual(row["compiled_artifacts"], str(artifact_dir))
        self.assertEqual(row["compiled_artifacts_resolved"], str(artifact_dir.resolve()))
        self.assertTrue(row["compiled_artifacts_path_exists"])
        self.assertTrue(row["compiled_artifacts_path_nonempty"])

    def test_compiled_artifact_path_evidence_rejects_empty_directory(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            artifact_dir = Path(tmpdir) / "artifacts"
            artifact_dir.mkdir()

            evidence = self.benchmark._compiled_artifact_path_evidence(
                str(artifact_dir)
            )

        self.assertTrue(evidence["compiled_artifacts_path_exists"])
        self.assertFalse(evidence["compiled_artifacts_path_nonempty"])

    def test_invalid_per_length_compiled_artifact_mapping_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            args = _args(Path(tmpdir))
            args.compiled_artifacts_by_len = ["131072"]

            with self.assertRaisesRegex(ValueError, "LEN=PATH"):
                self.benchmark._compiled_artifacts_by_len(args)

    def test_run_case_can_emit_end_to_end_max_tokens_variant(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            spec = next(
                item
                for item in self.benchmark._variant_specs()
                if item["name"] == "A_single512_old_chunked"
            )

            row = self.benchmark._run_case(
                _args(Path(tmpdir)),
                spec,
                128,
                self.benchmark._tile_cases(spec, _args(Path(tmpdir)))[0],
                32,
                repetition=2,
            )

        self.assertEqual(row["max_tokens"], 32)
        self.assertEqual(row["temperature"], 0)
        self.assertEqual(row["top_k"], 1)
        self.assertEqual(row["repetition"], 2)
        self.assertEqual(row["command"][row["command"].index("--max-tokens") + 1], "32")

    def test_prompt_generation_can_calibrate_with_tokenizer(self):
        prompt, token_count = self.benchmark._prompt_for_target_tokens(
            32,
            _WhitespaceTokenizer(),
        )

        self.assertEqual(token_count, 32)
        self.assertEqual(len(prompt.split()), 32)

    def test_prompt_generation_prefers_not_to_overshoot_target(self):
        _prompt, token_count = self.benchmark._prompt_for_target_tokens(
            31,
            _EvenTokenCountTokenizer(),
        )

        self.assertLessEqual(token_count, 31)
        self.assertEqual(token_count, 30)

    def test_load_tokenizer_skips_missing_absolute_paths(self):
        self.assertIsNone(self.benchmark._load_tokenizer("/definitely/missing/qwen"))

    def test_run_case_reuses_cached_tokenized_prompt(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            args = _args(Path(tmpdir))
            args._tokenizer = _WhitespaceTokenizer()
            spec = next(
                item
                for item in self.benchmark._variant_specs()
                if item["name"] == "A_single512_old_chunked"
            )

            first = self.benchmark._run_case(
                args,
                spec,
                32,
                self.benchmark._tile_cases(spec, args)[0],
                1,
            )
            second = self.benchmark._run_case(
                args,
                spec,
                32,
                self.benchmark._tile_cases(spec, args)[0],
                32,
            )

        self.assertEqual(first["prompt_token_count"], 32)
        self.assertEqual(second["prompt_token_count"], 32)
        self.assertEqual(first["prompt_sha256"], second["prompt_sha256"])

    def test_262k_recovery_profiles_use_262k_bucket_and_expected_blocks(self):
        specs = {
            item["name"]: item
            for item in self.benchmark._variant_specs()
            if item["name"].startswith(("I_262k", "J_262k"))
        }

        self.assertEqual(
            self.benchmark._tile_cases(specs["I_262k_recovery_block256"], _args(Path(os.devnull)))[0][
                "block_size"
            ],
            256,
        )
        self.assertEqual(
            self.benchmark._tile_cases(specs["J_262k_recovery_block128"], _args(Path(os.devnull)))[0][
                "block_size"
            ],
            128,
        )
        for name, spec in specs.items():
            self.assertTrue(self.benchmark._spec_supports_prompt_len(spec, 262144))
            self.assertFalse(self.benchmark._spec_supports_prompt_len(spec, 131072))
            expected_profile = "262k" if name.startswith("I_262k") else "short"
            self.assertEqual(spec["cte"], ["--cte-bucket-profile", expected_profile])
            self.assertNotIn("--cold-zero-conv-fast-path", spec["flags"])

    def test_kernel_toggle_variants_cover_old_fused_and_pytorch_chunk_paths(self):
        specs = {item["name"]: item for item in self.benchmark._variant_specs()}

        self.assertEqual(
            specs["A_single512_old_chunked"]["env"],
            {
                "USE_NKI_FUSED": "0",
                "USE_NKI_CHUNKED": "1",
                "USE_PYTORCH_CHUNK": "0",
            },
        )
        self.assertEqual(
            specs["E_short_text_compact_fused"]["env"],
            {
                "USE_NKI_FUSED": "1",
                "USE_NKI_CHUNKED": "0",
                "USE_PYTORCH_CHUNK": "0",
            },
        )
        self.assertEqual(
            specs["K_short_text_compact_pytorch_chunk"]["env"],
            {
                "USE_NKI_FUSED": "0",
                "USE_NKI_CHUNKED": "0",
                "USE_PYTORCH_CHUNK": "1",
            },
        )
        self.assertNotIn(
            "--cold-zero-conv-fast-path",
            specs["F_short_text_compact_fused_cold_zero"]["flags"],
        )
        self.assertNotIn(
            "--cold-zero-conv-fast-path",
            specs["G_tile_block_sweep"]["flags"],
        )
        self.assertIn(
            "--cold-zero-conv-fast-path",
            specs["M_short_text_compact_fused_cold_zero_ablation"]["flags"],
        )
        self.assertTrue(specs["K_short_text_compact_pytorch_chunk"]["optional"])
        self.assertTrue(specs["M_short_text_compact_fused_cold_zero_ablation"]["optional"])

    def test_dense_mask_fallback_variant_disables_chunked_prefill(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            args = _args(Path(tmpdir))
            spec = next(
                item
                for item in self.benchmark._variant_specs()
                if item["name"] == "L_small_dense_mask_fallback"
            )

            row = self.benchmark._run_case(
                args,
                spec,
                512,
                self.benchmark._tile_cases(spec, args)[0],
                1,
            )

        self.assertTrue(spec["optional"])
        self.assertFalse(row["enable_vllm_chunked_prefill"])
        self.assertNotIn("--enable-vllm-chunked-prefill", row["command"])
        self.assertIn("--no-compact-cte-attention-mask", row["command"])
        self.assertTrue(self.benchmark._spec_supports_prompt_len(spec, 256))
        self.assertFalse(self.benchmark._spec_supports_prompt_len(spec, 1024))

    def test_include_dense_fallback_adds_only_dense_optional_variant(self):
        args = _args(Path(os.devnull))
        args.include_dense_fallback = True

        names = [spec["name"] for spec in self.benchmark._selected_specs(args)]

        self.assertIn("L_small_dense_mask_fallback", names)
        self.assertNotIn("K_short_text_compact_pytorch_chunk", names)

    def test_extracts_prefill_and_generation_metrics(self):
        stdout = "\n".join(
            [
                'COLD_PREFILL_METRICS {"prefill_latency_ms": 100.0, "decode_tok_per_s": null}',
                'GENERATION_METRICS {"first_token_latency_ms": 100.0, "decode_tok_per_s": 42.0}',
                'GDN_STATE_DIFF {"recurrent_max_abs_diff": 0.001, "conv_max_abs_diff": 0.002}',
            ]
        )

        prefill = self.benchmark._extract_metrics(stdout)
        generation = self.benchmark._extract_generation_metrics(stdout)
        state_diff = self.benchmark._extract_gdn_state_diff(stdout)

        self.assertEqual(prefill["prefill_latency_ms"], 100.0)
        self.assertIsNone(prefill["decode_tok_per_s"])
        self.assertEqual(generation["first_token_latency_ms"], 100.0)
        self.assertEqual(generation["decode_tok_per_s"], 42.0)
        self.assertEqual(state_diff["recurrent_max_abs_diff"], 0.001)
        self.assertEqual(state_diff["conv_max_abs_diff"], 0.002)

    def test_metric_extractors_use_last_prefixed_runtime_line(self):
        stdout = "\n".join(
            [
                'COLD_PREFILL_METRICS {"prefill_latency_ms": 500.0}',
                'COLD_PREFILL_METRICS {"prefill_latency_ms": 100.0}',
                'GDN_STATE_DIFF {"recurrent_max_abs_diff": 9.0, "conv_max_abs_diff": 9.0}',
                'GDN_STATE_DIFF {"recurrent_max_abs_diff": 0.001, "conv_max_abs_diff": 0.002}',
            ]
        )

        prefill = self.benchmark._extract_metrics(stdout)
        state_diff = self.benchmark._extract_gdn_state_diff(stdout)

        self.assertEqual(prefill["prefill_latency_ms"], 100.0)
        self.assertEqual(state_diff["recurrent_max_abs_diff"], 0.001)
        self.assertEqual(state_diff["conv_max_abs_diff"], 0.002)

    def test_extracts_gdn_state_diff_from_enginecore_prefixed_line(self):
        stdout = (
            '(EngineCore_DP0 pid=123) GDN_STATE_DIFF '
            '{"recurrent_max_abs_diff": 0.001, "conv_max_abs_diff": 0.002}'
        )

        state_diff = self.benchmark._extract_gdn_state_diff(stdout)

        self.assertEqual(state_diff["recurrent_max_abs_diff"], 0.001)
        self.assertEqual(state_diff["conv_max_abs_diff"], 0.002)

    def test_gdn_state_diff_aliases_are_normalised_with_raw_payload(self):
        stdout = "\n".join(
            [
                (
                    'GDN_STATE_DIFF {"source": "debug-hook", '
                    '"gdn_recurrent_state_max_abs_diff": 0.003, '
                    '"gdn_conv_state_max_abs_diff": 0.004}'
                ),
            ]
        )

        state_diff = self.benchmark._extract_gdn_state_diff(stdout)

        self.assertEqual(state_diff["source"], "debug-hook")
        self.assertEqual(state_diff["recurrent_state_max_abs_diff"], 0.003)
        self.assertEqual(state_diff["recurrent_max_abs_diff"], 0.003)
        self.assertEqual(state_diff["conv_state_max_abs_diff"], 0.004)
        self.assertEqual(state_diff["conv_max_abs_diff"], 0.004)
        self.assertEqual(
            state_diff["raw"]["gdn_recurrent_state_max_abs_diff"],
            0.003,
        )

    def test_loads_gdn_state_diff_sidecar_from_keyed_mapping(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            sidecar_path = Path(tmpdir) / "state_diff.json"
            sidecar_path.write_text(
                (
                    '{"A_single512_old_chunked|128|1|0": '
                    '{"recurrent_max_abs_diff": 0.001, "conv_max_abs_diff": 0.002}}'
                ),
                encoding="utf-8",
            )

            sidecar = self.benchmark._load_gdn_state_diff_sidecar(sidecar_path)
            state_diff = self.benchmark._gdn_state_diff_from_sidecar(
                sidecar,
                variant="A_single512_old_chunked",
                prompt_len=128,
                max_tokens=1,
                repetition=0,
                tile_case={"kernel_q_tile_size": 128, "kernel_kv_tile_size": 1024, "block_size": 128},
            )

        self.assertEqual(state_diff["source"], "gdn_state_diff_json")
        self.assertEqual(state_diff["recurrent_state_max_abs_diff"], 0.001)
        self.assertEqual(state_diff["conv_state_max_abs_diff"], 0.002)

    def test_loads_gdn_state_diff_sidecar_from_row_records_with_tile_key(self):
        records = [
            {
                "variant": "G_tile_block_sweep",
                "target_prompt_tokens": 2048,
                "max_tokens": 32,
                "repetition": 2,
                "kernel_q_tile_size": 128,
                "kernel_kv_tile_size": 2048,
                "block_size": 128,
                "gdn_state_diff": {
                    "gdn_recurrent_state_max_abs_diff": 0.003,
                    "gdn_conv_state_max_abs_diff": 0.004,
                },
            }
        ]

        sidecar = self.benchmark._state_diff_sidecar_from_records(records)
        state_diff = self.benchmark._gdn_state_diff_from_sidecar(
            sidecar,
            variant="G_tile_block_sweep",
            prompt_len=2048,
            max_tokens=32,
            repetition=2,
            tile_case={
                "kernel_q_tile_size": 128,
                "kernel_kv_tile_size": 2048,
                "block_size": 128,
            },
        )

        self.assertEqual(state_diff["recurrent_state_max_abs_diff"], 0.003)
        self.assertEqual(state_diff["conv_state_max_abs_diff"], 0.004)

    def test_run_case_merges_generation_and_state_diff_into_metrics(self):
        stdout = "\n".join(
            [
                "TOKENS [1, 2]",
                'COLD_PREFILL_METRICS {"prefill_latency_ms": 500.0, "decode_tok_per_s": null}',
                'GENERATION_METRICS {"prefill_latency_ms": 100.0, "first_token_latency_ms": 100.0, "decode_tok_per_s": 42.0}',
                'GDN_STATE_DIFF {"recurrent_max_abs_diff": 0.001, "conv_max_abs_diff": 0.002}',
            ]
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            args = _args(Path(tmpdir))
            args.dry_run = False
            artifact_dir = Path(tmpdir) / "artifacts"
            artifact_dir.mkdir()
            (artifact_dir / "manifest.json").write_text("{}", encoding="utf-8")
            args.compiled_artifacts = str(artifact_dir)
            spec = next(
                item
                for item in self.benchmark._variant_specs()
                if item["name"] == "A_single512_old_chunked"
            )
            with patch.object(
                self.benchmark.subprocess,
                "run",
                return_value=SimpleNamespace(returncode=0, stdout=stdout),
            ):
                row = self.benchmark._run_case(
                    args,
                    spec,
                    128,
                    self.benchmark._tile_cases(spec, args)[0],
                    2,
                )

        self.assertTrue(row["artifact_load_success"])
        self.assertEqual(row["token_ids"], [1, 2])
        self.assertEqual(row["generation_metrics"]["decode_tok_per_s"], 42.0)
        self.assertEqual(row["metrics"]["prefill_latency_ms"], 100.0)
        self.assertEqual(row["metrics"]["decode_tok_per_s"], 42.0)
        self.assertEqual(row["metrics"]["first_token_latency_ms"], 100.0)
        self.assertEqual(row["gdn_state_diff"]["recurrent_max_abs_diff"], 0.001)
        self.assertEqual(row["gdn_state_diff"]["recurrent_state_max_abs_diff"], 0.001)
        self.assertEqual(row["metrics"]["gdn_state_diff"]["conv_max_abs_diff"], 0.002)

    def test_run_case_uses_sidecar_state_diff_when_stdout_is_missing_it(self):
        stdout = "\n".join(
            [
                "TOKENS [1, 2]",
                'COLD_PREFILL_METRICS {"prefill_latency_ms": 100.0}',
            ]
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            args = _args(Path(tmpdir))
            args.dry_run = False
            args._gdn_state_diff_sidecar = {
                "A_single512_old_chunked|128|1|0": {
                    "recurrent_max_abs_diff": 0.005,
                    "conv_max_abs_diff": 0.006,
                }
            }
            spec = next(
                item
                for item in self.benchmark._variant_specs()
                if item["name"] == "A_single512_old_chunked"
            )
            with patch.object(
                self.benchmark.subprocess,
                "run",
                return_value=SimpleNamespace(returncode=0, stdout=stdout),
            ):
                row = self.benchmark._run_case(
                    args,
                    spec,
                    128,
                    self.benchmark._tile_cases(spec, args)[0],
                    1,
                )

        self.assertEqual(row["gdn_state_diff"]["recurrent_max_abs_diff"], 0.005)
        self.assertEqual(row["metrics"]["gdn_state_diff"]["conv_max_abs_diff"], 0.006)

    def test_run_matrix_collects_rows_after_failure(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            args = _args(Path(tmpdir))
            args.dry_run = False
            args.variants = ["A_single512_old_chunked"]
            args.max_tokens_values = [1, 32]
            responses = [
                SimpleNamespace(returncode=7, stdout="runtime failed"),
                SimpleNamespace(
                    returncode=0,
                    stdout='COLD_PREFILL_METRICS {"prefill_latency_ms": 100.0}',
                ),
            ]
            with patch.object(
                self.benchmark.subprocess,
                "run",
                side_effect=responses,
            ):
                rows, returncode = self.benchmark._run_matrix(args)

        self.assertEqual(returncode, 7)
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]["returncode"], 7)
        self.assertIsNone(rows[0]["metrics"])
        self.assertEqual(rows[1]["returncode"], 0)
        self.assertEqual(rows[1]["metrics"]["prefill_latency_ms"], 100.0)

    def test_run_matrix_fail_fast_stops_after_failure(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            args = _args(Path(tmpdir))
            args.dry_run = False
            args.fail_fast = True
            args.variants = ["A_single512_old_chunked"]
            args.max_tokens_values = [1, 32]
            responses = [
                SimpleNamespace(returncode=7, stdout="runtime failed"),
                SimpleNamespace(
                    returncode=0,
                    stdout='COLD_PREFILL_METRICS {"prefill_latency_ms": 100.0}',
                ),
            ]
            with patch.object(
                self.benchmark.subprocess,
                "run",
                side_effect=responses,
            ):
                rows, returncode = self.benchmark._run_matrix(args)

        self.assertEqual(returncode, 7)
        self.assertEqual(len(rows), 1)


if __name__ == "__main__":
    unittest.main()
