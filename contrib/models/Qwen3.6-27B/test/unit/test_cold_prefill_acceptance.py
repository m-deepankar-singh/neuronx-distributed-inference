# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU-only tests for Qwen3.6 cold-prefill acceptance gates."""

import importlib.util
import tempfile
import unittest
from pathlib import Path


_REPO_ROOT = Path(__file__).resolve().parents[5]
_ACCEPTANCE_PATH = _REPO_ROOT / "validation_scripts" / "qwen36_cold_prefill_acceptance.py"


def _load_acceptance():
    spec = importlib.util.spec_from_file_location(
        "qwen36_cold_prefill_acceptance",
        _ACCEPTANCE_PATH,
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _row(variant: str, prompt_len: int, latency_ms: float, tps: float, tokens=None):
    seq_len = max(prompt_len, 2048)
    return {
        "benchmark_schema_version": 1,
        "benchmark_script": "/tmp/qwen36_cold_prefill_benchmark.py",
        "runner": "/tmp/run_offline_inference.py",
        "variant": variant,
        "target_prompt_tokens": prompt_len,
        "prompt_sha256": f"prompt-{prompt_len}",
        "model_path": "/tmp/model",
        "compiled_artifacts": "/tmp/artifacts",
        "compiled_artifacts_resolved": "/tmp/artifacts",
        "compiled_artifacts_path_exists": True,
        "compiled_artifacts_path_nonempty": True,
        "max_model_len": seq_len,
        "seq_len": seq_len,
        "tensor_parallel_size": 4,
        "logical_nc_config": 2,
        "max_num_seqs": 1,
        "ctx_batch_size": 1,
        "returncode": 0,
        "artifact_load_success": True,
        "token_ids": [1, 2, 3] if tokens is None else tokens,
        "temperature": 0,
        "top_k": 1,
        "prompt_file": f"/tmp/prompts/{prompt_len}.txt",
        "command": ["python", "run_offline_inference.py", "--max-tokens", "1"],
        "cte_args": ["--cte-bucket", "512"],
        "flags": [],
        "env": {},
        "repetition": 0,
        "elapsed_seconds": latency_ms / 1000.0,
        "enable_vllm_chunked_prefill": True,
        "output_tail": ["runtime completed normally"],
        "metrics": {
            "prefill_latency_ms": latency_ms,
            "actual_tok_per_s": tps,
            "bucket_tok_per_s": tps,
            "compact_mask_enabled": True,
            "chunked_prefill_enabled": True,
            "num_cte_chunks": 1,
        },
    }


def _row_with_max_tokens(
    variant: str,
    prompt_len: int,
    latency_ms: float,
    tps: float,
    max_tokens: int,
):
    row = _row(variant, prompt_len, latency_ms, tps)
    row["max_tokens"] = max_tokens
    return row


def _row_with_generation_metrics(
    variant: str,
    prompt_len: int,
    *,
    first_token_ms: float,
    decode_tps: float,
    end_to_end_tps: float,
    tokens=None,
):
    row = _row_with_max_tokens(variant, prompt_len, 200.0, 1000.0, 32)
    if tokens is not None:
        row["token_ids"] = tokens
    row["metrics"].update(
        {
            "first_token_latency_ms": first_token_ms,
            "decode_tok_per_s": decode_tps,
            "end_to_end_generated_tok_per_s": end_to_end_tps,
        }
    )
    return row


def _row_with_state_diff(
    variant: str,
    prompt_len: int,
    *,
    recurrent_diff: float,
    conv_diff: float,
):
    row = _row(variant, prompt_len, 50.0, 2000.0)
    row["metrics"]["gdn_state_diff"] = {
        "recurrent_max_abs_diff": recurrent_diff,
        "conv_max_abs_diff": conv_diff,
    }
    return row


def _row_with_hbm(
    variant: str,
    prompt_len: int,
    *,
    latency_ms: float,
    tps: float,
    used_bytes: int,
):
    row = _row(variant, prompt_len, latency_ms, tps)
    row["metrics"]["hbm_usage"] = {"bytes_used": used_bytes}
    return row


def _with_feature_profile(
    row,
    *,
    cte_buckets,
    text_only: bool,
    compact: bool,
    cold_zero: bool,
):
    row["metrics"].update(
        {
            "cte_buckets": list(cte_buckets),
            "text_only_cte_enabled": text_only,
            "compact_mask_enabled": compact,
            "cold_zero_conv_fast_path_enabled": cold_zero,
        }
    )
    return row


def _row_with_tile(
    variant: str,
    prompt_len: int,
    latency_ms: float,
    tps: float,
    *,
    q_tile: int,
    kv_tile: int,
    block_size: int,
):
    row = _row(variant, prompt_len, latency_ms, tps)
    tile_case = {
        "kernel_q_tile_size": q_tile,
        "kernel_kv_tile_size": kv_tile,
        "block_size": block_size,
    }
    row.update(tile_case)
    row["metrics"].update(tile_case)
    return row


def _long_artifact_row(
    variant: str,
    prompt_len: int,
    latency_ms: float,
    tps: float,
    *,
    q_tile: int,
    kv_tile: int,
    block_size: int,
    cte_buckets,
):
    row = _row_with_tile(
        variant,
        prompt_len,
        latency_ms,
        tps,
        q_tile=q_tile,
        kv_tile=kv_tile,
        block_size=block_size,
    )
    row["metrics"].update(
        {
            "cte_buckets": list(cte_buckets),
            "text_only_cte_enabled": True,
            "compact_mask_enabled": True,
        }
    )
    return row


def _strict_final_row(
    variant: str,
    prompt_len: int,
    *,
    latency_ms: float,
    tps: float,
    max_tokens: int,
    cte_buckets,
    text_only: bool,
    compact: bool,
    cold_zero: bool,
    gdn_cte_kernel: str,
    hbm_used_bytes: int,
    tile_case=None,
    chunked: bool = True,
    dense_mask_fallback: bool = False,
):
    row = _row_with_max_tokens(variant, prompt_len, latency_ms, tps, max_tokens)
    seq_artifact = {
        2048: "/tmp/artifacts-2k",
        8192: "/tmp/artifacts-8k",
        32768: "/tmp/artifacts-32k",
        131072: "/tmp/artifacts-128k",
        262144: "/tmp/artifacts-262k",
    }.get(row["seq_len"], "/tmp/artifacts")
    row["compiled_artifacts"] = seq_artifact
    row["compiled_artifacts_resolved"] = seq_artifact
    row["compiled_artifacts_path_exists"] = True
    row["compiled_artifacts_path_nonempty"] = True
    row["prompt_token_count"] = prompt_len
    selected_cte_bucket = next(
        (bucket for bucket in cte_buckets if bucket >= min(prompt_len, max(cte_buckets))),
        cte_buckets[-1],
    )
    row["metrics"].update(
        {
            "actual_prompt_len": prompt_len,
            "selected_cte_bucket": selected_cte_bucket,
            "padding_tokens": max(selected_cte_bucket - min(prompt_len, selected_cte_bucket), 0),
            "padding_ratio": 0.0,
            "tensor_parallel_size": 4,
            "logical_nc_config": 2,
            "max_num_seqs": 1,
            "ctx_batch_size": 1,
            "block_size": 128,
            "kernel_q_tile_size": 128,
            "kernel_kv_tile_size": 1024,
            "cte_buckets": list(cte_buckets),
            "text_only_cte_enabled": text_only,
            "compact_mask_enabled": compact,
            "cold_zero_conv_fast_path_enabled": cold_zero,
            "use_nki_fused": gdn_cte_kernel == "fused_initial_state",
            "gdn_cte_kernel": gdn_cte_kernel,
            "hbm_usage": {"bytes_used": hbm_used_bytes},
            "chunked_prefill_enabled": chunked,
            "dense_cte_mask_fallback": dense_mask_fallback,
            "cte_attention_mask_path": (
                "dense_4d_fallback"
                if dense_mask_fallback
                else "neuron_chunked_prefill"
                if chunked
                else "compact_2d"
            ),
            "gdn_state_diff": {
                "recurrent_max_abs_diff": 0.001,
                "conv_max_abs_diff": 0.001,
            },
        }
    )
    if max_tokens == 32:
        row["metrics"].update(
            {
                "first_token_latency_ms": latency_ms,
                "decode_tok_per_s": 50.0,
                "end_to_end_generated_tok_per_s": 25.0,
            }
        )
    if tile_case is not None:
        q_tile, kv_tile, block_size = tile_case
        row.update(
            {
                "kernel_q_tile_size": q_tile,
                "kernel_kv_tile_size": kv_tile,
                "block_size": block_size,
            }
        )
        row["metrics"].update(
            {
                "kernel_q_tile_size": q_tile,
                "kernel_kv_tile_size": kv_tile,
                "block_size": block_size,
            }
        )
    return row


def _strict_final_complete_rows(acceptance):
    rows = []
    variant_profiles = {
        "A_single512_old_chunked": {
            "latency_ms": 100.0,
            "tps": 420.0,
            "cte_buckets": [512],
            "text_only": False,
            "compact": False,
            "cold_zero": False,
            "kernel": "nki_chunked",
        },
        "B_short_buckets_old_chunked": {
            "latency_ms": 90.0,
            "tps": 520.0,
            "cte_buckets": [128, 256, 512, 1024],
            "text_only": False,
            "compact": False,
            "cold_zero": False,
            "kernel": "nki_chunked",
        },
        "C_short_text_only_old_chunked": {
            "latency_ms": 80.0,
            "tps": 620.0,
            "cte_buckets": [128, 256, 512, 1024],
            "text_only": True,
            "compact": False,
            "cold_zero": False,
            "kernel": "nki_chunked",
        },
        "D_short_text_compact_old_chunked": {
            "latency_ms": 70.0,
            "tps": 720.0,
            "cte_buckets": [128, 256, 512, 1024],
            "text_only": True,
            "compact": True,
            "cold_zero": False,
            "kernel": "nki_chunked",
        },
        "E_short_text_compact_fused": {
            "latency_ms": 65.0,
            "tps": 820.0,
            "cte_buckets": [128, 256, 512, 1024],
            "text_only": True,
            "compact": True,
            "cold_zero": False,
            "kernel": "fused_initial_state",
        },
        "F_short_text_compact_fused_cold_zero": {
            "latency_ms": 60.0,
            "tps": 920.0,
            "cte_buckets": [128, 256, 512, 1024],
            "text_only": True,
            "compact": True,
            "cold_zero": True,
            "kernel": "fused_initial_state",
        },
        "G_tile_block_sweep": {
            "latency_ms": 55.0,
            "tps": 1020.0,
            "cte_buckets": [128, 256, 512, 1024],
            "text_only": True,
            "compact": True,
            "cold_zero": True,
            "kernel": "fused_initial_state",
        },
    }
    default_tile = (128, 1024, 128)
    required_tile_cases = list(acceptance.REQUIRED_TILE_SWEEP_CASES)

    for prompt_len in acceptance.MANDATORY_PROMPT_LENGTHS:
        for variant in acceptance.FEATURE_MATRIX_VARIANTS:
            profile = variant_profiles[variant]
            tile_cases = (
                required_tile_cases
                if variant == "G_tile_block_sweep"
                and prompt_len in acceptance.REQUIRED_TILE_SWEEP_PROMPTS
                else [default_tile]
            )
            for tile_case in tile_cases:
                for max_tokens in (1, 32):
                    for sample in range(3):
                        rows.append(
                            _strict_final_row(
                                variant,
                                prompt_len,
                                latency_ms=profile["latency_ms"] + sample,
                                tps=profile["tps"],
                                max_tokens=max_tokens,
                                cte_buckets=profile["cte_buckets"],
                                text_only=profile["text_only"],
                                compact=profile["compact"],
                                cold_zero=profile["cold_zero"],
                                gdn_cte_kernel=profile["kernel"],
                                hbm_used_bytes=10000
                                if variant == "A_single512_old_chunked"
                                else 9000,
                                tile_case=tile_case
                                if variant == "G_tile_block_sweep"
                                else None,
                            )
                        )

    long_profiles = {
        "H_128k_candidate": (131072, [256, 512, 1024, 2048], (128, 1024, 128)),
        "I_262k_recovery_block256": (262144, [256], (128, 1024, 256)),
        "J_262k_recovery_block128": (262144, [256], (128, 1024, 128)),
    }
    for prompt_len in (131072, 262144):
        for max_tokens in (1, 32):
            for sample in range(3):
                rows.append(
                    _strict_final_row(
                        "A_single512_old_chunked",
                        prompt_len,
                        latency_ms=100.0 + sample,
                        tps=420.0,
                        max_tokens=max_tokens,
                        cte_buckets=[512],
                        text_only=False,
                        compact=False,
                        cold_zero=False,
                        gdn_cte_kernel="nki_chunked",
                        hbm_used_bytes=10000,
                    )
                )

    for prompt_len in acceptance.DENSE_FALLBACK_PROMPT_LENGTHS:
        for max_tokens in (1, 32):
            for sample in range(3):
                rows.append(
                    _strict_final_row(
                        acceptance.DENSE_FALLBACK_VARIANT,
                        prompt_len,
                        latency_ms=110.0 + sample,
                        tps=410.0,
                        max_tokens=max_tokens,
                        cte_buckets=[512],
                        text_only=True,
                        compact=False,
                        cold_zero=False,
                        gdn_cte_kernel="nki_chunked",
                        hbm_used_bytes=12000,
                        chunked=False,
                        dense_mask_fallback=True,
                    )
                )

    for variant, (prompt_len, cte_buckets, tile_case) in long_profiles.items():
        for max_tokens in (1, 32):
            for sample in range(3):
                rows.append(
                    _strict_final_row(
                        variant,
                        prompt_len,
                        latency_ms=80.0 + sample,
                        tps=1000.0,
                        max_tokens=max_tokens,
                        cte_buckets=cte_buckets,
                        text_only=True,
                        compact=True,
                        cold_zero=False,
                        gdn_cte_kernel="fused_initial_state",
                        hbm_used_bytes=9000,
                        tile_case=tile_case,
                    )
                )
    return rows


def _strict_final_hybrid_apc_report(
    *,
    full_prefix_exact=True,
    partial_prefix_exact=True,
    include_cold_metrics=True,
):
    report = {
        "full_prefix_exact": full_prefix_exact,
        "partial_prefix_exact": partial_prefix_exact,
        "validation_config": {
            "enable_vllm_chunked_prefill": True,
            "text_only_cte": True,
            "compact_cte_attention_mask": True,
            "cold_zero_conv_fast_path": True,
            "hybrid_apc_require_vllm_metadata": True,
            "max_num_seqs": 1,
            "block_size": 128,
            "gdn_checkpoint_interval": 128,
            "kernel_q_tile_size": 128,
            "kernel_kv_tile_size": 1024,
        },
    }
    if include_cold_metrics:
        report["cold_prefill_metrics"] = {
            "cold_full": {
                "actual_prompt_len": 384,
                "prefill_latency_ms": 100.0,
                "actual_tok_per_s": 3840.0,
                "bucket_tok_per_s": 5120.0,
            },
            "cold_partial": {
                "actual_prompt_len": 384,
                "prefill_latency_ms": 105.0,
                "actual_tok_per_s": 3657.0,
                "bucket_tok_per_s": 4876.0,
            },
        }
    return report


class TestColdPrefillAcceptance(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.acceptance = _load_acceptance()

    def test_strict_final_can_pass_with_complete_goal_evidence(self):
        report = self.acceptance.evaluate(
            _strict_final_complete_rows(self.acceptance),
            baseline_variant="A_single512_old_chunked",
            candidate_variant="F_short_text_compact_fused_cold_zero",
            short_latency_speedup=1.5,
            two_k_regression_tolerance=0.0,
            baseline_cold_tok_per_s_target=420.0,
            strict_final=True,
            hybrid_apc_report=_strict_final_hybrid_apc_report(),
        )

        self.assertTrue(report["passed"], report["failures"])
        audit = {item["name"]: item for item in report["audit_checklist"]}
        self.assertTrue(audit)
        self.assertTrue(
            all(item["status"] == "pass" for item in audit.values()),
            audit,
        )
        self.assertEqual(
            audit["prefill latency p50/p95 summaries"]["status"],
            "pass",
        )
        self.assertEqual(
            audit["G tile-sweep latency p50/p95 summaries"]["status"],
            "pass",
        )
        self.assertEqual(audit["strict-final launch shape"]["status"], "pass")
        self.assertEqual(audit["artifact load success evidence"]["status"], "pass")
        self.assertEqual(
            audit["compiled artifact path existence evidence"]["status"],
            "pass",
        )
        self.assertEqual(audit["runtime returncode evidence"]["status"], "pass")
        self.assertEqual(
            audit["runtime output tail DMA-spill evidence"]["status"],
            "pass",
        )
        self.assertTrue(
            any(
                check["variant"] == self.acceptance.DENSE_FALLBACK_VARIANT
                and check["present"]
                for check in report["hbm_usage_checks"]
            )
        )
        self.assertFalse(
            any(
                check["variant"] == self.acceptance.DENSE_FALLBACK_VARIANT
                for check in report["hbm_checks"]
            )
        )

    def test_acceptance_passes_for_speedup_and_exact_tokens(self):
        rows = [
            _row("A_single512_old_chunked", 128, 100.0, 1000.0),
            _row("F_short_text_compact_fused_cold_zero", 128, 60.0, 1600.0),
            _row("A_single512_old_chunked", 2048, 500.0, 4096.0),
            _row("F_short_text_compact_fused_cold_zero", 2048, 500.0, 4096.0),
            _row("A_single512_old_chunked", 8192, 2000.0, 4096.0),
            _row("F_short_text_compact_fused_cold_zero", 8192, 2000.0, 4096.0),
        ]

        report = self.acceptance.evaluate(
            rows,
            baseline_variant="A_single512_old_chunked",
            candidate_variant="F_short_text_compact_fused_cold_zero",
            short_latency_speedup=1.5,
            two_k_regression_tolerance=0.0,
        )

        self.assertTrue(report["passed"], report)
        self.assertEqual(report["short_results"][0]["speedup"], 100.0 / 60.0)
        self.assertEqual(report["short_results"][0]["baseline_p95_ms"], 100.0)
        self.assertTrue(report["latency_summaries"])
        audit = {item["name"]: item for item in report["audit_checklist"]}
        self.assertEqual(audit["short prompt p50 latency speedup"]["status"], "pass")
        self.assertEqual(audit["token exactness vs baseline"]["status"], "pass")
        self.assertEqual(audit["fixed prompt suite token counts"]["status"], "skip")
        self.assertEqual(audit["baseline cold tok/s reproduction"]["status"], "skip")

    def test_strict_final_requires_hybrid_apc_exactness_report(self):
        report = self.acceptance.evaluate(
            _strict_final_complete_rows(self.acceptance),
            baseline_variant="A_single512_old_chunked",
            candidate_variant="F_short_text_compact_fused_cold_zero",
            short_latency_speedup=1.5,
            two_k_regression_tolerance=0.0,
            baseline_cold_tok_per_s_target=420.0,
            strict_final=True,
        )

        self.assertFalse(report["passed"])
        self.assertTrue(
            any(
                "hybrid APC exactness report" in failure
                for failure in report["failures"]
            )
        )
        audit = {item["name"]: item for item in report["audit_checklist"]}
        self.assertEqual(
            audit["hybrid APC partial-prefix exactness"]["status"],
            "fail",
        )

    def test_strict_final_fails_when_hybrid_apc_partial_prefix_mismatches(self):
        report = self.acceptance.evaluate(
            _strict_final_complete_rows(self.acceptance),
            baseline_variant="A_single512_old_chunked",
            candidate_variant="F_short_text_compact_fused_cold_zero",
            short_latency_speedup=1.5,
            two_k_regression_tolerance=0.0,
            baseline_cold_tok_per_s_target=420.0,
            strict_final=True,
            hybrid_apc_report=_strict_final_hybrid_apc_report(
                partial_prefix_exact=False,
            ),
        )

        self.assertFalse(report["passed"])
        self.assertTrue(
            any(
                "hybrid APC partial-prefix token exactness failed" in failure
                for failure in report["failures"]
            )
        )

    def test_strict_final_requires_hybrid_apc_cold_prefill_metrics(self):
        report = self.acceptance.evaluate(
            _strict_final_complete_rows(self.acceptance),
            baseline_variant="A_single512_old_chunked",
            candidate_variant="F_short_text_compact_fused_cold_zero",
            short_latency_speedup=1.5,
            two_k_regression_tolerance=0.0,
            baseline_cold_tok_per_s_target=420.0,
            strict_final=True,
            hybrid_apc_report=_strict_final_hybrid_apc_report(
                include_cold_metrics=False,
            ),
        )

        self.assertFalse(report["passed"])
        self.assertTrue(
            any(
                "missing cold_partial COLD_PREFILL_METRICS" in failure
                for failure in report["failures"]
            )
        )

    def test_strict_final_requires_hybrid_apc_strict_config(self):
        hybrid_report = _strict_final_hybrid_apc_report()
        hybrid_report["validation_config"]["cold_zero_conv_fast_path"] = False

        report = self.acceptance.evaluate(
            _strict_final_complete_rows(self.acceptance),
            baseline_variant="A_single512_old_chunked",
            candidate_variant="F_short_text_compact_fused_cold_zero",
            short_latency_speedup=1.5,
            two_k_regression_tolerance=0.0,
            baseline_cold_tok_per_s_target=420.0,
            strict_final=True,
            hybrid_apc_report=hybrid_report,
        )

        self.assertFalse(report["passed"])
        self.assertTrue(
            any(
                "strict-final cold-zero/chunked validation config" in failure
                for failure in report["failures"]
            )
        )

    def test_strict_final_requires_greedy_sampling_config(self):
        rows = _strict_final_complete_rows(self.acceptance)
        rows[0]["temperature"] = 0.7

        report = self.acceptance.evaluate(
            rows,
            baseline_variant="A_single512_old_chunked",
            candidate_variant="F_short_text_compact_fused_cold_zero",
            short_latency_speedup=1.5,
            two_k_regression_tolerance=0.0,
            baseline_cold_tok_per_s_target=420.0,
            strict_final=True,
            hybrid_apc_report=_strict_final_hybrid_apc_report(),
        )

        self.assertFalse(report["passed"])
        self.assertTrue(
            any(
                "temperature=0 and top_k=1" in failure
                for failure in report["failures"]
            )
        )
        audit = {item["name"]: item for item in report["audit_checklist"]}
        self.assertEqual(audit["greedy sampling config"]["status"], "fail")

    def test_strict_final_requires_complete_cold_prefill_metrics(self):
        rows = _strict_final_complete_rows(self.acceptance)
        rows[0]["metrics"].pop("padding_ratio")

        report = self.acceptance.evaluate(
            rows,
            baseline_variant="A_single512_old_chunked",
            candidate_variant="F_short_text_compact_fused_cold_zero",
            short_latency_speedup=1.5,
            two_k_regression_tolerance=0.0,
            baseline_cold_tok_per_s_target=420.0,
            strict_final=True,
            hybrid_apc_report=_strict_final_hybrid_apc_report(),
        )

        self.assertFalse(report["passed"])
        self.assertTrue(
            any(
                "missing cold-prefill instrumentation metrics" in failure
                and "padding_ratio" in failure
                for failure in report["failures"]
            )
        )
        audit = {item["name"]: item for item in report["audit_checklist"]}
        self.assertEqual(
            audit["cold-prefill instrumentation metrics"]["status"],
            "fail",
        )

    def test_strict_final_requires_benchmark_row_provenance(self):
        rows = _strict_final_complete_rows(self.acceptance)
        rows[0].pop("command")

        report = self.acceptance.evaluate(
            rows,
            baseline_variant="A_single512_old_chunked",
            candidate_variant="F_short_text_compact_fused_cold_zero",
            short_latency_speedup=1.5,
            two_k_regression_tolerance=0.0,
            baseline_cold_tok_per_s_target=420.0,
            strict_final=True,
            hybrid_apc_report=_strict_final_hybrid_apc_report(),
        )

        self.assertFalse(report["passed"])
        self.assertTrue(
            any(
                "missing benchmark row provenance" in failure
                for failure in report["failures"]
            )
        )
        audit = {item["name"]: item for item in report["audit_checklist"]}
        self.assertEqual(audit["benchmark row provenance"]["status"], "fail")

    def test_strict_final_requires_objective_launch_shape(self):
        rows = _strict_final_complete_rows(self.acceptance)
        rows[0]["tensor_parallel_size"] = 8
        rows[0]["metrics"]["tensor_parallel_size"] = 8

        report = self.acceptance.evaluate(
            rows,
            baseline_variant="A_single512_old_chunked",
            candidate_variant="F_short_text_compact_fused_cold_zero",
            short_latency_speedup=1.5,
            two_k_regression_tolerance=0.0,
            baseline_cold_tok_per_s_target=420.0,
            strict_final=True,
            hybrid_apc_report=_strict_final_hybrid_apc_report(),
        )

        self.assertFalse(report["passed"])
        self.assertTrue(
            any(
                "did not use strict-final launch shape" in failure
                and "tensor_parallel_size" in failure
                for failure in report["failures"]
            )
        )
        audit = {item["name"]: item for item in report["audit_checklist"]}
        self.assertEqual(audit["strict-final launch shape"]["status"], "fail")

    def test_strict_final_requires_positive_artifact_load_evidence(self):
        rows = _strict_final_complete_rows(self.acceptance)
        rows[0].pop("artifact_load_success")

        report = self.acceptance.evaluate(
            rows,
            baseline_variant="A_single512_old_chunked",
            candidate_variant="F_short_text_compact_fused_cold_zero",
            short_latency_speedup=1.5,
            two_k_regression_tolerance=0.0,
            baseline_cold_tok_per_s_target=420.0,
            strict_final=True,
            hybrid_apc_report=_strict_final_hybrid_apc_report(),
        )

        self.assertFalse(report["passed"])
        self.assertTrue(
            any(
                "missing successful artifact load/run evidence" in failure
                for failure in report["failures"]
            )
        )
        audit = {item["name"]: item for item in report["audit_checklist"]}
        self.assertEqual(audit["artifact load success evidence"]["status"], "fail")

    def test_strict_final_requires_compiled_artifact_path_evidence(self):
        rows = _strict_final_complete_rows(self.acceptance)
        rows[0].pop("compiled_artifacts_path_exists")

        report = self.acceptance.evaluate(
            rows,
            baseline_variant="A_single512_old_chunked",
            candidate_variant="F_short_text_compact_fused_cold_zero",
            short_latency_speedup=1.5,
            two_k_regression_tolerance=0.0,
            baseline_cold_tok_per_s_target=420.0,
            strict_final=True,
            hybrid_apc_report=_strict_final_hybrid_apc_report(),
        )

        self.assertFalse(report["passed"])
        self.assertTrue(
            any(
                "missing compiled artifact path existence/non-empty evidence" in failure
                for failure in report["failures"]
            )
        )
        audit = {item["name"]: item for item in report["audit_checklist"]}
        self.assertEqual(
            audit["compiled artifact path existence evidence"]["status"],
            "fail",
        )

    def test_strict_final_requires_nonempty_compiled_artifact_path_evidence(self):
        rows = _strict_final_complete_rows(self.acceptance)
        rows[0]["compiled_artifacts_path_nonempty"] = False

        report = self.acceptance.evaluate(
            rows,
            baseline_variant="A_single512_old_chunked",
            candidate_variant="F_short_text_compact_fused_cold_zero",
            short_latency_speedup=1.5,
            two_k_regression_tolerance=0.0,
            baseline_cold_tok_per_s_target=420.0,
            strict_final=True,
            hybrid_apc_report=_strict_final_hybrid_apc_report(),
        )

        self.assertFalse(report["passed"])
        self.assertTrue(
            any(
                "missing compiled artifact path existence/non-empty evidence"
                in failure
                for failure in report["failures"]
            )
        )

    def test_strict_final_requires_explicit_successful_returncode(self):
        rows = _strict_final_complete_rows(self.acceptance)
        rows[0].pop("returncode")

        report = self.acceptance.evaluate(
            rows,
            baseline_variant="A_single512_old_chunked",
            candidate_variant="F_short_text_compact_fused_cold_zero",
            short_latency_speedup=1.5,
            two_k_regression_tolerance=0.0,
            baseline_cold_tok_per_s_target=420.0,
            strict_final=True,
            hybrid_apc_report=_strict_final_hybrid_apc_report(),
        )

        self.assertFalse(report["passed"])
        self.assertTrue(
            any(
                "missing successful runtime returncode evidence" in failure
                for failure in report["failures"]
            )
        )
        audit = {item["name"]: item for item in report["audit_checklist"]}
        self.assertEqual(audit["runtime returncode evidence"]["status"], "fail")

    def test_strict_final_requires_runtime_output_tail_for_dma_spill_audit(self):
        rows = _strict_final_complete_rows(self.acceptance)
        rows[0].pop("output_tail")

        report = self.acceptance.evaluate(
            rows,
            baseline_variant="A_single512_old_chunked",
            candidate_variant="F_short_text_compact_fused_cold_zero",
            short_latency_speedup=1.5,
            two_k_regression_tolerance=0.0,
            baseline_cold_tok_per_s_target=420.0,
            strict_final=True,
            hybrid_apc_report=_strict_final_hybrid_apc_report(),
        )

        self.assertFalse(report["passed"])
        self.assertTrue(
            any(
                "missing runtime output_tail evidence for DMA-spill audit"
                in failure
                for failure in report["failures"]
            )
        )
        audit = {item["name"]: item for item in report["audit_checklist"]}
        self.assertEqual(
            audit["runtime output tail DMA-spill evidence"]["status"],
            "fail",
        )

    def test_strict_final_requires_consistent_artifacts_per_seq_len(self):
        rows = _strict_final_complete_rows(self.acceptance)
        for row in rows:
            if row["seq_len"] == 8192:
                row["compiled_artifacts"] = "/tmp/other-artifacts-8k"
                break

        report = self.acceptance.evaluate(
            rows,
            baseline_variant="A_single512_old_chunked",
            candidate_variant="F_short_text_compact_fused_cold_zero",
            short_latency_speedup=1.5,
            two_k_regression_tolerance=0.0,
            baseline_cold_tok_per_s_target=420.0,
            strict_final=True,
            hybrid_apc_report=_strict_final_hybrid_apc_report(),
        )

        self.assertFalse(report["passed"])
        self.assertTrue(
            any(
                "one compiled artifact for seq_len=8192" in failure
                for failure in report["failures"]
            )
        )
        audit = {item["name"]: item for item in report["audit_checklist"]}
        self.assertEqual(
            audit["compiled artifact consistency by seq_len"]["status"],
            "fail",
        )

    def test_strict_final_checks_baseline_target_on_each_short_prompt(self):
        rows = _strict_final_complete_rows(self.acceptance)
        for row in rows:
            if (
                row["variant"] == "A_single512_old_chunked"
                and row["target_prompt_tokens"] == 384
                and row["max_tokens"] == 1
            ):
                row["metrics"]["actual_tok_per_s"] = 200.0

        report = self.acceptance.evaluate(
            rows,
            baseline_variant="A_single512_old_chunked",
            candidate_variant="F_short_text_compact_fused_cold_zero",
            short_latency_speedup=1.5,
            two_k_regression_tolerance=0.0,
            baseline_cold_tok_per_s_target=420.0,
            strict_final=True,
            hybrid_apc_report=_strict_final_hybrid_apc_report(),
        )

        self.assertFalse(report["passed"])
        self.assertTrue(
            any(
                "prompt=384 baseline cold tok/s" in failure
                for failure in report["failures"]
            )
        )
        audit = {item["name"]: item for item in report["audit_checklist"]}
        self.assertEqual(
            audit["baseline short prompt tok/s reproduction"]["status"],
            "fail",
        )

    def test_acceptance_can_require_baseline_cold_tok_per_s_target(self):
        rows = [
            _row("A_single512_old_chunked", 128, 100.0, 420.0),
            _row("F_short_text_compact_fused_cold_zero", 128, 60.0, 700.0),
        ]

        report = self.acceptance.evaluate(
            rows,
            baseline_variant="A_single512_old_chunked",
            candidate_variant="F_short_text_compact_fused_cold_zero",
            short_latency_speedup=1.5,
            two_k_regression_tolerance=0.0,
            baseline_cold_tok_per_s_target=420.0,
            baseline_cold_tok_per_s_tolerance=0.05,
        )

        self.assertTrue(report["passed"], report)
        audit = {item["name"]: item for item in report["audit_checklist"]}
        self.assertEqual(audit["baseline cold tok/s reproduction"]["status"], "pass")
        self.assertEqual(
            report["baseline_cold_tok_per_s_gate"]["baseline_p50_actual_tok_per_s"],
            420.0,
        )

    def test_acceptance_fails_when_baseline_cold_tok_per_s_target_misses(self):
        rows = [
            _row("A_single512_old_chunked", 128, 100.0, 300.0),
            _row("F_short_text_compact_fused_cold_zero", 128, 60.0, 700.0),
        ]

        report = self.acceptance.evaluate(
            rows,
            baseline_variant="A_single512_old_chunked",
            candidate_variant="F_short_text_compact_fused_cold_zero",
            short_latency_speedup=1.5,
            two_k_regression_tolerance=0.0,
            baseline_cold_tok_per_s_target=420.0,
            baseline_cold_tok_per_s_tolerance=0.05,
        )

        self.assertFalse(report["passed"])
        self.assertTrue(any("baseline cold tok/s" in failure for failure in report["failures"]))

    def test_acceptance_checks_prompt_token_count_tolerance_when_present(self):
        rows = [
            _row("A_single512_old_chunked", 128, 100.0, 1000.0),
            _row("F_short_text_compact_fused_cold_zero", 128, 60.0, 1600.0),
        ]
        rows[0]["metrics"]["actual_prompt_len"] = 128
        rows[1]["metrics"]["actual_prompt_len"] = 132

        report = self.acceptance.evaluate(
            rows,
            baseline_variant="A_single512_old_chunked",
            candidate_variant="F_short_text_compact_fused_cold_zero",
            short_latency_speedup=1.5,
            two_k_regression_tolerance=0.0,
            prompt_token_tolerance=0.05,
        )

        self.assertTrue(report["passed"], report)
        audit = {item["name"]: item for item in report["audit_checklist"]}
        self.assertEqual(audit["fixed prompt suite token counts"]["status"], "pass")
        self.assertEqual(len(report["prompt_token_checks"]), 2)

    def test_acceptance_fails_when_prompt_token_count_misses_target(self):
        rows = [
            _row("A_single512_old_chunked", 128, 100.0, 1000.0),
            _row("F_short_text_compact_fused_cold_zero", 128, 60.0, 1600.0),
        ]
        rows[0]["metrics"]["actual_prompt_len"] = 128
        rows[1]["metrics"]["actual_prompt_len"] = 180

        report = self.acceptance.evaluate(
            rows,
            baseline_variant="A_single512_old_chunked",
            candidate_variant="F_short_text_compact_fused_cold_zero",
            short_latency_speedup=1.5,
            two_k_regression_tolerance=0.0,
            prompt_token_tolerance=0.05,
        )

        self.assertFalse(report["passed"])
        self.assertTrue(any("actual prompt" in failure for failure in report["failures"]))

    def test_acceptance_can_require_prompt_token_counts(self):
        rows = [
            _row("A_single512_old_chunked", 128, 100.0, 1000.0),
            _row("F_short_text_compact_fused_cold_zero", 128, 60.0, 1600.0),
        ]

        report = self.acceptance.evaluate(
            rows,
            baseline_variant="A_single512_old_chunked",
            candidate_variant="F_short_text_compact_fused_cold_zero",
            short_latency_speedup=1.5,
            two_k_regression_tolerance=0.0,
            require_prompt_token_counts=True,
        )

        self.assertFalse(report["passed"])
        self.assertTrue(any("missing actual prompt token count" in failure for failure in report["failures"]))

    def test_load_all_rows_combines_json_array_and_jsonl_files(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            array_path = Path(tmpdir) / "rows.json"
            jsonl_path = Path(tmpdir) / "rows.jsonl"
            array_path.write_text('[{"variant": "A"}]\n', encoding="utf-8")
            jsonl_path.write_text('{"variant": "B"}\n{"variant": "C"}\n', encoding="utf-8")

            rows = self.acceptance._load_all_rows([array_path, jsonl_path])

        self.assertEqual([row["variant"] for row in rows], ["A", "B", "C"])

    def test_acceptance_fails_on_bad_speedup_or_token_mismatch(self):
        rows = [
            _row("A_single512_old_chunked", 128, 100.0, 1000.0, tokens=[1, 2]),
            _row(
                "F_short_text_compact_fused_cold_zero",
                128,
                90.0,
                900.0,
                tokens=[9, 9],
            ),
        ]

        report = self.acceptance.evaluate(
            rows,
            baseline_variant="A_single512_old_chunked",
            candidate_variant="F_short_text_compact_fused_cold_zero",
            short_latency_speedup=1.5,
            two_k_regression_tolerance=0.0,
        )

        self.assertFalse(report["passed"])
        self.assertTrue(any("speedup" in failure for failure in report["failures"]))
        self.assertTrue(any("token IDs differ" in failure for failure in report["failures"]))

    def test_acceptance_fails_on_bucket_tok_regression(self):
        rows = [
            _row("A_single512_old_chunked", 128, 100.0, 1000.0),
            _row("F_short_text_compact_fused_cold_zero", 128, 60.0, 1600.0),
        ]
        rows[0]["metrics"]["bucket_tok_per_s"] = 2000.0
        rows[1]["metrics"]["bucket_tok_per_s"] = 1000.0

        report = self.acceptance.evaluate(
            rows,
            baseline_variant="A_single512_old_chunked",
            candidate_variant="F_short_text_compact_fused_cold_zero",
            short_latency_speedup=1.5,
            two_k_regression_tolerance=0.0,
            bucket_tok_regression_tolerance=0.20,
        )

        self.assertFalse(report["passed"])
        self.assertTrue(any("bucket tok/s regressed" in failure for failure in report["failures"]))

    def test_acceptance_fails_when_baseline_tokens_missing(self):
        rows = [
            _row("A_single512_old_chunked", 128, 100.0, 1000.0),
            _row("F_short_text_compact_fused_cold_zero", 128, 60.0, 1600.0),
        ]
        rows[0].pop("token_ids")

        report = self.acceptance.evaluate(
            rows,
            baseline_variant="A_single512_old_chunked",
            candidate_variant="F_short_text_compact_fused_cold_zero",
            short_latency_speedup=1.5,
            two_k_regression_tolerance=0.0,
        )

        self.assertFalse(report["passed"])
        self.assertTrue(any("baseline tokens missing" in failure for failure in report["failures"]))

    def test_acceptance_fails_when_candidate_uses_different_prompt_digest(self):
        rows = [
            _row("A_single512_old_chunked", 128, 100.0, 1000.0),
            _row("F_short_text_compact_fused_cold_zero", 128, 60.0, 1600.0),
        ]
        rows[1]["prompt_sha256"] = "different-prompt"

        report = self.acceptance.evaluate(
            rows,
            baseline_variant="A_single512_old_chunked",
            candidate_variant="F_short_text_compact_fused_cold_zero",
            short_latency_speedup=1.5,
            two_k_regression_tolerance=0.0,
        )

        self.assertFalse(report["passed"])
        self.assertTrue(any("same prompt/model/artifact/length" in failure for failure in report["failures"]))

    def test_strict_final_checks_generation_rows_use_same_controlled_inputs(self):
        rows = [
            _row("A_single512_old_chunked", 128, 100.0, 420.0),
            _row("F_short_text_compact_fused_cold_zero", 128, 60.0, 800.0),
            _row_with_generation_metrics(
                "A_single512_old_chunked",
                128,
                first_token_ms=100.0,
                decode_tps=40.0,
                end_to_end_tps=8.0,
            ),
            _row_with_generation_metrics(
                "F_short_text_compact_fused_cold_zero",
                128,
                first_token_ms=60.0,
                decode_tps=42.0,
                end_to_end_tps=9.0,
            ),
        ]
        rows[-1]["prompt_sha256"] = "different-generation-prompt"

        report = self.acceptance.evaluate(
            rows,
            baseline_variant="A_single512_old_chunked",
            candidate_variant="F_short_text_compact_fused_cold_zero",
            short_latency_speedup=1.5,
            two_k_regression_tolerance=0.0,
            strict_final=True,
            baseline_cold_tok_per_s_target=420.0,
        )

        self.assertFalse(report["passed"])
        self.assertTrue(any("same prompt/model/artifact/length" in failure for failure in report["failures"]))
        audit = {item["name"]: item for item in report["audit_checklist"]}
        self.assertEqual(
            audit["controlled prompt/model/artifact/length inputs"]["status"],
            "fail",
        )
        self.assertTrue(
            any(
                check["variant"] == "F_short_text_compact_fused_cold_zero"
                and check["max_tokens"] == 32
                and check["matches_baseline"] is False
                for check in audit["controlled prompt/model/artifact/length inputs"][
                    "evidence"
                ]
            )
        )

    def test_acceptance_flags_artifact_load_failures_and_dma_spill(self):
        rows = [
            _row("A_single512_old_chunked", 128, 100.0, 1000.0),
            _row("F_short_text_compact_fused_cold_zero", 128, 60.0, 1600.0),
        ]
        rows[1]["artifact_load_success"] = False
        rows[1]["output_tail"] = ["NRT failure: DMA spill-ring allocation failed"]

        report = self.acceptance.evaluate(
            rows,
            baseline_variant="A_single512_old_chunked",
            candidate_variant="F_short_text_compact_fused_cold_zero",
            short_latency_speedup=1.5,
            two_k_regression_tolerance=0.0,
        )

        self.assertFalse(report["passed"])
        self.assertTrue(any("artifact load/run" in failure for failure in report["failures"]))
        self.assertTrue(any("DMA spill" in failure for failure in report["failures"]))

    def test_acceptance_can_require_128k_and_262k_artifact_rows(self):
        rows = [
            _row("A_single512_old_chunked", 128, 100.0, 1000.0),
            _row("F_short_text_compact_fused_cold_zero", 128, 60.0, 1600.0),
            _row("A_single512_old_chunked", 131072, 1000.0, 131072.0),
            _long_artifact_row(
                "H_128k_candidate",
                131072,
                900.0,
                140000.0,
                q_tile=128,
                kv_tile=1024,
                block_size=128,
                cte_buckets=[256, 512, 1024, 2048],
            ),
            _row("A_single512_old_chunked", 262144, 2000.0, 131072.0),
            _long_artifact_row(
                "I_262k_recovery_block256",
                262144,
                1800.0,
                145000.0,
                q_tile=128,
                kv_tile=1024,
                block_size=256,
                cte_buckets=[256],
            ),
            _long_artifact_row(
                "J_262k_recovery_block128",
                262144,
                1700.0,
                154000.0,
                q_tile=128,
                kv_tile=1024,
                block_size=128,
                cte_buckets=[256],
            ),
        ]

        report = self.acceptance.evaluate(
            rows,
            baseline_variant="A_single512_old_chunked",
            candidate_variant="F_short_text_compact_fused_cold_zero",
            short_latency_speedup=1.5,
            two_k_regression_tolerance=0.0,
            require_128k=True,
            require_262k=True,
        )

        self.assertTrue(report["passed"], report)
        self.assertTrue(report["long_artifact_requirements"]["saw_128k_candidate"])
        self.assertTrue(report["long_artifact_requirements"]["saw_262k_block256"])
        self.assertTrue(report["long_artifact_requirements"]["saw_262k_block128"])
        audit = {item["name"]: item for item in report["audit_checklist"]}
        self.assertEqual(
            audit["long-artifact tile/block launch profiles"]["status"],
            "pass",
        )

    def test_acceptance_requires_long_artifact_tile_block_profiles(self):
        rows = [
            _row("A_single512_old_chunked", 128, 100.0, 1000.0),
            _row("F_short_text_compact_fused_cold_zero", 128, 60.0, 1600.0),
            _row("A_single512_old_chunked", 131072, 1000.0, 131072.0),
            _long_artifact_row(
                "H_128k_candidate",
                131072,
                900.0,
                140000.0,
                q_tile=128,
                kv_tile=1024,
                block_size=256,
                cte_buckets=[256, 512, 1024, 2048],
            ),
        ]

        report = self.acceptance.evaluate(
            rows,
            baseline_variant="A_single512_old_chunked",
            candidate_variant="F_short_text_compact_fused_cold_zero",
            short_latency_speedup=1.5,
            two_k_regression_tolerance=0.0,
            require_128k=True,
        )

        self.assertFalse(report["passed"])
        self.assertTrue(
            any(
                "H_128k_candidate prompt=131072" in failure
                and "required long-artifact tile/block profile" in failure
                for failure in report["failures"]
            )
        )
        audit = {item["name"]: item for item in report["audit_checklist"]}
        self.assertEqual(
            audit["long-artifact tile/block launch profiles"]["status"],
            "fail",
        )
        self.assertEqual(
            report["long_artifact_profile_checks"][0]["expected_tile_case"][
                "block_size"
            ],
            128,
        )

    def test_acceptance_requires_long_artifact_cte_buckets_and_flags(self):
        rows = [
            _row("A_single512_old_chunked", 128, 100.0, 1000.0),
            _row("F_short_text_compact_fused_cold_zero", 128, 60.0, 1600.0),
            _row("A_single512_old_chunked", 262144, 2000.0, 131072.0),
            _long_artifact_row(
                "I_262k_recovery_block256",
                262144,
                1800.0,
                145000.0,
                q_tile=128,
                kv_tile=1024,
                block_size=256,
                cte_buckets=[256, 512],
            ),
        ]
        rows[-1]["metrics"]["text_only_cte_enabled"] = False

        report = self.acceptance.evaluate(
            rows,
            baseline_variant="A_single512_old_chunked",
            candidate_variant="F_short_text_compact_fused_cold_zero",
            short_latency_speedup=1.5,
            two_k_regression_tolerance=0.0,
            require_262k=True,
        )

        self.assertFalse(report["passed"])
        self.assertTrue(
            any("required long-artifact CTE buckets" in failure for failure in report["failures"])
        )
        self.assertTrue(
            any("text-only CTE and compact CTE mask" in failure for failure in report["failures"])
        )
        check = report["long_artifact_profile_checks"][0]
        self.assertEqual(check["expected_cte_buckets"], [256])
        self.assertFalse(check["matches_required_flags"])

    def test_acceptance_fails_when_required_262k_artifact_row_missing(self):
        rows = [
            _row("A_single512_old_chunked", 128, 100.0, 1000.0),
            _row("F_short_text_compact_fused_cold_zero", 128, 60.0, 1600.0),
        ]

        report = self.acceptance.evaluate(
            rows,
            baseline_variant="A_single512_old_chunked",
            candidate_variant="F_short_text_compact_fused_cold_zero",
            short_latency_speedup=1.5,
            two_k_regression_tolerance=0.0,
            require_262k=True,
        )

        self.assertFalse(report["passed"])
        self.assertTrue(any("262K block256" in failure for failure in report["failures"]))

    def test_long_artifact_row_without_baseline_skips_exactness(self):
        rows = [
            _row("A_single512_old_chunked", 128, 100.0, 1000.0),
            _row("F_short_text_compact_fused_cold_zero", 128, 60.0, 1600.0),
            _long_artifact_row(
                "H_128k_candidate",
                131072,
                900.0,
                140000.0,
                q_tile=128,
                kv_tile=1024,
                block_size=128,
                cte_buckets=[256, 512, 1024, 2048],
            ),
        ]

        report = self.acceptance.evaluate(
            rows,
            baseline_variant="A_single512_old_chunked",
            candidate_variant="F_short_text_compact_fused_cold_zero",
            short_latency_speedup=1.5,
            two_k_regression_tolerance=0.0,
            require_128k=True,
        )

        self.assertTrue(report["passed"], report)
        self.assertTrue(any("token exactness skipped" in warning for warning in report["warnings"]))

    def test_strict_final_fails_long_artifact_without_baseline_exactness(self):
        rows = [
            _row("A_single512_old_chunked", 128, 100.0, 1000.0),
            _row("F_short_text_compact_fused_cold_zero", 128, 60.0, 1600.0),
            _row("H_128k_candidate", 131072, 900.0, 140000.0),
        ]

        report = self.acceptance.evaluate(
            rows,
            baseline_variant="A_single512_old_chunked",
            candidate_variant="F_short_text_compact_fused_cold_zero",
            short_latency_speedup=1.5,
            two_k_regression_tolerance=0.0,
            strict_final=True,
            baseline_cold_tok_per_s_target=420.0,
        )

        self.assertFalse(report["passed"])
        self.assertTrue(
            any(
                "strict final token exactness requires a baseline row for prompt=131072"
                in failure
                for failure in report["failures"]
            )
        )

    def test_acceptance_requires_short_prompt_rows(self):
        rows = [
            _row("A_single512_old_chunked", 8192, 1000.0, 8192.0),
            _row("F_short_text_compact_fused_cold_zero", 8192, 900.0, 9000.0),
        ]

        report = self.acceptance.evaluate(
            rows,
            baseline_variant="A_single512_old_chunked",
            candidate_variant="F_short_text_compact_fused_cold_zero",
            short_latency_speedup=1.5,
            two_k_regression_tolerance=0.0,
        )

        self.assertFalse(report["passed"])
        self.assertTrue(any("no short-prompt rows" in failure for failure in report["failures"]))

    def test_acceptance_checks_gdn_state_diffs_when_present(self):
        rows = [
            _row("A_single512_old_chunked", 128, 100.0, 1000.0),
            _row_with_state_diff(
                "F_short_text_compact_fused_cold_zero",
                128,
                recurrent_diff=0.001,
                conv_diff=0.002,
            ),
        ]

        report = self.acceptance.evaluate(
            rows,
            baseline_variant="A_single512_old_chunked",
            candidate_variant="F_short_text_compact_fused_cold_zero",
            short_latency_speedup=1.5,
            two_k_regression_tolerance=0.0,
            recurrent_state_tolerance=0.01,
            conv_state_tolerance=0.01,
        )

        self.assertTrue(report["passed"], report)
        self.assertEqual(len(report["gdn_state_checks"]), 1)

    def test_acceptance_can_require_gdn_state_diffs(self):
        rows = [
            _row("A_single512_old_chunked", 128, 100.0, 1000.0),
            _row("F_short_text_compact_fused_cold_zero", 128, 60.0, 1600.0),
        ]

        report = self.acceptance.evaluate(
            rows,
            baseline_variant="A_single512_old_chunked",
            candidate_variant="F_short_text_compact_fused_cold_zero",
            short_latency_speedup=1.5,
            two_k_regression_tolerance=0.0,
            require_gdn_state_diff=True,
        )

        self.assertFalse(report["passed"])
        self.assertTrue(any("GDN state-diff gate skipped" in failure for failure in report["failures"]))

    def test_strict_final_requires_gdn_state_diff_on_each_candidate_row(self):
        rows = [
            _row("A_single512_old_chunked", 128, 100.0, 420.0),
            _row_with_state_diff(
                "B_short_buckets_old_chunked",
                128,
                recurrent_diff=0.001,
                conv_diff=0.002,
            ),
            _row("F_short_text_compact_fused_cold_zero", 128, 60.0, 800.0),
        ]

        report = self.acceptance.evaluate(
            rows,
            baseline_variant="A_single512_old_chunked",
            candidate_variant="F_short_text_compact_fused_cold_zero",
            short_latency_speedup=1.5,
            two_k_regression_tolerance=0.0,
            strict_final=True,
            baseline_cold_tok_per_s_target=420.0,
        )

        self.assertFalse(report["passed"])
        self.assertTrue(
            any(
                "F_short_text_compact_fused_cold_zero prompt=128 missing GDN recurrent/conv state diff fields"
                in failure
                for failure in report["failures"]
            )
        )

    def test_acceptance_checks_reported_gdn_cte_kernel_profiles(self):
        rows = [
            _row("A_single512_old_chunked", 128, 100.0, 1000.0),
            _row("F_short_text_compact_fused_cold_zero", 128, 60.0, 1600.0),
        ]
        rows[0]["metrics"]["gdn_cte_kernel"] = "fused_initial_state"
        rows[1]["metrics"]["gdn_cte_kernel"] = "fused_initial_state"

        report = self.acceptance.evaluate(
            rows,
            baseline_variant="A_single512_old_chunked",
            candidate_variant="F_short_text_compact_fused_cold_zero",
            short_latency_speedup=1.5,
            two_k_regression_tolerance=0.0,
        )

        self.assertFalse(report["passed"])
        self.assertTrue(
            any(
                "A_single512_old_chunked prompt=128" in failure
                and "expected nki_chunked" in failure
                for failure in report["failures"]
            )
        )
        audit = {item["name"]: item for item in report["audit_checklist"]}
        self.assertEqual(audit["GDN CTE kernel launch profiles"]["status"], "fail")

    def test_acceptance_checks_reported_a_to_g_launch_profiles(self):
        rows = [
            _with_feature_profile(
                _row("A_single512_old_chunked", 128, 100.0, 1000.0),
                cte_buckets=[128, 256, 512, 1024],
                text_only=False,
                compact=False,
                cold_zero=False,
            ),
            _with_feature_profile(
                _row("F_short_text_compact_fused_cold_zero", 128, 60.0, 1600.0),
                cte_buckets=[128, 256, 512, 1024],
                text_only=True,
                compact=True,
                cold_zero=True,
            ),
        ]

        report = self.acceptance.evaluate(
            rows,
            baseline_variant="A_single512_old_chunked",
            candidate_variant="F_short_text_compact_fused_cold_zero",
            short_latency_speedup=1.5,
            two_k_regression_tolerance=0.0,
        )

        self.assertFalse(report["passed"])
        self.assertTrue(
            any(
                "A_single512_old_chunked prompt=128" in failure
                and "required A-G launch profile" in failure
                and "cte_buckets" in failure
                for failure in report["failures"]
            )
        )
        audit = {item["name"]: item for item in report["audit_checklist"]}
        self.assertEqual(audit["A-G launch profiles"]["status"], "fail")
        self.assertEqual(
            report["feature_launch_profile_checks"][0]["expected_profile"][
                "cte_buckets"
            ],
            [512],
        )

    def test_strict_final_requires_gdn_cte_kernel_metrics(self):
        rows = [
            _row("A_single512_old_chunked", 128, 100.0, 420.0),
            _row("F_short_text_compact_fused_cold_zero", 128, 60.0, 800.0),
        ]

        report = self.acceptance.evaluate(
            rows,
            baseline_variant="A_single512_old_chunked",
            candidate_variant="F_short_text_compact_fused_cold_zero",
            short_latency_speedup=1.5,
            two_k_regression_tolerance=0.0,
            strict_final=True,
            baseline_cold_tok_per_s_target=420.0,
        )

        self.assertFalse(report["passed"])
        self.assertTrue(
            any("missing GDN CTE kernel metric" in failure for failure in report["failures"])
        )
        audit = {item["name"]: item for item in report["audit_checklist"]}
        self.assertEqual(audit["GDN CTE kernel launch profiles"]["status"], "fail")

    def test_strict_final_requires_a_to_g_launch_profile_metrics(self):
        rows = [
            _row("A_single512_old_chunked", 128, 100.0, 420.0),
            _row("F_short_text_compact_fused_cold_zero", 128, 60.0, 800.0),
        ]

        report = self.acceptance.evaluate(
            rows,
            baseline_variant="A_single512_old_chunked",
            candidate_variant="F_short_text_compact_fused_cold_zero",
            short_latency_speedup=1.5,
            two_k_regression_tolerance=0.0,
            strict_final=True,
            baseline_cold_tok_per_s_target=420.0,
        )

        self.assertFalse(report["passed"])
        self.assertTrue(
            any("missing A-G launch profile fields" in failure for failure in report["failures"])
        )
        audit = {item["name"]: item for item in report["audit_checklist"]}
        self.assertEqual(audit["A-G launch profiles"]["status"], "fail")

    def test_long_context_gate_allows_neuron_chunked_path_without_compact_flag(self):
        rows = [
            _row("A_single512_old_chunked", 128, 100.0, 1000.0),
            _row("F_short_text_compact_fused_cold_zero", 128, 60.0, 1600.0),
            _row("A_single512_old_chunked", 8192, 1000.0, 8192.0),
        ]
        rows[-1]["metrics"]["compact_mask_enabled"] = False
        rows[-1]["metrics"]["chunked_prefill_enabled"] = True

        report = self.acceptance.evaluate(
            rows,
            baseline_variant="A_single512_old_chunked",
            candidate_variant="F_short_text_compact_fused_cold_zero",
            short_latency_speedup=1.5,
            two_k_regression_tolerance=0.0,
        )

        self.assertTrue(report["passed"], report)

    def test_acceptance_checks_hbm_regression_when_requested(self):
        rows = [
            _row_with_hbm(
                "A_single512_old_chunked",
                128,
                latency_ms=100.0,
                tps=1000.0,
                used_bytes=1000,
            ),
            _row_with_hbm(
                "F_short_text_compact_fused_cold_zero",
                128,
                latency_ms=60.0,
                tps=1600.0,
                used_bytes=1100,
            ),
        ]

        report = self.acceptance.evaluate(
            rows,
            baseline_variant="A_single512_old_chunked",
            candidate_variant="F_short_text_compact_fused_cold_zero",
            short_latency_speedup=1.5,
            two_k_regression_tolerance=0.0,
            hbm_regression_tolerance=0.05,
            require_hbm_usage=True,
        )

        self.assertFalse(report["passed"])
        self.assertTrue(any("HBM usage regressed" in failure for failure in report["failures"]))

    def test_strict_final_requires_hbm_usage_on_each_row(self):
        rows = [
            _row_with_hbm(
                "A_single512_old_chunked",
                128,
                latency_ms=100.0,
                tps=420.0,
                used_bytes=1000,
            ),
            _row("F_short_text_compact_fused_cold_zero", 128, 60.0, 800.0),
            _row_with_hbm(
                "A_single512_old_chunked",
                128,
                latency_ms=220.0,
                tps=300.0,
                used_bytes=1200,
            ),
            _row_with_generation_metrics(
                "F_short_text_compact_fused_cold_zero",
                128,
                first_token_ms=60.0,
                decode_tps=42.0,
                end_to_end_tps=7.0,
            ),
        ]
        rows[-2]["max_tokens"] = 32

        report = self.acceptance.evaluate(
            rows,
            baseline_variant="A_single512_old_chunked",
            candidate_variant="F_short_text_compact_fused_cold_zero",
            short_latency_speedup=1.5,
            two_k_regression_tolerance=0.0,
            strict_final=True,
            baseline_cold_tok_per_s_target=420.0,
        )

        self.assertFalse(report["passed"])
        audit = {item["name"]: item for item in report["audit_checklist"]}
        self.assertEqual(audit["HBM usage present on every row"]["status"], "fail")
        self.assertTrue(
            any(
                "F_short_text_compact_fused_cold_zero prompt=128 max_tokens=None missing HBM usage"
                in failure
                for failure in report["failures"]
            )
        )
        self.assertTrue(
            any(
                "F_short_text_compact_fused_cold_zero prompt=128 max_tokens=32 missing HBM usage"
                in failure
                for failure in report["failures"]
            )
        )

    def test_acceptance_parses_xla_hbm_memory_info(self):
        row = _row("A_single512_old_chunked", 128, 100.0, 1000.0)
        row["metrics"]["hbm_usage"] = {"kb_total": 4096, "kb_free": 1024}

        self.assertEqual(self.acceptance._hbm_used_bytes(row), 3072 * 1024)

    def test_acceptance_filters_to_prefill_max_tokens_rows(self):
        rows = [
            _row_with_max_tokens("A_single512_old_chunked", 128, 100.0, 1000.0, 1),
            _row_with_max_tokens(
                "F_short_text_compact_fused_cold_zero",
                128,
                60.0,
                1600.0,
                1,
            ),
            _row_with_max_tokens("A_single512_old_chunked", 128, 1000.0, 100.0, 32),
            _row_with_max_tokens(
                "F_short_text_compact_fused_cold_zero",
                128,
                1000.0,
                100.0,
                32,
            ),
        ]

        report = self.acceptance.evaluate(
            rows,
            baseline_variant="A_single512_old_chunked",
            candidate_variant="F_short_text_compact_fused_cold_zero",
            short_latency_speedup=1.5,
            two_k_regression_tolerance=0.0,
            prefill_max_tokens=1,
        )

        self.assertTrue(report["passed"], report)
        self.assertEqual(report["prefill_max_tokens"], 1)
        self.assertEqual(report["short_results"][0]["baseline_p50_ms"], 100.0)

    def test_acceptance_reports_latency_p50_p95_across_repetitions(self):
        rows = [
            _row_with_max_tokens("A_single512_old_chunked", 128, 100.0, 1000.0, 1),
            _row_with_max_tokens("A_single512_old_chunked", 128, 120.0, 900.0, 1),
            _row_with_max_tokens("A_single512_old_chunked", 128, 140.0, 800.0, 1),
            _row_with_max_tokens(
                "F_short_text_compact_fused_cold_zero",
                128,
                50.0,
                2000.0,
                1,
            ),
            _row_with_max_tokens(
                "F_short_text_compact_fused_cold_zero",
                128,
                60.0,
                1800.0,
                1,
            ),
            _row_with_max_tokens(
                "F_short_text_compact_fused_cold_zero",
                128,
                70.0,
                1700.0,
                1,
            ),
            _row_with_max_tokens("A_single512_old_chunked", 2048, 500.0, 4096.0, 1),
            _row_with_max_tokens(
                "F_short_text_compact_fused_cold_zero",
                2048,
                500.0,
                4096.0,
                1,
            ),
        ]

        report = self.acceptance.evaluate(
            rows,
            baseline_variant="A_single512_old_chunked",
            candidate_variant="F_short_text_compact_fused_cold_zero",
            short_latency_speedup=1.5,
            two_k_regression_tolerance=0.0,
            prefill_max_tokens=1,
        )

        self.assertTrue(report["passed"], report)
        short = report["short_results"][0]
        self.assertEqual(short["baseline_p50_ms"], 120.0)
        self.assertEqual(short["candidate_p50_ms"], 60.0)
        self.assertEqual(short["baseline_p95_ms"], 140.0)
        self.assertEqual(short["candidate_p95_ms"], 70.0)
        baseline_summary = next(
            item
            for item in report["latency_summaries"]
            if item["variant"] == "A_single512_old_chunked"
            and item["prompt_len"] == 128
        )
        self.assertEqual(baseline_summary["sample_count"], 3)
        self.assertEqual(baseline_summary["p50_ms"], 120.0)
        self.assertEqual(baseline_summary["p95_ms"], 140.0)

    def test_strict_final_requires_repeated_samples_for_p50_p95(self):
        rows = []
        for _index in range(2):
            rows.extend(
                [
                    _row_with_max_tokens(
                        "A_single512_old_chunked",
                        128,
                        100.0,
                        420.0,
                        1,
                    ),
                    _row_with_max_tokens(
                        "F_short_text_compact_fused_cold_zero",
                        128,
                        60.0,
                        800.0,
                        1,
                    ),
                    _row_with_generation_metrics(
                        "A_single512_old_chunked",
                        128,
                        first_token_ms=100.0,
                        decode_tps=40.0,
                        end_to_end_tps=8.0,
                    ),
                    _row_with_generation_metrics(
                        "F_short_text_compact_fused_cold_zero",
                        128,
                        first_token_ms=60.0,
                        decode_tps=42.0,
                        end_to_end_tps=9.0,
                    ),
                ]
            )

        report = self.acceptance.evaluate(
            rows,
            baseline_variant="A_single512_old_chunked",
            candidate_variant="F_short_text_compact_fused_cold_zero",
            short_latency_speedup=1.5,
            two_k_regression_tolerance=0.0,
            strict_final=True,
            baseline_cold_tok_per_s_target=420.0,
            strict_min_samples=3,
        )

        self.assertFalse(report["passed"])
        self.assertEqual(report["strict_min_samples"], 3)
        self.assertTrue(
            any(
                "at least 3 prefill samples for A_single512_old_chunked prompt=128; saw 2"
                in failure
                for failure in report["failures"]
            )
        )
        self.assertTrue(
            any(
                "at least 3 max_tokens=32 generation samples for "
                "F_short_text_compact_fused_cold_zero prompt=128; saw 2"
                in failure
                for failure in report["failures"]
            )
        )
        audit = {item["name"]: item for item in report["audit_checklist"]}
        self.assertEqual(audit["strict-final repeated samples"]["status"], "fail")
        self.assertEqual(
            audit["prefill latency p50/p95 summaries"]["status"],
            "fail",
        )

    def test_strict_final_requires_repeated_samples_per_tile_case(self):
        rows = []
        for _index in range(3):
            rows.append(
                _row_with_tile(
                    "G_tile_block_sweep",
                    2048,
                    450.0,
                    4500.0,
                    q_tile=128,
                    kv_tile=512,
                    block_size=128,
                )
            )
        rows.append(
            _row_with_tile(
                "G_tile_block_sweep",
                2048,
                460.0,
                4400.0,
                q_tile=128,
                kv_tile=1024,
                block_size=128,
            )
        )
        rows.extend(
            [
                _row("A_single512_old_chunked", 128, 100.0, 420.0),
                _row("F_short_text_compact_fused_cold_zero", 128, 60.0, 800.0),
            ]
        )

        report = self.acceptance.evaluate(
            rows,
            baseline_variant="A_single512_old_chunked",
            candidate_variant="F_short_text_compact_fused_cold_zero",
            short_latency_speedup=1.5,
            two_k_regression_tolerance=0.0,
            strict_final=True,
            baseline_cold_tok_per_s_target=420.0,
            strict_min_samples=3,
        )

        self.assertFalse(report["passed"])
        self.assertTrue(
            any(
                "G_tile_block_sweep prompt=2048" in failure
                and "kernel_kv_tile_size': 1024" in failure
                and "saw 1" in failure
                for failure in report["failures"]
            )
        )
        tile_checks = [
            check
            for check in report["strict_sample_count_checks"]
            if check["kind"] == "tile_prefill"
            and check["prompt_len"] == 2048
            and check["tile_case"]["kernel_kv_tile_size"] == 1024
            and check["tile_case"]["block_size"] == 128
        ]
        self.assertEqual(tile_checks[0]["sample_count"], 1)
        self.assertFalse(tile_checks[0]["present"])
        audit = {item["name"]: item for item in report["audit_checklist"]}
        self.assertEqual(
            audit["G tile-sweep latency p50/p95 summaries"]["status"],
            "fail",
        )

    def test_strict_final_rejects_dense_sxs_mask_on_long_context_rows(self):
        rows = _strict_final_complete_rows(self.acceptance)
        for row in rows:
            if (
                row["variant"] == "F_short_text_compact_fused_cold_zero"
                and row["target_prompt_tokens"] == 8192
                and row["max_tokens"] == 1
            ):
                row["metrics"]["dense_cte_mask_fallback"] = True
                row["metrics"]["cte_attention_mask_path"] = "dense_4d_fallback"
                break

        report = self.acceptance.evaluate(
            rows,
            baseline_variant="A_single512_old_chunked",
            candidate_variant="F_short_text_compact_fused_cold_zero",
            short_latency_speedup=1.5,
            two_k_regression_tolerance=0.0,
            baseline_cold_tok_per_s_target=420.0,
            strict_final=True,
            hybrid_apc_report=_strict_final_hybrid_apc_report(),
        )

        self.assertFalse(report["passed"])
        self.assertTrue(
            any(
                "F_short_text_compact_fused_cold_zero prompt=8192 "
                "reported dense SxS CTE attention-mask fallback" in failure
                for failure in report["failures"]
            )
        )
        audit = {item["name"]: item for item in report["audit_checklist"]}
        self.assertEqual(
            audit["8K+ compact/chunked path avoids dense SxS fallback"]["status"],
            "fail",
        )

    def test_strict_final_requires_generation_rows_per_tile_case(self):
        rows = []
        for _index in range(3):
            row = _row_with_generation_metrics(
                "G_tile_block_sweep",
                2048,
                first_token_ms=450.0,
                decode_tps=32.0,
                end_to_end_tps=7.0,
            )
            row.update(
                {
                    "kernel_q_tile_size": 128,
                    "kernel_kv_tile_size": 512,
                    "block_size": 128,
                }
            )
            row["metrics"].update(
                {
                    "kernel_q_tile_size": 128,
                    "kernel_kv_tile_size": 512,
                    "block_size": 128,
                }
            )
            rows.append(row)
        rows.extend(
            [
                _row("A_single512_old_chunked", 128, 100.0, 420.0),
                _row("F_short_text_compact_fused_cold_zero", 128, 60.0, 800.0),
            ]
        )

        report = self.acceptance.evaluate(
            rows,
            baseline_variant="A_single512_old_chunked",
            candidate_variant="F_short_text_compact_fused_cold_zero",
            short_latency_speedup=1.5,
            two_k_regression_tolerance=0.0,
            strict_final=True,
            baseline_cold_tok_per_s_target=420.0,
            strict_min_samples=3,
        )

        self.assertFalse(report["passed"])
        audit = {item["name"]: item for item in report["audit_checklist"]}
        self.assertEqual(
            audit["G max_tokens=32 tile-sweep generation cases"]["status"],
            "fail",
        )
        self.assertTrue(
            any(
                "G_tile_block_sweep prompt=2048 max_tokens=32" in failure
                and "kernel_kv_tile_size': 1024" in failure
                for failure in report["failures"]
            )
        )
        checks = [
            check
            for check in report["strict_sample_count_checks"]
            if check["kind"] == "tile_generation"
            and check["prompt_len"] == 2048
            and check["tile_case"]["kernel_kv_tile_size"] == 1024
            and check["tile_case"]["block_size"] == 128
        ]
        self.assertEqual(checks[0]["sample_count"], 0)
        self.assertFalse(checks[0]["present"])

    def test_strict_final_requires_small_dense_fallback_exactness(self):
        rows = _strict_final_complete_rows(self.acceptance)
        rows = [
            row
            for row in rows
            if not (
                row["variant"] == self.acceptance.DENSE_FALLBACK_VARIANT
                and row["target_prompt_tokens"] == 256
                and row["max_tokens"] == 1
            )
        ]

        report = self.acceptance.evaluate(
            rows,
            baseline_variant="A_single512_old_chunked",
            candidate_variant="F_short_text_compact_fused_cold_zero",
            short_latency_speedup=1.5,
            two_k_regression_tolerance=0.0,
            strict_final=True,
            baseline_cold_tok_per_s_target=420.0,
        )

        self.assertFalse(report["passed"])
        self.assertTrue(
            any(
                "L_small_dense_mask_fallback prompt=256 max_tokens=1 row"
                in failure
                for failure in report["failures"]
            )
        )
        audit = {item["name"]: item for item in report["audit_checklist"]}
        self.assertEqual(
            audit["small dense mask fallback exactness"]["status"],
            "fail",
        )

    def test_strict_final_fails_when_tokens_differ_from_dense_fallback(self):
        rows = _strict_final_complete_rows(self.acceptance)
        for row in rows:
            if (
                row["variant"] == "F_short_text_compact_fused_cold_zero"
                and row["target_prompt_tokens"] == 512
                and row["max_tokens"] == 32
            ):
                row["token_ids"] = [9, 9, 9]
                break

        report = self.acceptance.evaluate(
            rows,
            baseline_variant="A_single512_old_chunked",
            candidate_variant="F_short_text_compact_fused_cold_zero",
            short_latency_speedup=1.5,
            two_k_regression_tolerance=0.0,
            strict_final=True,
            baseline_cold_tok_per_s_target=420.0,
        )

        self.assertFalse(report["passed"])
        self.assertTrue(
            any(
                "F_short_text_compact_fused_cold_zero prompt=512 max_tokens=32 "
                "token IDs differ from dense fallback" in failure
                for failure in report["failures"]
            )
        )

    def test_strict_final_requires_repeated_dense_fallback_samples(self):
        rows = _strict_final_complete_rows(self.acceptance)
        kept_dense = 0
        filtered = []
        for row in rows:
            if (
                row["variant"] == self.acceptance.DENSE_FALLBACK_VARIANT
                and row["target_prompt_tokens"] == 512
                and row["max_tokens"] == 32
            ):
                kept_dense += 1
                if kept_dense > 2:
                    continue
            filtered.append(row)

        report = self.acceptance.evaluate(
            filtered,
            baseline_variant="A_single512_old_chunked",
            candidate_variant="F_short_text_compact_fused_cold_zero",
            short_latency_speedup=1.5,
            two_k_regression_tolerance=0.0,
            strict_final=True,
            baseline_cold_tok_per_s_target=420.0,
        )

        self.assertFalse(report["passed"])
        self.assertTrue(
            any(
                "max_tokens=32 generation samples for "
                "L_small_dense_mask_fallback prompt=512; saw 2"
                in failure
                for failure in report["failures"]
            )
        )

    def test_acceptance_reports_generation_metric_summaries(self):
        rows = [
            _row_with_max_tokens("A_single512_old_chunked", 128, 100.0, 1000.0, 1),
            _row_with_max_tokens(
                "F_short_text_compact_fused_cold_zero",
                128,
                60.0,
                1600.0,
                1,
            ),
            _row_with_generation_metrics(
                "F_short_text_compact_fused_cold_zero",
                128,
                first_token_ms=60.0,
                decode_tps=42.0,
                end_to_end_tps=7.0,
            ),
        ]

        report = self.acceptance.evaluate(
            rows,
            baseline_variant="A_single512_old_chunked",
            candidate_variant="F_short_text_compact_fused_cold_zero",
            short_latency_speedup=1.5,
            two_k_regression_tolerance=0.0,
        )

        self.assertTrue(report["passed"], report)
        summary = next(
            item
            for item in report["generation_summaries"]
            if item["variant"] == "F_short_text_compact_fused_cold_zero"
        )
        self.assertEqual(summary["first_token_p50_ms"], 60.0)
        self.assertEqual(summary["decode_tok_per_s_p50"], 42.0)
        self.assertEqual(summary["end_to_end_generated_tok_per_s_p50"], 7.0)

    def test_strict_final_requires_generation_rows_and_metrics(self):
        rows = [
            _row("A_single512_old_chunked", 128, 100.0, 420.0),
            _row("F_short_text_compact_fused_cold_zero", 128, 60.0, 800.0),
            _row_with_max_tokens(
                "A_single512_old_chunked",
                128,
                220.0,
                600.0,
                32,
            ),
            _row_with_generation_metrics(
                "F_short_text_compact_fused_cold_zero",
                128,
                first_token_ms=60.0,
                decode_tps=42.0,
                end_to_end_tps=7.0,
            ),
        ]

        report = self.acceptance.evaluate(
            rows,
            baseline_variant="A_single512_old_chunked",
            candidate_variant="F_short_text_compact_fused_cold_zero",
            short_latency_speedup=1.5,
            two_k_regression_tolerance=0.0,
            strict_final=True,
            baseline_cold_tok_per_s_target=420.0,
        )

        self.assertFalse(report["passed"])
        audit = {item["name"]: item for item in report["audit_checklist"]}
        self.assertEqual(audit["A-G max_tokens=32 generation rows"]["status"], "fail")
        self.assertEqual(audit["generation latency/decode metrics"]["status"], "fail")
        self.assertTrue(
            any(
                "A_single512_old_chunked prompt=128 max_tokens=32 missing generation metrics"
                in failure
                for failure in report["failures"]
            )
        )
        self.assertTrue(
            any(
                "G_tile_block_sweep prompt=32768 max_tokens=32 row"
                in failure
                for failure in report["failures"]
            )
        )

    def test_strict_final_checks_generation_token_exactness(self):
        rows = [
            _row("A_single512_old_chunked", 128, 100.0, 420.0),
            _row("F_short_text_compact_fused_cold_zero", 128, 60.0, 800.0),
            _row_with_generation_metrics(
                "A_single512_old_chunked",
                128,
                first_token_ms=100.0,
                decode_tps=20.0,
                end_to_end_tps=6.0,
                tokens=[1, 2, 3],
            ),
            _row_with_generation_metrics(
                "F_short_text_compact_fused_cold_zero",
                128,
                first_token_ms=60.0,
                decode_tps=42.0,
                end_to_end_tps=7.0,
                tokens=[1, 2, 4],
            ),
        ]

        report = self.acceptance.evaluate(
            rows,
            baseline_variant="A_single512_old_chunked",
            candidate_variant="F_short_text_compact_fused_cold_zero",
            short_latency_speedup=1.5,
            two_k_regression_tolerance=0.0,
            strict_final=True,
            baseline_cold_tok_per_s_target=420.0,
        )

        self.assertFalse(report["passed"])
        audit = {item["name"]: item for item in report["audit_checklist"]}
        self.assertEqual(audit["max_tokens=32 token exactness vs baseline"]["status"], "fail")
        self.assertTrue(
            any(
                "F_short_text_compact_fused_cold_zero prompt=128 max_tokens=32 token IDs differ from baseline"
                in failure
                for failure in report["failures"]
            )
        )

    def test_strict_final_checks_long_artifact_generation_rows_and_exactness(self):
        rows = [
            _row("A_single512_old_chunked", 128, 100.0, 420.0),
            _row("F_short_text_compact_fused_cold_zero", 128, 60.0, 800.0),
            _row_with_generation_metrics(
                "A_single512_old_chunked",
                131072,
                first_token_ms=1000.0,
                decode_tps=20.0,
                end_to_end_tps=3.0,
                tokens=[1, 2, 3],
            ),
            _row_with_generation_metrics(
                "H_128k_candidate",
                131072,
                first_token_ms=900.0,
                decode_tps=22.0,
                end_to_end_tps=3.5,
                tokens=[1, 2, 4],
            ),
        ]

        report = self.acceptance.evaluate(
            rows,
            baseline_variant="A_single512_old_chunked",
            candidate_variant="F_short_text_compact_fused_cold_zero",
            short_latency_speedup=1.5,
            two_k_regression_tolerance=0.0,
            strict_final=True,
            baseline_cold_tok_per_s_target=420.0,
        )

        self.assertFalse(report["passed"])
        self.assertIn(
            {"variant": "H_128k_candidate", "prompt_len": 131072},
            report["long_artifact_generation_targets"],
        )
        self.assertTrue(
            any(
                "H_128k_candidate prompt=131072 max_tokens=32 token IDs differ from baseline"
                in failure
                for failure in report["failures"]
            )
        )
        self.assertTrue(
            any(
                "I_262k_recovery_block256 prompt=262144 max_tokens=32 long-artifact row"
                in failure
                for failure in report["failures"]
            )
        )

    def test_acceptance_reports_feature_delta_checks(self):
        rows = [
            _row("A_single512_old_chunked", 128, 100.0, 1000.0),
            _row("B_short_buckets_old_chunked", 128, 90.0, 1100.0),
            _row("C_short_text_only_old_chunked", 128, 80.0, 1200.0),
            _row("D_short_text_compact_old_chunked", 128, 70.0, 1300.0),
            _row("E_short_text_compact_fused", 128, 65.0, 1400.0),
            _row("F_short_text_compact_fused_cold_zero", 128, 60.0, 1500.0),
        ]

        report = self.acceptance.evaluate(
            rows,
            baseline_variant="A_single512_old_chunked",
            candidate_variant="F_short_text_compact_fused_cold_zero",
            short_latency_speedup=1.5,
            two_k_regression_tolerance=0.0,
        )

        self.assertTrue(report["passed"], report)
        deltas = {item["name"]: item for item in report["feature_delta_checks"]}
        self.assertEqual(deltas["dynamic CTE buckets"]["status"], "pass")
        self.assertEqual(deltas["text-only CTE"]["status"], "pass")
        self.assertEqual(deltas["cold-zero conv fast path"]["status"], "pass")
        self.assertEqual(deltas["tile/block sweep"]["status"], "skip")
        self.assertEqual(
            deltas["text-only CTE"]["prompt_checks"][0]["latency_ratio"],
            80.0 / 90.0,
        )

    def test_tile_sweep_feature_delta_uses_best_tile_case(self):
        rows = [
            _row("A_single512_old_chunked", 128, 100.0, 1000.0),
            _row("F_short_text_compact_fused_cold_zero", 128, 60.0, 1500.0),
            _row_with_tile(
                "G_tile_block_sweep",
                128,
                200.0,
                500.0,
                q_tile=128,
                kv_tile=512,
                block_size=128,
            ),
            _row_with_tile(
                "G_tile_block_sweep",
                128,
                55.0,
                1700.0,
                q_tile=128,
                kv_tile=1024,
                block_size=128,
            ),
        ]

        report = self.acceptance.evaluate(
            rows,
            baseline_variant="A_single512_old_chunked",
            candidate_variant="F_short_text_compact_fused_cold_zero",
            short_latency_speedup=1.5,
            two_k_regression_tolerance=0.0,
            require_feature_deltas=True,
        )

        self.assertTrue(report["passed"], report)
        deltas = {item["name"]: item for item in report["feature_delta_checks"]}
        tile_delta = deltas["tile/block sweep"]
        self.assertEqual(tile_delta["status"], "pass")
        self.assertEqual(len(tile_delta["tile_case_checks"]), 2)
        self.assertEqual(tile_delta["prompt_checks"][0]["target_p50_ms"], 55.0)
        self.assertEqual(
            tile_delta["prompt_checks"][0]["target_tile_case"],
            {
                "kernel_q_tile_size": 128,
                "kernel_kv_tile_size": 1024,
                "block_size": 128,
            },
        )
        self.assertTrue(tile_delta["prompt_checks"][0]["is_best_tile_case"])

    def test_feature_delta_checks_include_hbm_direction(self):
        rows = [
            _row_with_hbm(
                "B_short_buckets_old_chunked",
                128,
                latency_ms=90.0,
                tps=1100.0,
                used_bytes=2000,
            ),
            _row_with_hbm(
                "C_short_text_only_old_chunked",
                128,
                latency_ms=80.0,
                tps=1200.0,
                used_bytes=1500,
            ),
            _row("A_single512_old_chunked", 128, 100.0, 1000.0),
            _row("F_short_text_compact_fused_cold_zero", 128, 60.0, 1600.0),
        ]

        report = self.acceptance.evaluate(
            rows,
            baseline_variant="A_single512_old_chunked",
            candidate_variant="F_short_text_compact_fused_cold_zero",
            short_latency_speedup=1.5,
            two_k_regression_tolerance=0.0,
        )

        deltas = {item["name"]: item for item in report["feature_delta_checks"]}
        text_only = deltas["text-only CTE"]["prompt_checks"][0]
        self.assertEqual(text_only["source_hbm_used_bytes"], 2000.0)
        self.assertEqual(text_only["target_hbm_used_bytes"], 1500.0)
        self.assertEqual(text_only["hbm_delta_bytes"], -500.0)

    def test_feature_delta_regresses_when_hbm_increases(self):
        rows = [
            _row("A_single512_old_chunked", 128, 100.0, 1000.0),
            _row_with_hbm(
                "B_short_buckets_old_chunked",
                128,
                latency_ms=90.0,
                tps=1100.0,
                used_bytes=2000,
            ),
            _row_with_hbm(
                "C_short_text_only_old_chunked",
                128,
                latency_ms=80.0,
                tps=1200.0,
                used_bytes=2500,
            ),
            _row("F_short_text_compact_fused_cold_zero", 128, 60.0, 1600.0),
        ]

        report = self.acceptance.evaluate(
            rows,
            baseline_variant="A_single512_old_chunked",
            candidate_variant="F_short_text_compact_fused_cold_zero",
            short_latency_speedup=1.5,
            two_k_regression_tolerance=0.0,
            require_feature_deltas=True,
        )

        self.assertFalse(report["passed"])
        deltas = {item["name"]: item for item in report["feature_delta_checks"]}
        text_only = deltas["text-only CTE"]
        self.assertEqual(text_only["status"], "regressed")
        self.assertEqual(text_only["prompt_checks"][0]["hbm_delta_bytes"], 500.0)
        self.assertTrue(
            any(
                "text-only CTE feature delta regressed" in failure
                for failure in report["failures"]
            )
        )

    def test_acceptance_marks_feature_delta_regression(self):
        rows = [
            _row("A_single512_old_chunked", 128, 100.0, 1000.0),
            _row("B_short_buckets_old_chunked", 128, 120.0, 900.0),
            _row("F_short_text_compact_fused_cold_zero", 128, 60.0, 1600.0),
        ]

        report = self.acceptance.evaluate(
            rows,
            baseline_variant="A_single512_old_chunked",
            candidate_variant="F_short_text_compact_fused_cold_zero",
            short_latency_speedup=1.5,
            two_k_regression_tolerance=0.0,
        )

        deltas = {item["name"]: item for item in report["feature_delta_checks"]}
        self.assertEqual(deltas["dynamic CTE buckets"]["status"], "regressed")

    def test_acceptance_can_require_feature_delta_checks(self):
        rows = [
            _row("A_single512_old_chunked", 128, 100.0, 1000.0),
            _row("B_short_buckets_old_chunked", 128, 120.0, 900.0),
            _row("F_short_text_compact_fused_cold_zero", 128, 60.0, 1600.0),
        ]

        report = self.acceptance.evaluate(
            rows,
            baseline_variant="A_single512_old_chunked",
            candidate_variant="F_short_text_compact_fused_cold_zero",
            short_latency_speedup=1.5,
            two_k_regression_tolerance=0.0,
            require_feature_deltas=True,
        )

        self.assertFalse(report["passed"])
        self.assertTrue(any("feature delta regressed" in failure for failure in report["failures"]))

    def test_strict_final_requires_all_optional_goal_gates(self):
        rows = [
            _row("A_single512_old_chunked", 128, 100.0, 1000.0),
            _row("F_short_text_compact_fused_cold_zero", 128, 60.0, 1600.0),
        ]

        report = self.acceptance.evaluate(
            rows,
            baseline_variant="A_single512_old_chunked",
            candidate_variant="F_short_text_compact_fused_cold_zero",
            short_latency_speedup=1.5,
            two_k_regression_tolerance=0.0,
            strict_final=True,
        )

        self.assertFalse(report["passed"])
        self.assertTrue(report["strict_final"])
        self.assertTrue(report["require_feature_deltas"])
        audit = {item["name"]: item for item in report["audit_checklist"]}
        self.assertEqual(audit["fixed prompt suite rows"]["status"], "fail")
        self.assertEqual(audit["A-G fixed prompt matrix rows"]["status"], "fail")
        self.assertEqual(audit["2K/8K tile sweep cases"]["status"], "fail")
        self.assertEqual(audit["G max_tokens=32 tile-sweep generation cases"]["status"], "fail")
        self.assertEqual(audit["fixed prompt suite token counts"]["status"], "fail")
        self.assertEqual(audit["HBM usage regression"]["status"], "fail")
        self.assertEqual(audit["GDN recurrent/conv state diff bounds"]["status"], "fail")
        self.assertEqual(audit["128K artifact load/run"]["status"], "fail")
        self.assertEqual(audit["262K block256 artifact load/run"]["status"], "fail")
        self.assertEqual(audit["262K block128 comparison load/run"]["status"], "fail")
        self.assertEqual(audit["strict-final repeated samples"]["status"], "fail")
        self.assertEqual(audit["A-G launch profiles"]["status"], "fail")
        self.assertEqual(audit["GDN CTE kernel launch profiles"]["status"], "fail")
        self.assertEqual(
            audit["hybrid APC partial-prefix exactness"]["status"],
            "fail",
        )
        self.assertIn(32768, report["mandatory_prompt_lengths"])
        self.assertIn("G_tile_block_sweep", report["feature_matrix_variants"])
        self.assertEqual(
            report["expected_feature_launch_profile_by_variant"][
                "F_short_text_compact_fused_cold_zero"
            ]["cold_zero_conv_fast_path_enabled"],
            True,
        )
        self.assertEqual(
            report["expected_gdn_cte_kernel_by_variant"]["E_short_text_compact_fused"],
            "fused_initial_state",
        )
        self.assertIn(8192, report["required_tile_sweep_prompts"])
        self.assertIn(
            {
                "kernel_q_tile_size": 128,
                "kernel_kv_tile_size": 2048,
                "block_size": 128,
            },
            report["required_tile_sweep_cases"],
        )
        self.assertTrue(
            any(
                failure
                == (
                    "strict final acceptance requires "
                    "A_single512_old_chunked prompt=256 row"
                )
                for failure in report["failures"]
            )
        )
        self.assertTrue(
            any(
                failure
                == (
                    "strict final acceptance requires "
                    "G_tile_block_sweep prompt=32768 matrix row"
                )
                for failure in report["failures"]
            )
        )
        self.assertTrue(
            any(
                "strict final acceptance requires G_tile_block_sweep prompt=8192 "
                in failure
                and "kernel_kv_tile_size" in failure
                for failure in report["failures"]
            )
        )
        self.assertTrue(
            any("--baseline-cold-tok-per-s-target" in failure for failure in report["failures"])
        )
        self.assertTrue(
            any("262K block128 comparison" in failure for failure in report["failures"])
        )
        self.assertTrue(any("2048-token baseline" in failure for failure in report["failures"]))
        self.assertTrue(any("8K+ prompt row" in failure for failure in report["failures"]))
        self.assertTrue(any("feature delta missing" in failure for failure in report["failures"]))

    def test_strict_final_tracks_present_and_missing_tile_sweep_cases(self):
        rows = [
            _row("A_single512_old_chunked", 128, 100.0, 1000.0),
            _row("F_short_text_compact_fused_cold_zero", 128, 60.0, 1600.0),
            _row_with_tile(
                "G_tile_block_sweep",
                2048,
                450.0,
                4500.0,
                q_tile=128,
                kv_tile=512,
                block_size=128,
            ),
        ]

        report = self.acceptance.evaluate(
            rows,
            baseline_variant="A_single512_old_chunked",
            candidate_variant="F_short_text_compact_fused_cold_zero",
            short_latency_speedup=1.5,
            two_k_regression_tolerance=0.0,
            strict_final=True,
            baseline_cold_tok_per_s_target=420.0,
        )

        self.assertFalse(report["passed"])
        self.assertEqual(report["goal_baseline_cold_tok_per_s_target"], 420.0)
        checks = {
            (
                item["prompt_len"],
                item["tile_case"]["kernel_q_tile_size"],
                item["tile_case"]["kernel_kv_tile_size"],
                item["tile_case"]["block_size"],
            ): item["present"]
            for item in report["tile_sweep_case_checks"]
        }
        self.assertTrue(checks[(2048, 128, 512, 128)])
        self.assertFalse(checks[(2048, 128, 1024, 128)])
        self.assertFalse(checks[(8192, 128, 512, 128)])

    def test_strict_final_requires_documented_baseline_target(self):
        rows = [
            _row("A_single512_old_chunked", 128, 100.0, 1000.0),
            _row("F_short_text_compact_fused_cold_zero", 128, 60.0, 1600.0),
        ]

        report = self.acceptance.evaluate(
            rows,
            baseline_variant="A_single512_old_chunked",
            candidate_variant="F_short_text_compact_fused_cold_zero",
            short_latency_speedup=1.5,
            two_k_regression_tolerance=0.0,
            strict_final=True,
            baseline_cold_tok_per_s_target=1000.0,
        )

        self.assertFalse(report["passed"])
        self.assertTrue(
            any("documented ~420 tok/s baseline" in failure for failure in report["failures"])
        )


if __name__ == "__main__":
    unittest.main()
