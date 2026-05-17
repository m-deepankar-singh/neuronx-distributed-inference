# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import argparse
import importlib.util
import json
import os
import subprocess
import unittest
from types import SimpleNamespace


_CONTRIB_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
_RUNNER_PATH = os.path.join(_CONTRIB_ROOT, "vllm", "run_offline_inference.py")
_START_SERVER_PATH = os.path.join(_CONTRIB_ROOT, "vllm", "start_vllm_server.sh")


def _load_runner():
    spec = importlib.util.spec_from_file_location("qwen36_run_offline_inference", _RUNNER_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _args(**overrides):
    defaults = dict(
        cte_bucket=512,
        cte_buckets=None,
        cte_bucket_profile="single",
        seq_len=2048,
        tensor_parallel_size=4,
        max_num_seqs=1,
        ctx_batch_size=1,
        max_model_len=2048,
        compiled_max_prompt_length=None,
        max_tokens=1,
        logical_nc_config=2,
        block_size=128,
        enable_prefix_caching=False,
        enable_hybrid_apc=False,
        enable_vllm_chunked_prefill=True,
        kernel_q_tile_size=128,
        kernel_kv_tile_size=1024,
        hybrid_gdn_recurrent_cache_dtype=None,
        gdn_recurrent_cache_dtype="float32",
        hybrid_gdn_conv_cache_dtype=None,
        gdn_conv_cache_dtype="bfloat16",
        gdn_checkpoint_interval=256,
        max_gdn_checkpoint_slots=8,
        hybrid_cache_mode="all",
        hybrid_cache_prefix_boundary_only=True,
        hybrid_cache_validate_exact=False,
        hybrid_apc_require_vllm_metadata=False,
        text_only_cte=True,
        compact_cte_attention_mask=True,
        cold_zero_conv_fast_path=False,
    )
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


def _launcher_line(stdout: str, name: str) -> str:
    prefix = f"{name}="
    for line in stdout.splitlines():
        if line.startswith(prefix):
            return line[len(prefix) :]
    raise AssertionError(f"{name} not found in launcher output:\n{stdout}")


def _run_launcher_dry_run(*extra_args: str, env: dict[str, str] | None = None):
    command = [
        "bash",
        _START_SERVER_PATH,
        "--model-path",
        "/tmp/model",
        "--dry-run",
        *extra_args,
    ]
    run_env = os.environ.copy()
    if env is not None:
        run_env.update(env)
    return subprocess.run(
        command,
        env=run_env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )


class TestVllmServingConfig(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.runner = _load_runner()

    def test_cte_bucket_list_is_sorted_unique_and_128_aligned(self):
        args = _args(cte_buckets=["512,128", "256", "256"])

        self.assertEqual(self.runner._cte_buckets(args), [128, 256, 512])

    def test_cte_bucket_rejects_non_128_aligned_value(self):
        with self.assertRaisesRegex(ValueError, "128-aligned"):
            self.runner._cte_buckets(_args(cte_buckets=["192"]))

    def test_short_profile_builds_dynamic_bucket_config(self):
        config = self.runner._override_config(_args(cte_bucket_profile="short"))
        neuron_config = config["override_neuron_config"]

        self.assertEqual(neuron_config["context_encoding_buckets"], [128, 256, 512, 1024])
        self.assertEqual(neuron_config["max_context_length"], 1024)
        self.assertTrue(neuron_config["enable_bucketing"])
        self.assertEqual(config["max_prompt_length"], 1024)

    def test_compiled_max_prompt_length_can_exceed_logical_cte_bucket(self):
        config = self.runner._override_config(
            _args(cte_bucket=512, compiled_max_prompt_length=1024)
        )
        neuron_config = config["override_neuron_config"]

        self.assertEqual(neuron_config["context_encoding_buckets"], [512])
        self.assertEqual(neuron_config["max_context_length"], 1024)
        self.assertEqual(config["max_prompt_length"], 1024)

    def test_compiled_max_prompt_length_must_cover_cte_bucket(self):
        with self.assertRaisesRegex(ValueError, "compiled-max-prompt-length"):
            self.runner._override_config(
                _args(cte_bucket_profile="short", compiled_max_prompt_length=512)
            )

    def test_named_cte_profiles_match_cold_prefill_plan(self):
        expected = {
            "short": [128, 256, 512, 1024],
            "general": [256, 512, 1024, 2048],
            "long": [4096, 8192, 16384, 32768],
            "262k": [256],
        }

        for profile, buckets in expected.items():
            with self.subTest(profile=profile):
                self.assertEqual(
                    self.runner._cte_buckets(
                        _args(cte_bucket_profile=profile, seq_len=max(buckets))
                    ),
                    buckets,
                )

    def test_cold_prefill_metrics_report_padding_and_throughput(self):
        args = _args(cte_bucket_profile="short")

        metrics = self.runner._cold_prefill_metrics(
            args,
            actual_prompt_len=384,
            elapsed_seconds=0.25,
            generated_token_count=1,
            hbm_usage={"bytes_used": 123},
        )

        self.assertEqual(metrics["actual_prompt_len"], 384)
        self.assertEqual(metrics["selected_cte_bucket"], 512)
        self.assertEqual(metrics["selected_cte_buckets"], [512])
        self.assertEqual(metrics["num_cte_chunks"], 1)
        self.assertEqual(metrics["bucket_work_tokens"], 512)
        self.assertEqual(metrics["padding_tokens"], 128)
        self.assertEqual(metrics["ctx_batch_size"], 1)
        self.assertEqual(metrics["block_size"], 128)
        self.assertTrue(metrics["text_only_cte_enabled"])
        self.assertTrue(metrics["compact_mask_enabled"])
        self.assertTrue(metrics["chunked_prefill_enabled"])
        self.assertEqual(metrics["cte_attention_mask_path"], "neuron_chunked_prefill")
        self.assertFalse(metrics["dense_cte_mask_fallback"])
        self.assertEqual(metrics["actual_tok_per_s"], 1536)
        self.assertEqual(metrics["bucket_tok_per_s"], 2048)
        self.assertEqual(metrics["request_latency_ms"], 250.0)
        self.assertEqual(metrics["first_token_latency_ms"], 250.0)
        self.assertEqual(metrics["generated_tokens"], 1)
        self.assertEqual(metrics["end_to_end_generated_tok_per_s"], 4.0)
        self.assertIsNone(metrics["decode_tok_per_s"])
        self.assertEqual(metrics["hbm_usage"], {"bytes_used": 123})

    def test_neuron_monitor_hbm_parser_reports_peak_runtime_bytes(self):
        stdout = "\n".join(
            [
                json.dumps(
                    {
                        "neuron_runtime_data": [
                            {
                                "report": {
                                    "memory_used": {
                                        "neuron_runtime_used_bytes": {
                                            "neuron_device": 1024,
                                            "usage_breakdown": {
                                                "neuroncore_memory_usage": {
                                                    "0": {"tensors": 64},
                                                    "1": {"tensors": 128},
                                                }
                                            },
                                        }
                                    }
                                }
                            }
                        ]
                    }
                ),
                json.dumps(
                    {
                        "neuron_runtime_data": [
                            {
                                "report": {
                                    "memory_used": {
                                        "neuron_runtime_used_bytes": {
                                            "neuron_device": 2048,
                                            "usage_breakdown": {
                                                "neuroncore_memory_usage": {
                                                    "0": {"tensors": 256},
                                                }
                                            },
                                        }
                                    }
                                }
                            }
                        ]
                    }
                ),
            ]
        )

        usage = self.runner._parse_neuron_monitor_hbm(stdout)

        self.assertEqual(usage["source"], "neuron-monitor")
        self.assertEqual(usage["bytes_used"], 2048)
        self.assertEqual(usage["neuron_device_bytes_used"], 2048)
        self.assertEqual(usage["tensor_bytes"], 256)
        self.assertEqual(usage["samples"], 2)

    def test_neuron_monitor_hbm_parser_returns_none_without_runtime_samples(self):
        stdout = json.dumps(
            {
                "neuron_runtime_data": [],
                "neuron_hardware_info": {"neuron_device_memory_size": 103079215104},
            }
        )

        self.assertIsNone(self.runner._parse_neuron_monitor_hbm(stdout))

    def test_cold_prefill_metrics_account_for_chunked_long_prompts(self):
        args = _args(cte_bucket_profile="short")

        metrics = self.runner._cold_prefill_metrics(
            args,
            actual_prompt_len=2300,
            elapsed_seconds=1.0,
        )

        self.assertEqual(metrics["selected_cte_buckets"], [1024, 1024, 256])
        self.assertEqual(metrics["selected_cte_bucket"], 256)
        self.assertEqual(metrics["num_cte_chunks"], 3)
        self.assertEqual(metrics["bucket_work_tokens"], 2304)
        self.assertEqual(metrics["padding_tokens"], 4)
        self.assertEqual(metrics["bucket_tok_per_s"], 2304)

    def test_cold_prefill_metrics_reports_small_dense_mask_fallback(self):
        args = _args(
            enable_vllm_chunked_prefill=False,
            compact_cte_attention_mask=False,
        )

        metrics = self.runner._cold_prefill_metrics(
            args,
            actual_prompt_len=512,
            elapsed_seconds=0.5,
        )

        self.assertFalse(metrics["chunked_prefill_enabled"])
        self.assertFalse(metrics["compact_mask_enabled"])
        self.assertEqual(metrics["cte_attention_mask_path"], "dense_4d_fallback")
        self.assertTrue(metrics["dense_cte_mask_fallback"])

    def test_generation_metrics_use_vllm_request_timestamps(self):
        output = SimpleNamespace(
            metrics=SimpleNamespace(
                arrival_time=10.0,
                first_token_time=10.25,
                finished_time=11.0,
            )
        )

        metrics = self.runner._generation_metrics_from_vllm_output(
            output,
            request_start_time=9.5,
            elapsed_seconds=1.2,
            generated_token_count=5,
            max_tokens=32,
        )

        self.assertEqual(metrics["first_token_latency_ms"], 250.0)
        self.assertEqual(metrics["prefill_latency_ms"], 250.0)
        self.assertEqual(metrics["decode_tokens"], 4)
        self.assertEqual(metrics["decode_latency_ms"], 750.0)
        self.assertAlmostEqual(metrics["decode_tok_per_s"], 4 / 0.75)
        self.assertAlmostEqual(metrics["end_to_end_generated_tok_per_s"], 5 / 1.2)
        self.assertEqual(metrics["metrics_source"], "vllm_request_metrics")

    def test_generation_prefill_latency_overwrites_elapsed_request_latency(self):
        metrics = {
            "prefill_latency_ms": 1200.0,
            "request_latency_ms": 1200.0,
            "decode_tok_per_s": None,
        }
        generation_metrics = {
            "prefill_latency_ms": 250.0,
            "first_token_latency_ms": 250.0,
            "decode_tok_per_s": 42.0,
            "request_latency_ms": 1200.0,
        }

        self.runner._merge_generation_metrics(metrics, generation_metrics)

        self.assertEqual(metrics["prefill_latency_ms"], 250.0)
        self.assertEqual(metrics["first_token_latency_ms"], 250.0)
        self.assertEqual(metrics["decode_tok_per_s"], 42.0)
        self.assertEqual(metrics["request_latency_ms"], 1200.0)

    def test_generation_metrics_missing_vllm_timestamps_stay_null_for_decode(self):
        metrics = self.runner._generation_metrics_from_vllm_output(
            SimpleNamespace(metrics=None),
            request_start_time=9.5,
            elapsed_seconds=1.2,
            generated_token_count=5,
            max_tokens=32,
        )

        self.assertIsNone(metrics["first_token_latency_ms"])
        self.assertIsNone(metrics["prefill_latency_ms"])
        self.assertEqual(metrics["decode_tokens"], 4)
        self.assertIsNone(metrics["decode_latency_ms"])
        self.assertIsNone(metrics["decode_tok_per_s"])
        self.assertAlmostEqual(metrics["end_to_end_generated_tok_per_s"], 5 / 1.2)
        self.assertEqual(metrics["metrics_source"], "missing_vllm_request_metrics")

    def test_gdn_kernel_defaults_to_fused_initial_state(self):
        self.assertEqual(
            self.runner._gdn_cte_kernel_from_env({}),
            "fused_initial_state",
        )
        self.assertTrue(self.runner._use_nki_fused_from_env({}))

    def test_effective_fused_metric_follows_selected_gdn_kernel(self):
        self.assertEqual(
            self.runner._gdn_cte_kernel_from_env({"USE_PYTORCH_CHUNK": "1"}),
            "pytorch_chunk",
        )
        self.assertFalse(
            self.runner._use_nki_fused_from_env({"USE_PYTORCH_CHUNK": "1"})
        )
        self.assertFalse(
            self.runner._use_nki_fused_from_env(
                {"USE_NKI_FUSED": "0", "USE_NKI_CHUNKED": "1"}
            )
        )

    def test_text_only_and_compact_mask_flags_are_forwarded(self):
        config = self.runner._override_config(
            _args(
                text_only_cte=False,
                compact_cte_attention_mask=False,
                cold_zero_conv_fast_path=True,
            )
        )

        self.assertFalse(config["use_text_only_cte_inputs"])
        self.assertFalse(config["use_compact_cte_attention_mask"])
        self.assertTrue(config["use_cold_zero_conv_fast_path"])

    def test_hybrid_apc_requires_checkpoint_interval_equal_block_size(self):
        with self.assertRaisesRegex(ValueError, "gdn-checkpoint-interval"):
            self.runner._override_config(
                _args(
                    enable_hybrid_apc=True,
                    enable_prefix_caching=True,
                    block_size=128,
                    gdn_checkpoint_interval=256,
                )
            )

    def test_hybrid_apc_enables_prefix_caching_and_slots(self):
        args = _args(
            enable_hybrid_apc=True,
            enable_prefix_caching=False,
            block_size=256,
            gdn_checkpoint_interval=256,
            max_gdn_checkpoint_slots=3,
        )

        config = self.runner._override_config(args)

        self.assertTrue(args.enable_prefix_caching)
        self.assertTrue(config["use_hybrid_apc_manager"])
        self.assertEqual(config["max_gdn_checkpoint_slots"], 3)

    def test_hybrid_apc_can_require_vllm_metadata(self):
        config = self.runner._override_config(
            _args(
                enable_hybrid_apc=True,
                block_size=256,
                gdn_checkpoint_interval=256,
                hybrid_apc_require_vllm_metadata=True,
            )
        )

        self.assertTrue(config["hybrid_apc_require_vllm_metadata"])
        self.assertFalse(config["hybrid_apc_allow_local_hash_fallback"])
        self.assertTrue(config["hybrid_apc_require_attention_block_refs"])


class TestStartVllmServerDryRun(unittest.TestCase):
    def test_short_profile_dry_run_emits_cold_prefill_config(self):
        proc = _run_launcher_dry_run(
            "--max-model-len",
            "2048",
            "--seq-len",
            "2048",
            "--cte-bucket-profile",
            "short",
            "--enable-vllm-chunked-prefill",
            "--block-size",
            "128",
            "--kernel-q-tile-size",
            "128",
            "--kernel-kv-tile-size",
            "1024",
            "--text-only-cte",
            "--compact-cte-attention-mask",
            "--cold-zero-conv-fast-path",
        )

        self.assertEqual(proc.returncode, 0, proc.stdout)
        cold_prefill_config = json.loads(
            _launcher_line(proc.stdout, "COLD_PREFILL_CONFIG")
        )

        self.assertEqual(cold_prefill_config["cte_buckets"], [128, 256, 512, 1024])
        self.assertEqual(cold_prefill_config["block_size"], 128)
        self.assertEqual(cold_prefill_config["kernel_q_tile_size"], 128)
        self.assertEqual(cold_prefill_config["kernel_kv_tile_size"], 1024)
        self.assertTrue(cold_prefill_config["text_only_cte_enabled"])
        self.assertTrue(cold_prefill_config["compact_mask_enabled"])
        self.assertTrue(cold_prefill_config["cold_zero_conv_fast_path_enabled"])
        self.assertEqual(cold_prefill_config["gdn_cte_kernel"], "fused_initial_state")
        self.assertIn("--enable-chunked-prefill", _launcher_line(proc.stdout, "VLLM_COMMAND"))

    def test_262k_profile_dry_run_emits_recovery_launch_shape(self):
        proc = _run_launcher_dry_run(
            "--max-model-len",
            "262144",
            "--seq-len",
            "262144",
            "--cte-bucket-profile",
            "262k",
            "--enable-vllm-chunked-prefill",
            "--block-size",
            "256",
            "--kernel-q-tile-size",
            "128",
            "--kernel-kv-tile-size",
            "1024",
            "--text-only-cte",
            "--compact-cte-attention-mask",
        )

        self.assertEqual(proc.returncode, 0, proc.stdout)
        cold_prefill_config = json.loads(
            _launcher_line(proc.stdout, "COLD_PREFILL_CONFIG")
        )

        self.assertEqual(cold_prefill_config["cte_buckets"], [256])
        self.assertEqual(cold_prefill_config["block_size"], 256)
        self.assertEqual(cold_prefill_config["max_model_len"], 262144)
        self.assertEqual(cold_prefill_config["seq_len"], 262144)
        self.assertTrue(cold_prefill_config["chunked_prefill_enabled"])

    def test_128k_dry_run_emits_long_context_launch_shape(self):
        proc = _run_launcher_dry_run(
            "--max-model-len",
            "131072",
            "--seq-len",
            "131072",
            "--cte-buckets",
            "256,512,1024,2048",
            "--enable-vllm-chunked-prefill",
            "--block-size",
            "128",
            "--kernel-q-tile-size",
            "128",
            "--kernel-kv-tile-size",
            "1024",
            "--text-only-cte",
            "--compact-cte-attention-mask",
        )

        self.assertEqual(proc.returncode, 0, proc.stdout)
        cold_prefill_config = json.loads(
            _launcher_line(proc.stdout, "COLD_PREFILL_CONFIG")
        )

        self.assertEqual(
            cold_prefill_config["cte_buckets"],
            [256, 512, 1024, 2048],
        )
        self.assertEqual(cold_prefill_config["block_size"], 128)
        self.assertEqual(cold_prefill_config["kernel_q_tile_size"], 128)
        self.assertEqual(cold_prefill_config["kernel_kv_tile_size"], 1024)
        self.assertEqual(cold_prefill_config["max_model_len"], 131072)
        self.assertEqual(cold_prefill_config["seq_len"], 131072)
        self.assertTrue(cold_prefill_config["text_only_cte_enabled"])
        self.assertTrue(cold_prefill_config["compact_mask_enabled"])
        self.assertTrue(cold_prefill_config["chunked_prefill_enabled"])

    def test_dry_run_logs_selected_gdn_kernel_toggle(self):
        proc = _run_launcher_dry_run(
            "--cte-bucket-profile",
            "short",
            "--seq-len",
            "2048",
            env={
                "USE_NKI_FUSED": "0",
                "USE_NKI_CHUNKED": "1",
                "USE_PYTORCH_CHUNK": "0",
            },
        )

        self.assertEqual(proc.returncode, 0, proc.stdout)
        self.assertEqual(_launcher_line(proc.stdout, "GDN_CTE_KERNEL"), "nki_chunked")
        cold_prefill_config = json.loads(
            _launcher_line(proc.stdout, "COLD_PREFILL_CONFIG")
        )
        self.assertFalse(cold_prefill_config["use_nki_fused"])
        self.assertEqual(cold_prefill_config["gdn_cte_kernel"], "nki_chunked")


if __name__ == "__main__":
    unittest.main()
