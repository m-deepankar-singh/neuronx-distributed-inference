# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU-only tests for hybrid APC validation metric emission."""

import argparse
import importlib.util
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path


_REPO_ROOT = Path(__file__).resolve().parents[5]
_VALIDATION_PATH = _REPO_ROOT / "validation_scripts" / "qwen36_hybrid_apc_validation.py"


def _load_validation():
    spec = importlib.util.spec_from_file_location(
        "qwen36_hybrid_apc_validation",
        _VALIDATION_PATH,
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _FakeRunner:
    def __init__(self):
        self.hbm_usage = None

    def _prompt_token_count(self, model_path, prompt):
        self.model_path = model_path
        self.prompt = prompt
        return 384

    def _cold_prefill_metrics(
        self,
        runner_args,
        *,
        actual_prompt_len,
        elapsed_seconds,
        generated_token_count=None,
        hbm_usage=None,
    ):
        self.hbm_usage = hbm_usage
        return {
            "actual_prompt_len": actual_prompt_len,
            "elapsed_seconds": elapsed_seconds,
            "generated_tokens": generated_token_count,
            "hbm_usage": hbm_usage,
            "ctx_batch_size": runner_args.ctx_batch_size,
        }

    def _hbm_usage_if_available(self):
        return self.hbm_usage


class TestHybridApcValidationMetrics(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.validation = _load_validation()

    def test_cold_prefill_metrics_passes_hbm_usage_to_runner(self):
        runner = _FakeRunner()
        args = argparse.Namespace(model_path="/tmp/model")
        runner_args = argparse.Namespace(ctx_batch_size=1)
        hbm_usage = {"bytes_used": 123}

        metrics = self.validation._cold_prefill_metrics(
            runner,
            runner_args,
            args,
            "hello world",
            elapsed_seconds=0.25,
            generated_token_count=4,
            hbm_usage=hbm_usage,
        )

        self.assertEqual(metrics["actual_prompt_len"], 384)
        self.assertEqual(metrics["generated_tokens"], 4)
        self.assertEqual(metrics["hbm_usage"], hbm_usage)
        self.assertIs(runner.hbm_usage, hbm_usage)

    def test_emit_cold_prefill_metrics_prints_prefixed_json(self):
        stream = StringIO()

        with redirect_stdout(stream):
            self.validation._emit_cold_prefill_metrics(
                "cold_full",
                {"actual_prompt_len": 128, "hbm_usage": {"bytes_used": 99}},
            )

        line = stream.getvalue().strip()
        self.assertTrue(line.startswith("COLD_PREFILL_METRICS "))
        payload = json.loads(line.split(" ", 1)[1])
        self.assertEqual(payload["request_label"], "cold_full")
        self.assertEqual(payload["actual_prompt_len"], 128)
        self.assertEqual(payload["hbm_usage"], {"bytes_used": 99})

    def test_run_exactness_writes_json_report(self):
        args = argparse.Namespace(
            shared_prefix="shared",
            suffix_a=" A",
            suffix_b=" B",
            max_model_len=2048,
            seq_len=2048,
            cte_bucket=512,
            cte_buckets=None,
            cte_bucket_profile="short",
            tensor_parallel_size=4,
            logical_nc_config=2,
            max_num_seqs=1,
            ctx_batch_size=1,
            block_size=128,
            gdn_checkpoint_interval=128,
            max_gdn_checkpoint_slots=8,
            enable_vllm_chunked_prefill=True,
            kernel_q_tile_size=128,
            kernel_kv_tile_size=1024,
            text_only_cte=True,
            compact_cte_attention_mask=True,
            cold_zero_conv_fast_path=True,
            hybrid_apc_require_vllm_metadata=True,
            max_tokens=32,
            output_json=None,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            args.output_json = Path(tmpdir) / "hybrid_apc.json"
            original_build_llm = self.validation._build_llm
            original_generate = self.validation._generate
            original_cold_prefill_metrics = self.validation._cold_prefill_metrics

            try:
                self.validation._build_llm = lambda args, enable_hybrid_apc: (
                    f"llm-{enable_hybrid_apc}",
                    "sampling",
                    _FakeRunner(),
                    argparse.Namespace(ctx_batch_size=1),
                )
                self.validation._generate = lambda llm, sampling, prompt: {
                    "tokens": [1, 2, 3],
                    "elapsed_seconds": 0.1,
                }
                self.validation._cold_prefill_metrics = (
                    lambda runner, runner_args, args, prompt, elapsed_seconds, generated_token_count=None, hbm_usage=None: {
                        "actual_prompt_len": 128,
                        "generated_tokens": generated_token_count,
                    }
                )

                with redirect_stdout(StringIO()):
                    returncode = self.validation.run_exactness(args)
            finally:
                self.validation._build_llm = original_build_llm
                self.validation._generate = original_generate
                self.validation._cold_prefill_metrics = original_cold_prefill_metrics

            report = json.loads(args.output_json.read_text(encoding="utf-8"))

        self.assertEqual(returncode, 0)
        self.assertTrue(report["full_prefix_exact"])
        self.assertTrue(report["partial_prefix_exact"])
        self.assertEqual(
            report["cold_prefill_metrics"]["cold_full"]["actual_prompt_len"],
            128,
        )
        self.assertTrue(report["validation_config"]["cold_zero_conv_fast_path"])
        self.assertTrue(report["validation_config"]["hybrid_apc_require_vllm_metadata"])
        self.assertEqual(report["validation_config"]["max_num_seqs"], 1)


if __name__ == "__main__":
    unittest.main()
