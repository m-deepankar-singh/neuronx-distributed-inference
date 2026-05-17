# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU-only tests for Qwen3.6 minimal OpenAI server cold-prefill metrics."""

import importlib.util
import json
import os
import unittest
from pathlib import Path
from unittest.mock import patch


_CONTRIB_ROOT = Path(__file__).resolve().parents[2]
_SERVER_PATH = _CONTRIB_ROOT / "scripts" / "openai_compat_server.py"


def _load_server():
    spec = importlib.util.spec_from_file_location("qwen36_openai_compat_server", _SERVER_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TestOpenAICompatColdPrefillMetrics(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = _load_server()

    def test_metrics_use_launcher_config_and_prefill_latency(self):
        metrics = self.server._cold_prefill_metrics(
            request_id="req-a",
            route="/v1/completions",
            actual_prompt_len=19,
            prefill_elapsed_seconds=0.5,
            config={
                "cte_buckets": [4, 8],
                "chunked_prefill_enabled": True,
                "ctx_batch_size": 1,
                "block_size": 128,
                "kernel_q_tile_size": 128,
                "kernel_kv_tile_size": 1024,
            },
            fallback_chunk_size=8,
        )

        self.assertEqual(metrics["selected_cte_buckets"], [8, 8, 4])
        self.assertEqual(metrics["bucket_work_tokens"], 20)
        self.assertEqual(metrics["padding_tokens"], 1)
        self.assertEqual(metrics["prefill_latency_ms"], 500.0)
        self.assertEqual(metrics["actual_tok_per_s"], 38.0)
        self.assertEqual(metrics["bucket_tok_per_s"], 40.0)
        self.assertEqual(metrics["block_size"], 128)

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

        usage = self.server._parse_neuron_monitor_hbm(stdout)

        self.assertEqual(usage["source"], "neuron-monitor")
        self.assertEqual(usage["bytes_used"], 2048)
        self.assertEqual(usage["neuron_device_bytes_used"], 2048)
        self.assertEqual(usage["tensor_bytes"], 256)
        self.assertEqual(usage["samples"], 2)

    def test_metrics_fall_back_to_chunk_size_when_no_cte_config(self):
        metrics = self.server._cold_prefill_metrics(
            request_id="req-b",
            route="/v1/chat/completions",
            actual_prompt_len=5,
            prefill_elapsed_seconds=1.0,
            config={},
            fallback_chunk_size=8,
        )

        self.assertEqual(metrics["selected_cte_bucket"], 8)
        self.assertEqual(metrics["padding_tokens"], 3)

    def test_load_cold_prefill_config_ignores_invalid_env(self):
        with patch.dict(os.environ, {"QWEN36_COLD_PREFILL_CONFIG": "not-json"}):
            self.assertEqual(self.server._load_cold_prefill_config(), {})

    def test_generation_metrics_report_first_token_and_decode_rate(self):
        metrics = self.server._generation_metrics(
            request_id="req-c",
            route="/v1/chat/completions",
            prompt_tokens=128,
            completion_tokens=5,
            max_tokens=5,
            prefill_elapsed_seconds=0.4,
            request_elapsed_seconds=0.9,
            first_token_id=42,
            finish_reason="length",
        )

        self.assertEqual(metrics["request_id"], "req-c")
        self.assertEqual(metrics["first_token_latency_ms"], 400.0)
        self.assertEqual(metrics["prefill_latency_ms"], 400.0)
        self.assertEqual(metrics["request_latency_ms"], 900.0)
        self.assertEqual(metrics["decode_latency_ms"], 500.0)
        self.assertEqual(metrics["decode_tokens"], 4)
        self.assertAlmostEqual(metrics["decode_tok_per_s"], 8.0)
        self.assertAlmostEqual(metrics["end_to_end_generated_tok_per_s"], 5 / 0.9)


if __name__ == "__main__":
    unittest.main()
