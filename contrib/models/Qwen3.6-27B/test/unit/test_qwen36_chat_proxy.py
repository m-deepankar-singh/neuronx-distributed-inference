# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU-only tests for the Qwen3.6 guarded chat proxy helpers."""

import importlib.util
import unittest
from pathlib import Path


_CONTRIB_ROOT = Path(__file__).resolve().parents[2]
_PROXY_PATH = _CONTRIB_ROOT / "vllm" / "qwen36_chat_proxy.py"


def _load_proxy():
    spec = importlib.util.spec_from_file_location("qwen36_chat_proxy", _PROXY_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TestQwen36ChatProxy(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.proxy = _load_proxy()

    def test_cold_prefill_request_metrics_use_launcher_config(self):
        metrics = self.proxy._cold_prefill_request_metrics(
            request_id="req-a",
            route="/v1/chat/completions",
            prompt_text="one two three four five",
            tokenizer=None,
            config={
                "cte_buckets": [4, 8],
                "ctx_batch_size": 1,
                "block_size": 128,
                "kernel_q_tile_size": 128,
                "kernel_kv_tile_size": 1024,
                "text_only_cte_enabled": True,
                "compact_mask_enabled": True,
                "chunked_prefill_enabled": True,
                "cold_zero_conv_fast_path_enabled": False,
                "use_nki_fused": True,
                "gdn_cte_kernel": "fused_initial_state",
                "max_model_len": 2048,
                "seq_len": 2048,
            },
        )

        self.assertEqual(metrics["request_id"], "req-a")
        self.assertEqual(metrics["actual_prompt_len"], 5)
        self.assertEqual(metrics["selected_cte_bucket"], 8)
        self.assertEqual(metrics["bucket_work_tokens"], 8)
        self.assertEqual(metrics["padding_tokens"], 3)
        self.assertEqual(metrics["actual_prompt_len_source"], "whitespace")
        self.assertTrue(metrics["chunked_prefill_enabled"])

    def test_cold_prefill_request_metrics_chunk_long_prompts(self):
        metrics = self.proxy._cold_prefill_request_metrics(
            request_id="req-b",
            route="/v1/completions",
            prompt_text=" ".join(str(i) for i in range(19)),
            tokenizer=None,
            config={"cte_buckets": [4, 8], "chunked_prefill_enabled": True},
        )

        self.assertEqual(metrics["selected_cte_buckets"], [8, 8, 4])
        self.assertEqual(metrics["num_cte_chunks"], 3)
        self.assertEqual(metrics["bucket_work_tokens"], 20)
        self.assertEqual(metrics["padding_tokens"], 1)

    def test_normalize_messages_preserves_non_system_order(self):
        messages = [
            {"role": "developer", "content": "dev"},
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi"},
        ]

        normalized = self.proxy._normalize_messages_for_qwen(messages)

        self.assertEqual(normalized[0], {"role": "system", "content": "dev"})
        self.assertEqual(normalized[1:], messages[1:])


if __name__ == "__main__":
    unittest.main()
