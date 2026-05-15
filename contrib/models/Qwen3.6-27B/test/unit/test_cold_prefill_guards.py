# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU-only tests for Qwen3.6 cold-prefill guardrails."""

import os
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch


_SRC_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "src")
)
if _SRC_ROOT not in sys.path:
    sys.path.insert(0, _SRC_ROOT)

from cold_prefill_utils import (  # noqa: E402
    depthwise_causal_conv1d_from_zero,
    depthwise_causal_conv1d_with_state,
    prepare_cte_attention_mask,
    safe_cold_zero_conv_fast_path,
    validate_text_only_cte_vision_inputs,
)


class TestColdZeroConvFastPath(unittest.TestCase):
    def test_fast_conv_matches_stateful_conv_for_zero_prefix(self):
        torch.manual_seed(0)
        mixed = torch.randn(2, 5, 8)
        conv_weight = torch.randn(5, 1, 4)
        conv_state = torch.zeros(2, 5, 3)

        fast = depthwise_causal_conv1d_from_zero(mixed, conv_weight)
        stateful = depthwise_causal_conv1d_with_state(
            mixed,
            conv_weight,
            conv_state,
        )

        self.assertTrue(torch.allclose(fast, stateful, atol=1e-6, rtol=1e-6))

    def test_fast_conv_diverges_when_prefix_state_is_nonzero(self):
        torch.manual_seed(1)
        mixed = torch.randn(1, 4, 6)
        conv_weight = torch.randn(4, 1, 4)
        conv_state = torch.randn(1, 4, 3)

        fast = depthwise_causal_conv1d_from_zero(mixed, conv_weight)
        stateful = depthwise_causal_conv1d_with_state(
            mixed,
            conv_weight,
            conv_state,
        )

        self.assertFalse(torch.allclose(fast, stateful, atol=1e-6, rtol=1e-6))

    def test_guard_requires_position_zero_and_caches(self):
        config = SimpleNamespace(
            use_cold_zero_conv_fast_path=True,
            use_hybrid_apc_manager=False,
        )
        recurrent_cache = torch.zeros(1, 2, 3, 4)
        conv_cache = torch.zeros(1, 5, 3)

        self.assertTrue(
            safe_cold_zero_conv_fast_path(
                config,
                torch.arange(8).unsqueeze(0),
                recurrent_cache,
                conv_cache,
            )
        )
        self.assertFalse(
            safe_cold_zero_conv_fast_path(
                config,
                torch.arange(4, 12).unsqueeze(0),
                recurrent_cache,
                conv_cache,
            )
        )
        self.assertFalse(
            safe_cold_zero_conv_fast_path(
                config,
                torch.arange(8).unsqueeze(0),
                None,
                conv_cache,
            )
        )

    def test_guard_blocks_hybrid_restore_hits(self):
        config = SimpleNamespace(
            use_cold_zero_conv_fast_path=True,
            use_hybrid_apc_manager=True,
        )
        recurrent_cache = torch.zeros(1, 2, 3, 4)
        conv_cache = torch.zeros(1, 5, 3)
        position_ids = torch.arange(8).unsqueeze(0)

        self.assertTrue(
            safe_cold_zero_conv_fast_path(
                config,
                position_ids,
                recurrent_cache,
                conv_cache,
                hybrid_restore_mask=torch.zeros(1, dtype=torch.int32),
            )
        )
        self.assertFalse(
            safe_cold_zero_conv_fast_path(
                config,
                position_ids,
                recurrent_cache,
                conv_cache,
                hybrid_restore_mask=torch.ones(1, dtype=torch.int32),
            )
        )
        self.assertFalse(
            safe_cold_zero_conv_fast_path(
                config,
                position_ids,
                recurrent_cache,
                conv_cache,
                hybrid_restore_mask=torch.zeros(1, dtype=torch.int32),
                hybrid_restore_prefix_lens=torch.ones(1, dtype=torch.int32),
            )
        )

    def test_guard_disables_fast_path_while_tracing(self):
        config = SimpleNamespace(
            use_cold_zero_conv_fast_path=True,
            use_hybrid_apc_manager=False,
        )
        recurrent_cache = torch.zeros(1, 2, 3, 4)
        conv_cache = torch.zeros(1, 5, 3)

        with patch("torch.jit.is_tracing", return_value=True):
            self.assertFalse(
                safe_cold_zero_conv_fast_path(
                    config,
                    torch.arange(8).unsqueeze(0),
                    recurrent_cache,
                    conv_cache,
                )
            )

    def test_guard_keeps_chunk_continuation_stateful_for_long_prompt(self):
        torch.manual_seed(2)
        channels = 4
        state_len = 3
        chunk_len = 1024
        prompt_len = chunk_len * 2
        mixed = torch.randn(1, channels, prompt_len)
        conv_weight = torch.randn(channels, 1, state_len + 1)
        recurrent_cache = torch.zeros(1, 2, 3, 4)
        initial_conv_state = torch.zeros(1, channels, state_len)
        fast_config = SimpleNamespace(
            use_cold_zero_conv_fast_path=True,
            use_hybrid_apc_manager=False,
        )
        stateful_config = SimpleNamespace(
            use_cold_zero_conv_fast_path=False,
            use_hybrid_apc_manager=False,
        )

        def run_chunks(config):
            conv_state = initial_conv_state.clone()
            decisions = []
            outputs = []
            for start in range(0, prompt_len, chunk_len):
                chunk = mixed[:, :, start : start + chunk_len]
                position_ids = torch.arange(start, start + chunk_len).unsqueeze(0)
                use_fast = safe_cold_zero_conv_fast_path(
                    config,
                    position_ids,
                    recurrent_cache,
                    conv_state,
                )
                decisions.append(use_fast)
                if use_fast:
                    outputs.append(depthwise_causal_conv1d_from_zero(chunk, conv_weight))
                    state_source = chunk
                else:
                    conv_input = torch.cat([conv_state, chunk], dim=-1)
                    outputs.append(
                        depthwise_causal_conv1d_with_state(
                            chunk,
                            conv_weight,
                            conv_state,
                            conv_input,
                        )
                    )
                    state_source = conv_input
                conv_state = state_source[:, :, -state_len:].contiguous()
            return decisions, torch.cat(outputs, dim=-1)

        fast_decisions, fast_outputs = run_chunks(fast_config)
        stateful_decisions, stateful_outputs = run_chunks(stateful_config)

        self.assertEqual(fast_decisions, [True, False])
        self.assertEqual(stateful_decisions, [False, False])
        self.assertTrue(
            torch.allclose(fast_outputs, stateful_outputs, atol=1e-6, rtol=1e-6)
        )


class TestCompactCteMaskGuard(unittest.TestCase):
    def test_compact_mask_keeps_long_cte_mask_2d(self):
        attention_mask = torch.ones(1, 4096, dtype=torch.int32)

        prepared = prepare_cte_attention_mask(
            attention_mask,
            is_for_context_encoding=True,
            seq_length=4096,
            use_compact_cte_attention_mask=True,
            use_neuron_cte_attention=False,
        )

        self.assertIs(prepared, attention_mask)
        self.assertEqual(tuple(prepared.shape), (1, 4096))

    def test_dense_long_cte_mask_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "Dense CTE attention masks"):
            prepare_cte_attention_mask(
                torch.ones(1, 4096, dtype=torch.int32),
                is_for_context_encoding=True,
                seq_length=4096,
                use_compact_cte_attention_mask=False,
                use_neuron_cte_attention=False,
            )

    def test_small_dense_fallback_builds_causal_4d_mask(self):
        prepared = prepare_cte_attention_mask(
            torch.tensor([[1, 1, 0, 0]], dtype=torch.int32),
            is_for_context_encoding=True,
            seq_length=4,
            use_compact_cte_attention_mask=False,
            use_neuron_cte_attention=False,
        )

        self.assertEqual(tuple(prepared.shape), (1, 1, 4, 4))
        self.assertEqual(prepared[0, 0, 0].tolist(), [1, 0, 0, 0])
        self.assertEqual(prepared[0, 0, 3].tolist(), [1, 1, 0, 0])


class TestTextOnlyCteVisionInputs(unittest.TestCase):
    def test_text_only_cte_accepts_empty_vision_tensors(self):
        config = SimpleNamespace(use_text_only_cte_inputs=True)

        validate_text_only_cte_vision_inputs(
            config,
            is_for_context_encoding=True,
            vision_embeddings=torch.zeros(0),
            vision_mask=torch.zeros(0, dtype=torch.int32),
        )

    def test_text_only_cte_rejects_dense_dummy_vision_tensors(self):
        config = SimpleNamespace(use_text_only_cte_inputs=True)

        with self.assertRaises(AssertionError):
            validate_text_only_cte_vision_inputs(
                config,
                is_for_context_encoding=True,
                vision_embeddings=torch.zeros(1, 4, 8),
                vision_mask=torch.zeros(1, 4, 1, dtype=torch.int32),
            )


if __name__ == "__main__":
    unittest.main()
