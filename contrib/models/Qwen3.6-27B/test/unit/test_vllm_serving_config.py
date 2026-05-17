# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import argparse
import importlib.util
import os
import unittest


_CONTRIB_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
_RUNNER_PATH = os.path.join(_CONTRIB_ROOT, "vllm", "run_offline_inference.py")


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
        hybrid_apc_reject_unbacked_attention_hits=True,
        hybrid_apc_disable_unbacked_prefix_reads=False,
        text_only_cte=True,
        compact_cte_attention_mask=True,
        cold_zero_conv_fast_path=False,
    )
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


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
        self.assertTrue(config["hybrid_apc_reject_unbacked_attention_hits"])

    def test_hybrid_apc_can_disable_unbacked_prefix_reads(self):
        config = self.runner._override_config(
            _args(
                enable_hybrid_apc=True,
                block_size=256,
                gdn_checkpoint_interval=256,
                hybrid_apc_disable_unbacked_prefix_reads=True,
            )
        )

        self.assertTrue(config["hybrid_apc_disable_unbacked_prefix_reads"])


if __name__ == "__main__":
    unittest.main()
