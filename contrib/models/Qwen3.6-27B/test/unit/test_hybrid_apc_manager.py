# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import sys
import unittest
import importlib.util


_CONTRIB_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _CONTRIB_ROOT not in sys.path:
    sys.path.insert(0, _CONTRIB_ROOT)

_HYBRID_APC_PATH = os.path.join(_CONTRIB_ROOT, "src", "hybrid_apc.py")
_SPEC = importlib.util.spec_from_file_location("qwen36_hybrid_apc", _HYBRID_APC_PATH)
_HYBRID_APC = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _HYBRID_APC
_SPEC.loader.exec_module(_HYBRID_APC)

HybridAPCMetadataStore = _HYBRID_APC.HybridAPCMetadataStore
estimate_qwen_gdn_checkpoint_bytes_per_rank = (
    _HYBRID_APC.estimate_qwen_gdn_checkpoint_bytes_per_rank
)
estimate_qwen_hybrid_cache_bytes_per_rank = (
    _HYBRID_APC.estimate_qwen_hybrid_cache_bytes_per_rank
)


def _store(**overrides):
    defaults = dict(
        required_gdn_layers=[0, 1, 2],
        block_size=128,
        checkpoint_interval=128,
        model_revision="rev-a",
        layout_version=1,
        tp_rank=0,
        recurrent_dtype="float32",
        conv_dtype="bfloat16",
    )
    defaults.update(overrides)
    return HybridAPCMetadataStore(**defaults)


def _insert(store, prefix_len, prefix_hash=None, **overrides):
    key = store.make_key(
        cumulative_prefix_hash=prefix_hash or f"h{prefix_len}",
        prefix_len=prefix_len,
        cache_salt=overrides.pop("cache_salt", "tenant-a"),
        model_revision=overrides.pop("model_revision", "rev-a"),
        layout_version=overrides.pop("layout_version", 1),
        tp_rank=overrides.pop("tp_rank", 0),
        recurrent_dtype=overrides.pop("recurrent_dtype", "float32"),
        conv_dtype=overrides.pop("conv_dtype", "bfloat16"),
    )
    checkpoint = store.insert(
        key=key,
        attention_block_refs=overrides.pop("attention_block_refs", range(prefix_len // 128)),
        gdn_checkpoint_slot=overrides.pop("gdn_checkpoint_slot", prefix_len // 128),
        **overrides,
    )
    return key, checkpoint


class TestHybridAPCMetadataStore(unittest.TestCase):
    def test_same_prefix_hash_and_salt_hits(self):
        store = _store()
        key, _checkpoint = _insert(store, 256)

        plan = store.compute_hit_plan(
            cumulative_hashes_by_prefix_len={128: "h128", 256: "h256"},
            attention_hit_len=256,
            request_prefix_len=300,
            cache_salt="tenant-a",
            model_revision="rev-a",
        )

        self.assertEqual(plan.checkpoint_key, key)
        self.assertEqual(plan.restore_checkpoint_prefix_len, 256)
        self.assertEqual(plan.usable_hit_len, 256)
        self.assertEqual(plan.residual_replay_len, 0)
        self.assertEqual(plan.suffix_len, 44)

    def test_same_tokens_with_different_salt_misses(self):
        store = _store()
        _insert(store, 128, prefix_hash="same")

        plan = store.compute_hit_plan(
            cumulative_hashes_by_prefix_len={128: "same"},
            attention_hit_len=128,
            request_prefix_len=128,
            cache_salt="tenant-b",
            model_revision="rev-a",
        )

        self.assertIsNone(plan.checkpoint_key)
        self.assertEqual(plan.usable_hit_len, 0)

    def test_same_last_block_with_different_cumulative_hash_misses(self):
        store = _store()
        _insert(store, 256, prefix_hash="parent-a+block-z")

        plan = store.compute_hit_plan(
            cumulative_hashes_by_prefix_len={256: "parent-b+block-z"},
            attention_hit_len=256,
            request_prefix_len=256,
            cache_salt="tenant-a",
            model_revision="rev-a",
        )

        self.assertIsNone(plan.checkpoint_key)

    def test_missing_recurrent_layer_invalidates_hit(self):
        store = _store()
        key, _checkpoint = _insert(store, 128)
        store.mark_invalid(key, state_kind="recurrent", layer_id=1)

        plan = store.compute_hit_plan(
            cumulative_hashes_by_prefix_len={128: "h128"},
            attention_hit_len=128,
            request_prefix_len=128,
            cache_salt="tenant-a",
            model_revision="rev-a",
        )

        self.assertIsNone(plan.checkpoint_key)

    def test_missing_conv_layer_invalidates_hit(self):
        store = _store()
        key, _checkpoint = _insert(store, 128)
        store.mark_invalid(key, state_kind="conv", layer_id=2)

        plan = store.compute_hit_plan(
            cumulative_hashes_by_prefix_len={128: "h128"},
            attention_hit_len=128,
            request_prefix_len=128,
            cache_salt="tenant-a",
            model_revision="rev-a",
        )

        self.assertIsNone(plan.checkpoint_key)

    def test_dtype_layout_and_revision_are_identity(self):
        store = _store()
        _insert(store, 128)

        for kwargs in (
            {"recurrent_dtype": "bfloat16"},
            {"conv_dtype": "float32"},
            {"layout_version": 2},
            {"model_revision": "rev-b"},
        ):
            with self.subTest(kwargs=kwargs):
                plan = store.compute_hit_plan(
                    cumulative_hashes_by_prefix_len={128: "h128"},
                    attention_hit_len=128,
                    request_prefix_len=128,
                    cache_salt="tenant-a",
                    **kwargs,
                )
                self.assertIsNone(plan.checkpoint_key)

    def test_refcount_blocks_lru_eviction(self):
        store = _store(max_checkpoints=2)
        key128, _ = _insert(store, 128)
        key256, _ = _insert(store, 256)
        store.inc_ref(key128)
        key384, _ = _insert(store, 384)

        self.assertIsNotNone(store.lookup(key128))
        self.assertIsNone(store.lookup(key256))
        self.assertIsNotNone(store.lookup(key384))

    def test_evicting_gdn_checkpoint_makes_hybrid_hit_fallback(self):
        store = _store()
        _key, checkpoint = _insert(store, 128)
        store.on_gdn_checkpoint_evicted(checkpoint.gdn_checkpoint_slot)

        plan = store.compute_hit_plan(
            cumulative_hashes_by_prefix_len={128: "h128"},
            attention_hit_len=128,
            request_prefix_len=128,
            cache_salt="tenant-a",
            model_revision="rev-a",
        )

        self.assertIsNone(plan.checkpoint_key)

    def test_evicting_attention_block_makes_hybrid_hit_fallback(self):
        store = _store()
        key, _checkpoint = _insert(store, 256, attention_block_refs=(7, 8))
        invalidated = store.on_attention_block_evicted(8)

        self.assertEqual(invalidated, [key])
        plan = store.compute_hit_plan(
            cumulative_hashes_by_prefix_len={256: "h256"},
            attention_hit_len=256,
            request_prefix_len=256,
            cache_salt="tenant-a",
            model_revision="rev-a",
        )
        self.assertIsNone(plan.checkpoint_key)

    def test_non_block_aligned_prompt_uses_checkpoint_boundary_in_v0(self):
        store = _store(allow_residual_replay=False)
        _insert(store, 256)

        plan = store.compute_hit_plan(
            cumulative_hashes_by_prefix_len={256: "h256"},
            attention_hit_len=300,
            request_prefix_len=384,
            cache_salt="tenant-a",
            model_revision="rev-a",
        )

        self.assertEqual(plan.usable_hit_len, 256)
        self.assertEqual(plan.restore_checkpoint_prefix_len, 256)
        self.assertEqual(plan.residual_replay_len, 0)
        self.assertEqual(plan.suffix_len, 128)

    def test_residual_replay_requires_explicit_enablement(self):
        store = _store(allow_residual_replay=True)
        _insert(store, 256)

        plan = store.compute_hit_plan(
            cumulative_hashes_by_prefix_len={256: "h256"},
            attention_hit_len=300,
            request_prefix_len=384,
            cache_salt="tenant-a",
            model_revision="rev-a",
        )

        self.assertEqual(plan.usable_hit_len, 300)
        self.assertEqual(plan.restore_checkpoint_prefix_len, 256)
        self.assertEqual(plan.residual_replay_len, 44)
        self.assertEqual(plan.suffix_len, 84)

    def test_request_lifecycle_releases_restored_ref_on_finish(self):
        store = _store()
        key, _checkpoint = _insert(store, 128)

        record = store.on_request_restore(request_id="req-1", checkpoint_key=key)
        self.assertEqual(record.state, "RESTORED_FROM_HYBRID_APC")
        self.assertEqual(store.lookup(key).refcount, 1)

        store.on_prefill_running("req-1")
        store.on_decode_running("req-1")
        finished = store.on_request_finish("req-1")

        self.assertEqual(finished.state, "FINISHED")
        self.assertEqual(store.lookup(key).refcount, 0)

    def test_request_cancel_releases_ref_and_drops_pending_commit(self):
        store = _store()
        restored_key, _checkpoint = _insert(store, 128)
        committed_key, _checkpoint = _insert(store, 256)

        store.on_request_restore(request_id="req-1", checkpoint_key=restored_key)
        store.on_checkpoint_committed(
            request_id="req-1",
            checkpoint_key=committed_key,
        )
        cancelled = store.on_request_cancel("req-1")

        self.assertEqual(cancelled.state, "CANCELLED")
        self.assertEqual(store.lookup(restored_key).refcount, 0)
        self.assertIsNone(store.lookup(committed_key))

    def test_qwen_hbm_estimator_uses_checkpoint_slots_not_token_slots(self):
        per_checkpoint = estimate_qwen_gdn_checkpoint_bytes_per_rank()
        totals = estimate_qwen_hybrid_cache_bytes_per_rank(
            max_context_len=1024,
            checkpoint_interval=256,
        )

        self.assertEqual(totals["num_gdn_checkpoints"], 4)
        self.assertEqual(totals["gdn_checkpoint_bytes"], per_checkpoint * 4)
        self.assertGreater(totals["gdn_checkpoint_bytes"], totals["attention_kv_bytes"])


if __name__ == "__main__":
    unittest.main()
