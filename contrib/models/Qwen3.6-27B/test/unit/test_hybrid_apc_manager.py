# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import sys
import unittest
import importlib.util
import types
from unittest.mock import patch

import torch


_CONTRIB_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _CONTRIB_ROOT not in sys.path:
    sys.path.insert(0, _CONTRIB_ROOT)
_VLLM_ROOT = os.path.join(_CONTRIB_ROOT, "vllm")
if _VLLM_ROOT not in sys.path:
    sys.path.insert(0, _VLLM_ROOT)

_HYBRID_APC_PATH = os.path.join(_CONTRIB_ROOT, "src", "hybrid_apc.py")
_SPEC = importlib.util.spec_from_file_location("qwen36_hybrid_apc", _HYBRID_APC_PATH)
_HYBRID_APC = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _HYBRID_APC
_SPEC.loader.exec_module(_HYBRID_APC)

HybridAPCMetadataStore = _HYBRID_APC.HybridAPCMetadataStore
HybridAPCHitPlan = _HYBRID_APC.HybridAPCHitPlan
HybridAPCSchedulerBridge = _HYBRID_APC.HybridAPCSchedulerBridge
HybridAPCSlotAllocator = _HYBRID_APC.HybridAPCSlotAllocator
apply_hybrid_apc_prefill_plan = _HYBRID_APC.apply_hybrid_apc_prefill_plan
build_cumulative_prefix_hashes = _HYBRID_APC.build_cumulative_prefix_hashes
estimate_qwen_gdn_checkpoint_bytes_per_rank = (
    _HYBRID_APC.estimate_qwen_gdn_checkpoint_bytes_per_rank
)
estimate_qwen_hybrid_cache_bytes_per_rank = (
    _HYBRID_APC.estimate_qwen_hybrid_cache_bytes_per_rank
)
import qwen36_hybrid_apc_scheduler_patch as _SCHEDULER_PATCH  # noqa: E402


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


class TestHybridAPCPrefillPlanInputs(unittest.TestCase):
    def test_prefill_plan_materializes_suffix_restore_and_commit(self):
        plan = HybridAPCHitPlan(
            attention_hit_len=2,
            recurrent_hit_len=2,
            conv_hit_len=2,
            usable_hit_len=2,
            restore_checkpoint_prefix_len=2,
            residual_replay_len=0,
            suffix_len=3,
            checkpoint_slot=5,
            checkpoint_key=None,
        )
        input_dict = {
            "input_ids": torch.tensor([[10, 11, 12, 13, 14]], dtype=torch.int32),
            "attention_mask": torch.tensor([[1, 1, 1, 1, 1]], dtype=torch.int32),
            "position_ids": torch.arange(5, dtype=torch.int32).unsqueeze(0),
            "slot_mapping": torch.tensor([[0, 1, 2, 3, 4]], dtype=torch.int32),
        }

        output = apply_hybrid_apc_prefill_plan(
            input_dict,
            plan=plan,
            commit_slot=7,
        )

        self.assertTrue(
            torch.equal(output["input_ids"], torch.tensor([[12, 13, 14]], dtype=torch.int32))
        )
        self.assertTrue(
            torch.equal(output["attention_mask"], torch.tensor([[1, 1, 1]], dtype=torch.int32))
        )
        self.assertTrue(
            torch.equal(output["position_ids"], torch.tensor([[2, 3, 4]], dtype=torch.int32))
        )
        self.assertTrue(
            torch.equal(output["slot_mapping"], torch.tensor([[2, 3, 4]], dtype=torch.int32))
        )
        self.assertTrue(
            torch.equal(output["computed_context_lens"], torch.tensor([[2]], dtype=torch.int32))
        )
        self.assertTrue(
            torch.equal(output["full_context_lens"], torch.tensor([[5]], dtype=torch.int32))
        )
        self.assertTrue(
            torch.equal(output["num_queries"], torch.tensor([[3]], dtype=torch.int32))
        )
        self.assertTrue(
            torch.equal(output["hybrid_restore_slot_ids"], torch.tensor([5], dtype=torch.int32))
        )
        self.assertTrue(
            torch.equal(output["hybrid_restore_mask"], torch.tensor([1], dtype=torch.int32))
        )
        self.assertTrue(
            torch.equal(output["hybrid_restore_prefix_lens"], torch.tensor([2], dtype=torch.int32))
        )
        self.assertTrue(
            torch.equal(output["hybrid_commit_slot_ids"], torch.tensor([7], dtype=torch.int32))
        )
        self.assertTrue(
            torch.equal(output["hybrid_commit_mask"], torch.tensor([1], dtype=torch.int32))
        )

    def test_prefill_plan_does_not_restore_without_checkpoint_slot(self):
        plan = HybridAPCHitPlan(
            attention_hit_len=0,
            recurrent_hit_len=0,
            conv_hit_len=0,
            usable_hit_len=0,
            restore_checkpoint_prefix_len=0,
            residual_replay_len=0,
            suffix_len=3,
            checkpoint_slot=None,
            checkpoint_key=None,
        )
        input_dict = {
            "input_ids": torch.tensor([[10, 11, 12]], dtype=torch.int32),
            "position_ids": torch.arange(3, dtype=torch.int32).unsqueeze(0),
        }

        output = apply_hybrid_apc_prefill_plan(input_dict, plan=plan)

        self.assertTrue(
            torch.equal(output["input_ids"], torch.tensor([[10, 11, 12]], dtype=torch.int32))
        )
        self.assertTrue(
            torch.equal(output["hybrid_restore_slot_ids"], torch.tensor([0], dtype=torch.int32))
        )
        self.assertTrue(
            torch.equal(output["hybrid_restore_mask"], torch.tensor([0], dtype=torch.int32))
        )
        self.assertTrue(
            torch.equal(output["hybrid_commit_mask"], torch.tensor([0], dtype=torch.int32))
        )

    def test_prefill_plan_rejects_restore_boundary_without_checkpoint_slot(self):
        plan = HybridAPCHitPlan(
            attention_hit_len=2,
            recurrent_hit_len=0,
            conv_hit_len=0,
            usable_hit_len=0,
            restore_checkpoint_prefix_len=2,
            residual_replay_len=0,
            suffix_len=1,
            checkpoint_slot=None,
            checkpoint_key=None,
        )

        with self.assertRaisesRegex(ValueError, "requires a checkpoint slot"):
            apply_hybrid_apc_prefill_plan(
                {"input_ids": torch.tensor([[10, 11, 12]], dtype=torch.int32)},
                plan=plan,
            )

    def test_prefill_plan_uses_restore_boundary_for_residual_replay(self):
        plan = HybridAPCHitPlan(
            attention_hit_len=5,
            recurrent_hit_len=4,
            conv_hit_len=4,
            usable_hit_len=5,
            restore_checkpoint_prefix_len=4,
            residual_replay_len=1,
            suffix_len=2,
            checkpoint_slot=9,
            checkpoint_key=None,
        )
        input_dict = {
            "input_ids": torch.tensor([[10, 11, 12, 13, 14, 15, 16]], dtype=torch.int32),
            "position_ids": torch.arange(7, dtype=torch.int32).unsqueeze(0),
        }

        output = apply_hybrid_apc_prefill_plan(
            input_dict,
            plan=plan,
            commit_slot=10,
        )

        self.assertTrue(
            torch.equal(output["input_ids"], torch.tensor([[14, 15, 16]], dtype=torch.int32))
        )
        self.assertTrue(
            torch.equal(output["position_ids"], torch.tensor([[4, 5, 6]], dtype=torch.int32))
        )
        self.assertTrue(
            torch.equal(output["computed_context_lens"], torch.tensor([[4]], dtype=torch.int32))
        )
        self.assertTrue(
            torch.equal(output["num_queries"], torch.tensor([[3]], dtype=torch.int32))
        )

    def test_prefill_plan_slices_flattened_slot_mapping_with_suffix(self):
        plan = HybridAPCHitPlan(
            attention_hit_len=5,
            recurrent_hit_len=4,
            conv_hit_len=4,
            usable_hit_len=5,
            restore_checkpoint_prefix_len=4,
            residual_replay_len=1,
            suffix_len=2,
            checkpoint_slot=9,
            checkpoint_key=None,
        )
        input_dict = {
            "input_ids": torch.tensor([[10, 11, 12, 13, 14, 15, 16]], dtype=torch.int32),
            "position_ids": torch.arange(7, dtype=torch.int32).unsqueeze(0),
            "slot_mapping": torch.arange(100, 107, dtype=torch.int32),
        }

        output = apply_hybrid_apc_prefill_plan(
            input_dict,
            plan=plan,
            commit_slot=10,
        )

        self.assertTrue(
            torch.equal(output["slot_mapping"], torch.tensor([104, 105, 106], dtype=torch.int32))
        )

    def test_prefill_plan_synthesizes_padding_suffix_slots_from_block_table(self):
        plan = HybridAPCHitPlan(
            attention_hit_len=4,
            recurrent_hit_len=4,
            conv_hit_len=4,
            usable_hit_len=4,
            restore_checkpoint_prefix_len=4,
            residual_replay_len=0,
            suffix_len=3,
            checkpoint_slot=9,
            checkpoint_key=None,
        )
        input_dict = {
            "input_ids": torch.tensor([[10, 11, 12, 13, 14, 15, 16]], dtype=torch.int32),
            "position_ids": torch.arange(7, dtype=torch.int32).unsqueeze(0),
            "slot_mapping": torch.full((1, 7), -1, dtype=torch.int32),
            "block_table": torch.tensor([[1, 3, 4, 5]], dtype=torch.int32),
        }

        output = apply_hybrid_apc_prefill_plan(
            input_dict,
            plan=plan,
            commit_slot=10,
            block_size=4,
        )

        self.assertTrue(
            torch.equal(output["slot_mapping"], torch.tensor([[12, 13, 14]], dtype=torch.int32))
        )

    def test_prefill_plan_repairs_negative_active_slots_from_block_table(self):
        plan = HybridAPCHitPlan(
            attention_hit_len=4,
            recurrent_hit_len=0,
            conv_hit_len=0,
            usable_hit_len=0,
            restore_checkpoint_prefix_len=0,
            residual_replay_len=0,
            suffix_len=6,
            checkpoint_slot=None,
            checkpoint_key=None,
        )
        input_dict = {
            "input_ids": torch.tensor([[10, 11, 12, 13, 14, 15]], dtype=torch.int32),
            "position_ids": torch.arange(6, dtype=torch.int32).unsqueeze(0),
            "slot_mapping": torch.tensor([[-1, -1, 10, 11, 12, 13]], dtype=torch.int32),
            "block_table": torch.tensor([[2, 3]], dtype=torch.int32),
        }

        output = apply_hybrid_apc_prefill_plan(
            input_dict,
            plan=plan,
            block_size=4,
        )

        self.assertTrue(
            torch.equal(
                output["slot_mapping"],
                torch.tensor([[8, 9, 10, 11, 12, 13]], dtype=torch.int32),
            )
        )

    def test_prefill_plan_rebuilds_unbacked_attention_hit_slots(self):
        plan = HybridAPCHitPlan(
            attention_hit_len=4,
            recurrent_hit_len=0,
            conv_hit_len=0,
            usable_hit_len=0,
            restore_checkpoint_prefix_len=0,
            residual_replay_len=0,
            suffix_len=6,
            checkpoint_slot=None,
            checkpoint_key=None,
        )
        input_dict = {
            "input_ids": torch.tensor([[10, 11, 12, 13, 14, 15]], dtype=torch.int32),
            "position_ids": torch.arange(6, dtype=torch.int32).unsqueeze(0),
            "slot_mapping": torch.tensor([[12, 13, 14, 15, 16, 17]], dtype=torch.int32),
            "block_table": torch.tensor([[2, 3]], dtype=torch.int32),
        }

        output = apply_hybrid_apc_prefill_plan(
            input_dict,
            plan=plan,
            block_size=4,
        )

        self.assertTrue(
            torch.equal(
                output["slot_mapping"],
                torch.tensor([[8, 9, 10, 11, 12, 13]], dtype=torch.int32),
            )
        )


class TestHybridAPCSchedulerBridge(unittest.TestCase):
    def tearDown(self):
        _SCHEDULER_PATCH.clear_hybrid_apc_gdn_checkpoint_registry()

    def test_slot_allocator_validates_lifecycle(self):
        allocator = HybridAPCSlotAllocator(num_slots=2)

        with self.assertRaisesRegex(ValueError, "outside"):
            allocator.validate_slot_range(2)
        with self.assertRaisesRegex(ValueError, "not reserved"):
            allocator.mark_committed(1)

        slot = allocator.reserve()
        allocator.mark_committed(slot)

        self.assertEqual(allocator.committed_slots, (slot,))

    def test_cumulative_prefix_hash_includes_parent_prefix(self):
        tokens_a = torch.tensor([[1, 2, 3, 4]], dtype=torch.int32)
        tokens_b = torch.tensor([[9, 8, 3, 4]], dtype=torch.int32)

        hashes_a = build_cumulative_prefix_hashes(tokens_a, block_size=2)
        hashes_b = build_cumulative_prefix_hashes(tokens_b, block_size=2)

        self.assertNotEqual(hashes_a[4], hashes_b[4])

    def test_bridge_prepares_warm_suffix_and_commits_checkpoint(self):
        store = _store()
        allocator = HybridAPCSlotAllocator(num_slots=4)
        input_ids = torch.arange(256, dtype=torch.int32).unsqueeze(0)
        hashes = build_cumulative_prefix_hashes(input_ids, block_size=128)
        restored_key, _checkpoint = _insert(
            store,
            128,
            prefix_hash=hashes[128],
            gdn_checkpoint_slot=3,
        )
        bridge = HybridAPCSchedulerBridge(
            store=store,
            slot_allocator=allocator,
            cache_salt="tenant-a",
            model_revision="rev-a",
            reject_unbacked_attention_hits=False,
        )

        prepared = bridge.prepare_request(
            request_id="req-warm",
            input_dict={
                "input_ids": input_ids,
                "attention_mask": torch.ones((1, 256), dtype=torch.int32),
                "position_ids": torch.arange(256, dtype=torch.int32).unsqueeze(0),
            },
            attention_hit_len=128,
            cumulative_hashes_by_prefix_len=hashes,
            attention_block_refs_by_prefix_len={256: (11, 12)},
        )

        self.assertEqual(prepared.plan.restore_checkpoint_prefix_len, 128)
        self.assertEqual(prepared.commit_prefix_len, 256)
        self.assertEqual(prepared.commit_slot, 0)
        self.assertTrue(
            torch.equal(
                prepared.input_dict["input_ids"],
                torch.arange(128, 256, dtype=torch.int32).unsqueeze(0),
            )
        )
        self.assertTrue(
            torch.equal(
                prepared.input_dict["position_ids"],
                torch.arange(128, 256, dtype=torch.int32).unsqueeze(0),
            )
        )
        self.assertTrue(
            torch.equal(
                prepared.input_dict["hybrid_restore_mask"],
                torch.tensor([1], dtype=torch.int32),
            )
        )
        self.assertEqual(store.lookup(restored_key).refcount, 1)

        committed = bridge.commit_prefill(prepared)
        self.assertIsNotNone(committed)
        self.assertEqual(committed.gdn_checkpoint_slot, 0)
        self.assertEqual(committed.attention_block_refs, (11, 12))
        self.assertEqual(allocator.committed_slots, (0,))
        self.assertIsNotNone(store.lookup(prepared.commit_key))

        fake_scheduler = types.SimpleNamespace(
            cache_config=types.SimpleNamespace(block_size=128),
            vllm_config=types.SimpleNamespace(
                model_config=types.SimpleNamespace(
                    hf_config=types.SimpleNamespace(
                        hybrid_apc_model_revision="rev-a",
                        hybrid_apc_layout_version=1,
                        hybrid_recurrent_cache_dtype="float32",
                        hybrid_conv_cache_dtype="bfloat16",
                        tp_rank=0,
                    )
                )
            ),
        )
        fake_request = types.SimpleNamespace(
            prompt_token_ids=list(range(300)),
            num_tokens=300,
            cache_salt="tenant-a",
        )
        self.assertEqual(
            _SCHEDULER_PATCH.backed_gdn_prefix_hit_len(fake_scheduler, fake_request),
            256,
        )

        store.mark_invalid(prepared.commit_key, state_kind="conv")
        self.assertEqual(
            _SCHEDULER_PATCH.backed_gdn_prefix_hit_len(fake_scheduler, fake_request),
            0,
        )

        bridge.finish_request("req-warm")
        self.assertEqual(store.lookup(restored_key).refcount, 0)

    def test_bridge_misses_without_gdn_checkpoint_and_cancels_reserved_slot(self):
        store = _store()
        allocator = HybridAPCSlotAllocator(num_slots=2)
        input_ids = torch.arange(256, dtype=torch.int32).unsqueeze(0)
        bridge = HybridAPCSchedulerBridge(
            store=store,
            slot_allocator=allocator,
            cache_salt="tenant-a",
            model_revision="rev-a",
            reject_unbacked_attention_hits=False,
        )

        prepared = bridge.prepare_request(
            request_id="req-cold",
            input_dict={"input_ids": input_ids},
            attention_hit_len=128,
        )

        self.assertEqual(prepared.plan.restore_checkpoint_prefix_len, 0)
        self.assertEqual(prepared.commit_slot, 0)
        self.assertTrue(
            torch.equal(prepared.input_dict["input_ids"], input_ids)
        )
        self.assertTrue(
            torch.equal(
                prepared.input_dict["hybrid_restore_mask"],
                torch.tensor([0], dtype=torch.int32),
            )
        )

        cancelled = bridge.cancel_request(prepared)
        self.assertEqual(cancelled.state, "CANCELLED")
        self.assertEqual(allocator.reserved_slots, ())
        self.assertEqual(allocator.free_slots, (1, 0))
        self.assertEqual(len(store), 0)

    def test_bridge_rejects_attention_hit_without_gdn_checkpoint_by_default(self):
        bridge = HybridAPCSchedulerBridge(
            store=_store(),
            slot_allocator=HybridAPCSlotAllocator(num_slots=2),
            cache_salt="tenant-a",
            model_revision="rev-a",
        )

        with self.assertRaisesRegex(ValueError, "without a matching GDN checkpoint"):
            bridge.prepare_request(
                request_id="req-unbacked-hit",
                input_dict={
                    "input_ids": torch.arange(256, dtype=torch.int32).unsqueeze(0)
                },
                attention_hit_len=128,
            )

    def test_bridge_env_can_allow_unbacked_attention_fallback(self):
        bridge = HybridAPCSchedulerBridge(
            store=_store(),
            slot_allocator=HybridAPCSlotAllocator(num_slots=2),
            cache_salt="tenant-a",
            model_revision="rev-a",
        )

        with patch.dict(
            os.environ,
            {"QWEN36_ALLOW_UNBACKED_HYBRID_APC_FALLBACK": "1"},
        ):
            prepared = bridge.prepare_request(
                request_id="req-unbacked-fallback",
                input_dict={
                    "input_ids": torch.arange(256, dtype=torch.int32).unsqueeze(0)
                },
                attention_hit_len=128,
            )

        self.assertIsNone(prepared.plan.checkpoint_key)
        self.assertEqual(prepared.input_dict["hybrid_restore_mask"].item(), 0)

    def test_bridge_does_not_commit_mid_prompt_checkpoint_boundary(self):
        store = _store()
        allocator = HybridAPCSlotAllocator(num_slots=2)
        input_ids = torch.arange(192, dtype=torch.int32).unsqueeze(0)
        hashes = build_cumulative_prefix_hashes(input_ids[:, :128], block_size=128)
        bridge = HybridAPCSchedulerBridge(
            store=store,
            slot_allocator=allocator,
            cache_salt="tenant-a",
            model_revision="rev-a",
        )

        prepared = bridge.prepare_request(
            request_id="req-mid-boundary",
            input_dict={"input_ids": input_ids},
            attention_hit_len=0,
            cumulative_hashes_by_prefix_len=hashes,
        )

        self.assertEqual(prepared.commit_prefix_len, 128)
        self.assertIsNone(prepared.commit_slot)
        self.assertEqual(prepared.input_dict["hybrid_commit_mask"].item(), 0)
        self.assertEqual(allocator.reserved_slots, ())
        self.assertEqual(allocator.free_slots, (0, 1))
        self.assertIsNone(bridge.commit_prefill(prepared))

    def test_bridge_can_require_scheduler_prefix_hashes(self):
        bridge = HybridAPCSchedulerBridge(
            store=_store(),
            slot_allocator=HybridAPCSlotAllocator(num_slots=2),
            cache_salt="tenant-a",
            model_revision="rev-a",
            allow_local_hash_fallback=False,
        )

        with self.assertRaisesRegex(ValueError, "requires vLLM cumulative prefix hashes"):
            bridge.prepare_request(
                request_id="req-strict",
                input_dict={"input_ids": torch.arange(128, dtype=torch.int32).unsqueeze(0)},
                attention_hit_len=0,
            )

    def test_bridge_can_require_attention_refs_on_commit(self):
        store = _store()
        input_ids = torch.arange(128, dtype=torch.int32).unsqueeze(0)
        hashes = build_cumulative_prefix_hashes(input_ids, block_size=128)
        bridge = HybridAPCSchedulerBridge(
            store=store,
            slot_allocator=HybridAPCSlotAllocator(num_slots=2),
            cache_salt="tenant-a",
            model_revision="rev-a",
            allow_local_hash_fallback=False,
            require_attention_block_refs=True,
        )

        prepared = bridge.prepare_request(
            request_id="req-refs",
            input_dict={"input_ids": input_ids},
            attention_hit_len=0,
            cumulative_hashes_by_prefix_len=hashes,
        )

        with self.assertRaisesRegex(ValueError, "requires real attention block refs"):
            bridge.commit_prefill(prepared)

        committed = bridge.commit_prefill(prepared, attention_block_refs=(31,))

        self.assertEqual(committed.attention_block_refs, (31,))

    def test_bridge_salt_mismatch_does_not_restore_slot_zero(self):
        store = _store()
        allocator = HybridAPCSlotAllocator(num_slots=2)
        input_ids = torch.arange(128, dtype=torch.int32).unsqueeze(0)
        hashes = build_cumulative_prefix_hashes(input_ids, block_size=128)
        _insert(store, 128, prefix_hash=hashes[128], cache_salt="tenant-a")
        bridge = HybridAPCSchedulerBridge(
            store=store,
            slot_allocator=allocator,
            cache_salt="tenant-b",
            model_revision="rev-a",
            reject_unbacked_attention_hits=False,
        )

        prepared = bridge.prepare_request(
            request_id="req-salt",
            input_dict={"input_ids": input_ids},
            attention_hit_len=128,
            cumulative_hashes_by_prefix_len=hashes,
        )

        self.assertIsNone(prepared.plan.checkpoint_key)
        self.assertEqual(prepared.plan.restore_checkpoint_prefix_len, 0)
        self.assertTrue(
            torch.equal(
                prepared.input_dict["hybrid_restore_slot_ids"],
                torch.tensor([0], dtype=torch.int32),
            )
        )
        self.assertTrue(
            torch.equal(
                prepared.input_dict["hybrid_restore_mask"],
                torch.tensor([0], dtype=torch.int32),
            )
        )

    def test_bridge_env_can_disable_restore_and_commit(self):
        store = _store()
        allocator = HybridAPCSlotAllocator(num_slots=2)
        input_ids = torch.arange(128, dtype=torch.int32).unsqueeze(0)
        hashes = build_cumulative_prefix_hashes(input_ids, block_size=128)
        restored_key, _checkpoint = _insert(
            store,
            128,
            prefix_hash=hashes[128],
            gdn_checkpoint_slot=1,
        )
        bridge = HybridAPCSchedulerBridge(
            store=store,
            slot_allocator=allocator,
            cache_salt="tenant-a",
            model_revision="rev-a",
        )

        with patch.dict(
            os.environ,
            {
                "QWEN36_DISABLE_HYBRID_GDN_RESTORE": "1",
                "QWEN36_DISABLE_HYBRID_GDN_COMMIT": "1",
            },
        ):
            prepared = bridge.prepare_request(
                request_id="req-disabled",
                input_dict={"input_ids": input_ids},
                attention_hit_len=128,
                cumulative_hashes_by_prefix_len=hashes,
            )

        self.assertIsNone(prepared.plan.checkpoint_key)
        self.assertIsNone(prepared.commit_slot)
        self.assertEqual(store.lookup(restored_key).refcount, 0)
        self.assertEqual(allocator.reserved_slots, ())
        self.assertTrue(torch.equal(prepared.input_dict["input_ids"], input_ids))
        self.assertEqual(prepared.input_dict["hybrid_restore_mask"].item(), 0)
        self.assertEqual(prepared.input_dict["hybrid_commit_mask"].item(), 0)
        self.assertTrue(
            torch.equal(
                prepared.input_dict["hybrid_restore_mask"],
                torch.tensor([0], dtype=torch.int32),
            )
        )

    def test_bridge_skips_commit_when_checkpoint_already_exists(self):
        store = _store()
        allocator = HybridAPCSlotAllocator(num_slots=2)
        input_ids = torch.arange(128, dtype=torch.int32).unsqueeze(0)
        hashes = build_cumulative_prefix_hashes(input_ids, block_size=128)
        _insert(store, 128, prefix_hash=hashes[128], gdn_checkpoint_slot=1)
        bridge = HybridAPCSchedulerBridge(
            store=store,
            slot_allocator=allocator,
            cache_salt="tenant-a",
            model_revision="rev-a",
        )

        prepared = bridge.prepare_request(
            request_id="req-existing",
            input_dict={"input_ids": input_ids},
            attention_hit_len=128,
            cumulative_hashes_by_prefix_len=hashes,
        )

        self.assertIsNone(prepared.commit_slot)
        self.assertEqual(prepared.input_dict["hybrid_commit_mask"].item(), 0)
        self.assertEqual(allocator.free_slots, (0, 1))
        self.assertIsNone(bridge.commit_prefill(prepared))

    def test_bridge_finish_releases_uncommitted_reserved_slot(self):
        store = _store()
        allocator = HybridAPCSlotAllocator(num_slots=2)
        input_ids = torch.arange(128, dtype=torch.int32).unsqueeze(0)
        bridge = HybridAPCSchedulerBridge(
            store=store,
            slot_allocator=allocator,
            cache_salt="tenant-a",
            model_revision="rev-a",
        )

        prepared = bridge.prepare_request(
            request_id="req-no-commit",
            input_dict={"input_ids": input_ids},
            attention_hit_len=0,
        )

        self.assertEqual(prepared.commit_slot, 0)
        self.assertEqual(allocator.reserved_slots, (0,))
        finished = bridge.finish_request("req-no-commit")

        self.assertEqual(finished.state, "FINISHED")
        self.assertEqual(allocator.reserved_slots, ())
        self.assertEqual(allocator.free_slots, (1, 0))


if __name__ == "__main__":
    unittest.main()
