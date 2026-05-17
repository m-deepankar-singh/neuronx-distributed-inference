# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import importlib.util
import os
import sys
import types
import unittest
from unittest.mock import patch


_CONTRIB_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
_PATCH_PATH = os.path.join(
    _CONTRIB_ROOT,
    "vllm",
    "qwen36_hybrid_apc_scheduler_patch.py",
)
_SCHEDULER_MODULE = "vllm.v1.core.sched.scheduler"


def _load_patch_module():
    spec = importlib.util.spec_from_file_location("qwen36_scheduler_patch", _PATCH_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _scheduler(
    *,
    use_hybrid_apc=True,
    disable_unbacked_prefix_reads=False,
    block_size=2,
    model_revision="rev-a",
):
    hf_config_kwargs = dict(
        use_hybrid_apc_manager=use_hybrid_apc,
        hybrid_apc_disable_unbacked_prefix_reads=disable_unbacked_prefix_reads,
        hybrid_apc_layout_version=1,
        hybrid_recurrent_cache_dtype="float32",
        hybrid_conv_cache_dtype="bfloat16",
        tp_rank=0,
    )
    if model_revision is not None:
        hf_config_kwargs["hybrid_apc_model_revision"] = model_revision
    hf_config = types.SimpleNamespace(**hf_config_kwargs)
    model_config = types.SimpleNamespace(hf_config=hf_config)
    vllm_config = types.SimpleNamespace(model_config=model_config)
    cache_config = types.SimpleNamespace(block_size=block_size)
    return types.SimpleNamespace(vllm_config=vllm_config, cache_config=cache_config)


class TestQwen36HybridAPCSchedulerPatch(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.patch = _load_patch_module()

    def tearDown(self):
        self.patch.clear_hybrid_apc_gdn_checkpoint_registry()
        sys.modules.pop(_SCHEDULER_MODULE, None)
        sys.meta_path = [
            finder
            for finder in sys.meta_path
            if not getattr(finder, "_qwen36_hybrid_apc_import_hook", False)
        ]

    def test_config_flag_disables_prefix_reads_for_hybrid_apc(self):
        scheduler = _scheduler(disable_unbacked_prefix_reads=True)

        self.assertTrue(self.patch.should_disable_unbacked_prefix_reads(scheduler))

    def test_env_flag_disables_prefix_reads_for_hybrid_apc(self):
        scheduler = _scheduler()

        with patch.dict(
            os.environ,
            {"QWEN36_HYBRID_APC_DISABLE_UNBACKED_PREFIX_READS": "1"},
        ):
            self.assertTrue(self.patch.should_disable_unbacked_prefix_reads(scheduler))

    def test_env_flag_wins_when_artifact_config_is_stale(self):
        scheduler = _scheduler(use_hybrid_apc=False)

        with patch.dict(
            os.environ,
            {"QWEN36_HYBRID_APC_DISABLE_UNBACKED_PREFIX_READS": "1"},
        ):
            self.assertTrue(self.patch.should_disable_unbacked_prefix_reads(scheduler))

    def test_non_hybrid_apc_model_is_not_changed(self):
        scheduler = _scheduler(
            use_hybrid_apc=False,
            disable_unbacked_prefix_reads=True,
        )

        self.assertFalse(self.patch.should_disable_unbacked_prefix_reads(scheduler))

    def test_patch_marks_request_skip_reading_prefix_cache(self):
        calls = []

        class FakeScheduler:
            def __init__(self):
                self.vllm_config = _scheduler(
                    disable_unbacked_prefix_reads=True
                ).vllm_config

            def add_request(self, request):
                calls.append(request.skip_reading_prefix_cache)

        installed = self.patch.patch_scheduler_class(FakeScheduler)
        request = types.SimpleNamespace(skip_reading_prefix_cache=False)

        FakeScheduler().add_request(request)

        self.assertTrue(installed)
        self.assertEqual(calls, [True])
        self.assertTrue(request.skip_reading_prefix_cache)

    def test_registered_gdn_checkpoint_allows_prefix_read(self):
        scheduler = _scheduler(block_size=2)
        token_ids = [10, 11, 12, 13, 14]
        hashes = self.patch._local_cumulative_prefix_hashes(
            token_ids,
            block_size=2,
            max_prefix_len=4,
        )
        self.patch.register_hybrid_apc_gdn_checkpoint(
            self.patch.HybridGDNPrefixKey(
                cumulative_prefix_hash=hashes[4],
                prefix_len=4,
                block_size=2,
                cache_salt=None,
                model_revision="rev-a",
                layout_version=1,
                tp_rank=0,
                recurrent_dtype="float32",
                conv_dtype="bfloat16",
            )
        )
        request = types.SimpleNamespace(
            prompt_token_ids=token_ids,
            num_tokens=len(token_ids),
            cache_salt=None,
        )

        with patch.dict(
            os.environ,
            {"QWEN36_HYBRID_APC_DISABLE_UNBACKED_PREFIX_READS": "1"},
        ):
            self.assertEqual(
                self.patch.backed_gdn_prefix_hit_len(scheduler, request),
                4,
            )
            self.assertFalse(
                self.patch.should_disable_unbacked_prefix_reads(scheduler, request)
            )

    def test_mismatched_gdn_checkpoint_keeps_prefix_read_disabled(self):
        scheduler = _scheduler(block_size=2)
        token_ids = [10, 11, 12, 13, 14]
        hashes = self.patch._local_cumulative_prefix_hashes(
            token_ids,
            block_size=2,
            max_prefix_len=4,
        )
        self.patch.register_hybrid_apc_gdn_checkpoint(
            self.patch.HybridGDNPrefixKey(
                cumulative_prefix_hash=hashes[4],
                prefix_len=4,
                block_size=2,
                cache_salt="tenant-a",
                model_revision="rev-a",
                layout_version=1,
                tp_rank=0,
                recurrent_dtype="float32",
                conv_dtype="bfloat16",
            )
        )
        request = types.SimpleNamespace(
            prompt_token_ids=token_ids,
            num_tokens=len(token_ids),
            cache_salt="tenant-b",
        )

        with patch.dict(
            os.environ,
            {"QWEN36_HYBRID_APC_DISABLE_UNBACKED_PREFIX_READS": "1"},
        ):
            self.assertEqual(
                self.patch.backed_gdn_prefix_hit_len(scheduler, request),
                0,
            )
            self.assertTrue(
                self.patch.should_disable_unbacked_prefix_reads(scheduler, request)
            )

    def test_missing_model_revision_defaults_to_unknown(self):
        scheduler = _scheduler(block_size=2, model_revision=None)
        token_ids = [10, 11, 12, 13, 14]
        hashes = self.patch._local_cumulative_prefix_hashes(
            token_ids,
            block_size=2,
            max_prefix_len=4,
        )
        self.patch.register_hybrid_apc_gdn_checkpoint(
            self.patch.HybridGDNPrefixKey(
                cumulative_prefix_hash=hashes[4],
                prefix_len=4,
                block_size=2,
                cache_salt=None,
                model_revision="unknown",
                layout_version=1,
                tp_rank=0,
                recurrent_dtype="float32",
                conv_dtype="bfloat16",
            )
        )
        request = types.SimpleNamespace(
            prompt_token_ids=token_ids,
            num_tokens=len(token_ids),
            cache_salt=None,
        )

        self.assertEqual(
            self.patch.backed_gdn_prefix_hit_len(scheduler, request),
            4,
        )

    def test_import_hook_does_not_import_scheduler_immediately(self):
        installed = self.patch.install_import_hook()

        self.assertFalse(installed)
        self.assertNotIn(_SCHEDULER_MODULE, sys.modules)
        self.assertTrue(
            any(
                getattr(finder, "_qwen36_hybrid_apc_import_hook", False)
                for finder in sys.meta_path
            )
        )

    def test_import_hook_patches_already_loaded_scheduler_module(self):
        calls = []

        class FakeScheduler:
            def __init__(self):
                self.vllm_config = _scheduler(
                    disable_unbacked_prefix_reads=True
                ).vllm_config

            def add_request(self, request):
                calls.append(request.skip_reading_prefix_cache)

        module = types.SimpleNamespace(Scheduler=FakeScheduler)
        sys.modules[_SCHEDULER_MODULE] = module

        installed = self.patch.install_import_hook()
        request = types.SimpleNamespace(skip_reading_prefix_cache=False)
        FakeScheduler().add_request(request)

        self.assertTrue(installed)
        self.assertEqual(calls, [True])
        self.assertTrue(request.skip_reading_prefix_cache)


if __name__ == "__main__":
    unittest.main()
