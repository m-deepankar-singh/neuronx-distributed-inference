# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import importlib.util
import os
import types
import unittest
from unittest.mock import patch


_CONTRIB_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
_PATCH_PATH = os.path.join(
    _CONTRIB_ROOT,
    "vllm",
    "qwen36_hybrid_apc_scheduler_patch.py",
)


def _load_patch_module():
    spec = importlib.util.spec_from_file_location("qwen36_scheduler_patch", _PATCH_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _scheduler(*, use_hybrid_apc=True, disable_unbacked_prefix_reads=False):
    hf_config = types.SimpleNamespace(
        use_hybrid_apc_manager=use_hybrid_apc,
        hybrid_apc_disable_unbacked_prefix_reads=disable_unbacked_prefix_reads,
    )
    model_config = types.SimpleNamespace(hf_config=hf_config)
    vllm_config = types.SimpleNamespace(model_config=model_config)
    return types.SimpleNamespace(vllm_config=vllm_config)


class TestQwen36HybridAPCSchedulerPatch(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.patch = _load_patch_module()

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


if __name__ == "__main__":
    unittest.main()
