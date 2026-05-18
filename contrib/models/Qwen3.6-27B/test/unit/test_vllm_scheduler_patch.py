# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import importlib.util
import os
import sys
import types
import unittest
from dataclasses import dataclass
from unittest.mock import patch


_CONTRIB_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
_PATCH_PATH = os.path.join(
    _CONTRIB_ROOT,
    "vllm",
    "qwen36_hybrid_apc_scheduler_patch.py",
)
_SCHEDULER_MODULE = "vllm.v1.core.sched.scheduler"
_VLLM_NEURON_RUNNER_MODULE = "vllm_neuron.worker.neuronx_distributed_model_runner"


def _load_patch_module():
    spec = importlib.util.spec_from_file_location("qwen36_scheduler_patch", _PATCH_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _scheduler(
    *,
    use_hybrid_apc=True,
    disable_unbacked_prefix_reads=False,
    enable_backed_prefix_reads=False,
    use_qwen_hybrid_chunked_prefill=False,
    block_size=2,
    model_revision="rev-a",
    additional_config=None,
    max_num_seqs=1,
):
    hf_config_kwargs = dict(
        use_hybrid_apc_manager=use_hybrid_apc,
        hybrid_apc_disable_unbacked_prefix_reads=disable_unbacked_prefix_reads,
        hybrid_apc_enable_backed_prefix_reads=enable_backed_prefix_reads,
        use_qwen_hybrid_chunked_prefill=use_qwen_hybrid_chunked_prefill,
        hybrid_apc_layout_version=1,
        hybrid_recurrent_cache_dtype="float32",
        hybrid_conv_cache_dtype="bfloat16",
        tp_rank=0,
    )
    if model_revision is not None:
        hf_config_kwargs["hybrid_apc_model_revision"] = model_revision
    hf_config = types.SimpleNamespace(**hf_config_kwargs)
    model_config = types.SimpleNamespace(hf_config=hf_config)
    vllm_config = types.SimpleNamespace(
        model_config=model_config,
        additional_config=additional_config or {},
    )
    cache_config = types.SimpleNamespace(block_size=block_size)
    scheduler_config = types.SimpleNamespace(max_num_seqs=max_num_seqs)
    return types.SimpleNamespace(
        vllm_config=vllm_config,
        cache_config=cache_config,
        scheduler_config=scheduler_config,
    )


class TestQwen36HybridAPCSchedulerPatch(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.patch = _load_patch_module()

    def tearDown(self):
        self.patch.clear_hybrid_apc_gdn_checkpoint_registry()
        sys.modules.pop(_SCHEDULER_MODULE, None)
        sys.modules.pop(_VLLM_NEURON_RUNNER_MODULE, None)
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

    def test_registered_gdn_checkpoint_keeps_prefix_read_disabled_without_cte_support(self):
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
            self.assertTrue(
                self.patch.should_disable_unbacked_prefix_reads(scheduler, request)
            )

    def test_registered_gdn_checkpoint_allows_prefix_read_when_cte_supports_it(self):
        scheduler = _scheduler(
            block_size=2,
            enable_backed_prefix_reads=True,
            use_qwen_hybrid_chunked_prefill=True,
        )
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

    def test_additional_config_allows_prefix_read_when_hf_config_is_stale(self):
        scheduler = _scheduler(
            block_size=2,
            enable_backed_prefix_reads=False,
            use_qwen_hybrid_chunked_prefill=False,
            additional_config={
                "use_hybrid_apc_manager": True,
                "hybrid_apc_disable_unbacked_prefix_reads": True,
                "hybrid_apc_enable_backed_prefix_reads": True,
                "use_qwen_hybrid_chunked_prefill": True,
            },
        )
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

        self.assertFalse(
            self.patch.should_disable_unbacked_prefix_reads(scheduler, request)
        )
        authorized = self.patch.pop_hybrid_apc_authorized_prefix_key(
            prefix_len=4,
            cache_salt=None,
            model_revision="rev-a",
            layout_version=1,
            tp_rank=0,
            recurrent_dtype="float32",
            conv_dtype="bfloat16",
        )
        self.assertIsNotNone(authorized)
        self.assertEqual(authorized.cumulative_prefix_hash, hashes[4])

    def test_authorized_prefix_read_can_be_request_scoped(self):
        key = self.patch.HybridGDNPrefixKey(
            cumulative_prefix_hash="hash-a",
            prefix_len=4,
            block_size=2,
            cache_salt=None,
            model_revision="rev-a",
            layout_version=1,
            tp_rank=0,
            recurrent_dtype="float32",
            conv_dtype="bfloat16",
        )

        self.patch.authorize_hybrid_apc_prefix_read(key, request_id="req-a")

        self.assertIsNone(
            self.patch.pop_hybrid_apc_authorized_prefix_key(
                prefix_len=4,
                request_id="req-b",
                cache_salt=None,
                model_revision="rev-a",
                layout_version=1,
                tp_rank=0,
                recurrent_dtype="float32",
                conv_dtype="bfloat16",
            )
        )
        self.assertEqual(
            self.patch.pop_hybrid_apc_authorized_prefix_key(
                prefix_len=4,
                request_id="req-a",
                cache_salt=None,
                model_revision="rev-a",
                layout_version=1,
                tp_rank=0,
                recurrent_dtype="float32",
                conv_dtype="bfloat16",
            ),
            key,
        )

    def test_scheduler_authorizes_backed_prefix_read_by_request_id(self):
        scheduler = _scheduler(
            block_size=2,
            enable_backed_prefix_reads=True,
            use_qwen_hybrid_chunked_prefill=True,
        )
        token_ids = [10, 11, 12, 13, 14]
        hashes = self.patch._local_cumulative_prefix_hashes(
            token_ids,
            block_size=2,
            max_prefix_len=4,
        )
        key = self.patch.HybridGDNPrefixKey(
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
        self.patch.register_hybrid_apc_gdn_checkpoint(key)
        request = types.SimpleNamespace(
            request_id="req-a",
            prompt_token_ids=token_ids,
            num_tokens=len(token_ids),
            cache_salt=None,
        )

        with patch.dict(
            os.environ,
            {"QWEN36_HYBRID_APC_DISABLE_UNBACKED_PREFIX_READS": "1"},
        ):
            self.assertFalse(
                self.patch.should_disable_unbacked_prefix_reads(scheduler, request)
            )

        self.assertIsNone(
            self.patch.pop_hybrid_apc_authorized_prefix_key(
                prefix_len=4,
                request_id="req-b",
                cache_salt=None,
                model_revision="rev-a",
                layout_version=1,
                tp_rank=0,
                recurrent_dtype="float32",
                conv_dtype="bfloat16",
            )
        )
        self.assertEqual(
            self.patch.pop_hybrid_apc_authorized_prefix_key(
                prefix_len=4,
                request_id="req-a",
                cache_salt=None,
                model_revision="rev-a",
                layout_version=1,
                tp_rank=0,
                recurrent_dtype="float32",
                conv_dtype="bfloat16",
            ),
            key,
        )

    def test_backed_prefix_read_allows_batched_scheduler_when_configured(self):
        scheduler = _scheduler(
            block_size=2,
            enable_backed_prefix_reads=True,
            use_qwen_hybrid_chunked_prefill=True,
            max_num_seqs=2,
        )
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
            self.assertFalse(
                self.patch.should_disable_unbacked_prefix_reads(scheduler, request)
            )
        self.assertIsNotNone(
            self.patch.pop_hybrid_apc_authorized_prefix_key(
                prefix_len=4,
                cache_salt=None,
                model_revision="rev-a",
                layout_version=1,
                tp_rank=0,
                recurrent_dtype="float32",
                conv_dtype="bfloat16",
            )
        )

    def test_env_backed_prefix_override_allows_batched_scheduler(self):
        scheduler = _scheduler(
            block_size=2,
            enable_backed_prefix_reads=False,
            use_qwen_hybrid_chunked_prefill=False,
            max_num_seqs=2,
        )
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
            {
                "QWEN36_HYBRID_APC_DISABLE_UNBACKED_PREFIX_READS": "1",
                "QWEN36_HYBRID_APC_ENABLE_BACKED_PREFIX_READS": "1",
            },
        ):
            self.assertFalse(
                self.patch.should_disable_unbacked_prefix_reads(scheduler, request)
            )
        self.assertIsNotNone(
            self.patch.pop_hybrid_apc_authorized_prefix_key(
                prefix_len=4,
                cache_salt=None,
                model_revision="rev-a",
                layout_version=1,
                tp_rank=0,
                recurrent_dtype="float32",
                conv_dtype="bfloat16",
            )
        )

    def test_additional_config_overrides_scheduler_registry_key_metadata(self):
        scheduler = _scheduler(
            block_size=2,
            model_revision="stale-rev",
            additional_config={
                "hybrid_apc_model_revision": "runtime-rev",
                "hybrid_apc_layout_version": 2,
                "tp_rank": 3,
                "hybrid_recurrent_cache_dtype": "bf16",
                "hybrid_conv_cache_dtype": "float32",
            },
        )
        token_ids = [10, 11, 12, 13, 14]
        hashes = self.patch._local_cumulative_prefix_hashes(
            token_ids,
            block_size=2,
            max_prefix_len=4,
        )
        key = self.patch.HybridGDNPrefixKey(
            cumulative_prefix_hash=hashes[4],
            prefix_len=4,
            block_size=2,
            cache_salt=None,
            model_revision="runtime-rev",
            layout_version=2,
            tp_rank=3,
            recurrent_dtype="bfloat16",
            conv_dtype="float32",
        )
        self.patch.register_hybrid_apc_gdn_checkpoint(key)
        request = types.SimpleNamespace(
            prompt_token_ids=token_ids,
            num_tokens=len(token_ids),
            cache_salt=None,
        )

        self.assertEqual(self.patch.backed_gdn_prefix_hit(scheduler, request), key)

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

    def test_runner_patch_exposes_request_ids_during_model_execution(self):
        class FakeRunner:
            def __init__(self):
                self.model = types.SimpleNamespace(model=types.SimpleNamespace())

            def _execute_model_for_text(self, model_input, intermediate_tensors=None):
                self.seen_request_ids = getattr(
                    self.model.model,
                    "_qwen36_vllm_request_ids",
                    None,
                )
                self.seen_cached_request_ids = getattr(
                    self.model.model,
                    "_qwen36_vllm_cached_request_ids",
                    None,
                )
                self.seen_prefill_completion_state = getattr(
                    self.model.model,
                    "_qwen36_vllm_prefill_completion_state",
                    None,
                )
                return self.seen_request_ids

        installed = self.patch.patch_neuron_model_runner_class(FakeRunner)
        runner = FakeRunner()
        result = runner._execute_model_for_text(
            types.SimpleNamespace(
                request_ids=["req-a"],
                _qwen36_cached_request_ids=("req-a",),
                prefill_completion_state="done",
            )
        )

        self.assertTrue(installed)
        self.assertEqual(result, ("req-a",))
        self.assertEqual(runner.seen_request_ids, ("req-a",))
        self.assertEqual(runner.seen_cached_request_ids, ("req-a",))
        self.assertEqual(runner.seen_prefill_completion_state, "done")
        self.assertFalse(hasattr(runner.model, "_qwen36_vllm_request_ids"))
        self.assertFalse(hasattr(runner.model.model, "_qwen36_vllm_request_ids"))
        self.assertFalse(hasattr(runner.model.model, "_qwen36_vllm_cached_request_ids"))

    def test_runner_patch_attaches_scheduler_request_sources_to_model_input(self):
        @dataclass(frozen=True)
        class FrozenModelInput:
            request_ids: list[str]

        class FakeRunner:
            def __init__(self):
                self.model = types.SimpleNamespace(model=types.SimpleNamespace())

            def _prepare_model_input(self, scheduler_output):
                del scheduler_output
                return FrozenModelInput(request_ids=["cached-1", "new-1"])

            def _execute_model_for_text(self, model_input, intermediate_tensors=None):
                del intermediate_tensors
                return model_input

        installed = self.patch.patch_neuron_model_runner_class(FakeRunner)
        runner = FakeRunner()
        scheduler_output = types.SimpleNamespace(
            scheduled_cached_reqs=types.SimpleNamespace(req_ids=["cached-1"]),
            scheduled_new_reqs=[types.SimpleNamespace(req_id="new-1")],
        )
        model_input = runner._prepare_model_input(scheduler_output)

        self.assertTrue(installed)
        self.assertEqual(model_input._qwen36_cached_request_ids, ("cached-1",))
        self.assertEqual(model_input._qwen36_new_request_ids, ("new-1",))

    def test_import_hook_patches_already_loaded_neuron_runner_module(self):
        class FakeRunner:
            def __init__(self):
                self.model = types.SimpleNamespace(model=types.SimpleNamespace())

            def _execute_model_for_text(self, model_input, intermediate_tensors=None):
                return getattr(self.model.model, "_qwen36_vllm_request_ids", None)

        module = types.SimpleNamespace(NeuronxDistributedModelRunner=FakeRunner)
        sys.modules[_VLLM_NEURON_RUNNER_MODULE] = module

        installed = self.patch.install_import_hook()
        runner = FakeRunner()
        result = runner._execute_model_for_text(
            types.SimpleNamespace(request_ids=("req-a",))
        )

        self.assertTrue(installed)
        self.assertEqual(result, ("req-a",))
        self.assertFalse(hasattr(runner.model, "_qwen36_vllm_request_ids"))
        self.assertFalse(hasattr(runner.model.model, "_qwen36_vllm_request_ids"))


if __name__ == "__main__":
    unittest.main()
