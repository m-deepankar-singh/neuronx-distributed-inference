# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import importlib.util
import os
import sys
import types
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch import nn


_REPO_ROOT = Path(__file__).resolve().parents[5]
_QWEN_MODEL_PATH = (
    _REPO_ROOT / "contrib" / "models" / "Qwen3.6-27B" / "src" / "modeling_qwen35.py"
)


def _package(name):
    module = types.ModuleType(name)
    module.__path__ = []
    return module


def _module(name, **attrs):
    module = types.ModuleType(name)
    for key, value in attrs.items():
        setattr(module, key, value)
    return module


def _jit(*_args, **_kwargs):
    def decorator(fn):
        return fn

    return decorator


class _FakeDecoderModelInstance:
    def get(self, bucket_rank, **kwargs):
        del bucket_rank, kwargs
        num_outputs = 1 if not self.neuron_config.output_logits else 2
        kvs = self.module.kv_mgr.past_key_values
        aliases = {param: num_outputs + i for i, param in enumerate(kvs)}
        self.input_output_aliases = aliases
        return self.module, aliases


class _FakeModelWrapper:
    def input_generator(self):
        return self._base_inputs

    def pad_inputs(self, *args, pad_type="first_fit"):
        del pad_type
        return args


def _fake_modules():
    return {
        "nki": _module("nki", jit=_jit),
        "neuronxcc": _package("neuronxcc"),
        "neuronxcc.nki": _package("neuronxcc.nki"),
        "neuronxcc.nki._private_kernels": _package("neuronxcc.nki._private_kernels"),
        "neuronxcc.nki._private_kernels.attention": _module(
            "neuronxcc.nki._private_kernels.attention",
            attention_isa_kernel=lambda *args, **kwargs: None,
        ),
        "neuronx_distributed": _package("neuronx_distributed"),
        "neuronx_distributed.parallel_layers": _package(
            "neuronx_distributed.parallel_layers"
        ),
        "neuronx_distributed.parallel_layers.parallel_state": _module(
            "neuronx_distributed.parallel_layers.parallel_state",
            get_tensor_model_parallel_rank=lambda: 0,
        ),
        "neuronx_distributed.parallel_layers.layers": _module(
            "neuronx_distributed.parallel_layers.layers",
            ColumnParallelLinear=nn.Linear,
            ParallelEmbedding=nn.Embedding,
            RowParallelLinear=nn.Linear,
        ),
        "neuronx_distributed.parallel_layers.mappings": _module(
            "neuronx_distributed.parallel_layers.mappings",
            _gather_along_dim=lambda tensor, *_args, **_kwargs: tensor,
        ),
        "neuronx_distributed.utils": _module(
            "neuronx_distributed.utils",
            cpu_mode=lambda: True,
        ),
        "transformers": _package("transformers"),
        "transformers.models": _package("transformers.models"),
        "transformers.models.qwen3_moe": _package("transformers.models.qwen3_moe"),
        "transformers.models.qwen3_moe.modeling_qwen3_moe": _module(
            "transformers.models.qwen3_moe.modeling_qwen3_moe",
            Qwen3MoeRMSNorm=nn.LayerNorm,
        ),
        "src": _package("src"),
        "src.nki_kernels": _package("src.nki_kernels"),
        "src.nki_kernels.nki_deltanet": _module(
            "src.nki_kernels.nki_deltanet",
            deltanet_recurrent_fwd=lambda *args, **kwargs: None,
            deltanet_recurrent_fwd_state=lambda *args, **kwargs: None,
        ),
        "src.nki_kernels.nki_deltanet_chunked": _module(
            "src.nki_kernels.nki_deltanet_chunked",
            deltanet_chunk_step=lambda *args, **kwargs: None,
        ),
        "src.nki_kernels.nki_deltanet_fused": _module(
            "src.nki_kernels.nki_deltanet_fused",
            deltanet_fused_chunked_fwd=lambda *args, **kwargs: None,
            _make_lower_mask=lambda *args, **kwargs: None,
            _make_lower_mask_diag=lambda *args, **kwargs: None,
            _make_identity=lambda *args, **kwargs: None,
        ),
        "src.hybrid_apc": _module(
            "src.hybrid_apc",
            HybridAPCMetadataStore=object,
            HybridAPCSchedulerBridge=object,
            HybridAPCSlotAllocator=object,
        ),
        "neuronx_distributed_inference": _package("neuronx_distributed_inference"),
        "neuronx_distributed_inference.models": _package(
            "neuronx_distributed_inference.models"
        ),
        "neuronx_distributed_inference.models.config": _module(
            "neuronx_distributed_inference.models.config",
            InferenceConfig=object,
            NeuronConfig=object,
        ),
        "neuronx_distributed_inference.models.model_base": _module(
            "neuronx_distributed_inference.models.model_base",
            NeuronBaseForCausalLM=object,
            NeuronBaseModel=nn.Module,
            mask_padded_logits=lambda logits, *_args, **_kwargs: logits,
        ),
        "neuronx_distributed_inference.models.model_wrapper": _module(
            "neuronx_distributed_inference.models.model_wrapper",
            CONTEXT_ENCODING_MODEL_TAG="context_encoding_model",
            TOKEN_GENERATION_MODEL_TAG="token_generation_model",
            DecoderModelInstance=_FakeDecoderModelInstance,
            ModelWrapper=_FakeModelWrapper,
        ),
        "neuronx_distributed_inference.utils": _package(
            "neuronx_distributed_inference.utils"
        ),
        "neuronx_distributed_inference.utils.distributed": _module(
            "neuronx_distributed_inference.utils.distributed",
            get_tp_group=lambda *_args, **_kwargs: None,
        ),
        "neuronx_distributed_inference.modules": _package(
            "neuronx_distributed_inference.modules"
        ),
        "neuronx_distributed_inference.modules.async_execution": _module(
            "neuronx_distributed_inference.modules.async_execution",
            finish_hybrid_apc_request=lambda *args, **kwargs: None,
            prepare_hybrid_apc_request_for_execution=lambda *args, **kwargs: None,
        ),
        "neuronx_distributed_inference.modules.custom_calls": _module(
            "neuronx_distributed_inference.modules.custom_calls",
            CustomRMSNorm=nn.LayerNorm,
        ),
        "neuronx_distributed_inference.modules.attention": _package(
            "neuronx_distributed_inference.modules.attention"
        ),
        "neuronx_distributed_inference.modules.attention.attention_base": _module(
            "neuronx_distributed_inference.modules.attention.attention_base",
            NeuronAttentionBase=nn.Module,
        ),
        "neuronx_distributed_inference.modules.attention.utils": _module(
            "neuronx_distributed_inference.modules.attention.utils",
            RotaryEmbedding=object,
        ),
        "neuronx_distributed_inference.modules.kvcache": _package(
            "neuronx_distributed_inference.modules.kvcache"
        ),
        "neuronx_distributed_inference.modules.kvcache.block_kv_cache_manager": _module(
            "neuronx_distributed_inference.modules.kvcache.block_kv_cache_manager",
            BlockKVCacheManager=object,
        ),
        "neuronx_distributed_inference.modules.kvcache.kv_cache_manager": _module(
            "neuronx_distributed_inference.modules.kvcache.kv_cache_manager",
            KVCacheManager=object,
        ),
        "neuronx_distributed_inference.models.layer_boundary_marker": _module(
            "neuronx_distributed_inference.models.layer_boundary_marker",
            ModuleMarkerEndWrapper=object,
            ModuleMarkerStartWrapper=object,
        ),
    }


def _load_qwen_module():
    spec = importlib.util.spec_from_file_location(
        "qwen36_model_aliases_under_test",
        _QWEN_MODEL_PATH,
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    with patch.dict(sys.modules, _fake_modules()):
        spec.loader.exec_module(module)
    return module


def _make_instance(qwen_module, *, output_logits, on_device_sampling_config=None):
    kv0, kv1, state, checkpoint = (torch.nn.Parameter(torch.zeros(1)) for _ in range(4))
    module = SimpleNamespace(
        kv_mgr=SimpleNamespace(past_key_values=[kv0, kv1]),
        config=SimpleNamespace(use_hybrid_cache_manager=False),
        _deltanet_state_params=[state],
        _hybrid_gdn_checkpoint_params=[checkpoint],
    )
    instance = qwen_module.Qwen35DecoderModelInstance.__new__(
        qwen_module.Qwen35DecoderModelInstance
    )
    instance.neuron_config = SimpleNamespace(
        output_logits=output_logits,
        on_device_sampling_config=on_device_sampling_config,
    )
    instance.module = module
    return instance, (kv0, kv1, state, checkpoint)


def _make_wrapper(qwen_module, *, tag, use_hybrid_apc_manager=True):
    wrapper = qwen_module.Qwen35ModelWrapper.__new__(qwen_module.Qwen35ModelWrapper)
    wrapper.tag = tag
    wrapper.config = SimpleNamespace(
        hidden_size=8,
        neuron_config=SimpleNamespace(torch_dtype=torch.bfloat16),
        use_text_only_cte_inputs=True,
        use_hybrid_apc_manager=use_hybrid_apc_manager,
    )
    wrapper._base_inputs = [
        (
            torch.ones((1, 1), dtype=torch.int32),  # input_ids
            torch.ones((1, 1), dtype=torch.int32),  # attention_mask
            torch.ones((1, 1), dtype=torch.int32),  # position_ids
            torch.zeros((1,), dtype=torch.int32),  # seq_ids
            torch.ones((1, 3), dtype=torch.float32),  # sampling_params
            torch.empty(0),
            torch.zeros((1,), dtype=torch.int32),  # adapter_ids
            torch.empty(0),
            torch.empty(0),
            torch.empty(0),
            torch.empty(0),
            torch.zeros((1, 1), dtype=torch.int32),  # slot_mapping
            torch.zeros((1, 1), dtype=torch.int32),  # block_table
            torch.ones((1, 1), dtype=torch.int32),  # num_queries
            torch.zeros((1, 1), dtype=torch.int32),  # computed_context_lens
        )
    ]
    return wrapper


class TestQwen36ModelAliases(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.qwen_module = _load_qwen_module()

    def test_host_logits_aliases_after_single_trace_output(self):
        instance, (kv0, kv1, state, checkpoint) = _make_instance(
            self.qwen_module,
            output_logits=True,
            on_device_sampling_config=None,
        )

        _module, aliases = instance.get(bucket_rank=0)

        self.assertEqual(aliases[kv0], 1)
        self.assertEqual(aliases[kv1], 2)
        self.assertEqual(aliases[state], 3)
        self.assertEqual(aliases[checkpoint], 4)

    def test_on_device_logits_aliases_after_tokens_and_logits(self):
        instance, (kv0, kv1, state, checkpoint) = _make_instance(
            self.qwen_module,
            output_logits=True,
            on_device_sampling_config=object(),
        )

        _module, aliases = instance.get(bucket_rank=0)

        self.assertEqual(aliases[kv0], 2)
        self.assertEqual(aliases[kv1], 3)
        self.assertEqual(aliases[state], 4)
        self.assertEqual(aliases[checkpoint], 5)

    def test_gathered_logits_mask_only_actual_vocab_padding(self):
        lm_head = SimpleNamespace(pad_size=248320, gather_output=True)
        config = SimpleNamespace(vocab_size=248320)

        self.assertEqual(
            self.qwen_module._effective_lm_head_pad_size(
                lm_head, torch.empty(1, 1, 248320), config
            ),
            0,
        )
        self.assertEqual(
            self.qwen_module._effective_lm_head_pad_size(
                lm_head, torch.empty(1, 1, 248336), config
            ),
            16,
        )

    def test_sharded_logits_keep_lm_head_pad_size(self):
        lm_head = SimpleNamespace(pad_size=128, gather_output=False)
        config = SimpleNamespace(vocab_size=248320)

        self.assertEqual(
            self.qwen_module._effective_lm_head_pad_size(
                lm_head, torch.empty(1, 1, 62080), config
            ),
            128,
        )

    def test_fused_deltanet_decay_bound_recovers_finite_deltas(self):
        g = torch.tensor(
            [[[[0.0, float("-inf"), float("-inf"), -1.0]]]],
            dtype=torch.float32,
        )

        bounded = self.qwen_module._bound_fused_deltanet_log_decay(
            g,
            batch_size=1,
            num_heads=1,
            total_seq_len=4,
            chunk_size=4,
        )
        bounded_cumsum = bounded.reshape(1, 1, 1, 4).cumsum(dim=-1)

        self.assertTrue(torch.isfinite(bounded).all())
        self.assertTrue(torch.isfinite(bounded_cumsum).all())
        self.assertGreaterEqual(
            float(bounded_cumsum.min().item()),
            self.qwen_module.FUSED_DELTANET_DECAY_MIN,
        )
        self.assertLessEqual(
            float(bounded_cumsum.max().item()),
            self.qwen_module.FUSED_DELTANET_DECAY_MAX,
        )

    def test_legacy_tkg_args_are_env_gated(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertFalse(self.qwen_module._use_legacy_tkg_args())
        with patch.dict(os.environ, {"QWEN36_TKG_LEGACY_ARGS": "1"}, clear=True):
            self.assertTrue(self.qwen_module._use_legacy_tkg_args())

    def test_legacy_tkg_uses_prefix_contract_for_cte_trace_args(self):
        wrapper = _make_wrapper(
            self.qwen_module,
            tag=self.qwen_module.CONTEXT_ENCODING_MODEL_TAG,
        )

        with patch.dict(os.environ, {"QWEN36_TKG_LEGACY_ARGS": "1"}, clear=True):
            generated = wrapper.input_generator()[0]

        self.assertEqual(len(generated), 24)
        self.assertEqual(generated[11].shape, (1, 1))
        self.assertEqual(generated[12].shape, (1, 1))
        self.assertEqual(generated[13].shape, (1, 1))
        self.assertEqual(generated[14].shape, (1, 1))

    def test_legacy_tkg_trace_args_keep_prefix_metadata(self):
        wrapper = _make_wrapper(
            self.qwen_module,
            tag=self.qwen_module.TOKEN_GENERATION_MODEL_TAG,
        )

        with patch.dict(os.environ, {"QWEN36_TKG_LEGACY_ARGS": "1"}, clear=True):
            generated = wrapper.input_generator()[0]

        self.assertEqual(len(generated), 24)
        self.assertEqual(generated[11].shape, (1, 1))
        self.assertEqual(generated[12].shape, (1, 1))
        self.assertEqual(generated[13].shape, (1, 1))
        self.assertEqual(generated[14].shape, (1, 1))

    def test_nonlegacy_tkg_trace_args_keep_prefix_and_hybrid_metadata(self):
        wrapper = _make_wrapper(
            self.qwen_module,
            tag=self.qwen_module.TOKEN_GENERATION_MODEL_TAG,
        )

        with patch.dict(os.environ, {}, clear=True):
            generated = wrapper.input_generator()[0]

        self.assertEqual(len(generated), 29)
        self.assertEqual(generated[11].shape, (1, 1))
        self.assertEqual(generated[14].shape, (1, 1))
        self.assertEqual(generated[24].shape, (1,))

    def test_tkg_token_guard_rejects_out_of_vocab_id(self):
        with self.assertRaisesRegex(ValueError, "out-of-vocab token id"):
            self.qwen_module._validate_qwen36_tkg_input_ids(
                torch.tensor([[2143289344]], dtype=torch.int32),
                248320,
            )

    def test_tkg_token_guard_accepts_valid_vocab_id(self):
        self.qwen_module._validate_qwen36_tkg_input_ids(
            torch.tensor([[42]], dtype=torch.int32),
            248320,
        )

    def test_prefill_detection_keeps_nonzero_multi_token_suffix_on_cte(self):
        self.assertTrue(
            self.qwen_module._qwen36_is_prefill_request(
                torch.ones((1, 207), dtype=torch.int32),
                torch.arange(207, 414, dtype=torch.int32).reshape(1, -1),
            )
        )

    def test_prefill_detection_keeps_one_token_nonzero_decode_on_tkg(self):
        self.assertFalse(
            self.qwen_module._qwen36_is_prefill_request(
                torch.ones((1, 1), dtype=torch.int32),
                torch.tensor([[207]], dtype=torch.int32),
            )
        )

    def test_flattened_slot_mapping_is_normalized_before_batch_chunking(self):
        flattened = torch.arange(256, 719, dtype=torch.int32)

        normalized = self.qwen_module._normalize_qwen36_slot_mapping(
            flattened,
            batch_size=1,
            active_tokens=463,
        )

        self.assertEqual(normalized.shape, (1, 463))
        self.assertTrue(torch.equal(normalized[0], flattened))

    def test_flattened_decode_slot_mapping_is_normalized_by_batch(self):
        flattened = torch.tensor([1488, 1489], dtype=torch.int32)

        normalized = self.qwen_module._normalize_qwen36_slot_mapping(
            flattened,
            batch_size=2,
            active_tokens=1,
        )

        self.assertTrue(
            torch.equal(
                normalized,
                torch.tensor([[1488], [1489]], dtype=torch.int32),
            )
        )

    def test_stage_builders_keep_cte_and_tkg_contracts_explicit(self):
        wrapper = _make_wrapper(
            self.qwen_module,
            tag=self.qwen_module.TOKEN_GENERATION_MODEL_TAG,
        )
        prefix_args = wrapper._base_inputs[0]
        mrope = torch.zeros((0,), dtype=torch.int32)
        vision_embeddings = torch.zeros((0,), dtype=torch.bfloat16)
        vision_mask = torch.zeros((0,), dtype=torch.int32)

        with patch.dict(os.environ, {"QWEN36_TKG_LEGACY_ARGS": "1"}, clear=True):
            cte_args = self.qwen_module.build_cte_args(
                wrapper.config,
                prefix_args,
                mrope,
                vision_embeddings,
                vision_mask,
            )
            tkg_args = self.qwen_module.build_tkg_args(
                wrapper.config,
                prefix_args,
                mrope,
                vision_embeddings,
                vision_mask,
            )

        self.assertEqual(len(cte_args), 24)
        self.assertEqual(len(tkg_args), 24)
        self.assertEqual(cte_args[13].shape, (1, 1))
        self.assertEqual(tkg_args[13].shape, (1, 1))


if __name__ == "__main__":
    unittest.main()
