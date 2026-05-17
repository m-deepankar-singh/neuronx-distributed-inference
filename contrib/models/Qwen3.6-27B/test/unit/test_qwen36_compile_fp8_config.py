# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import argparse
import importlib.util
import os
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch


_REPO_ROOT = Path(__file__).resolve().parents[5]
_REPO_SRC = _REPO_ROOT / "src"
if str(_REPO_SRC) not in sys.path:
    sys.path.insert(0, str(_REPO_SRC))

_COMPILE_PATH = (
    _REPO_ROOT
    / "contrib"
    / "models"
    / "Qwen3.6-27B"
    / "test"
    / "integration"
    / "qwen36_27b_compile_fp8.py"
)
_SPEC = importlib.util.spec_from_file_location(
    "qwen36_compile_fp8_under_test",
    _COMPILE_PATH,
)
_COMPILE = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _COMPILE
_SPEC.loader.exec_module(_COMPILE)


class _FakeQwen35InferenceConfig:
    def __init__(self, *, neuron_config, **config_dict):
        self.neuron_config = neuron_config
        self.config_dict = config_dict


class _FakeNeuronConfig:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)
        self.output_logits = kwargs.get("output_logits", False)
        self.on_device_sampling_config = kwargs.get("on_device_sampling_config")


class _FakeOnDeviceSamplingConfig:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


class _FakeChunkedPrefillConfig:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


def _fake_config_module():
    module = types.ModuleType("neuronx_distributed_inference.models.config")
    module.NeuronConfig = _FakeNeuronConfig
    module.OnDeviceSamplingConfig = _FakeOnDeviceSamplingConfig
    module.ChunkedPrefillConfig = _FakeChunkedPrefillConfig
    return module


def _fake_qwen_module():
    module = types.ModuleType("src.modeling_qwen35")
    module.Qwen35InferenceConfig = _FakeQwen35InferenceConfig
    return module


def _args(**overrides):
    defaults = dict(
        model_path="/tmp/qwen36",
        quantized_checkpoints_path="/tmp/qwen36-fp8",
        weight_dtype="fp8_mlp_only",
        seq_len=2048,
        cte_bucket=512,
        cte_buckets=["256,512"],
        prefix_buckets=None,
        block_size=256,
        pa_num_blocks=8,
        tp_degree=4,
        logical_nc_config=2,
        enable_prefix_caching=True,
        enable_hybrid_apc=True,
        enable_vllm_chunked_prefill=False,
        disable_on_device_sampling=True,
        kernel_q_tile_size=128,
        kernel_kv_tile_size=1024,
        disable_static_hybrid_cache=False,
        gdn_checkpoint_interval=256,
        max_gdn_checkpoint_slots=8,
        gdn_recurrent_cache_dtype="float32",
        gdn_conv_cache_dtype="bfloat16",
        hybrid_cache_mode="all",
        hybrid_apc_require_vllm_metadata=False,
    )
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


class TestQwen36CompileFp8Config(unittest.TestCase):
    def test_fp8_environment_defaults_are_set_without_overriding_existing(self):
        with patch.dict(os.environ, {}, clear=True):
            _COMPILE._ensure_fp8_environment()
            self.assertEqual(os.environ["XLA_HANDLE_SPECIAL_SCALAR"], "1")
            self.assertEqual(os.environ["UNSAFE_FP8FNCAST"], "1")

        with patch.dict(
            os.environ,
            {
                "XLA_HANDLE_SPECIAL_SCALAR": "custom",
                "UNSAFE_FP8FNCAST": "custom",
            },
            clear=True,
        ):
            _COMPILE._ensure_fp8_environment()
            self.assertEqual(os.environ["XLA_HANDLE_SPECIAL_SCALAR"], "custom")
            self.assertEqual(os.environ["UNSAFE_FP8FNCAST"], "custom")

    def test_host_sampling_compile_keeps_output_logits_enabled(self):
        with patch.object(
            _COMPILE,
            "_load_text_config",
            return_value={"num_hidden_layers": 2},
        ), patch.dict(
            sys.modules,
            {
                "neuronx_distributed_inference.models.config": _fake_config_module(),
                "src.modeling_qwen35": _fake_qwen_module(),
            },
        ):
            config, _modules = _COMPILE._build_config(
                _args(disable_on_device_sampling=True),
            )

        self.assertTrue(config.neuron_config.output_logits)
        self.assertIsNone(config.neuron_config.on_device_sampling_config)
        self.assertEqual(config.neuron_config.pa_num_blocks, 9)
        self.assertTrue(config.neuron_config.quantized)

    def test_on_device_sampling_compile_uses_sampler_config(self):
        with patch.object(
            _COMPILE,
            "_load_text_config",
            return_value={"num_hidden_layers": 2},
        ), patch.dict(
            sys.modules,
            {
                "neuronx_distributed_inference.models.config": _fake_config_module(),
                "src.modeling_qwen35": _fake_qwen_module(),
            },
        ):
            config, _modules = _COMPILE._build_config(
                _args(disable_on_device_sampling=False),
            )

        self.assertIsNotNone(config.neuron_config.on_device_sampling_config)
        self.assertEqual(config.neuron_config.pa_num_blocks, 9)

    def test_bf16_control_compile_disables_quantization_and_keeps_host_logits(self):
        with patch.object(
            _COMPILE,
            "_load_text_config",
            return_value={"num_hidden_layers": 2},
        ), patch.dict(
            sys.modules,
            {
                "neuronx_distributed_inference.models.config": _fake_config_module(),
                "src.modeling_qwen35": _fake_qwen_module(),
            },
        ):
            config, modules = _COMPILE._build_config(
                _args(
                    disable_on_device_sampling=True,
                    weight_dtype="bf16_control",
                    quantized_checkpoints_path=None,
                ),
            )

        self.assertTrue(config.neuron_config.output_logits)
        self.assertFalse(config.neuron_config.quantized)
        self.assertIsNone(config.neuron_config.on_device_sampling_config)
        self.assertGreater(len(modules), 0)

    def test_pa_num_blocks_rejects_user_blocks_below_sequence_requirement(self):
        with self.assertRaisesRegex(ValueError, "need at least 8"):
            _COMPILE._pa_num_blocks(_args(pa_num_blocks=7))


if __name__ == "__main__":
    unittest.main()
