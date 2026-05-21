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
        max_context_length=None,
        cte_bucket=512,
        cte_buckets=["256,512"],
        prefix_buckets=None,
        block_size=256,
        pa_num_blocks=8,
        pa_headroom_blocks=0,
        tp_degree=4,
        logical_nc_config=2,
        max_num_seqs=1,
        ctx_batch_size=1,
        skip_warmup=False,
        enable_prefix_caching=True,
        enable_hybrid_apc=True,
        enable_vllm_chunked_prefill=False,
        deltanet_cte_backend="env",
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
        hybrid_apc_enable_backed_prefix_reads=False,
        quantize_edge_mlp_layers=False,
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
        self.assertEqual(config.neuron_config.pa_num_blocks, 8)
        self.assertTrue(config.neuron_config.quantized)

    def test_compile_can_trace_batched_token_generation(self):
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
                _args(
                    disable_on_device_sampling=True,
                    weight_dtype="bf16_control",
                    quantized_checkpoints_path=None,
                    max_num_seqs=2,
                    ctx_batch_size=1,
                    skip_warmup=True,
                    pa_num_blocks=16,
                ),
            )

        self.assertEqual(config.neuron_config.batch_size, 2)
        self.assertEqual(config.neuron_config.ctx_batch_size, 1)
        self.assertEqual(config.neuron_config.tkg_batch_size, 2)
        self.assertEqual(config.neuron_config.pa_num_blocks, 16)
        self.assertTrue(config.neuron_config.skip_warmup)

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
        self.assertEqual(config.neuron_config.pa_num_blocks, 8)

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

    def test_fp8_mlp_only_keeps_edge_mlp_layers_in_bf16_by_default(self):
        with patch.object(
            _COMPILE,
            "_load_text_config",
            return_value={"num_hidden_layers": 4},
        ), patch.dict(
            sys.modules,
            {
                "neuronx_distributed_inference.models.config": _fake_config_module(),
                "src.modeling_qwen35": _fake_qwen_module(),
            },
        ):
            _config, modules = _COMPILE._build_config(
                _args(disable_on_device_sampling=True),
            )

        self.assertIn("layers.0.mlp", modules)
        self.assertIn("layers.3.mlp", modules)
        self.assertNotIn("layers.1.mlp", modules)

    def test_long_prefix_buckets_must_fit_max_context_length(self):
        with self.assertRaisesRegex(ValueError, "Largest prefix bucket"):
            _COMPILE._validate_prefix_buckets_fit_context(
                _args(enable_prefix_caching=True),
                max_context_length=512,
                prefix_buckets=[512, 131072],
            )

    def test_pa_num_blocks_rejects_user_blocks_below_sequence_requirement(self):
        with self.assertRaisesRegex(ValueError, "need at least 8"):
            _COMPILE._pa_num_blocks(_args(pa_num_blocks=7))

    def test_pa_headroom_blocks_extend_default_pa_capacity(self):
        args = _args(
            seq_len=4096,
            block_size=32,
            max_num_seqs=2,
            pa_num_blocks=None,
            pa_headroom_blocks=32,
        )

        self.assertEqual(_COMPILE._pa_min_blocks(args), 256)
        self.assertEqual(_COMPILE._pa_requested_blocks(args), 288)
        self.assertEqual(_COMPILE._pa_num_blocks(args), 288)

    def test_base_compile_work_dir_defaults_next_to_artifacts(self):
        with self.subTest("default"), patch.dict(os.environ, {}, clear=True):
            work_dir = _COMPILE._configure_base_compile_work_dir(
                Path("/tmp/qwen_artifacts/model_a"),
                None,
            )

            self.assertEqual(
                work_dir,
                Path("/tmp/qwen_artifacts/_nxd_model_workdir").resolve(),
            )
            self.assertEqual(os.environ["BASE_COMPILE_WORK_DIR"], str(work_dir))

        with self.subTest("existing env"), patch.dict(
            os.environ,
            {"BASE_COMPILE_WORK_DIR": "/tmp/existing_nxd_workdir"},
            clear=True,
        ):
            work_dir = _COMPILE._configure_base_compile_work_dir(
                Path("/tmp/qwen_artifacts/model_a"),
                None,
            )

            self.assertEqual(work_dir, Path("/tmp/existing_nxd_workdir").resolve())
            self.assertEqual(os.environ["BASE_COMPILE_WORK_DIR"], str(work_dir))

        with self.subTest("explicit override"), patch.dict(
            os.environ,
            {"BASE_COMPILE_WORK_DIR": "/tmp/existing_nxd_workdir"},
            clear=True,
        ):
            work_dir = _COMPILE._configure_base_compile_work_dir(
                Path("/tmp/qwen_artifacts/model_a"),
                "/tmp/explicit_nxd_workdir",
            )

            self.assertEqual(work_dir, Path("/tmp/explicit_nxd_workdir").resolve())
            self.assertEqual(os.environ["BASE_COMPILE_WORK_DIR"], str(work_dir))

    def test_deltanet_cte_backend_preserves_environment_by_default(self):
        with patch.dict(
            os.environ,
            {
                "USE_NKI_FUSED": "custom",
                "USE_NKI_CHUNKED": "custom",
            },
            clear=True,
        ):
            _COMPILE._configure_deltanet_cte_backend("env")

            self.assertEqual(os.environ["USE_NKI_FUSED"], "custom")
            self.assertEqual(os.environ["USE_NKI_CHUNKED"], "custom")

    def test_deltanet_cte_backend_can_force_nki_chunked(self):
        with patch.dict(
            os.environ,
            {
                "USE_NKI_FUSED": "1",
                "USE_PYTORCH_CHUNK": "1",
                "DELTANET_SEQUENTIAL": "1",
            },
            clear=True,
        ):
            _COMPILE._configure_deltanet_cte_backend("nki_chunked")

            self.assertEqual(os.environ["USE_NKI_FUSED"], "0")
            self.assertEqual(os.environ["USE_NKI_CHUNKED"], "1")
            self.assertNotIn("USE_PYTORCH_CHUNK", os.environ)
            self.assertNotIn("DELTANET_SEQUENTIAL", os.environ)

    def test_deltanet_cte_backend_can_force_pytorch_chunk(self):
        with patch.dict(os.environ, {"USE_NKI_CHUNKED": "1"}, clear=True):
            _COMPILE._configure_deltanet_cte_backend("pytorch_chunk")

            self.assertEqual(os.environ["USE_NKI_FUSED"], "0")
            self.assertEqual(os.environ["USE_PYTORCH_CHUNK"], "1")
            self.assertNotIn("USE_NKI_CHUNKED", os.environ)

    def test_backed_prefix_read_compile_flag_is_forwarded(self):
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
                _args(hybrid_apc_enable_backed_prefix_reads=True),
            )

        self.assertTrue(config.config_dict["hybrid_apc_enable_backed_prefix_reads"])

    def test_vllm_chunked_prefill_uses_qwen_flags_not_nxdi_chunked_prefill(self):
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
                _args(enable_vllm_chunked_prefill=True),
            )

        self.assertTrue(config.neuron_config.is_block_kv_layout)
        self.assertIsNone(getattr(config.neuron_config, "chunked_prefill_config", None))
        self.assertTrue(config.config_dict["use_qwen_hybrid_chunked_prefill"])
        self.assertTrue(config.config_dict["use_qwen_hybrid_chunked_prefill_nki"])


if __name__ == "__main__":
    unittest.main()
