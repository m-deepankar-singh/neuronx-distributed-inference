import sys
import types
import importlib
import inspect
from unittest.mock import MagicMock, Mock

import torch


def _stub_lora_import_cycle():
    package_name = "neuronx_distributed_inference.modules.lora_serving"
    module_name = f"{package_name}.lora_module"

    package = types.ModuleType(package_name)
    package.__path__ = []
    lora_module = types.ModuleType(module_name)
    lora_module.is_lora_module = lambda module: False

    sys.modules[package_name] = package
    sys.modules[module_name] = lora_module


def _run_o_proj_contract_checks():
    _stub_lora_import_cycle()

    from neuronx_distributed_inference.modules.attention import gqa

    hidden_size = 16
    head_dim = 256
    num_attention_heads = 8
    tp_degree = 2
    batch_size = 1
    seq_len = 8
    heads_per_core = num_attention_heads // tp_degree
    nd = heads_per_core * head_dim

    calls = []

    class FakeOutputProjection:
        def __getitem__(self, logical_nc_config):
            def call(**kwargs):
                calls.append((logical_nc_config, kwargs))
                return torch.rand((batch_size, seq_len, hidden_size))

            return call

    class FakeGatedOutputProjection:
        def __getitem__(self, logical_nc_config):
            def call(**kwargs):
                calls.append((logical_nc_config, kwargs))
                return torch.rand((batch_size, seq_len, hidden_size))

            return call

    gqa.output_projection_cte = FakeOutputProjection()
    gqa.qwen_gated_output_projection_cte = FakeGatedOutputProjection()
    gqa.reduce_from_tensor_model_parallel_region = lambda x, process_group=None: x

    module = Mock(spec=gqa.GroupQueryAttention_O)
    module.num_attention_heads = num_attention_heads
    module.tp_degree = tp_degree
    module.head_dim = head_dim
    module.logical_nc_config = 2
    module.bias = False
    module.quantized = True
    module.rpl_reduce_dtype = torch.float32
    module.sequence_parallel_enabled = False
    module.tensor_model_parallel_group = None
    module.o_proj = Mock()
    module.o_proj.weight = Mock()
    module.o_proj.weight.shape = (nd, hidden_size)
    module.o_proj.weight.dtype = torch.float8_e4m3fn
    module.o_proj.weight.data = torch.rand((nd, hidden_size)).to(torch.float8_e4m3fn)
    module.o_proj.bias = None
    module.o_proj.scale = Mock()
    weight_scales = torch.rand((128, hidden_size), dtype=torch.float32)
    module.o_proj.scale.data = weight_scales

    attention_output = torch.rand((batch_size, seq_len, nd))
    out = gqa.GroupQueryAttention_O._kernel_o_proj(module, attention_output)
    logical_nc_config, kwargs = calls[-1]

    assert logical_nc_config == 2
    assert kwargs["attention"].shape == (
        batch_size,
        seq_len,
        heads_per_core * 2,
        head_dim // 2,
    )
    assert kwargs["quantization_type"] == gqa.QuantizationType.ROW
    torch.testing.assert_close(kwargs["weight_scales"], weight_scales)
    assert out.shape == (batch_size, seq_len, hidden_size)

    gate = torch.rand((batch_size, seq_len, nd))
    out = gqa.GroupQueryAttention_O._kernel_gated_o_proj(module, attention_output, gate)
    logical_nc_config, kwargs = calls[-1]

    assert logical_nc_config == 2
    assert kwargs["attention"].shape == (
        batch_size,
        seq_len,
        heads_per_core * 2,
        head_dim // 2,
    )
    assert kwargs["gate"].shape == kwargs["attention"].shape
    torch.testing.assert_close(kwargs["weight_scales"], weight_scales)
    assert out.shape == (batch_size, seq_len, hidden_size)

    module.o_proj.scale = None
    try:
        gqa.GroupQueryAttention_O._kernel_o_proj(module, attention_output)
    except RuntimeError as exc:
        assert "requires o_proj.scale" in str(exc)
    else:
        raise AssertionError("quantized output-projection path accepted missing o_proj.scale")


def _run_nkilib_source_check():
    mod = importlib.import_module(
        "nkilib.core.output_projection.output_projection_cte.output_projection_cte"
    )
    source = inspect.getsource(mod.output_projection_cte)
    assert "ROW: attention is [B, S, N, D]" in source, (
        "Imported NKILib output_projection_cte does not contain ROW layout "
        f"support; imported from {mod.__file__}"
    )


def _run_quantized_hook_shape_checks():
    from neuronx_distributed_inference.modules.attention import utils

    utils.quantized_weight_cache.clear()
    utils.quantized_scale_cache.clear()

    tp_degree = 4
    o_prefix = "model.layers.0.self_attn.o_proj."
    o_weight = torch.zeros((16, 32), dtype=torch.float8_e4m3fn)
    o_scale = torch.arange(16, dtype=torch.float32).reshape(16, 1)
    state_dict = {
        o_prefix + "weight": o_weight,
        o_prefix + "scale": o_scale,
    }

    transformed_weight = utils._get_weight_from_state_dict_quantized(
        o_prefix, state_dict, tensor_grp_size=tp_degree
    )
    transformed_scale = utils._get_scale_from_state_dict_quantized(
        o_prefix, state_dict, tensor_grp_size=tp_degree
    )

    assert transformed_weight.shape == (512, 16)
    assert transformed_weight.dtype == torch.float8_e4m3fn
    assert transformed_scale.shape == (128, 16)
    torch.testing.assert_close(transformed_scale[0], o_scale.reshape(16))
    torch.testing.assert_close(transformed_scale[-1], o_scale.reshape(16))

    qkv_prefix = "model.layers.0.self_attn.Wqkv."
    qkv_weight = torch.zeros((24, 16), dtype=torch.float8_e4m3fn)
    qkv_scale = torch.arange(24, dtype=torch.float32).reshape(24, 1)
    state_dict = {
        qkv_prefix + "weight": qkv_weight,
        qkv_prefix + "scale": qkv_scale,
    }
    utils.quantized_weight_cache.clear()
    utils.quantized_scale_cache.clear()

    transformed_weight = utils._get_weight_from_state_dict_quantized(
        qkv_prefix, state_dict, tensor_grp_size=tp_degree
    )
    transformed_scale = utils._get_scale_from_state_dict_quantized(
        qkv_prefix, state_dict, tensor_grp_size=tp_degree
    )

    assert transformed_weight.shape == (16, 512)
    assert transformed_weight.dtype == torch.float8_e4m3fn
    assert transformed_scale.shape == (128, 512)


if __name__ == "__main__":
    _run_nkilib_source_check()
    _run_o_proj_contract_checks()
    _run_quantized_hook_shape_checks()
    print("outproj_nki_contract: PASS")
