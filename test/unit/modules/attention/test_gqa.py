from unittest.mock import MagicMock, Mock, call, patch

import pytest
import torch

from neuronx_distributed_inference.modules.attention import gqa


def _bind_qkv_kernel_helpers(qkv_proj):
    for method_name in (
        "_expected_local_fused_qkv_size",
        "_validate_qkv_kernel_layout",
        "_qkv_kernel_quantization_args",
    ):
        setattr(
            qkv_proj,
            method_name,
            getattr(gqa.GroupQueryAttention_QKV, method_name).__get__(
                qkv_proj, gqa.GroupQueryAttention_QKV
            ),
        )


@pytest.mark.parametrize(
    "batch_size, seq_len, fuse_rope",
    # fmt: off
    [
        (1, 8, True),    # bs=1, context encoding, fuse rope enabled
        (2, 8, True),    # bs=2, context encoding, fuse rope enabled
        (1, 8, False),   # bs=1, context encoding, fuse rope disabled
        (2, 8, False),   # bs=2, context encoding, fuse rope disabled
        (1, 1, False),   # bs=1, token gen, fuse rope disabled
        (2, 1, False),   # bs=2, token gen, fuse rope disabled
    ],
    # fmt: on
)
@patch('neuronx_distributed_inference.modules.attention.gqa.qkv_kernel')
def test_kernel_qkv_forward_rope_fusion(mock_qkv_kernel, batch_size, seq_len, fuse_rope):
    """Test that qkv_kernel is called with correct arguments when rope fusion is enabled."""
    
    # Test parameters
    hidden_size = 16
    head_dim = 4
    num_attention_heads = 8
    num_key_value_heads = 2
    tp_degree = 2
    
    # Prepare inputs
    hidden_states = torch.rand((batch_size, seq_len, hidden_size))
    cos_cache = torch.rand((batch_size, seq_len, head_dim)) if fuse_rope else None
    sin_cache = torch.rand((batch_size, seq_len, head_dim)) if fuse_rope else None
    
    # Mock qkv kernel
    fused_qkv_size = (num_attention_heads + 2 * num_key_value_heads) * head_dim // tp_degree
    QKV = torch.rand((batch_size, seq_len, fused_qkv_size))
    
    mock_kernel_call = MagicMock(return_value=QKV)
    mock_qkv_kernel.__getitem__ = MagicMock(return_value=mock_kernel_call)
    
    # Create a mock GroupQueryAttention_QKV instance
    qkv_proj = Mock(spec=gqa.GroupQueryAttention_QKV)
    qkv_proj.num_attention_heads = num_attention_heads
    qkv_proj.num_key_value_heads = num_key_value_heads
    qkv_proj.tp_degree = tp_degree
    qkv_proj.head_dim = head_dim
    qkv_proj.fused_rmsnorm = False
    qkv_proj.fused_rmsnorm_skip_gamma = False
    qkv_proj.logical_nc_config = 1
    qkv_proj.bias = False
    qkv_proj.seq_len_threshold_for_cc_tiling = 16834
    qkv_proj.tiling_factor = 1
    qkv_proj.qkv_kernel_nbsd_layout = False
    qkv_proj.qkv_nki_kernel_enabled = True
    qkv_proj.rms_norm_eps = 1e-6
    _bind_qkv_kernel_helpers(qkv_proj)
    
    # Create a mock weight with correct shape (transposed for qkv_nki_kernel_enabled=True)
    qkv_proj.Wqkv = Mock()
    qkv_proj.Wqkv.weight = Mock()
    qkv_proj.Wqkv.weight.shape = (hidden_size, fused_qkv_size)
    qkv_proj.Wqkv.weight.dtype = torch.float32
    qkv_proj.Wqkv.bias = None
    
    # Mock _split_fused_qkv to return Q, K, V
    Q = torch.rand((batch_size, seq_len, num_attention_heads * head_dim // tp_degree))
    K = torch.rand((batch_size, seq_len, num_key_value_heads * head_dim // tp_degree))
    V = torch.rand((batch_size, seq_len, num_key_value_heads * head_dim // tp_degree))
    qkv_proj._split_fused_qkv = Mock(return_value=(Q, K, V))
    
    # Call the real _kernel_qkv_forward method with our mock instance
    result = gqa.GroupQueryAttention_QKV._kernel_qkv_forward(
        qkv_proj, hidden_states, None, None, cos_cache, sin_cache
    )
    
    # Verify the kernel was called
    mock_qkv_kernel.__getitem__.assert_called_once_with(qkv_proj.logical_nc_config)
    mock_kernel_call.assert_called_once()
    
    # Check the kernel arguments
    kernel_kwargs = mock_kernel_call.call_args.kwargs
    assert kernel_kwargs["quantization_type"] == gqa.QuantizationType.NONE
    assert kernel_kwargs["qkv_w_scale"] is None
    assert kernel_kwargs["qkv_in_scale"] is None
    
    if fuse_rope:
        # When rope fusion is enabled, cos_cache and sin_cache should be passed
        torch.testing.assert_close(kernel_kwargs["cos_cache"], cos_cache)
        torch.testing.assert_close(kernel_kwargs["sin_cache"], sin_cache)
        assert kernel_kwargs["num_q_heads"] == num_attention_heads // tp_degree
        assert kernel_kwargs["num_kv_heads"] == num_key_value_heads // tp_degree
    else:
        # When rope fusion is disabled, cos_cache and sin_cache should NOT be passed
        assert kernel_kwargs["cos_cache"] is None
        assert kernel_kwargs["sin_cache"] is None
    
    # Verify result is a tuple with Q, K, V, residual
    assert len(result) == 4
    Q, K, V, residual = result
    assert Q.shape == (batch_size, seq_len, num_attention_heads * head_dim // tp_degree)
    assert K.shape == (batch_size, seq_len, num_key_value_heads * head_dim // tp_degree)
    assert V.shape == (batch_size, seq_len, num_key_value_heads * head_dim // tp_degree)
    assert residual is None


def test_fused_qkv_nki_installs_quantized_layout_hook(monkeypatch):
    class FakeProcessGroup:
        def size(self):
            return 4

    class FakeColumnParallelLinear(torch.nn.Module):
        def __init__(
            self,
            input_size,
            output_size,
            bias,
            gather_output,
            dtype,
            sequence_parallel_enabled,
            tensor_model_parallel_group,
            rank_ordering=None,
        ):
            super().__init__()
            del gather_output, sequence_parallel_enabled, rank_ordering
            self.input_size = input_size
            self.output_size = output_size
            self.output_size_per_partition = output_size // tensor_model_parallel_group.size()
            self.tensor_parallel_group = tensor_model_parallel_group
            self.weight = torch.nn.Parameter(
                torch.empty((self.output_size_per_partition, input_size), dtype=dtype),
                requires_grad=False,
            )
            setattr(self.weight, "partition_dim", 0)
            self.bias = None if not bias else torch.nn.Parameter(
                torch.empty(self.output_size_per_partition, dtype=dtype),
                requires_grad=False,
            )

    monkeypatch.setattr(gqa, "ColumnParallelLinear", FakeColumnParallelLinear)

    qkv_proj = gqa.GroupQueryAttention_QKV(
        hidden_size=5120,
        head_dim=256,
        num_attention_heads=24,
        num_key_value_heads=4,
        tp_degree=4,
        dtype=torch.float32,
        bias=False,
        gather_output=False,
        fused_qkv=True,
        tensor_model_parallel_group=FakeProcessGroup(),
        qkv_nki_kernel_enabled=True,
    )

    assert tuple(qkv_proj.Wqkv.weight.shape) == (5120, 2048)
    assert qkv_proj.Wqkv.weight.partition_dim == 1
    assert (
        qkv_proj.Wqkv.post_create_quantized_module_hook
        is gqa.preprocess_quantized_qkv_nki_layer
    )


def test_qkv_nki_state_dict_loaders_transpose_weight_and_broadcast_scale():
    prefix = "layers.0.self_attn.Wqkv."
    gqa._qkv_nki_weight_cache.clear()
    gqa._qkv_nki_scale_cache.clear()

    weight = torch.arange(24, dtype=torch.float32).reshape(6, 4)
    scale = torch.arange(1, 7, dtype=torch.float32).reshape(6, 1)
    state_dict = {
        prefix + "weight": weight,
        prefix + "scale": scale,
    }

    loaded_weight = gqa._get_qkv_nki_weight_from_state_dict(prefix, state_dict)
    loaded_scale = gqa._get_qkv_nki_scale_from_state_dict(prefix, state_dict)

    torch.testing.assert_close(loaded_weight, weight.t().contiguous())
    assert tuple(loaded_scale.shape) == (128, 6)
    torch.testing.assert_close(loaded_scale[0], scale.squeeze(1))
    torch.testing.assert_close(loaded_scale[127], scale.squeeze(1))


@patch('neuronx_distributed_inference.modules.attention.gqa.qkv_kernel')
def test_quantized_qkv_forward_passes_row_scale(mock_qkv_kernel, monkeypatch):
    class FakeQuantizedParallelLinear:
        pass

    monkeypatch.setattr(gqa, "BaseQuantizeParallelLinear", FakeQuantizedParallelLinear)

    batch_size = 1
    seq_len = 8
    hidden_size = 5120
    head_dim = 256
    num_attention_heads = 24
    num_key_value_heads = 4
    tp_degree = 4
    fused_qkv_size = (num_attention_heads + 2 * num_key_value_heads) * head_dim // tp_degree

    hidden_states = torch.rand((batch_size, seq_len, hidden_size))
    QKV = torch.rand((batch_size, seq_len, fused_qkv_size))
    mock_kernel_call = MagicMock(return_value=QKV)
    mock_qkv_kernel.__getitem__ = MagicMock(return_value=mock_kernel_call)

    qkv_proj = Mock(spec=gqa.GroupQueryAttention_QKV)
    qkv_proj.num_attention_heads = num_attention_heads
    qkv_proj.num_key_value_heads = num_key_value_heads
    qkv_proj.tp_degree = tp_degree
    qkv_proj.head_dim = head_dim
    qkv_proj.fused_rmsnorm = False
    qkv_proj.fused_rmsnorm_skip_gamma = False
    qkv_proj.logical_nc_config = 1
    qkv_proj.bias = False
    qkv_proj.qkv_kernel_nbsd_layout = False
    qkv_proj.rms_norm_eps = 1e-6
    _bind_qkv_kernel_helpers(qkv_proj)

    wqkv = FakeQuantizedParallelLinear()
    wqkv.weight = Mock()
    wqkv.weight.shape = (hidden_size, fused_qkv_size)
    wqkv.weight.data = torch.empty((hidden_size, fused_qkv_size))
    wqkv.scale = torch.ones((128, fused_qkv_size), dtype=torch.float32)
    wqkv.bias = None
    qkv_proj.Wqkv = wqkv
    qkv_proj._split_fused_qkv = Mock(
        return_value=(
            torch.rand((batch_size, seq_len, num_attention_heads * head_dim // tp_degree)),
            torch.rand((batch_size, seq_len, num_key_value_heads * head_dim // tp_degree)),
            torch.rand((batch_size, seq_len, num_key_value_heads * head_dim // tp_degree)),
        )
    )

    gqa.GroupQueryAttention_QKV._kernel_qkv_forward(
        qkv_proj, hidden_states, None, None, None, None
    )

    kernel_kwargs = mock_kernel_call.call_args.kwargs
    assert kernel_kwargs["quantization_type"] == gqa.QuantizationType.ROW
    torch.testing.assert_close(kernel_kwargs["qkv_w_scale"], wqkv.scale)
    assert kernel_kwargs["qkv_in_scale"] is None


def test_kernel_qkv_forward_rejects_untransposed_weight():
    qkv_proj = Mock(spec=gqa.GroupQueryAttention_QKV)
    qkv_proj.num_attention_heads = 24
    qkv_proj.num_key_value_heads = 4
    qkv_proj.tp_degree = 4
    qkv_proj.head_dim = 256
    _bind_qkv_kernel_helpers(qkv_proj)

    hidden_size = 5120
    fused_qkv_size = (24 + 2 * 4) * 256 // 4
    qkv_proj.Wqkv = Mock()
    qkv_proj.Wqkv.weight = Mock()
    qkv_proj.Wqkv.weight.shape = (fused_qkv_size, hidden_size)

    with pytest.raises(RuntimeError, match="transposed"):
        qkv_proj._validate_qkv_kernel_layout(hidden_size)


def test_quantized_qkv_forward_rejects_untransposed_scale(monkeypatch):
    class FakeQuantizedParallelLinear:
        pass

    monkeypatch.setattr(gqa, "BaseQuantizeParallelLinear", FakeQuantizedParallelLinear)

    qkv_proj = Mock(spec=gqa.GroupQueryAttention_QKV)
    qkv_proj.num_attention_heads = 24
    qkv_proj.num_key_value_heads = 4
    qkv_proj.tp_degree = 4
    qkv_proj.head_dim = 256
    _bind_qkv_kernel_helpers(qkv_proj)

    hidden_size = 5120
    fused_qkv_size = (24 + 2 * 4) * 256 // 4
    wqkv = FakeQuantizedParallelLinear()
    wqkv.weight = Mock()
    wqkv.weight.shape = (hidden_size, fused_qkv_size)
    wqkv.scale = torch.ones((fused_qkv_size, 1), dtype=torch.float32)
    qkv_proj.Wqkv = wqkv

    with pytest.raises(RuntimeError, match="row-scale layout"):
        qkv_proj._qkv_kernel_quantization_args(fused_qkv_size)
