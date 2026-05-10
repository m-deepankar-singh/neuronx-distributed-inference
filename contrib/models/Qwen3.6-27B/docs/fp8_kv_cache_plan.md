# Qwen3.6-27B FP8 KV Cache Plan

## Scope

This is a prep/design document only. It does not enable FP8 KV cache.

Baseline target:

- model: `contrib/models/Qwen3.6-27B`
- serving baseline: `qwen36-27b-vllm-apc-baseline-v3`
- artifact: `/opt/dlami/nvme/qwen_artifacts/qwen36_27b_128k_fp8_mlp_only_vllm_statereset_run1`
- current cache manager: `HybridDeltaNetCacheManager`

## Found API

NxDI exposes two related KV quantization paths.

### Simple config flag

`NeuronConfig.kv_cache_quant` is parsed in
`src/neuronx_distributed_inference/models/config.py`:

```text
self.kv_cache_quant = kwargs.pop("kv_cache_quant", False)
```

The public docs describe this as the simple user-facing switch. When enabled,
NxDI quantizes the KV cache to `torch.float8_e4m3fn` and dequantizes before
use. The docs also require:

```text
XLA_HANDLE_SPECIAL_SCALAR=1
```

`src/neuronx_distributed_inference/inference_demo.py` additionally sets:

```text
XLA_HANDLE_SPECIAL_SCALAR=1
UNSAFE_FP8FNCAST=1
```

when `kv_cache_quant` is enabled.

### Explicit `KVQuantizationConfig`

`NeuronConfig.kv_quant_config` is parsed in
`src/neuronx_distributed_inference/models/config.py`:

```text
self.kv_quant_config = kwargs.pop("kv_quant_config", None)
if type(self.kv_quant_config) is dict:
    self.kv_quant_config = KVQuantizationConfig(**self.kv_quant_config)
```

`inference_demo.py` constructs this object from:

```text
k_quant_method
v_quant_method
kv_quant_dtype
kv_direct_cast
```

The migration docs say NxDI uses FP8 for KV quantization and direct cast.

## Current NxDI Cache Behavior

`KVCacheManager` stores cache tensors using `self.cache_dtype`. If
`kv_quant_config` is present:

```text
self.cache_dtype = self.kv_quant_config.quant_dtype
```

On reads:

```text
k_cache = self._dequantize_cache(k_cache, idx, is_key=True)
v_cache = self._dequantize_cache(v_cache, idx, is_key=False)
```

On writes:

```text
latest_k = self._quantize_cache(latest_k, idx, is_key=True)
latest_v = self._quantize_cache(latest_v, idx, is_key=False)
```

`BlockKVCacheManager` follows the same pattern for block-layout KV.

## Current Qwen3.6 Blocker

`HybridDeltaNetCacheManager._validate_hybrid_config()` explicitly rejects KV
quantization:

```text
if getattr(nc, "kv_quant_config", None) is not None or getattr(nc, "kv_cache_quant", False):
    unsupported.append("KV cache quantization")
```

This was correct for initial bring-up. For FP8 KV, remove this rejection only
after attention-layer-only quantization is implemented and tested.

## Hybrid Compatibility Design

Qwen3.6 cache entries are layer-type dependent:

- GQA/full-attention layers: standard `(k_cache, v_cache)`
- GDN layers: `(recurrent_state, conv_state)`

FP8 KV must apply **only** to full-attention K/V tensors. GDN state must not be
quantized by the KV path.

Recommended state dtypes:

- attention K/V cache: FP8 E4M3 through `kv_quant_config` or equivalent direct
  cast path
- GDN recurrent state: FP32 for numerical safety
- GDN conv state: BF16 is acceptable unless validation shows drift

Implementation approach:

1. Keep `HybridDeltaNetCacheManager` layer-aware allocation.
2. For full-attention layers, allocate K/V with `cache_dtype`, as it already
   does through `cache_dtype = getattr(self, "cache_dtype", dtype)`.
3. For GDN layers, allocate recurrent/conv states with explicit non-FP8 dtypes
   and do not call `_quantize_cache()` / `_dequantize_cache()` on them.
4. In `update_qwen_chunked_kv_by_layer_id()`, replace the current direct
   `latest_k.to(k_cache.dtype)` / `latest_v.to(v_cache.dtype)` cast with the
   same quantization helper used by `KVCacheManager` when `kv_quant_config` is
   present. Direct `to(float8)` may be acceptable only if using the documented
   direct-cast path.
5. In `get_cache()` for attention layers, rely on inherited
   `get_kv_by_layer_id()` dequantization. GDN layers continue to bypass KV
   quantization.
6. Keep vLLM APC/mamba state handling separate: standard APC manages attention
   blocks; Qwen3.6 still needs GDN recurrent/conv state preservation.

## Calibration Requirements

Docs for `kv_cache_quant=True` describe direct FP8 cache quantization, not an
offline calibration artifact. The explicit `KVQuantizationConfig` path supports
methods/scales, but the migration docs state NxDI's KV cache quantization uses
direct cast.

Initial experiment should therefore use direct cast first:

```text
kv_cache_quant=True
kv_quant_config={"quant_dtype": torch.float8_e4m3fn, "direct_cast": True}
```

If quality fails, investigate static scale buffers via `KVCacheManager`:

```text
_init_scale_buffers()
_quantize_cache()
_dequantize_cache()
```

## Compile Config Delta From Baseline v3

Environment:

```bash
export XLA_HANDLE_SPECIAL_SCALAR=1
export UNSAFE_FP8FNCAST=1
```

Neuron config delta:

```python
kv_cache_quant=True
# or, explicitly:
kv_quant_config={
    "quant_dtype": torch.float8_e4m3fn,
    "direct_cast": True,
}
```

Qwen3.6 model config delta:

```python
use_hybrid_cache_manager=True
use_qwen_hybrid_chunked_prefill=True
use_qwen_hybrid_chunked_prefill_nki=True
```

Plus the implementation change that allows KV quantization in
`HybridDeltaNetCacheManager` only for full-attention layers.

## Expected Benefit

Primary benefit is memory headroom and long-context/concurrency capacity:

- attention KV memory roughly halves for GQA layers
- 128K context has more room for batching/APC/block tables
- less K/V read/write bandwidth during attention cache update/decode

Expected speedup:

- decode: `1.1x-1.3x` if attention KV bandwidth is a meaningful fraction
- prefill: likely smaller, because current cold prefill is dominated by GDN and
  orchestration rather than attention KV storage

This is not expected to match the gains from FP8 weight GEMM or speculative
decoding by itself.

## Validation Plan

1. Unit tests:
   - full-attention layers allocate FP8 K/V when enabled
   - GDN recurrent/conv state remains non-FP8
   - `get_cache()` dequantizes only attention K/V
   - `update_qwen_chunked_kv_by_layer_id()` quantizes only attention K/V
   - unsupported-mode tests updated so KV quantization is no longer rejected
     when the new flag is enabled

2. Compile/load:
   - compile from baseline v3 with FP8 KV enabled
   - load on `trn2.3xlarge`
   - verify HBM peak at 128K

3. Correctness:
   - smoke prompt
   - first-token finite logits once a logits-capable harness exists
   - MMLU-Lite 200 delta < 1pp
   - GSM8K 50 delta < 1pp
   - long-context 16K/64K/128K no invalid IDs

4. Performance:
   - 16K and 64K prefill tok/s
   - 128K max-context smoke
   - decode tok/s
   - APC hit/miss behavior under repeated-prefix traffic

## Risks

- Current OpenAI/vLLM proxy does not expose usable logprobs for cosine
  validation, so quality validation needs a logits-capable path first.
- FP8 KV may interact with APC and chunked prefill differently than standard
  non-hybrid KV because GDN state is not part of normal KV blocks.
- BF16/FP8 cache state may drift on reasoning/tool-use tasks; keep GDN
  recurrent state FP32 and validate long generations.

## Recommendation

FP8 KV is worth a focused implementation after the logits validation gap is
closed. It should be treated as a memory/concurrency optimization first and a
decode-speed optimization second. Do not enable it by simply removing the
hybrid-manager rejection; implement attention-layer-only quantization and keep
GDN state out of the FP8 path.
