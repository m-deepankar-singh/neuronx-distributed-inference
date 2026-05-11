# Qwen3.6-27B MTP Speculative Decoding Scouting

Date: 2026-05-11
Branch inspected: `codex/qwen36-gdn-stable-fused-prefill`
Target: `contrib/models/Qwen3.6-27B`
Remote model path inspected: `/opt/dlami/nvme/models/Qwen3.6-27B`

## Summary

Phase A result: **MTP is not ready to enable as a config-only feature on Neuron**.

The Qwen3.6-27B checkpoint does include native MTP heads, and vLLM 0.16.0 recognizes `qwen3_next_mtp`. The blocking gap is in the Neuron/NxDI execution path: the local NxDI tree supports speculative decoding through vanilla draft-model, fused speculation, EAGLE, and Medusa-style paths, but there is no MTP/NextN draft execution path. The Qwen3.6 contrib model also currently skips `mtp.*` weights during conversion and explicitly rejects speculative decoding in hybrid-cache mode.

Go/no-go: **No-go for Phase B as written if the expectation is a quick integration.** Proceed only as a larger implementation project that adds Qwen MTP model plumbing and Neuron speculative execution support.

## Evidence

### vLLM frontend support exists

The installed remote vLLM is `0.16.0`. Its speculative config includes `qwen3_next_mtp` in `MTPModelTypes` and normalizes model-specific MTP methods to internal method `mtp`.

Remote inspection showed:

```text
vllm 0.16.0
MTPModelTypes typing.Literal[..., 'qwen3_next_mtp', ..., 'mtp', ...]
HAS qwen3_next_mtp True
```

The public vLLM Qwen3-Next recipe documents:

```bash
--speculative-config '{"method": "qwen3_next_mtp", "num_speculative_tokens": 2}'
```

### NxDI supports speculation, but not Qwen MTP

Local grep found NxDI support for:

- `speculation_length`
- `enable_fused_speculation`
- `enable_eagle_speculation`
- `is_medusa`
- EAGLE hidden-state/token-tree paths
- Medusa cache/sampling paths

Key locations:

```text
src/neuronx_distributed_inference/models/config.py:242-250
src/neuronx_distributed_inference/models/model_base.py:3147-3165
src/neuronx_distributed_inference/models/model_base.py:3173-3182
src/neuronx_distributed_inference/modules/eagle/hidden_state.py
src/neuronx_distributed_inference/modules/kvcache/kv_cache_manager.py:265-317
```

The installed `neuronx_distributed_inference` package search showed no `qwen3_next_mtp`, `mtp`, `nextn`, or `NextN` implementation. AWS's Trainium speculative decoding blog lists NxDI modes as vanilla draft-model, fused speculation, EAGLE, and Medusa; it does not list native Qwen MTP.

### Qwen3.6 checkpoint has MTP heads

Remote `config.json` inspection:

```text
TOP_MODEL_TYPE qwen3_5
TEXT_MODEL_TYPE qwen3_5_text
text_config.mtp_num_hidden_layers 1
text_config.mtp_use_dedicated_embeddings False
```

Remote `model.safetensors.index.json` contains 15 MTP-like weights:

```text
mtp.fc.weight
mtp.layers.0.input_layernorm.weight
mtp.layers.0.mlp.down_proj.weight
mtp.layers.0.mlp.gate_proj.weight
mtp.layers.0.mlp.up_proj.weight
mtp.layers.0.post_attention_layernorm.weight
mtp.layers.0.self_attn.k_norm.weight
mtp.layers.0.self_attn.k_proj.weight
mtp.layers.0.self_attn.o_proj.weight
mtp.layers.0.self_attn.q_norm.weight
mtp.layers.0.self_attn.q_proj.weight
mtp.layers.0.self_attn.v_proj.weight
mtp.norm.weight
mtp.pre_fc_norm_embedding.weight
mtp.pre_fc_norm_hidden.weight
```

### Current Qwen3.6 contrib code blocks MTP today

The Qwen3.6 converter explicitly skips MTP weights:

```text
contrib/models/Qwen3.6-27B/src/modeling_qwen35.py:3009
elif k.startswith("mtp."):
    continue  # Skip MTP
```

The hybrid cache config validator rejects speculation:

```text
contrib/models/Qwen3.6-27B/src/modeling_qwen35.py:1894-1897
if nc.enable_fused_speculation or nc.speculation_length > 0 or nc.is_medusa:
    unsupported.append("speculative decoding")
if getattr(nc, "enable_eagle_speculation", False) or getattr(nc, "is_eagle_draft", False):
    unsupported.append("EAGLE speculation")
```

This means even if vLLM sends a speculative config, the current Qwen3.6 NxDI model will not load or execute MTP weights.

## API Gap

The gap is both model-side and backend-side.

Model-side gaps:

1. Define Qwen MTP modules in `modeling_qwen35.py`.
2. Stop skipping `mtp.*` weights and convert/shard them correctly.
3. Add MTP forward semantics matching Qwen3.6's HF/vLLM implementation.
4. Thread MTP through the hybrid cache state model without resetting GDN recurrent/conv state.

NxDI/backend gaps:

1. Add a native MTP/NextN speculative draft graph or adapt existing Medusa/fused speculation to use Qwen MTP heads.
2. Teach the compiled speculation graph to output draft tokens and verification logits in the shape expected by vLLM's speculative scheduler.
3. Integrate acceptance bookkeeping with hybrid KV + GDN state updates.
4. Decide interaction with vLLM APC/chunked prefill. Current Qwen3.6 hybrid path rejects speculation, and EAGLE3 has explicit prefix-caching limitations elsewhere in NxDI.

vLLM-Neuron gap:

1. vLLM parses `qwen3_next_mtp`, but Neuron backend does not expose a corresponding NxDI MTP worker path.
2. The Neuron platform plugin currently loads our custom NxDI model artifact. That artifact contains no MTP graph or weights today.

## Recommendation

Do not proceed to compile attempts for MTP yet. A config-only run is expected to fail or silently fall back because the compiled artifact has no MTP graph.

Recommended next step is a design spike, not Phase B implementation:

1. Read vLLM's GPU Qwen3-Next MTP implementation (`vllm/model_executor/models/qwen3_next.py` and MTP proposer code) and map exact tensor contracts.
2. Implement a CPU-only Qwen3.6 MTP module loader for `mtp.*` weights in contrib and validate one-step MTP logits against HF/vLLM CPU reference.
3. Decide whether to adapt NxDI Medusa flow or add a new `is_mtp`/`mtp_speculation_length` path. Medusa is closer structurally because it proposes multiple tokens with extra heads, but Qwen MTP is a transformer-like next-token stack, not simple Medusa heads.
4. Only after CPU MTP logits match should we attempt a Neuron compile.

Estimated scope: **2-4 weeks**, not 3 days.

## Phase A Verdict

- Native checkpoint MTP heads available: **yes**.
- vLLM frontend recognizes `qwen3_next_mtp`: **yes**.
- NxDI native Qwen MTP execution support: **no evidence found**.
- Current Qwen3.6 contrib loads MTP weights: **no, it skips them**.
- Current Qwen3.6 hybrid path allows speculation: **no, it rejects it**.

Final decision: **No-go for direct Phase B. Go for a design/CPU-reference MTP integration spike first.**
