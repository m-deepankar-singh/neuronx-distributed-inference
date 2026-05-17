# Qwen3.6 Hybrid APC Goal State

## Objective

Make Qwen3.6-27B Hybrid APC on Trainium correct first, then measure cold-prefill performance. Correctness means:

- BF16 host-logits path emits finite real-token outputs.
- Cold and warm Hybrid APC outputs match for the tested prompts.
- Attention KV prefix reuse is only used when matching GDN recurrent/conv checkpoint state is available.
- Any unsupported scheduler/cache state fails fast instead of silently producing wrong tokens.

## Current Status

The active branch is `experimental`. The latest pushed guard patch is `939dba5`.

The base BF16 host-logits path is no longer the blocker when using the per-chunk DeltaNet CTE path:

- Fused CTE artifact goes NaN around 105-106 tokens.
- Per-chunk CTE artifact stays finite through the long BF16 control.
- Hybrid APC decode passes short exactness after suffix slot mapping repair, but drifts on longer decode once vLLM schedules an attention prefix hit that does not have a matching GDN checkpoint.

Useful artifacts on the Trainium instance:

- Base BF16 no-prefix fused CTE artifact:
  `/home/ubuntu/qwen_artifacts/qwen36_27b_2048_bf16_host_logits_no_prefix_353306f`
- Base BF16 no-prefix per-chunk CTE artifact:
  `/home/ubuntu/qwen_artifacts/qwen36_27b_2048_bf16_host_logits_no_prefix_nki_chunked_6f575ef`
- Hybrid APC BF16 host-logits per-chunk CTE artifact:
  `/home/ubuntu/qwen_artifacts/qwen36_27b_2048_bf16_hybrid_apc_host_logits_nki_chunked_4434edf`

Important validation logs:

- BF16 fused boundary NaN:
  `/home/ubuntu/validation_logs/host_logits_controls/bf16_no_prefix_length_sweep_boundary_6f575ef.log`
- BF16 per-chunk long finite control:
  `/home/ubuntu/validation_logs/host_logits_controls/bf16_no_prefix_nki_chunked_long_6f575ef.log`
- Hybrid APC CTE-only pass:
  `/home/ubuntu/validation_logs/hybrid_apc_real_tokens/bf16_hybrid_apc_host_logits_nki_chunked_4434edf_cte_only.json`
- Hybrid APC decode4 exact pass after suffix slot mapping repair:
  `/home/ubuntu/validation_logs/hybrid_apc_real_tokens/bf16_hybrid_apc_host_logits_nki_chunked_b59e3a2_decode4.json`
- Hybrid APC decode22 exact pass:
  `/home/ubuntu/validation_logs/hybrid_apc_real_tokens/bf16_hybrid_apc_host_logits_nki_chunked_b59e3a2_decode22.json`
- Hybrid APC decode24/32 warm drift evidence:
  `/home/ubuntu/validation_logs/hybrid_apc_real_tokens/bf16_hybrid_apc_host_logits_nki_chunked_b35841d_decode24_topk.log`
  `/home/ubuntu/validation_logs/hybrid_apc_real_tokens/bf16_hybrid_apc_host_logits_nki_chunked_fd3b906_decode24_guard.json`

## Exact Current Problem

The remaining correctness bug is a scheduler/state contract mismatch:

```text
vLLM attention APC says: prefix attention KV can be reused
Hybrid APC metadata says: no matching GDN checkpoint is available
Neuron execution still receives a prefix-cache-shaped request
```

For Qwen GDN, attention KV reuse without the matching recurrent/conv checkpoint is invalid. It can produce warm logits that are finite but semantically wrong. This is why the current failure is exactness drift, not primarily NaN/OOB.

NVIDIA/vLLM avoids this class of bug by keeping prefix cache ownership inside its scheduler/KV cache manager. For hybrid models, the usable cache hit must be the intersection of all required cache groups. The Neuron path has to recreate that contract across vLLM-Neuron, NxDI trace inputs/outputs, block tables, slot mapping, and GDN checkpoint metadata.

Reference:

- vLLM PagedAttention: https://docs.vllm.ai/en/stable/design/paged_attention/
- vLLM prefix caching: https://docs.vllm.ai/en/stable/design/prefix_caching/

## Current Guard Patch

The safety patch in `939dba5` adds:

- `hybrid_apc_reject_unbacked_attention_hits=True` by default.
- A bridge guard that raises when `attention_hit_len > 0` but no matching GDN checkpoint exists.
- Debug kill switches:
  - `QWEN36_DISABLE_HYBRID_GDN_RESTORE=1`
  - `QWEN36_DISABLE_HYBRID_GDN_COMMIT=1`

The guard does not deliver the final performance fix by itself. It prevents silent wrong output and proves the scheduler needs to intersect attention and GDN cache eligibility before block/slot allocation.

Verification:

- Remote focused unit suite passed after pulling `939dba5`:
  `74 passed` across `test_hybrid_apc_manager.py`, `test_config.py`, and `test_vllm_serving_config.py`.
- Existing BF16 Hybrid APC per-chunk artifact now fails fast, as expected, instead of silently drifting:
  `/home/ubuntu/validation_logs/hybrid_apc_real_tokens/bf16_hybrid_apc_host_logits_nki_chunked_939dba5_reject_unbacked.log`
- The explicit validation error is:
  `hybrid APC received an attention prefix hit without a matching GDN checkpoint; scheduler must intersect attention KV hits with GDN checkpoint hits or disable prefix reuse for this request`

## Recommended Next Work

1. Push and verify the guard patch.
2. Run the existing BF16 Hybrid APC validation with the per-chunk artifact. Expected behavior is now a fast, explicit error on the unbacked attention hit instead of warm drift.
3. Implement the real scheduler fallback:

```text
usable_prefix_hit = attention_kv_hit intersect gdn_checkpoint_hit
if gdn_checkpoint_hit is missing:
    force prefix hit to 0 before block/slot allocation
```

4. After fallback passes exactness without silent drift, compile one real chunked-prefill artifact that creates checkpoint-boundary prefill calls at 256-token boundaries.
5. Only then run the cold-prefill performance gate.

Avoid additional compiles until the scheduler fallback is implemented or a compile is needed specifically for true chunked-prefill boundary behavior.
