# Qwen3.6 Hybrid APC Goal State

## Objective

Make Qwen3.6-27B Hybrid APC on Trainium correct first, then measure cold-prefill performance. Correctness means:

- BF16 host-logits path emits finite real-token outputs.
- Cold and warm Hybrid APC outputs match for the tested prompts.
- Attention KV prefix reuse is only used when matching GDN recurrent/conv checkpoint state is available.
- Unsupported scheduler/cache states fail fast instead of silently producing wrong tokens.

## Current Status

The active branch is `experimental`. The latest pushed code patch is `479755a`.

The base BF16 host-logits path is not the current blocker when using the per-chunk DeltaNet CTE path:

- Fused CTE artifact goes NaN around 105-106 tokens.
- Per-chunk CTE artifact stays finite through the long BF16 control.
- The old warm drift was caused by vLLM reusing attention KV without a matching GDN checkpoint.
- The new opt-in scheduler fallback disables those unbacked vLLM prefix reads before allocation, and the existing BF16 Hybrid APC per-chunk artifact now passes decode24 exactness without the model-side debug fallback.

Useful Trainium paths:

- Instance: `ubuntu@16.50.102.110`
- Key: `/Users/deepankarsingh1312/Downloads/trainium.pem`
- Remote repo: `/home/ubuntu/inferentia-gdn-experimental-test`
- Weights: `/home/ubuntu/models/Qwen3.6-27B`
- Main BF16 Hybrid APC artifact:
  `/home/ubuntu/qwen_artifacts/qwen36_27b_2048_bf16_hybrid_apc_host_logits_nki_chunked_4434edf`

## Confirmed Validation

Local focused tests after the latest patches:

```text
15 passed
```

Remote focused tests after pulling `479755a`:

```text
84 passed
```

Remote exactness command reused the existing artifact and did not compile:

```bash
QWEN36_HYBRID_APC_DEBUG=1 USE_NKI_FUSED=0 USE_NKI_CHUNKED=1 \
python3 validation_scripts/qwen36_hybrid_apc_validation.py exactness \
  --model-path /home/ubuntu/models/Qwen3.6-27B \
  --compiled-artifacts /home/ubuntu/qwen_artifacts/qwen36_27b_2048_bf16_hybrid_apc_host_logits_nki_chunked_4434edf \
  --max-model-len 2048 --seq-len 2048 --cte-buckets 256,512 \
  --max-tokens 24 --require-real-tokens --skip-fp8-env \
  --hybrid-apc-disable-unbacked-prefix-reads
```

Result:

```text
full_prefix_exact=True
partial_prefix_exact=True
real_generated_tokens_passed=True
```

Artifacts:

- JSON:
  `/home/ubuntu/validation_logs/hybrid_apc_real_tokens/bf16_hybrid_apc_host_logits_nki_chunked_479755a_scheduler_fallback_decode24.json`
- Log:
  `/home/ubuntu/validation_logs/hybrid_apc_real_tokens/bf16_hybrid_apc_host_logits_nki_chunked_479755a_scheduler_fallback_decode24.log`

The log shows the scheduler fallback path is active. Warm requests run with:

```text
restore_len=0
suffix_len=463
input_shape=(1, 463)
```

That means the invalid attention prefix hit is no longer being used; vLLM executes the full active prompt instead of reusing attention KV without GDN state.

## What Changed

`939dba5` added the production safety guard:

- `hybrid_apc_reject_unbacked_attention_hits=True` by default.
- Raise if `attention_hit_len > 0` and no matching GDN checkpoint exists.
- Debug kill switches:
  - `QWEN36_DISABLE_HYBRID_GDN_RESTORE=1`
  - `QWEN36_DISABLE_HYBRID_GDN_COMMIT=1`

`fb881a7` repaired the model-side no-restore fallback:

- If an unbacked attention hit is explicitly allowed for debugging, rebuild full active `slot_mapping` from `block_table`.
- This fixed token 0 being written to a suffix slot.

`553e4e1`, `f2ac367`, and `479755a` added the scheduler-side opt-in fallback:

- CLI/config/env flag: `--hybrid-apc-disable-unbacked-prefix-reads`
- Env flag: `QWEN36_HYBRID_APC_DISABLE_UNBACKED_PREFIX_READS=1`
- Lazy `sitecustomize.py` hook patches vLLM EngineCore without importing vLLM at Python startup.
- The env flag now wins even when the artifact's embedded config is stale, which is required for old compiled artifacts.

The failed intermediate attempt at `f2ac367` proved the lazy hook fixed stdout pollution but did not yet disable prefix reads, because `hf_config.use_hybrid_apc_manager` was stale/false inside EngineCore. `479755a` fixed that by letting the explicit env flag override stale artifact config.

## Exact Current Problem

Correctness is now protected, but the real performance path is not done.

The current passing mode intentionally disables unbacked vLLM attention prefix reads:

```text
attention KV prefix hit exists
GDN checkpoint hit is missing
scheduler marks request skip_reading_prefix_cache=True
vLLM recomputes the prompt as a no-prefix request
outputs are exact
```

This is correct but does not deliver the final cold-prefill speedup, because it avoids reuse when the GDN checkpoint side is not available.

The final implementation still needs a real cache eligibility contract:

```text
usable_prefix_hit = attention_kv_hit intersect gdn_checkpoint_hit
```

Only that usable prefix should be exposed to vLLM allocation. If the GDN checkpoint hit is missing, the request must be scheduled as no-prefix before block/slot allocation. If both attention KV and GDN checkpoint exist at the same boundary, the request may restore GDN state and reuse attention KV.

NVIDIA/vLLM avoids this class of bug because prefix-cache ownership is centralized in the scheduler/KV cache manager. For hybrid state models, the usable hit must be the intersection of all required cache groups. The Neuron path has to recreate that contract across vLLM-Neuron, NxDI trace inputs/outputs, block tables, slot mapping, and GDN checkpoint metadata.

References:

- vLLM PagedAttention: https://docs.vllm.ai/en/stable/design/paged_attention/
- vLLM prefix caching: https://docs.vllm.ai/en/stable/design/prefix_caching/

## Recommended Next Work

1. Keep the fail-fast guard enabled by default.
2. Keep `--hybrid-apc-disable-unbacked-prefix-reads` enabled for correctness validation on old artifacts.
3. Promote the fallback into a real scheduler/cache-manager contract that computes `usable_prefix_hit = attention_kv_hit intersect gdn_checkpoint_hit`.
4. Add a positive warm-prefix test where the prompt is chunked on checkpoint boundaries and a matching GDN checkpoint exists.
5. Compile only once for that true checkpoint-boundary path, then run decode exactness and the cold-prefill performance gate.

Avoid additional compiles until the scheduler can produce a request with both matching attention KV blocks and matching GDN checkpoint metadata.
