# Qwen3.6 Hybrid APC Goal State

## Objective

Make Qwen3.6-27B Hybrid APC on Trainium correct first, then measure cold-prefill performance. Correctness means:

- BF16 host-logits path emits finite real-token outputs.
- Cold and warm Hybrid APC outputs match for tested prompts.
- Attention KV prefix reuse is only used when matching GDN recurrent/conv checkpoint state is available.
- Unsupported scheduler/cache states fail fast or fall back to no-prefix execution instead of silently producing wrong tokens.

## Current Status

The active branch is `experimental`. The latest pushed code patch before this
round was `7d1138e`.

Useful Trainium paths:

- Instance: `ubuntu@16.50.102.110`
- Key: `/Users/deepankarsingh1312/Downloads/trainium.pem`
- Remote repo: `/home/ubuntu/inferentia-gdn-experimental-test`
- Weights: `/home/ubuntu/models/Qwen3.6-27B`
- Main BF16 Hybrid APC artifact:
  `/home/ubuntu/qwen_artifacts/qwen36_27b_2048_bf16_hybrid_apc_host_logits_nki_chunked_4434edf`

The base BF16 host-logits path is not the current blocker when using the per-chunk DeltaNet CTE path:

- Fused CTE artifact goes NaN around 105-106 tokens.
- Per-chunk CTE artifact stays finite through the long BF16 control.
- Real-token host-logits generation works on the existing BF16 artifact.
- The old warm drift was caused by vLLM reusing attention KV without a matching GDN checkpoint.

The correctness-safe fallback now works:

```text
attention KV prefix hit exists
GDN checkpoint hit is missing
scheduler sets request.skip_reading_prefix_cache=True
vLLM recomputes the active prompt as a no-prefix request
cold and warm outputs match
```

The real performance path is not done:

```text
attention KV prefix hit exists
matching GDN checkpoint exists
model restores GDN state and runs only the suffix
```

That backed-restore path now activates, but the warm partial-prefix output is still wrong. The current failure is no longer "scheduler cannot find a GDN checkpoint"; it is now in the CTE restore/padding/model execution contract after a valid 256-token restore is selected.

Follow-up inspection found an additional required condition: the current BF16 artifact has:

```text
use_qwen_hybrid_chunked_prefill=False
use_qwen_hybrid_chunked_prefill_nki=False
```

With that artifact, Qwen CTE restores GDN state but does not consume attention KV prefix state for full-attention layers. A registered GDN checkpoint is therefore not sufficient to make an attention prefix read safe.

## Confirmed Validation

Local focused tests after `7d1138e`:

```text
53 passed
```

Remote focused tests after `d93470b`:

```text
86 passed
```

Remote focused subset after `7d1138e`:

```text
45 passed
```

The no-compile safe fallback used the existing artifact and passed decode24 exactness:

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
  `/home/ubuntu/validation_logs/hybrid_apc_real_tokens/bf16_hybrid_apc_host_logits_nki_chunked_d93470b_scheduler_registry_decode24.json`
- Log:
  `/home/ubuntu/validation_logs/hybrid_apc_real_tokens/bf16_hybrid_apc_host_logits_nki_chunked_d93470b_scheduler_registry_decode24.log`

The passing fallback does not exercise restore:

```text
attention_hit_len=0
restore_len=0
```

That is correct for safety, but it does not prove the performance path.

## Backed Restore Evidence

A positive boundary prompt was constructed so prompt A commits a 256-token GDN checkpoint and prompt B reuses that checkpoint with a suffix:

```bash
SHARED=$(python3 -c "print('System: answer deterministically.\n'*36 + ' 1 2', end='')")
SUFFIX_B=$(python3 -c "print('\nUser: What is 19 * 29?\nAssistant:', end='')")

QWEN36_HYBRID_APC_DEBUG=1 USE_NKI_FUSED=0 USE_NKI_CHUNKED=1 \
python3 validation_scripts/qwen36_hybrid_apc_validation.py exactness \
  --model-path /home/ubuntu/models/Qwen3.6-27B \
  --compiled-artifacts /home/ubuntu/qwen_artifacts/qwen36_27b_2048_bf16_hybrid_apc_host_logits_nki_chunked_4434edf \
  --max-model-len 2048 --seq-len 2048 --cte-buckets 256,512 \
  --max-tokens 24 --require-real-tokens --skip-fp8-env \
  --hybrid-apc-disable-unbacked-prefix-reads \
  --shared-prefix "$SHARED" --suffix-a "" --suffix-b "$SUFFIX_B"
```

Under `d93470b`, the run still did not restore because the scheduler registry and the artifact config disagreed on model revision:

```text
artifact config hybrid_apc_model_revision="unknown"
scheduler registry default used config._name_or_path
registry lookup missed
restore_len=0
```

`7d1138e` fixed that by aligning the scheduler registry default to `"unknown"`.

Under `7d1138e`, the same boundary run exercised the backed restore:

```text
[hybrid_apc_debug] apply prompt_len=272 restore_len=256 suffix_len=16 restore_slot=0 commit_slot=None input_shape=(1, 272) output_shape=(1, 16)
[hybrid_apc_debug] prepare request_id=('seq_id', 0) attention_hit_len=256 request_prefix_len=272 restore_len=256 commit_prefix_len=256 restore_slot=0 commit_slot=None input_shape=(1, 272) prepared_shape=(1, 16) computed=tensor([[256]], dtype=torch.int32) num_queries=tensor([[16]], dtype=torch.int32) restore_mask=tensor([1], dtype=torch.int32) commit_mask=tensor([0], dtype=torch.int32)
[hybrid_apc_debug] qwen-cte-call input_shape=(1, 16) attention_shape=(1, 16) position_shape=(1, 16) position_minmax=256:271 slot_shape=(1, 16) slot_minmax=1792:1807 block_shape=(1, 8) block_minmax=0:7 num_queries=[16] computed=[256] restore_slots=[0] restore_mask=[1] restore_prefix=[256] commit_slots=[0] commit_mask=[0]
```

But exactness failed:

```text
full_prefix_exact=True
partial_prefix_exact=False
real_generated_tokens_passed=True
```

Artifacts:

- JSON:
  `/home/ubuntu/validation_logs/hybrid_apc_real_tokens/bf16_hybrid_apc_host_logits_nki_chunked_7d1138e_scheduler_registry_boundary_decode24.json`
- Log:
  `/home/ubuntu/validation_logs/hybrid_apc_real_tokens/bf16_hybrid_apc_host_logits_nki_chunked_7d1138e_scheduler_registry_boundary_decode24.log`

The warm partial output starts with control/sentinel-looking tokens instead of the cold deterministic answer:

```text
warm_partial tokens:
[220, 248068, 271, 248069, 271, 17, 271, 760, 1918, 314, 220, 16, 321, 220, 17, 369, 220, 17, 13, 248044]
```

The most suspicious debug lines are from padding:

```text
pad-pre tag=context_encoding_model input_shape=(1, 16) attention_shape=(1, 16) position_shape=(1, 16) slot_shape=(1, 16) slot_minmax=1792:1807 block_shape=(1, 8) block_minmax=0:7 prefill_len=16 prefix_len=256 prefill_bucket=512 prefix_bucket=0
pad-post tag=context_encoding_model adjusted_prefix_len=0 extra_prefill_slots=496 padded_input_shape=(1, 512) padded_attention_shape=(1,) padded_position_shape=(1, 512) padded_slot_shape=(1, 512) padded_slot_minmax=-1:1807 padded_block_shape=(1,) padded_block_minmax=0:0
```

This strongly suggests that the scheduler and model now select the right restore, but the CTE padding wrapper treats the 256-token restored prefix incorrectly. It pads the 16-token suffix to a 512-token CTE bucket while collapsing the prefix side to `adjusted_prefix_len=0`, `prefix_bucket=0`, `padded_attention_shape=(1,)`, and `padded_block_shape=(1,)`.

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

`553e4e1`, `f2ac367`, and `479755a` added the scheduler-side no-prefix fallback:

- CLI/config/env flag: `--hybrid-apc-disable-unbacked-prefix-reads`
- Env flag: `QWEN36_HYBRID_APC_DISABLE_UNBACKED_PREFIX_READS=1`
- Lazy `sitecustomize.py` hook patches vLLM EngineCore without importing vLLM at Python startup.
- The explicit env flag can override stale artifact config.

`d93470b` added backed GDN checkpoint tracking:

- Process-local scheduler registry in `contrib/models/Qwen3.6-27B/vllm/qwen36_hybrid_apc_scheduler_patch.py`.
- Model-side `HybridAPCSchedulerBridge.commit_prefill` publishes committed GDN checkpoint keys to the scheduler registry.
- `HybridAPCMetadataStore.mark_invalid` and `_delete_checkpoint` unpublish registry keys.
- Scheduler fallback now allows vLLM prefix reads only when `backed_gdn_prefix_hit_len(...) > 0`.
- Added unit coverage for registered/mismatched registry behavior and publish/unpublish bridge behavior.

`7d1138e` fixed the registry default:

- Scheduler registry now defaults missing `hybrid_apc_model_revision` to `"unknown"`, matching the Qwen model config embedded in the current artifact.
- Added test coverage for the missing model revision default.

The next local patch tightens the scheduler safety gate:

- A registered GDN checkpoint still proves that the GDN side is backed.
- vLLM attention prefix reads are only allowed when the compiled artifact also advertises backed CTE attention-prefix support.
- The scheduler checks `hybrid_apc_enable_backed_prefix_reads=True` and `use_qwen_hybrid_chunked_prefill=True` before exposing a backed prefix read.
- `QWEN36_HYBRID_APC_ENABLE_BACKED_PREFIX_READS=1` remains a debug override.
- This prevents the current artifact from taking the known-wrong "GDN restored, attention prefix missing" path.

## Exact Current Problem

The exact current problem is:

```text
Backed restore path activates at restore_len=256, but warm partial-prefix exactness fails.
The failure appears after scheduling/registry selection, inside the CTE restore padding/model input/full-attention prefix contract.
```

The likely bad contract is:

```text
suffix input length: 16
restored prefix length: 256
computed_context_lens: [256]
num_queries: [16]
slot_mapping: suffix slots only
block_table: 8 prefix blocks
current artifact: use_qwen_hybrid_chunked_prefill=False
padding wrapper: resets/collapses prefix side during CTE bucket padding
full-attention layers: do not consume attention KV prefix in CTE on this artifact
```

The current artifact does not support the full backed-prefix CTE path because Qwen chunked prefill was compiled off. Before compiling again, also inspect and fix the wrapper path that produced:

```text
adjusted_prefix_len=0
padded_attention_shape=(1,)
padded_block_shape=(1,)
```

Primary files to inspect:

- `src/neuronx_distributed_inference/models/model_wrapper.py`
- `contrib/models/Qwen3.6-27B/src/modeling_qwen35.py`
- `contrib/models/Qwen3.6-27B/vllm/qwen36_hybrid_apc_scheduler_patch.py`

## Why The Earlier Artifacts Did Not Work

The earlier artifacts failed for different reasons:

- Fused BF16 CTE artifact: numerical failure/NaNs around 105-106 tokens.
- FP8 path: still needs BF16 comparison before chasing FP8-specific NaNs.
- Old warm Hybrid APC runs: reused attention KV without matching GDN recurrent/conv state, so warm logits drifted.
- Existing BF16 per-chunk artifact: works for correctness when unbacked prefix reads are disabled, but the true backed restore path now exposes a CTE restore/padding contract issue.

The current issue does not prove that another compile is required. The run used an existing artifact and reached the right restore call. The next step is to fix or prove the Python-side CTE input/padding contract before spending another compile.

## Recommended Next Work

1. Do not compile again until the padding contract is understood.
2. Inspect `model_wrapper.py` around the `pad-pre` / `pad-post` debug path and find why `prefix_len=256` becomes `adjusted_prefix_len=0`.
3. Preserve `computed_context_lens=[256]`, `num_queries=[16]`, the suffix `slot_mapping`, and the required attention-prefix block table through CTE padding if the compiled NEFF expects prefix-cache CTE inputs.
4. Enable backed prefix reads only for an artifact compiled with the CTE full-attention prefix path.
5. Compare against the older working `contrib/qwen36-27b-vllm-apc-pr` branch for CTE/TKG argument and padding contracts. That branch compiled Qwen chunked prefill on by default.
6. Rerun the same 2K boundary validation after the safety gate. It should fall back to no-prefix on the current artifact and pass exactness without claiming perf.
7. Only if/when the corrected CTE prefix contract is implemented, compile once with the corrected signature and `hybrid_apc_enable_backed_prefix_reads=True`.
8. After BF16 backed restore exactness passes, revisit FP8 and TKG/on-device sampling separately.

The next concrete engineering target is specific: fix the CTE restore padding/input/full-attention-prefix contract so the 256-token backed GDN restore plus 16-token suffix produces the same logits/tokens as the cold 272-token prompt.

## NVIDIA/vLLM Comparison

NVIDIA vLLM avoids this failure class because cache ownership and kernel contracts are centralized:

- Attention KV is managed as paged blocks.
- Prefix cache hits are scheduled against block tables and slot mappings that CUDA/Triton kernels expect.
- Hybrid state such as Mamba/SSM cache is treated as separate model state, not normal attention KV.
- Decode and prefill kernel signatures are stable and typed by the GPU runner.

The Neuron path is rebuilding that contract across vLLM-Neuron, NxDI trace signatures, Qwen wrappers, block tables, slot mapping, and GDN checkpoint aliases. The correct architecture is still the same:

```text
usable_prefix_hit = attention_kv_hit intersect gdn_checkpoint_hit
```

Now that the scheduler can find a backed hit, the remaining work is to make the compiled CTE input contract honor that backed hit during suffix-only execution.

References:

- vLLM PagedAttention: https://docs.vllm.ai/en/stable/design/paged_attention/
- vLLM prefix caching: https://docs.vllm.ai/en/stable/design/prefix_caching/
