# Qwen3.6 Hybrid APC Goal State

## Objective

Make Qwen3.6-27B Hybrid APC on Trainium correct first, then measure cold-prefill performance. Correctness means:

- BF16 host-logits path emits finite real-token outputs.
- Cold and warm Hybrid APC outputs match for tested prompts.
- Attention KV prefix reuse is only used when matching GDN recurrent/conv checkpoint state is available.
- Unsupported scheduler/cache states fail fast or fall back to no-prefix execution instead of silently producing wrong tokens.

## Current Status

The active branch is `experimental`.

Useful Trainium paths:

- Instance: `ubuntu@16.50.102.110`
- Key: `/Users/deepankarsingh1312/Downloads/trainium.pem`
- Remote repo: `/home/ubuntu/inferentia-gdn-experimental-test`
- Weights: `/home/ubuntu/models/Qwen3.6-27B`
- Main BF16 Hybrid APC artifact:
  `/mnt/trainium_artifacts/qwen_artifacts/qwen36_27b_2048_bf16_hybrid_apc_backed_prefix_d061df5`

### 2026-05-18 Overnight Update

The new 2K BF16 Hybrid APC artifact compiled on the mounted NVMe, but normal
load warmup OOBs in the non-target CTE bucket:

```text
context_encoding_model/_tp0_bk2
neuron_config.buckets = [[256, 512]]
NRT_EXEC_OOB during warmup
```

Setting `skip_warmup=true` in the artifact config lets the artifact load and
serve. The backup before that local artifact edit is:

```text
/mnt/trainium_artifacts/qwen_artifacts/qwen36_27b_2048_bf16_hybrid_apc_backed_prefix_d061df5/neuron_config.json.bak_before_skip_warmup_6fa9ea1
```

With `--enable-vllm-chunked-prefill`, the safety fallback still passes
decode24 exactness and real-token generation:

```text
full_prefix_exact=true
partial_prefix_exact=true
real_generated_tokens_passed=true
```

Artifacts:

- JSON:
  `/home/ubuntu/validation_logs/hybrid_apc_real_tokens/bf16_hybrid_apc_backed_prefix_d061df5_boundary_decode24_chunked_6fa9ea1.json`
- Log:
  `/home/ubuntu/validation_logs/hybrid_apc_real_tokens/bf16_hybrid_apc_backed_prefix_d061df5_boundary_decode24_chunked_6fa9ea1.log`

That pass is not the performance path. The debug lines still show:

```text
attention_hit_len=0
restore_len=0
computed=tensor([[0]], dtype=torch.int32)
```

Forcing vLLM prefix reads proves the exact remaining contract bug. vLLM does
reuse the attention KV prefix and sends:

```text
input_shape=(1, 16)
computed=[256]
position_minmax=256:271
```

but Qwen Hybrid APC does not restore GDN state because vLLM-Neuron only passes
the suffix into CTE and does not pass the full prompt tokens, cumulative prefix
hash, or restore slot metadata. The forced-prefix run therefore mismatches:

```text
full_prefix_exact=true
partial_prefix_exact=false
real_generated_tokens_passed=true
restore_mask=[0]
```

Artifacts:

- JSON:
  `/home/ubuntu/validation_logs/hybrid_apc_real_tokens/bf16_hybrid_apc_backed_prefix_d061df5_force_prefix_reads_py_path_6fa9ea1.json`
- Log:
  `/home/ubuntu/validation_logs/hybrid_apc_real_tokens/bf16_hybrid_apc_backed_prefix_d061df5_force_prefix_reads_py_path_6fa9ea1.log`

Current interpretation:

```text
vLLM/NVIDIA contract:
  scheduler finds prefix KV hit
  runner passes num_computed_tokens + block table + suffix slots
  model-specific state cache is handled by a separate stable contract

Current Neuron/Qwen contract:
  vLLM-Neuron passes the attention suffix and computed_context_lens
  Qwen Hybrid APC needs full prompt/hash or scheduler-selected restore metadata
  without that, attention KV can be reused while GDN state is not restored
```

The next diagnostic patch is intentionally guarded by:

```text
QWEN36_HYBRID_APC_ALLOW_UNHASHED_SINGLE_PREFIX_RESTORE=1
```

It allows suffix-only restore only when exactly one valid GDN checkpoint exists
for the vLLM-reported prefix length. This is not the production contract; it is
to prove whether the CTE restore path itself becomes exact once GDN state is
restored. The production fix should pass full prompt hashes or explicit restore
metadata from vLLM/vLLM-Neuron into the Qwen model request.

After commit `1c3d9dd`, the guarded diagnostic passed with forced vLLM prefix
reads:

```text
full_prefix_exact=true
partial_prefix_exact=true
real_generated_tokens_passed=true
warm_partial elapsed: 1.3459s
cold_partial elapsed: 1.9542s
```

The key debug proof is:

```text
apply-suffix prompt_len=272 restore_len=256 suffix_len=16 restore_slot=0
attention_hit_len=256 restore_len=256 computed=tensor([[256]]) num_queries=tensor([[16]])
restore_mask=tensor([1])
qwen-cte-call input_shape=(1, 16) position_minmax=256:271 computed=[256] restore_prefix=[256]
pad-pre prefill_len=16 prefix_len=256 prefill_bucket=256 prefix_bucket=256
```

Artifacts:

- JSON:
  `/home/ubuntu/validation_logs/hybrid_apc_real_tokens/bf16_hybrid_apc_backed_prefix_d061df5_suffix_restore_diag_1c3d9dd.json`
- Log:
  `/home/ubuntu/validation_logs/hybrid_apc_real_tokens/bf16_hybrid_apc_backed_prefix_d061df5_suffix_restore_diag_1c3d9dd.log`

This proves the CTE restore/padding/model execution contract can be exact for
the 256-token backed prefix.

The next scheduler-gated run exposed a runtime config propagation bug: the
serving `additional_config` advertised
`hybrid_apc_enable_backed_prefix_reads=True` and
`use_qwen_hybrid_chunked_prefill=True`, but the scheduler gate only read
`vllm_config.model_config.hf_config`. It therefore printed:

```text
backed_hit_len=256
supports_backed=False
```

The fix is to have the scheduler gate read Hybrid APC safety flags from
`vllm_config.additional_config` before falling back to `hf_config`.

After that fix, the same 2K boundary validation passed without global
`QWEN36_HYBRID_APC_ENABLE_PREFIX_READS`:

```text
full_prefix_exact=true
partial_prefix_exact=true
real_generated_tokens_passed=true
cold_partial elapsed: 1.9555s
warm_partial elapsed: 1.3282s
```

Key debug proof:

```text
scheduler-decision backed_hit_len=256 supports_backed=True prompt_len=272 registry_size=1
apply-suffix prompt_len=272 restore_len=256 suffix_len=16 restore_slot=0
attention_hit_len=256 restore_len=256 computed=tensor([[256]]) num_queries=tensor([[16]])
qwen-cte-call input_shape=(1, 16) position_minmax=256:271 computed=[256] restore_mask=[1]
```

Artifacts:

- JSON:
  `/home/ubuntu/validation_logs/hybrid_apc_real_tokens/bf16_hybrid_apc_backed_prefix_d061df5_scheduler_gate_boundary_additional_config.json`
- Log:
  `/home/ubuntu/validation_logs/hybrid_apc_real_tokens/bf16_hybrid_apc_backed_prefix_d061df5_scheduler_gate_boundary_additional_config.log`

This proves the scheduler-gated backed-prefix path is correct for the current
single-request diagnostic.

The next patch removed the length-only dependency for this case. The scheduler
now records the exact GDN prefix key when it allows a backed prefix read, and
Qwen consumes that authorized key for suffix-only restore before falling back to
the guarded diagnostic path. The same boundary validation passed with
`QWEN36_HYBRID_APC_ALLOW_UNHASHED_SINGLE_PREFIX_RESTORE` unset:

```text
full_prefix_exact=true
partial_prefix_exact=true
real_generated_tokens_passed=true
cold_partial elapsed: 1.9669s
warm_partial elapsed: 1.3296s
```

Artifacts:

- JSON:
  `/home/ubuntu/validation_logs/hybrid_apc_real_tokens/bf16_hybrid_apc_backed_prefix_d061df5_scheduler_authorized_key_no_unhashed.json`
- Log:
  `/home/ubuntu/validation_logs/hybrid_apc_real_tokens/bf16_hybrid_apc_backed_prefix_d061df5_scheduler_authorized_key_no_unhashed.log`

The current remaining production hardening is for batched/concurrent serving:

```text
The scheduler/model process currently uses an in-process authorized-prefix key
handoff. This is correct for the single-request validation path and avoids
length-only restore, but concurrent requests should carry request-id scoped
restore metadata or a runner-provided restore slot/key to avoid queue-order
assumptions.
```

The scheduler now keeps backed prefix reads disabled when
`scheduler_config.max_num_seqs != 1`, so batched serving falls back safely until
request-scoped metadata is wired.

### 2026-05-18 Late Update

Current overnight target:

```text
Carry the scheduler-approved GDN restore key with the vLLM request identity,
then expose that request identity to Qwen request prep through the
vLLM-Neuron runner without changing the traced NEFF input signature.
```

The intended short patch is:

- Scheduler authorizes backed prefix reads by exact request id when available.
- vLLM-Neuron runner temporarily sets `_qwen36_vllm_request_ids` on the model
  during `_execute_model_for_text`.
- Qwen model request prep forwards the single request id as `hybrid_request_id`
  into `prepare_hybrid_apc_request_for_execution`.
- Hybrid APC suffix restore consumes the scheduler-authorized key with the same
  request id before falling back to the old length-only diagnostic gate.

This is still guarded by the single-request backed-prefix condition. It removes
the queue-order assumption from the proven path and prepares the code for a
later batched/concurrent validation.

### 2026-05-18 Request-Scoped Runner Patch

The request-scoped restore handoff is now wired for the proven single-request
path:

- Scheduler-authorized prefix reads are stored by exact vLLM request id when
  the scheduler request exposes one.
- The vLLM-Neuron runner import hook patches `_execute_model_for_text` and
  temporarily exposes `model_input.request_ids` on both the runner model wrapper
  and the nested Qwen model.
- Qwen request prep forwards that request id as `hybrid_request_id`.
- Hybrid APC suffix restore consumes the authorized key using the same request
  id.

One validation run caught the first implementation mistake: the request id was
only attached to the Neuron wrapper, while Qwen `_get_model_outputs` runs on the
nested model. That produced the old fallback request id:

```text
request_id=('seq_id', 0)
ValueError: suffix-only hybrid APC received an attention prefix hit without scheduler-authorized GDN checkpoint metadata
```

The fix attaches/restores the temporary request-id attribute on the wrapper and
nested `.model` chain.

After that fix, the 2K BF16 backed-prefix boundary validation passed without
`QWEN36_HYBRID_APC_ALLOW_UNHASHED_SINGLE_PREFIX_RESTORE`:

```text
full_prefix_exact=true
partial_prefix_exact=true
real_generated_tokens_passed=true
cold_partial elapsed: 1.9467s
warm_partial elapsed: 1.3388s
```

Key debug proof:

```text
Installed Qwen Hybrid APC vLLM-Neuron runner patch
scheduler-decision backed_hit_len=256 supports_backed=True prompt_len=272 registry_size=1
apply-suffix prompt_len=272 restore_len=256 suffix_len=16 restore_slot=0 input_shape=(1, 16)
prepare request_id='3-880fa588' attention_hit_len=256 request_prefix_len=272 restore_len=256
computed=tensor([[256]]) num_queries=tensor([[16]]) restore_mask=tensor([1])
qwen-cte-call input_shape=(1, 16) position_minmax=256:271 computed=[256] restore_mask=[1]
```

Artifacts:

- JSON:
  `/home/ubuntu/validation_logs/hybrid_apc_real_tokens/bf16_hybrid_apc_backed_prefix_d061df5_request_scoped_nested_model_no_unhashed.json`
- Log:
  `/home/ubuntu/validation_logs/hybrid_apc_real_tokens/bf16_hybrid_apc_backed_prefix_d061df5_request_scoped_nested_model_no_unhashed.log`

The remaining production work is not the single-request restore identity
anymore. It is proving and safely enabling this through batched/concurrent
serving, where `max_num_seqs > 1` is still intentionally disabled for backed
prefix reads.

### 2026-05-18 Batched Validation Prep

The next batched/concurrent slice added two safety pieces:

- Core Hybrid APC request prep now explicitly handles vectorized no-hit
  metadata. If a multi-request batch has `computed_context_lens=[0, ...]`, it
  bypasses Hybrid APC prep and lets the scheduler fallback run as normal. If
  any vectorized request has a nonzero attention hit, it fails fast because
  vectorized GDN restore is not wired yet.
- `validation_scripts/qwen36_hybrid_apc_validation.py` now has
  `batched-exactness`, which warms two distinct prefixes and then submits two
  partial prompts in a single `llm.generate([...])` group with configurable
  `--max-num-seqs`.

Focused local and remote tests for this slice:

```text
101 passed
```

The first full batched E2E attempt against the current 2K BF16 artifact did not
reach Hybrid APC restore. It failed in token generation because the artifact was
compiled for single-request TKG:

```text
sampling_params shape: [2, 3]
compiled input_shape_map only has sampling_params shape: [1, 3]
tkg_batch_size=1
max_num_seqs=2
```

The validation harness now preflights this and fails clearly:

```text
ValueError: batched generation requires a compiled artifact with
tkg_batch_size >= --max-num-seqs; got tkg_batch_size=1 and max_num_seqs=2
```

This means a real generated-token batched E2E proof needs either:

- a 2K BF16 artifact compiled with `tkg_batch_size >= 2` / compatible
  `max_num_seqs=2`, or
- a separate prefill-only batched validator that does not enter TKG.

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

The single-request backed performance path is now proven:

```text
attention KV prefix hit exists
matching GDN checkpoint exists
model restores GDN state and runs only the suffix
```

It is guarded to `max_num_seqs=1` because batched/concurrent serving still
needs vectorized restore prep and an artifact or harness that can prove multiple
scheduled requests.

## Overnight Operating Rule

If the same failure mode or error repeats more than twice, stop local trial-and-error and search the current NVIDIA/vLLM implementation or documentation before continuing. Compare the Neuron contract against NVIDIA/vLLM for:

- paged attention block tables
- slot mapping
- prefix caching
- Mamba/SSM or other hybrid state cache handling
- prefill/decode runner argument order
- sampler/logits contract

Then record the finding in this file before applying the next patch or starting another compile.

## Confirmed Validation

Local and remote focused tests after the scheduler/additional-config fix:

```text
59 passed
```

Remote 2K BF16 boundary validation after the scheduler/additional-config fix:

```text
full_prefix_exact=true
partial_prefix_exact=true
real_generated_tokens_passed=true
backed_hit_len=256 supports_backed=True
apply-suffix prompt_len=272 restore_len=256 suffix_len=16
```

Remote 2K BF16 boundary validation after scheduler-authorized key restore,
with `QWEN36_HYBRID_APC_ALLOW_UNHASHED_SINGLE_PREFIX_RESTORE` unset:

```text
full_prefix_exact=true
partial_prefix_exact=true
real_generated_tokens_passed=true
backed_hit_len=256 supports_backed=True
apply-suffix prompt_len=272 restore_len=256 suffix_len=16
```

Local focused tests after the single-request guard:

```text
61 passed
```

Remote scheduler subset after the single-request guard:

```text
13 passed
```

Local focused tests after additional-config key metadata hardening:

```text
62 passed
```

Remote scheduler subset after additional-config key metadata hardening:

```text
14 passed
```

Local focused tests after batched debug-override guard:

```text
63 passed
```

Remote scheduler subset after batched debug-override guard:

```text
15 passed
```

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

Local focused tests after `02e636c`:

```text
46 passed
```

Remote focused tests after `02e636c`:

```text
46 passed
```

The CTE-prefix implementation work after `02e636c` has remote focused unit coverage:

```text
90 passed
```

That covers:

- Prefix-cache bucket/padding behavior, including the 16-token suffix plus 256-token Hybrid APC restore case.
- Qwen compile config flag forwarding.
- vLLM serving config flag forwarding.
- Scheduler backed-prefix gating.
- Hybrid APC manager registry publish/unpublish behavior.

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

After `02e636c`, the same 2K checkpoint-boundary prompt also passes on the existing artifact without compiling:

```text
full_prefix_exact=True
partial_prefix_exact=True
real_generated_tokens_passed=True
```

Artifacts:

- JSON:
  `/home/ubuntu/validation_logs/hybrid_apc_real_tokens/bf16_hybrid_apc_host_logits_nki_chunked_02e636c_scheduler_backed_gate_boundary_decode24.json`
- Log:
  `/home/ubuntu/validation_logs/hybrid_apc_real_tokens/bf16_hybrid_apc_host_logits_nki_chunked_02e636c_scheduler_backed_gate_boundary_decode24.log`

The log confirms the scheduler safety gate avoided the backed path on this artifact:

```text
attention_hit_len=0
restore_len=0
computed=tensor([[0]], dtype=torch.int32)
```

So correctness is protected for both the normal validation prompt and the checkpoint-boundary prompt, but this is still a no-prefix fallback and not the final perf path.

The same no-compile checkpoint-boundary validation also passes from clean commit `d6df06a`:

```text
full_prefix_exact=True
partial_prefix_exact=True
real_generated_tokens_passed=True
```

Artifacts:

- JSON:
  `/home/ubuntu/validation_logs/hybrid_apc_real_tokens/bf16_hybrid_apc_host_logits_nki_chunked_d6df06a_cte_prefix_contract_boundary_decode24.json`
- Log:
  `/home/ubuntu/validation_logs/hybrid_apc_real_tokens/bf16_hybrid_apc_host_logits_nki_chunked_d6df06a_cte_prefix_contract_boundary_decode24.log`

The `d6df06a` debug log still shows the intended safety behavior on the old artifact:

```text
attention_hit_len=0
restore_len=0
computed=tensor([[0]], dtype=torch.int32)
```

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

`02e636c` tightened the scheduler safety gate:

- A registered GDN checkpoint still proves that the GDN side is backed.
- vLLM attention prefix reads are only allowed when the compiled artifact also advertises backed CTE attention-prefix support.
- The scheduler checks `hybrid_apc_enable_backed_prefix_reads=True` and `use_qwen_hybrid_chunked_prefill=True` before exposing a backed prefix read.
- `QWEN36_HYBRID_APC_ENABLE_BACKED_PREFIX_READS=1` remains a debug override.
- This prevents the current artifact from taking the known-wrong "GDN restored, attention prefix missing" path.

The CTE-prefix implementation patch adds the pieces needed for the next artifact:

- Hybrid APC restore padding now treats `input_ids`, `attention_mask`, `position_ids`, and `slot_mapping` as suffix-only tensors.
- The bucket selector no longer applies the 257-512 token no-prefix corner case or empty-prefill prefix stealing when `hybrid_restore_mask=1`.
- The failing 16-token suffix plus 256-token prefix case now pads to `[prefill_bucket=256, prefix_bucket=256]`.
- Qwen CTE full-attention layers can treat selected prefix KV blocks as a logical prefix and concatenate current suffix K/V for attention.
- Padded suffix K/V positions are masked out in that selected-prefix attention path.
- Backed prefix reads remain opt-in through `hybrid_apc_enable_backed_prefix_reads`; the default remains safe.

## Exact Current Problem

The exact current problem is:

```text
Correctness is protected by the scheduler fallback when no matching GDN
checkpoint exists.

The scheduler-gated, request-scoped backed-prefix path now works for the
single-request boundary case: vLLM reuses the 256-token attention prefix, Qwen
restores the scheduler-authorized GDN checkpoint key for the same request id,
runs the 16-token suffix, and cold/warm outputs match.

The remaining production problem is batched/concurrent serving. Backed prefix
reads are still disabled when scheduler `max_num_seqs != 1`, so the next proof
must show the request-scoped restore identity and vectorized metadata contract
are correct for multiple live requests before that guard is relaxed.
```

The proven backed run uses this contract:

```text
suffix input length: 16
restored prefix length: 256
computed_context_lens: [256]
num_queries: [16]
slot_mapping: suffix slots only
block_table: prefix plus suffix physical blocks
runtime config: use_qwen_hybrid_chunked_prefill=True
restore_mask: [1]
scheduler restore identity: exact authorized GDN prefix key
```

The current code fixes the known wrapper/model/scheduler contracts in code and
focused unit tests:

```text
16-token suffix plus 256-token restored prefix pads as prefill_bucket=256, prefix_bucket=256
Hybrid APC restore tensors stay suffix-only through padding
Qwen CTE full-attention layers can consume selected prefix KV
backed prefix reads remain opt-in
the scheduler gate honors serving additional_config, not only hf_config
the scheduler registry key also honors serving additional_config metadata
```

The next required proof is batched/concurrent serving on an artifact or harness
that can actually execute more than one scheduled request. The single-request
boundary path already passes without
QWEN36_HYBRID_APC_ALLOW_UNHASHED_SINGLE_PREFIX_RESTORE, and vectorized no-hit
fallback is now unit-covered.

Primary files changed or relevant:

- `src/neuronx_distributed_inference/modules/async_execution.py`
- `src/neuronx_distributed_inference/models/model_wrapper.py`
- `contrib/models/Qwen3.6-27B/src/modeling_qwen35.py`
- `contrib/models/Qwen3.6-27B/vllm/qwen36_hybrid_apc_scheduler_patch.py`
- `contrib/models/Qwen3.6-27B/vllm/run_offline_inference.py`
- `validation_scripts/qwen36_hybrid_apc_validation.py`

## Why The Earlier Artifacts Did Not Work

The earlier artifacts failed for different reasons:

- Fused BF16 CTE artifact: numerical failure/NaNs around 105-106 tokens.
- FP8 path: still needs BF16 comparison before chasing FP8-specific NaNs.
- Old warm Hybrid APC runs: reused attention KV without matching GDN recurrent/conv state, so warm logits drifted.
- Existing BF16 per-chunk artifact: works for correctness fallback and the
  single-request backed restore proof, but it cannot prove generated-token
  batched serving because its TKG trace is batch 1.

The existing BF16 per-chunk artifact was generated before the full
backed-prefix and batched proof path existed. It can prove the safety fallback
and single-request backed path, but not generated-token `max_num_seqs=2`.

## Recommended Next Work

1. Produce a runnable batched/concurrent proof. The current validation harness
   has a `batched-exactness` mode, but the current BF16 artifact cannot run it
   with generated tokens because `tkg_batch_size=1`. Either compile a small 2K
   BF16 artifact with `tkg_batch_size >= 2` / `max_num_seqs=2`, or add a
   prefill-only batched validator that avoids TKG.
2. Keep the current scheduler rule: vLLM prefix reads are allowed only when a
   matching GDN checkpoint exists and the runtime config advertises backed CTE
   prefix support.
3. After the batched generated-token or prefill-only proof exists, decide
   whether to relax the `max_num_seqs == 1` backed-prefix guard or keep
   fallback-only behavior for multi-request serving.
4. Keep rerunning the 2K checkpoint-boundary validation without
   `QWEN36_HYBRID_APC_ALLOW_UNHASHED_SINGLE_PREFIX_RESTORE`.
5. Expected backed-prefix debug stays:

```text
attention_hit_len=256
restore_len=256
computed=[256]
num_queries=[16]
restore_mask=[1]
prefill_bucket=256
prefix_bucket=256
```

6. If backed BF16 exactness passes, measure cold vs warm prefill performance.
7. If backed BF16 exactness fails, capture the first failing token/logit stage and inspect CTE prefix K/V selection before compiling again.
8. If the same failure repeats more than twice, search NVIDIA/vLLM implementation details and compare contracts before continuing.
9. After BF16 backed restore exactness and perf are understood, revisit FP8 and TKG/on-device sampling separately.

The next concrete engineering target is a runnable batched/concurrent proof.
The old compile/prove target is done for the 2K single-request BF16 boundary
case, but the current BF16 artifact cannot run generated-token
`max_num_seqs=2` validation because its TKG trace is batch 1.

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

The compiled CTE input contract now honors the single-request backed hit during
suffix-only execution. The remaining Neuron/vLLM gap is vectorized
batched/concurrent restore handling, plus a generated-token artifact with
`tkg_batch_size >= 2` or a prefill-only proof that avoids TKG.

References:

- vLLM PagedAttention: https://docs.vllm.ai/en/stable/design/paged_attention/
- vLLM prefix caching: https://docs.vllm.ai/en/stable/design/prefix_caching/
