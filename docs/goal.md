# Qwen3.6 Hybrid APC Goal State

## Objective

Make Qwen3.6-27B Hybrid APC on Trainium correct first, then measure cold-prefill performance. Correctness means:

- BF16 host-logits path emits finite real-token outputs.
- Cold and warm Hybrid APC outputs match for tested prompts.
- Attention KV prefix reuse is only used when matching GDN recurrent/conv checkpoint state is available.
- Unsupported scheduler/cache states fail fast or fall back to no-prefix execution instead of silently producing wrong tokens.

## Current Status

The active branch is `experimental`.

Current useful hosts and paths:

- Key: `/Users/deepankarsingh1312/Downloads/trainium.pem`
- Trn2 runtime host: `ubuntu@16.26.98.193`
- r7i compile host: `ubuntu@16.26.249.227`
- Remote repo on both hosts: `/home/ubuntu/inferentia-gdn-experimental-test`
- Weights on Trn2/r7i: `/home/ubuntu/models/Qwen3.6-27B`
- Current copied BF16 Hybrid APC artifact on Trn2:
  `/mnt/trainium_artifacts/qwen_artifacts/qwen36_27b_2048_bf16_hybrid_apc_backed_prefix_ctx2_tkg2_r7i_trn2_local_7306c2e`
- Latest copied-artifact validation logs:
  `/home/ubuntu/validation_logs/hybrid_apc_real_tokens/bf16_hybrid_apc_ctx2_tkg2_bucket_pad_probe_pathfix_20260518T215320Z.log`
  `/home/ubuntu/validation_logs/hybrid_apc_real_tokens/bf16_hybrid_apc_batched_ctx2_tkg2_bucket_pad_validation_simple_20260518T215629Z.log`

Current result:

```text
multi-CTE ctx2/tkg2 compile: passed on r7i
artifact copy to Trn2: passed
Trn2 artifact load: passed
vectorized Hybrid APC request prep: now runs past the old one-request guard
mixed cached-decode + prefill CTE padding: now pads [2,511] to compiled [2,512]
slot_mapping/block_table/seq_ids rank repair: passed the static Neuron checks
short ctx2/tkg2 probe with commit disabled: finite raw logits and exit 0
batched grouped Hybrid APC validation: no static CTE shape miss after padding fix
batched grouped Hybrid APC validation with commit enabled: fails on all-NaN logits
```

Current blocker:

```text
BF16 host-side logits from the compiled Hybrid APC ctx2/tkg2 Neuron Qwen graph
are all NaN. This is already true at NxDI raw output 0, before vLLM-Neuron
slices logits and before vLLM sampling.

nxdi_raw_output_debug count=1
raw_output[0] shape=(1, 1, 248320)
finite=0/248320 nan=248320

runner_hidden_states_before_prepare shape=(1, 248320)
finite=0 nan=248320

runner_logits_after_prepare shape=(1, 248320)
finite=0 nan=248320
```

The no-Hybrid BF16 host-logits control is finite, and the same Hybrid APC
artifact produces finite raw logits when GDN checkpoint commit is disabled.
So the remaining NaN issue is not the generic Qwen host-logits path, vLLM
sampling, vLLM-Neuron output slicing, or the static CTE bucket selector.

The request-prep and bucket-shape contract now get through validation,
including short batched CTE padded from `[2,16]` to `[2,256]` and grouped
batched CTE padded to compiled `[2,512]`. The remaining issue is specific to
the compiled Hybrid APC GDN checkpoint commit/restore contract that produces
raw all-NaN logits and, for two scheduled requests, only one logits row.

The most likely source is the traced GDN checkpoint commit path. The existing
compiled artifact still contains the old checkpoint-bank scatter behavior, so
a real commit-path fix requires patching the model code and compiling a new
artifact; it cannot be fully fixed by runtime Python padding against the
already-compiled NEFF.

### 2026-05-18 Batched CTE Bucket Padding Fix

The runtime wrapper now pads batched prefix-cache CTE inputs at the final
`ModelWrapper.pad_inputs` boundary instead of returning early for batch > 1.
It selects the target prefill/prefix bucket from the max active/prefix lengths
across rows and synthesizes the prefix attention mask for mixed rows.

Validation against the existing ctx2/tkg2 artifact:

```text
short probe:
  qwen-cte-call input_shape=(2, 16)
  pad-post padded_input_shape=(2, 256)
  raw_output[0] finite=248320/248320 nan=0
  infer_exit:0

batched grouped validation:
  qwen-cte-call input_shape=(2, 512)
  pad-post padded_input_shape=(2, 512)
  no "Input shape not found" static Neuron error
```

The batched validation still fails after the shape fix:

```text
raw_output[0] shape=(1, 1, 248320)
finite=0/248320 nan=248320
request_ids=['2-b99eee03', '3-897b6f9b']
prefill_completion_state=tensor([ True, False])
IndexError: index 1 is out of bounds for dimension 0 with size 1
validation_exit:1
```

Interpretation:

```text
Fixed:
  static batched CTE bucket mismatch ([2,16]/[2,511] reaching Neuron)

Still blocked:
  compiled Hybrid APC checkpoint commit path corrupts/returns all-NaN logits
  compiled graph returns one logits row for two scheduled rows

Applied code fix before recompile:
  HybridGDNCheckpointCache.commit_from_active_rows no longer uses scatter over
  all padded rows. It writes only rows enabled by commit_mask, so duplicate
  padded slot IDs such as commit_slot_ids=[0, 0] with commit_mask=[1, 0] cannot
  let an inactive padded row overwrite the active checkpoint row.

Validation:
  local qwen36 alias tests: 17 passed
  remote qwen36 alias tests on Trn2: 17 passed
  remote model wrapper tests on Trn2: 34 passed
  local async_execution tests: 33 passed

Remaining required step:
  compile a fresh Hybrid APC artifact with the commit-path fix. The existing
  ctx2/tkg2 artifact still has the old scatter behavior baked into its NEFF, so
  it can prove runtime padding but cannot prove the checkpoint commit fix.
```

### 2026-05-18 No-Hybrid BF16 Host-Logits Control

A smallest useful no-Hybrid BF16 host-logits control artifact compiled on Trn2:

```text
/mnt/trainium_artifacts/qwen_artifacts/qwen36_27b_2048_bf16_hostlogits_nohybrid_cte256_tkg1_8269f27
compile_exit:0
artifact size: 51G
ctx_batch_size=1
tkg_batch_size=1
context_encoding_buckets=[256]
enable_hybrid_apc=false
enable_prefix_caching=false
enable_vllm_chunked_prefill=false
output_logits=true
```

Compile log:

```text
/home/ubuntu/validation_logs/host_logits_controls/bf16_hostlogits_nohybrid_cte256_tkg1_8269f27_20260518T210539Z.compile.log
```

The raw-output debug inference passed and produced finite logits:

```text
SWEEP_CASE repeats=0 tokens=16
raw_output[0] shape=(1, 1, 248320) dtype=torch.float32
finite=248320/248320 nan=0 posinf=0 neginf=0 finite_min=-6.3125 finite_max=20.0
logits shape=(1, 1, 248320) dtype=torch.float32
finite=248320/248320 nan=0 posinf=0 neginf=0 finite_min=-6.3125 finite_max=20.0
SWEEP_RESULT repeats=0 tokens=16 output_tokens=[220]
```

Inference log:

```text
/home/ubuntu/validation_logs/host_logits_controls/bf16_hostlogits_nohybrid_cte256_tkg1_8269f27_20260518T212554Z.infer.log
```

Interpretation:

```text
BF16 no-Hybrid raw logits finite:
  generic host-logits / vLLM CPU sampler path is not the blocker.

Hybrid APC ctx2/tkg2 raw logits all NaN:
  focus on Hybrid APC/chunked compile/runtime contract, not sampler slicing.
```

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

- a 2K BF16 artifact compiled with both `ctx_batch_size >= 2` and
  `tkg_batch_size >= 2` / compatible `max_num_seqs=2`, or
- a separate prefill-only batched validator that does not enter TKG.

### 2026-05-18 Batched Compile Prep

The compile helper now accepts the missing batch shape knobs:

```text
--max-num-seqs
--ctx-batch-size
--skip-warmup
```

It maps them into the compiled Neuron config as:

```text
batch_size=max_num_seqs
ctx_batch_size=ctx_batch_size
tkg_batch_size=max_num_seqs
pa_num_blocks=(seq_len / block_size * max_num_seqs) + 1 null block
```

The intended next artifact is a 2K BF16 host-logits Hybrid APC compile with
`--max-num-seqs 2`, `--ctx-batch-size 1`, `--skip-warmup`, and 17 physical PA
blocks for two 2048-token sequences at block size 256. This should remove the known
`sampling_params [2,3]` vs compiled `[1,3]` TKG mismatch and let
`batched-exactness` reach the actual Hybrid APC restore/fallback logic.

Focused local test for this compile-helper change:

```text
contrib/models/Qwen3.6-27B/test/unit/test_qwen36_compile_fp8_config.py
8 passed
```

### 2026-05-18 Batch-2 TKG Artifact Result

The first batch-2 artifact compiled and loaded:

```text
/mnt/trainium_artifacts/qwen_artifacts/qwen36_27b_2048_bf16_hybrid_apc_backed_prefix_tkg2_fa5f87e
batch_size=2
ctx_batch_size=1
tkg_batch_size=2
pa_num_blocks=17
skip_warmup=True
```

This removed the old `sampling_params [2,3]` vs compiled `[1,3]` TKG mismatch.
The batched generated-token validation then failed later in vLLM-Neuron
host-logits sampling:

```text
IndexError: index 1 is out of bounds for dimension 0 with size 1
File: /vllm/vllm_neuron/worker/neuronx_distributed_model_runner.py
Function: _prepare_logits_for_sampling
Line: return hidden_states[reorder_indices]
```

The important runtime clue is that the grouped prefill for two 26-token
requests was packed into one CTE row:

```text
qwen-cte-call input_shape=(1, 52)
sampling_params shape=(1, 3)
num_queries=[26]
computed=[0]
```

vLLM-Neuron then tried to reorder logits for two live request ids, but the model
output only had one row. This strongly indicates the generated-token batched
proof needs `ctx_batch_size >= 2` as well as `tkg_batch_size >= 2`, unless
vLLM-Neuron host-logits sampling is patched to split packed CTE logits. The
validation preflight now rejects `ctx_batch_size < --max-num-seqs` for batched
generated-token runs.

### 2026-05-18 Ctx2/TKG2 Artifact Attempt

The next artifact attempt is in flight:

```text
/mnt/trainium_artifacts/qwen_artifacts/qwen36_27b_2048_bf16_hybrid_apc_backed_prefix_ctx2_tkg2_4244d86
batch_size=2
ctx_batch_size=2
tkg_batch_size=2
pa_num_blocks=17
skip_warmup=True
```

Last verified log before SSH stopped completing banner exchange:

```text
START_CTX2_TKG2_COMPILE_AND_VALIDATE date=2026-05-18T05:06:10+00:00
CONTEXT_TRACE_SHAPE ... ctx_batch_size=2 ... max_num_seqs=2 ... tkg_batch_size=2
INFO:Neuron:generating HLO: context_encoding_model, input example shape = torch.Size([2, 256])
```

This confirms the second compile is tracing CTE with a two-row prefill batch,
unlike the previous `ctx_batch_size=1` artifact that packed two requests into
`input_shape=(1, 52)` and failed in host-logits reorder. The remote instance
became temporarily unreachable via SSH while this compile was running; recheck
the log/status before deciding whether to recompile or patch vLLM-Neuron.

Potential prefill-only fallback note:

```text
Current vLLM SamplingParams validation requires max_tokens >= 1.
```

So a prefill-only batched proof cannot be implemented as a simple
`LLM.generate(..., SamplingParams(max_tokens=0))` path. It would need a
lower-level vLLM/vLLM-Neuron runner hook, a prompt-logprobs path that is proven
not to enter TKG, or a custom model-execute validator.

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
- `contrib/models/Qwen3.6-27B/test/integration/qwen36_27b_compile_fp8.py`

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

## Current Limitations

What is proven:

- Single-request BF16 Hybrid APC backed-prefix correctness is proven on the
  checkpoint-boundary prompt.
- Safety fallback is proven: if vLLM has an attention KV prefix hit but Qwen
  has no matching GDN checkpoint, the scheduler disables prefix reads and
  cold/warm outputs match.
- Request-scoped restore identity is proven for the single-request path.
- Vectorized no-hit fallback is unit-covered.
- Single-bucket ctx2/tkg2 compiles have succeeded for 256-only and 512-only.
- Combined multi-bucket `cte_buckets=256,512` ctx2/tkg2 BF16 compile succeeded
  on the r7i compile host.
- The combined artifact is portable to Trn2, loads there, and completes
  single-request bucket-aligned CTE + TKG generation.
- The generated-token batched harness now runs through vectorized request prep,
  pads mixed `[2,511]` CTE rows up to the compiled `[2,512]` bucket, repairs
  slot/block/row metadata, and avoids the previous static Neuron shape failures.
- The latest grouped validation reaches sampling and proves the output reaching
  vLLM is already all NaN before sampling.
- The `NXDI_RAW_OUTPUT_DEBUG=1` run proves the first raw NxDI output tensor is
  already all NaN, so the NaNs are not introduced by vLLM-Neuron's
  `output.logits[:, -1, :]`/chunked-prefill slicing or by vLLM's CPU sampler.

What is not proven yet:

- Batched/concurrent backed-prefix serving is not real-token proven.
- The `max_num_seqs > 1` backed-prefix guard should not be relaxed yet.
- The final grouped `llm.generate([prompt_a, prompt_b])` path gets past the
  vectorized metadata and static-shape blockers, but fails the real-token gate
  because every sampled token is `0`.
- BF16 host-logits correctness is not proven; NxDI raw output and the runner
  both see all-NaN logits.
- Cold-prefill performance has not been measured for the final batched path.
- FP8 is not validated for this path and should not be used to debug the
  serving contract.
- Fused CTE is not the current correctness path because the fused BF16 artifact
  previously produced NaNs around token 105-106.

Current practical constraints:

- Generated-token batch-2 validation now has a copied artifact with both
  `ctx_batch_size=2` and `tkg_batch_size=2`.
- The artifact with `tkg_batch_size=2` but `ctx_batch_size=1` failed because
  vLLM-Neuron packed two prefills into one CTE row and host-logits sampling
  tried to index a missing second output row.
- The smaller Trainium instance can compile single CTE buckets, but the combined
  `cte_buckets=256,512` ctx2/tkg2 compile overloaded it. Use the r7i compile
  host for large CPU/RAM/disk compile work, then copy the finished artifact to
  Trn2.
- The current blocker is no longer request metadata preparation, artifact shape,
  vLLM-Neuron output slicing, or sampling. It is all-NaN BF16 host logits from
  the compiled Qwen graph/artifact contract.

## Recommended Next Work

1. Build/run the smallest BF16 host-logits control artifact that can answer whether
   raw logits are finite without the current Hybrid APC/chunked-CTE artifact
   contract. Prefer a single CTE bucket and `ctx_batch_size=1`/`tkg_batch_size=1`
   first to reduce compile load. If that is finite, the remaining problem is the
   Hybrid APC/chunked-CTE artifact path. If it is also NaN, the issue is the
   Qwen host-logits path or artifact contract more generally.
2. If a non-Hybrid control is too broad, build a debug Hybrid artifact that
   returns or captures an earlier finite check: after final norm, after
   `lm_head`, after `mask_padded_logits`, and before returning logits.
3. Also test a compile-time DeltaNet CTE backend switch before spending time on
   FP8: current artifacts trace the fused chunked path. A control compiled with
   the alternate chunked backend can separate fused-CTE numerical NaNs from
   host-logits/output-alias contract bugs. Use
   `--deltanet-cte-backend nki_chunked` on
   `contrib/models/Qwen3.6-27B/test/integration/qwen36_27b_compile_fp8.py` for
   that control.
4. Preserve the current scheduler rule: vLLM prefix reads are allowed only when a
   matching GDN checkpoint exists and the runtime config advertises backed CTE
   prefix support.
5. Keep the vectorized CTE bucket-padding and row-repair tests in place; they
   cover the `[2,511]` to `[2,512]` fix, short slot mapping repair, and stale
   `seq_ids` repair.
6. Rerun the copied-artifact generated-token validation with
   `QWEN36_VLLM_LOGITS_DEBUG=1` after each logits/output-contract change.
7. Only after that generated-token proof passes with finite real tokens, decide
   whether to relax the
   `max_num_seqs == 1` backed-prefix guard or keep fallback-only behavior for
   multi-request serving.
8. Keep rerunning the 2K checkpoint-boundary validation without
   `QWEN36_HYBRID_APC_ALLOW_UNHASHED_SINGLE_PREFIX_RESTORE`.
9. Expected backed-prefix debug stays:

```text
attention_hit_len=256
restore_len=256
computed=[256]
num_queries=[16]
restore_mask=[1]
prefill_bucket=256
prefix_bucket=256
```

10. If backed BF16 exactness passes, measure cold vs warm prefill performance.
11. If backed BF16 exactness fails, capture the first failing token/logit stage
   and inspect CTE prefix K/V selection before compiling again.
12. If the same failure repeats more than twice, search NVIDIA/vLLM
    implementation details and compare contracts before continuing.
13. After BF16 backed restore exactness and perf are understood, revisit FP8 and
    TKG/on-device sampling separately.

The next concrete engineering target is no longer artifact production or
vectorized request prep. The combined 2K BF16 `ctx_batch_size=2` /
`tkg_batch_size=2` artifact exists, loads on Trn2, and reaches generation for the
grouped two-request path. The next target is finite BF16 raw logits from the
compiled Qwen graph.

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
suffix-only execution. The vectorized request-prep path now gets through the
previous metadata and static-shape blockers. The remaining Neuron/vLLM gap is
the host-logits/compiled-graph contract: raw NxDI output 0 is already all NaN,
while NVIDIA vLLM keeps model execution, logits computation, and sampling as
separate stable stages with explicit NaN accounting before bookkeeping.

## 2026-05-18 Trainium Recovery Note

The batch-2 ctx/tkg compile on `16.26.178.5` reached a useful intermediate
state before SSH became unresponsive:

```text
TKG priority compile completed successfully.
CTE batch-2 HLO compilation had started.
Validation had not started yet.
```

The instance was later stop/started and returned as `16.50.60.182`. The new
boot was reachable, with uptime about one minute, no active `neuronx-cc` or
validation processes, and root still nearly full:

```text
/dev/root 484G used 472G, avail 12G, 98%
/dev/nvme1n1 437.7G present, initially unmounted and empty
```

The previous compile script set Neuron cache and temp dirs to
`/mnt/trainium_artifacts`, but NxDI `ModelBuilder` still defaulted
`BASE_COMPILE_WORK_DIR` to `/tmp/nxd_model` on the nearly full root volume.
Patch the compile script to default `BASE_COMPILE_WORK_DIR` next to
`--compiled-path`, or pass it explicitly under `/mnt/trainium_artifacts`, before
restarting long compiles. This should reduce root pressure and make the run less
likely to starve SSH or wedge the guest.

AWS Neuron docs note that compilation worker count can affect host CPU/memory
pressure in supported stacks; this NxDI `ModelBuilder` version does not expose a
worker-count constructor argument, so the immediate mitigation is to keep all
large compiler work/cache/tmp paths off root and avoid aggressive SSH polling
while `neuronx-cc` is active.

## 2026-05-18 Staged Ctx2/TKG2 Compile Result

The current Trainium instance is:

```text
ubuntu@16.50.122.105
```

The 2K BF16 ctx2/tkg2 compile was split by CTE bucket to reduce compiler
pressure and warm the Neuron cache before trying the combined artifact.

The 256-only stage completed successfully:

```text
/mnt/trainium_artifacts/qwen_artifacts/qwen36_27b_2048_bf16_hybrid_apc_backed_prefix_ctx2_tkg2_256only_e6493d3
status: success
```

The 512-only stage also completed successfully:

```text
/mnt/trainium_artifacts/qwen_artifacts/qwen36_27b_2048_bf16_hybrid_apc_backed_prefix_ctx2_tkg2_512only_e6493d3
status: success
size: 51G
```

This confirms that both individual ctx2/tkg2 CTE shapes can compile on the
current instance when the compile workdir/cache/temp paths are under
`/mnt/trainium_artifacts`.

The combined `256,512` artifact is now running:

```text
pid: 4495
artifact: /mnt/trainium_artifacts/qwen_artifacts/qwen36_27b_2048_bf16_hybrid_apc_backed_prefix_ctx2_tkg2_staged_e6493d3
log: /home/ubuntu/validation_logs/hybrid_apc_real_tokens/bf16_hybrid_apc_backed_prefix_ctx2_tkg2_staged_e6493d3_compile_validate.log
status: /home/ubuntu/validation_logs/hybrid_apc_real_tokens/bf16_hybrid_apc_backed_prefix_ctx2_tkg2_staged_e6493d3_compile_validate.status
```

Last confirmed combined-run state before SSH timed out during compilation:

```text
CONTEXT_TRACE_SHAPE ctx_batch_size=2 tkg_batch_size=2
context_encoding_buckets=[256,512]
prefix_buckets=[256,512]
Generating 6 hlos for key: context_encoding_model
Starting compilation for the priority HLO
```

Two later SSH probes timed out while the combined compile was still presumed
active:

```text
ssh: connect to host 16.50.122.105 port 22: Operation timed out
```

Local AWS CLI credentials are not valid for checking EC2 health from this
machine. Do not assume the process is dead from SSH alone; this is the same
symptom seen when `neuronx-cc` starves sshd during heavy compilation.

The user later reported that the instance was terminated. Treat the in-flight
combined compile and the `/mnt/trainium_artifacts` artifacts from that instance
as lost unless the underlying volume was explicitly preserved and reattached.
The durable state is the pushed `experimental` branch and this report.

Replacement instance setup started on:

```text
ubuntu@16.50.246.35
```

Confirmed setup state:

```text
repo: /home/ubuntu/inferentia-gdn-experimental-test
branch: experimental
commit: 8c694f6a46f17a467907bba0323d3ce736b20ad0
scratch: /mnt/trainium_artifacts on /dev/nvme1n1
scratch free: 408G
fstab: UUID mount added with nofail
weights target: /home/ubuntu/models/Qwen3.6-27B
weights source: Qwen/Qwen3.6-27B
weights download status: success
weights size: 52G
weights safetensors: 15
weights log: /home/ubuntu/validation_logs/setup/qwen36_weights_download.log
256-only compile status: success
256-only artifact: /mnt/trainium_artifacts/qwen_artifacts/qwen36_27b_2048_bf16_hybrid_apc_backed_prefix_ctx2_tkg2_256only_8c694f6
256-only artifact size: 51G
512-only compile status: success
512-only artifact: /mnt/trainium_artifacts/qwen_artifacts/qwen36_27b_2048_bf16_hybrid_apc_backed_prefix_ctx2_tkg2_512only_8c694f6
512-only artifact size: 51G
512-only cache note: TKG reused cached NEFF from the 256-only stage
active combined pid: 6725
active combined artifact: /mnt/trainium_artifacts/qwen_artifacts/qwen36_27b_2048_bf16_hybrid_apc_backed_prefix_ctx2_tkg2_staged_3557925
active combined log: /home/ubuntu/validation_logs/hybrid_apc_real_tokens/bf16_hybrid_apc_backed_prefix_ctx2_tkg2_staged_3557925_compile_validate.log
active combined status: /home/ubuntu/validation_logs/hybrid_apc_real_tokens/bf16_hybrid_apc_backed_prefix_ctx2_tkg2_staged_3557925_compile_validate.status
active combined validation json: /home/ubuntu/validation_logs/hybrid_apc_real_tokens/bf16_hybrid_apc_backed_prefix_ctx2_tkg2_staged_3557925_batched_exactness.json
```

Latest confirmed state on `16.50.246.35`:

```text
remote time: 2026-05-18T14:00:04+00:00
repo branch: experimental
repo commit: 3557925
scratch: /mnt/trainium_artifacts mounted, 305G free
weights: /home/ubuntu/models/Qwen3.6-27B, 52G
main compile process: pid 6729, nice 10, elapsed 29:41
neuronx-cc processes: 13
status file: compile_started
artifact size: 16K
validation json: not created yet
```

The combined run generated all CTE/TKG HLOs, then the priority TKG compile
passed:

```text
Generated all HLOs in 130.22113156318665 seconds
Starting compilation for the priority HLO
'token_generation_model' is the priority model with bucket rank 0
Compiler status PASS
Compilation Successfully Completed for model.MODULE_e2b47141f2b9df7d4250+63d0419b.hlo_module.pb
Done compilation for the priority HLO in 1152.990659236908 seconds
Done optimizing weight layout for all HLOs in 26.17424511909485 seconds
Starting compilation for all HLOs
```

After that point, three lightweight SSH probes timed out during banner
exchange:

```text
Connection timed out during banner exchange
Connection to 16.50.246.35 port 22 timed out
```

Do not treat that alone as proof that the compile failed. The last successful
read showed the compiler alive and actively compiling CTE HLOs. This is the
same access-layer symptom seen when `neuronx-cc` saturates the host enough that
`sshd` stops responding promptly. Avoid rebooting, killing, or starting another
compile unless AWS console shows the instance is failed/terminated or SSH stays
unreachable after a long backoff.

Follow-up check from the local machine at `2026-05-18T20:07:59+0530` still
timed out during SSH banner exchange:

```text
Connection timed out during banner exchange
Connection to 16.50.246.35 port 22 timed out
```

No compile or validation status could be read from the host during that check.
The last authoritative remote evidence remains the `2026-05-18T14:00:04+00:00`
read above.

Remote `py_compile` passed for:

```text
contrib/models/Qwen3.6-27B/test/integration/qwen36_27b_compile_fp8.py
validation_scripts/qwen36_hybrid_apc_validation.py
```

Next new-instance bootstrap should:

```text
1. Clone/fetch origin/experimental.
2. Recreate /mnt/trainium_artifacts on the large scratch volume.
3. Put TMPDIR/TMP/TEMP, NEURON_COMPILE_CACHE_URL, and --base-compile-work-dir under /mnt/trainium_artifacts.
4. Download or mount /home/ubuntu/models/Qwen3.6-27B.
5. Re-run the staged ctx2/tkg2 compile plan: 256-only, then 512-only, then combined 256,512 only if the first two succeed.
```

If the combined artifact succeeds, the same job immediately runs
`batched-exactness` with `max_num_seqs=2`, `ctx_batch_size=2`,
`tkg_batch_size=2`, host-side BF16 logits, and real-token generation. If it
fails or wedges again, the confirmed state is:

```text
single-bucket ctx2/tkg2 compile works
combined multi-bucket ctx2/tkg2 compile still overloads or misses cache
```

That would make the next practical path either a prefill-only batched proof
that avoids generated-token TKG, or a larger/more isolated compile box for the
combined artifact.

## 2026-05-18 Artifact Portability Plan

It is valid to compile the multi-CTE artifact on a larger or more isolated
instance, then copy the finished artifact back to the smaller inference
instance, as long as the artifact is compiled for the smaller instance's exact
runtime contract. The larger instance should only be used to absorb compile
CPU/RAM/disk pressure; it should not change the serving shape.

AWS Neuron documents that the one-time `neuronx-cc` compilation can be
performed on another EC2 instance or even outside EC2, and that NEFFs can be
distributed to an inference fleet. Runtime loading still validates the NEFF
version and hardware/operator compatibility, and load can fail if the NEFF
needs more NeuronCores or memory than the target instance has.

For this project, the compile box should produce the exact artifact intended
for the smaller inference box:

```text
target: trn2
tp_degree: 4
logical_nc_config: 2
max_num_seqs: 2
ctx_batch_size: 2
tkg_batch_size: 2
seq_len: 2048
cte_buckets: 256,512
prefix_buckets: 256,512
block_size: 256
runtime num_gpu_blocks_override / user PA blocks: 16
compiled physical PA blocks: include the helper's null-block adjustment if applicable
sampling mode: host logits / disable on-device sampling
Hybrid APC flags: backed prefix reads enabled, static hybrid cache disabled
```

Do not compile a wider artifact just because the compile instance is larger. If
the large box compiles `tp_degree=8`, a different logical NeuronCore layout, a
different chip family target, or a different host-logits/sampling contract, the
smaller box should be expected to reject it or run the wrong validation path.

Copy the complete compiled artifact directory back to the smaller instance, not
just the Neuron compile cache:

```text
/mnt/trainium_artifacts/qwen_artifacts/<artifact>/
  neuron_config.json
  compiled NEFF directories
  sharded weights / metadata
```

The compile cache is optional and only helps future compiles. Inference should
load from `--compiled-artifacts <artifact_path>` and should not invoke
`neuronx-cc` on the smaller instance.

Concrete large-compile/small-inference runbook:

```bash
export REPO=/home/ubuntu/inferentia-gdn-experimental-test
export MODEL=/home/ubuntu/models/Qwen3.6-27B
export SCRATCH=/mnt/trainium_artifacts
export ART=$SCRATCH/qwen_artifacts/qwen36_27b_2048_bf16_hybrid_apc_backed_prefix_ctx2_tkg2_largebox_$(cd "$REPO" && git rev-parse --short HEAD)

mkdir -p "$SCRATCH"/{qwen_artifacts,neuron_compile_cache,tmp}
export TMPDIR=$SCRATCH/tmp
export TMP=$SCRATCH/tmp
export TEMP=$SCRATCH/tmp
export NEURON_COMPILE_CACHE_URL=$SCRATCH/neuron_compile_cache

cd "$REPO"
python3 contrib/models/Qwen3.6-27B/test/integration/qwen36_27b_compile_fp8.py \
  --repo-root "$REPO" \
  --model-path "$MODEL" \
  --compiled-path "$ART" \
  --base-compile-work-dir "$SCRATCH/qwen_artifacts/_nxd_model_workdir" \
  --weight-dtype bf16_control \
  --seq-len 2048 \
  --cte-buckets 256,512 \
  --prefix-buckets 256,512 \
  --block-size 256 \
  --pa-num-blocks 16 \
  --tp-degree 4 \
  --logical-nc-config 2 \
  --max-num-seqs 2 \
  --ctx-batch-size 2 \
  --enable-prefix-caching \
  --enable-hybrid-apc \
  --enable-vllm-chunked-prefill \
  --disable-on-device-sampling \
  --disable-static-hybrid-cache \
  --gdn-checkpoint-interval 256 \
  --max-gdn-checkpoint-slots 8 \
  --gdn-recurrent-cache-dtype float32 \
  --gdn-conv-cache-dtype bfloat16 \
  --hybrid-cache-mode all \
  --hybrid-apc-enable-backed-prefix-reads \
  --skip-warmup
```

Then copy the full `$ART` directory to the smaller inference instance under the
same or another stable path. Validate there without running the compile helper:

```bash
P2=$(python3 -c 'print("System B: answer deterministically.\n" * 62, end="")')

QWEN36_HYBRID_APC_DEBUG=1 \
python3 validation_scripts/qwen36_hybrid_apc_validation.py batched-exactness \
  --model-path /home/ubuntu/models/Qwen3.6-27B \
  --compiled-artifacts /mnt/trainium_artifacts/qwen_artifacts/<copied-artifact> \
  --max-model-len 2048 \
  --seq-len 2048 \
  --cte-buckets 256,512 \
  --align-prompts-to-cte-buckets \
  --tensor-parallel-size 4 \
  --max-num-seqs 2 \
  --logical-nc-config 2 \
  --ctx-batch-size 2 \
  --block-size 256 \
  --gdn-checkpoint-interval 256 \
  --max-gdn-checkpoint-slots 8 \
  --gdn-recurrent-cache-dtype float32 \
  --gdn-conv-cache-dtype bfloat16 \
  --hybrid-apc-enable-backed-prefix-reads \
  --enable-vllm-chunked-prefill \
  --num-gpu-blocks-override 16 \
  --max-tokens 8 \
  --require-real-tokens \
  --skip-fp8-env \
  --shared-prefix-2 "$P2"
```

If the copied artifact fails to load on the smaller box, first inspect:

```text
NEFF version mismatch -> compiler/runtime version mismatch
unsupported hardware/operator -> wrong target family or runtime stack
insufficient NeuronCores -> tp_degree/logical_nc_config too wide for target
insufficient memory -> artifact shape is too large for target device memory
```

## 2026-05-18 R7i Multi-CTE Compile Result

The r7i compile host proved the previously blocked combined multi-CTE compile.
The run was read-only monitored from the local machine and completed
successfully:

```text
compile host: ubuntu@16.26.249.227
instance: r7i.48xlarge
repo: /home/ubuntu/inferentia-gdn-experimental-test
branch/commit: experimental / 7306c2e
model: /home/ubuntu/models/Qwen3.6-27B
status: success
exit: COMPILE_EXIT=0
completed remote check: 2026-05-18T18:31:00+0000
artifact size: 51G
artifact path: /mnt/trainium_artifacts/qwen_artifacts/qwen36_27b_2048_bf16_hybrid_apc_backed_prefix_ctx2_tkg2_r7i_trn2_local_7306c2e
```

The compiled artifact contains the expected full artifact files:

```text
model.pt
neuron_config.json
weights/tp0_sharded_checkpoint.safetensors
weights/tp1_sharded_checkpoint.safetensors
weights/tp2_sharded_checkpoint.safetensors
weights/tp3_sharded_checkpoint.safetensors
```

The exact compile contract was:

```text
target: trn2 via NEURON_PLATFORM_TARGET_OVERRIDE=trn2
PYTHONPATH: local src and local contrib Qwen model path before site-packages
weight dtype: bf16_control
seq_len: 2048
cte_buckets: 256,512
prefix_buckets: 256,512
block_size: 256
user pa_num_blocks: 16
compiled physical PA blocks: 17 with the helper null-block adjustment
tp_degree: 4
logical_nc_config: 2
max_num_seqs: 2
ctx_batch_size: 2
tkg_batch_size: 2
prefix caching: enabled
Hybrid APC: enabled
vLLM chunked prefill: enabled
on-device sampling: disabled
static hybrid cache: disabled
GDN checkpoint interval: 256
GDN checkpoint slots: 8
GDN recurrent cache dtype: float32
GDN conv cache dtype: bfloat16
Hybrid APC backed prefix reads: enabled
warmup: skipped
```

Important compile milestones from the log:

```text
HLO generation succeeded: 6 CTE HLOs + 1 TKG HLO in 73.521s
TKG priority compile: PASS, 1262.218s
weight layout optimization: 20.278s
all six CTE HLOs: PASS
all-HLO compile phase: 616.075s
final save/package: completed, COMPILE_EXIT=0
```

This resolved the compile-capacity blocker from the smaller Trn2 host for the
2K ctx2/tkg2 multi-bucket BF16 artifact. The next section records the follow-up
copy and runtime validation on Trn2. That run proved artifact portability and
single-request generation, and narrowed the remaining blocker to vectorized
Hybrid APC request-prep metadata rather than `NRT_EXEC_OOB`.

## 2026-05-19 Trn2 Copy And Runtime Result

The r7i artifact was copied to the new Trn2 instance with a direct private-IP
transfer:

```text
source: ubuntu@16.26.249.227:/mnt/trainium_artifacts/qwen_artifacts/qwen36_27b_2048_bf16_hybrid_apc_backed_prefix_ctx2_tkg2_r7i_trn2_local_7306c2e
destination: ubuntu@16.26.98.193:/mnt/trainium_artifacts/qwen_artifacts/qwen36_27b_2048_bf16_hybrid_apc_backed_prefix_ctx2_tkg2_r7i_trn2_local_7306c2e
method: rsync over private IP 172.31.46.4
bytes transferred: 54.32G
average transfer rate: 205.18 MB/s
destination size: 51G
destination free disk after copy: 316G
```

The copied artifact contains:

```text
model.pt
neuron_config.json
weights/tp0_sharded_checkpoint.safetensors
weights/tp1_sharded_checkpoint.safetensors
weights/tp2_sharded_checkpoint.safetensors
weights/tp3_sharded_checkpoint.safetensors
```

The Trn2 checkout was updated to `experimental` commit `cccdfde`, which adds
`--align-prompts-to-cte-buckets` to the validation harness. This was needed
because the static Neuron trace only accepts compiled CTE sequence shapes. The
first copied-artifact runtime run sent a 463-token prompt into a trace compiled
for 256/512 and failed with:

```text
Input shape [[2, 463], ...] not found in input_shape_map
```

The validation harness now tokenizes prompts and pads token IDs to the next CTE
bucket before calling vLLM. Focused tests passed locally and on Trn2:

```text
contrib/models/Qwen3.6-27B/test/unit/test_hybrid_apc_validation.py
9 passed
```

With prompt bucket alignment enabled and the second synthetic prefix shortened
to fit under 512 tokens, the final copied-artifact validation reached the real
current Hybrid APC blocker:

```text
run id: bf16_hybrid_apc_batched_ctx2_tkg2_copied_r7i_db2ee21_20260518T190201Z
status: failed:1
artifact load: success
cold_partial_a: prompt_len=512, completed real generation
cold_partial_b: prompt_len=512, completed real generation
warmup_full_a: prompt_len=512, completed real generation
warmup_full_b: prompt_len=512, completed real generation
final grouped warm_partial_a/warm_partial_b: failed
```

The final failure is not compile capacity, artifact portability, or the old
non-bucket prompt shape problem. It is the intentional v0 guard in
`prepare_hybrid_apc_request_for_execution`:

```text
ValueError: hybrid APC v0 request prep supports one request at a time;
vectorized continuous-batching metadata is not wired yet
```

That means the project has now proven:

```text
multi-CTE ctx2/tkg2 artifact compiles on r7i
the artifact is portable to Trn2
Trn2 loads the copied artifact
single-request bucket-aligned CTE + TKG generation works from the copied artifact
batched grouped Hybrid APC still fails at vectorized request-prep metadata
```

The next implementation step is vectorizing Hybrid APC request preparation so
`computed_context_lens`, `request_id`/sequence identity, prefix hashes/full
input IDs, restore slots, restore masks, restore prefix lens, commit slots, and
commit masks are prepared per scheduled request instead of assuming one active
request. This should mirror vLLM's GPU-side separation: the scheduler maintains
per-request logical KV block state and prefix-cache hits, then passes per-request
block/slot metadata to the runner rather than collapsing the batch into a single
request. vLLM's prefix caching design uses hash-addressed KV blocks, and its
PagedAttention path maps logical request blocks to non-contiguous physical KV
blocks; the Neuron path needs the same per-row contract for the GDN
restore/commit metadata.

Concrete implementation checklist for the next patch:

```text
1. In prepare_hybrid_apc_request_for_execution, replace the current vectorized
   metadata guard with per-row preparation for batch-size > 1.
2. Derive a stable per-row request identity from explicit scheduler metadata
   when available, falling back to seq_ids only when that remains valid across
   the full request lifecycle.
3. For each row, read that row's attention hit length, full prefix length, full
   input IDs, and cumulative prefix hashes.
4. Call the Qwen Hybrid APC bridge independently per row so restore/commit
   decisions are not collapsed to one request.
5. Materialize batched tensors with shape [batch]:
   hybrid_restore_slot_ids, hybrid_restore_mask,
   hybrid_restore_prefix_lens, hybrid_commit_slot_ids, hybrid_commit_mask.
6. Preserve the no-hit fallback path for rows that do not restore, while still
   allowing other rows in the same batch to restore or commit.
7. Add unit tests covering a mixed batch:
   one no-hit row, one backed-prefix-hit row, and two commit slots.
8. Rerun the copied-artifact Trn2 validation:
   bf16_hybrid_apc_batched_ctx2_tkg2_copied_r7i_db2ee21 with
   --align-prompts-to-cte-buckets.
```

References:

- vLLM PagedAttention: https://docs.vllm.ai/en/stable/design/paged_attention/
- vLLM prefix caching: https://docs.vllm.ai/en/stable/design/prefix_caching/
- vLLM SamplingParams: https://docs.vllm.ai/en/latest/api/vllm/sampling_params/
- AWS Neuron compiler: https://awsdocs-neuron.readthedocs-hosted.com/en/latest/compiler/neuronx-cc/api-reference-guide/
- AWS Neuron compiler FAQ: https://awsdocs-neuron.readthedocs-hosted.com/en/latest/compiler/neuronx-cc/faq.html
- AWS Neuron runtime troubleshooting: https://awsdocs-neuron.readthedocs-hosted.com/en/latest/neuron-runtime/nrt-troubleshoot.html
- AWS Neuron runtime config: https://awsdocs-neuron.readthedocs-hosted.com/en/latest/neuron-runtime/nrt-configurable-parameters.html
- AWS Transformers NeuronX compilation worker count: https://awsdocs-neuron.readthedocs-hosted.com/en/v2.25.0/libraries/transformers-neuronx/transformers-neuronx-developer-guide.html#compilation-worker-count-support
