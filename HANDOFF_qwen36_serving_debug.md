# Handoff — Qwen3.6-27B FP8 hybrid serving garbage on Trainium

**Date:** 2026-06-04  ·  **Branch:** `codex/nki-deltanet-multihead-cte`  ·  **Owner:** Deepankar

---

## Latest session note — 2026-06-04/05 coherence-speed work

This section supersedes older conclusions below where they conflict. Do not
restart the same bisects unless a new artifact changes one of the facts here.

### Morning decision ledger — do not loop back here

This is the current coherence-speed ledger as of 2026-06-05. It records what
was actually tried, what it proved, and what remains.

Primary target:

- Restore coherent Qwen3.6 FP8 serving while preserving the fast QKV-NKI /
  CTE2048 cold-prefill path as much as possible.
- Required validation before calling the fix done: short prompts, exact boundary
  prompts, restored-prefix suffix prompts, multi-turn chat, serve-log scan for
  invalid token fallback/logit NaNs, and only then usage-accounted 16k prefill
  speed.

Cleared or deprioritized causes:

- Not a simple output alias order bug. The output order was verified as
  `[sample_token(int32), logits(fp32), KV..., deltanet_state..., checkpoint...]`;
  token/logits order is not swapped.
- Not just BF16 recurrent precision. CPU chunk gates passed through multiple
  chunk boundaries with bf16 round-trip tolerance, and the observed failure is a
  hard all-NaN cliff/sentinel-token path rather than gradual drift.
- Not standalone GDN fused math at the tested scale. The fused GDN validator and
  segmented GDN path were finite/coherent in isolation and for the first CTE2048
  chunk.
- Not monolithic cold segmented attention for the first 2048-token chunk. The
  full segmented-attention CTE2048 artifact returned coherent output at exact
  2048.
- Not a reason to keep chasing repeated `SCAN_STEPS`/solve-only compiles. `S=7`
  and `kkt_hier` are useful for exact/stable GDN chunk solve, but the remaining
  runtime evidence points at graph composition or the FP8 moat.

Proven failures:

- Full-FP8 CTE2048 with QKV-NKI and older qknormrope/qkvgate style paths
  produced mojibake/garbage past the short boundary. The fast ~3.4k tok/s
  numbers came from incoherent CTE2048 artifacts, not from a verified coherent
  path.
- Segmented-attention CTE2048 + GDN segment 512 + fp32 recurrent compiled and
  served coherently for short prompts and exact 2048, but failed after the
  internal 2048 boundary:
  `2500 = 2048 committed prefix + 452 suffix`, followed by all-NaN logits:
  `finite=0/248320 nan=248320`.
- The 2049 one-token suffix path was separately misrouted as token generation.
  Python routing fixes made it non-fatal, but logs still showed missing
  authoritative Hybrid APC prep on the promoted second step. That path must not
  be confused with the 2500 all-NaN suffix graph.
- Zeroing the GDN restore mask was not a complete fix. Older probe scripts
  passed, but exact boundary probes still failed at cold/restored cases,
  including all-NaN logits at 2048 in the full-FP8 CTE2048 graph.

External implementation comparison:

- Qwen FlashQLA explicitly optimizes **GDN Chunked Prefill** and reports its
  speedup by fusing the chunked GDN workflow, not by removing chunking:
  https://github.com/QwenLM/FlashQLA
- NVIDIA Megatron exposes GDN as a chunked rule with `chunk_size=64`,
  `initial_state`, and `output_final_state` in
  `core.ssm.gated_delta_net.torch_chunk_gated_delta_rule`:
  https://docs.nvidia.com/megatron-core/developer-guide/latest/apidocs/core/core.ssm.gated_delta_net.html
- NVIDIA NeMo's Qwen3.5 CP wrapper recovers dense sequence order, runs causal
  conv1d plus the FLA gated-delta rule, then restores layout; the important
  invariant is correct dense-order state/sequence mapping:
  https://docs.nvidia.com/nemo/automodel/nightly/apidocs/nemo_automodel/nemo_automodel.components.models.qwen3_5_moe.cp_linear_attn.html
- vLLM's state-passing path keeps `final_states` in `torch.float32` and handles
  chunk/sequence boundary state explicitly:
  https://docs.vllm.ai/en/v0.10.2/api/vllm/model_executor/layers/mamba/ops/ssd_state_passing.html

Current interpretation:

- The GPU/NVIDIA pattern supports chunked GDN with fp32 state and streamed/paged
  prefix handling. It does **not** validate our exact Neuron CTE2048 full graph.
- The useful question is no longer "should long context be chunked?" It is:
  does CTE2048 fail because of the full-FP8 moat, or because the 2048 compiled
  graph shape is numerically/semantically unsafe even without FP8 cache/weights?

Current isolation compile:

- Compile host: `ubuntu@16.51.94.87`
- Runtime host: `ubuntu@16.51.179.180`
- PID: `595791`
- Driver:
  `/home/ubuntu/tmp_compile_qwen32k_bf16_qkvnki_segcte2048_gdnseg512.sh`
- Artifact:
  `/mnt/trainium_artifacts/qwen_artifacts/qwen36_27b_32768_bf16control_qkvnki_tiled_segmented_cte512_nki_decode_stable_probe32k_gdnseg512_gdnrecfloat32_sampletokens_outlogits_b256_cte2048_pfx16k_slots64_async_cteargmaxsafe_20260605T004451Z_kkt_hier_scan7`
- Compile log:
  `/home/ubuntu/validation_logs/fp8_256k_decode_nki/qwen36_27b_32768_bf16control_qkvnki_tiled_segmented_cte512_nki_decode_stable_probe32k_gdnseg512_gdnrecfloat32_sampletokens_outlogits_b256_cte2048_pfx16k_slots64_async_cteargmaxsafe_20260605T004451Z_kkt_hier_scan7_compile.log`
- Purpose: keep QKV-NKI, segmented CTE, external CTE2048, GDN segment 512, fp32
  recurrent, and `kkt_hier/scan7`, but remove the full-FP8 moat:
  `WEIGHT_DTYPE=bf16_control`, `ENABLE_KV_CACHE_QUANT=0`,
  `bf16_weights,bf16_lm_head,bf16_kv_cache`.
- Latest poll: process alive, 18 loose NEFF files under the workdir, workdir
  `1.1G`, artifact disk `89G` free, compiler workers active for token-generation
  buckets, no `COMPILE_DONE` yet, and no hard error markers. Only current log
  warnings were missing optional MoE blockwise imports:
  `No module named 'neuronxcc.nki._private.blockwise_mm'` and
  `No module named 'neuronxcc.nki._private.blockwise_mm_bwd'`.

Decision after this compile:

- If BF16-control CTE2048 is coherent at 146/485/1225/2048/2049/2500/4096, the
  full-FP8 moat is the active corruption source. Re-enable FP8 one component at
  a time: KV cache, lm_head, then dense weights/MLP/out-proj as separate
  compiles.
- If BF16-control CTE2048 still produces all-NaN logits, stop spending compile
  hours on CTE2048 as the default coherent target. Optimize the coherent CTE512
  chunked structure and then recover speed with tighter/fused chunks, which is
  closer to the FlashQLA/NVIDIA direction.
- If BF16-control passes short/cold but fails restored-prefix suffix, the next
  code target is the Hybrid APC handoff: one authoritative prep path for
  same-request continuations, exact GDN checkpoint row selection, and segmented
  prefix-read finite probes.

Profiling answer:

- Neuron profiling can help once there is a coherent candidate whose throughput
  needs optimization, or if we capture the exact failing graph with runtime
  inspection and finite probes. It is not the first diagnostic for all-NaN
  logits because a normal profile reports engine/DMA timing, not the first layer
  where hidden states become NaN.
- If we do profile this bug, use `NEURON_RT_INSPECT_ENABLE=1`,
  `NEURON_RT_INSPECT_DEVICE_PROFILE=1`, `NEURON_RT_INSPECT_OUTPUT_DIR=<dir>`,
  capture with `neuron-explorer capture --profile-nth-exec=2
  --enable-dge-notifs`, then query the NEFF/NTFF with neuron-explorer. This
  should be paired with explicit finite-check instrumentation around
  attention/GDN/lm_head, otherwise the profile will be too coarse.

New errors logged per `AGENTS.md`:

- Remote poll command:
  `ssh ubuntu@16.51.94.87 pgrep -af "neuronx-cc compile"` failed with
  `pgrep: only one pattern can be provided`. Root cause: SSH/remote argument
  splitting gave `pgrep` more than one pattern. Mitigation: reran as
  `ssh ubuntu@16.51.94.87 'pgrep -af neuronx-cc'`, which returned the active
  compiler workers.
- Remote grep command:
  `ssh ubuntu@16.51.94.87 grep -E "Finished Compilation|COMPILE_DONE|..."
  <compile.log>` failed because the alternation pattern was not preserved as one
  remote argument. Exact stderr included `bash: line 1: COMPILE_DONE: command
  not found`, `grep: Compilation: No such file or directory`, and
  `alias: <compile.log>: not found`. Root cause: quoting error in the polling
  command, not a compile failure. Mitigation: reran with the entire remote grep
  command quoted; it returned only optional MoE import warnings and no hard
  compile errors.

BF16-control compile failure and recovery plan:

- Compile PID `595791` reached `INFO:Neuron:Finished Compilation for all HLOs
  in 420.3451347351074 seconds`, then failed during final weight sharding.
- Exact log:
  `/home/ubuntu/validation_logs/fp8_256k_decode_nki/qwen36_27b_32768_bf16control_qkvnki_tiled_segmented_cte512_nki_decode_stable_probe32k_gdnseg512_gdnrecfloat32_sampletokens_outlogits_b256_cte2048_pfx16k_slots64_async_cteargmaxsafe_20260605T004451Z_kkt_hier_scan7_compile.log`
- Exact exception:
  `FileNotFoundError: No such file or directory:
  /home/ubuntu/models/Qwen3.6-27B/model-00011-of-00015.safetensors`.
- Failing stage:
  `model.compile -> shard_weights -> model_builder.shard_checkpoint ->
  checkpoint_loader_fn -> load_state_dict -> load_safetensors_sharded`.
- Root cause: both compile and runtime `/home/ubuntu/models/Qwen3.6-27B`
  directories had configs/tokenizer/index only and no raw
  `model-*-of-00015.safetensors` shards. The compile could build HLOs from
  config, but could not produce sharded runtime weights.
- Mitigation applied: after approval, removed only stale/generated compile
  artifacts on the compile host: the failed BF16-control artifact/workdir,
  stale qkvonly segcte host-sampling artifact/workdir, and the older expanded
  attention CTE2048 artifact/workdir. Disk recovered from about `87G` free to
  `161G` free on `/mnt/trainium_artifacts/qwen_artifacts`.
- Recovery launched: raw Hugging Face shard download on compile host with
  `huggingface_hub.snapshot_download`, PID `605589`, log
  `/home/ubuntu/validation_logs/fp8_256k_decode_nki/qwen36_raw_weights_download_20260605T0105Z.log`,
  PID file
  `/home/ubuntu/validation_logs/fp8_256k_decode_nki/qwen36_raw_weights_download_20260605T0105Z.pid`,
  destination `/home/ubuntu/models/Qwen3.6-27B`.
- Current download evidence: process alive, log shows `Fetching 16 files`, model
  directory grew to `30G`, and disk had about `131G` free after partial
  download.
- Automation updated: `monitor-qwen-gdnseg512-expattn-compile` now monitors the
  raw-weight download, reruns
  `/home/ubuntu/tmp_compile_qwen32k_bf16_qkvnki_segcte2048_gdnseg512.sh` after
  all 15 raw shards exist, and then validates/rsyncs/serves/probes the artifact
  if compile succeeds.

Recovery progress after the missing-shard failure:

- Raw Hugging Face download completed successfully:
  `DOWNLOAD_DONE path=/home/ubuntu/models/Qwen3.6-27B`.
- Verified all 15 raw safetensor shards are present in
  `/home/ubuntu/models/Qwen3.6-27B`, including the previously missing
  `model-00011-of-00015.safetensors`.
- No `.incomplete` files remained under the local Hugging Face download cache.
- Disk after raw shard recovery: model directory `52G`, root/artifact volume
  about `109G` free.
- Reran the same BF16-control compile driver:
  `/home/ubuntu/tmp_compile_qwen32k_bf16_qkvnki_segcte2048_gdnseg512.sh`.
- New compile:
  - PID: `608233`
  - Artifact:
    `/mnt/trainium_artifacts/qwen_artifacts/qwen36_27b_32768_bf16control_qkvnki_tiled_segmented_cte512_nki_decode_stable_probe32k_gdnseg512_gdnrecfloat32_sampletokens_outlogits_b256_cte2048_pfx16k_slots64_async_cteargmaxsafe_20260605T011731Z_kkt_hier_scan7`
  - Workdir:
    `/mnt/trainium_artifacts/qwen_artifacts/_nxd_model_workdir_32768_bf16control_qkvnki_tiled_segmented_cte512_gdnseg512_gdnrecfloat32_sampletokens_outlogits_b256_cte2048_cteargmaxsafe_20260605T011731Z_kkt_hier_scan7`
  - Compile log:
    `/home/ubuntu/validation_logs/fp8_256k_decode_nki/qwen36_27b_32768_bf16control_qkvnki_tiled_segmented_cte512_nki_decode_stable_probe32k_gdnseg512_gdnrecfloat32_sampletokens_outlogits_b256_cte2048_pfx16k_slots64_async_cteargmaxsafe_20260605T011731Z_kkt_hier_scan7_compile.log`
  - Env log:
    `/home/ubuntu/validation_logs/fp8_256k_decode_nki/qwen36_27b_32768_bf16control_qkvnki_tiled_segmented_cte512_nki_decode_stable_probe32k_gdnseg512_gdnrecfloat32_sampletokens_outlogits_b256_cte2048_pfx16k_slots64_async_cteargmaxsafe_20260605T011731Z_kkt_hier_scan7_env.txt`
- First rerun evidence: process alive; log shows
  `WEIGHT_DTYPE_MODE bf16_control`, `FP8_MODE disabled_bf16_control`,
  `enable_kv_cache_quant=false`, `prefix_cte_attention_backend=segmented_cte`,
  `prefix_cte_attention_segment_size=512`, `QWEN36_PREFIX_ATTENTION_IMPL=expanded`,
  and HLO generation has started.
- Automation updated again to monitor the new PID/log/artifact.

### Active objective

Make Qwen3.6 FP8 output coherent at short and long prompts while preserving the
fast CTE2048/QKV-NKI cold-prefill path as much as possible.

User constraints from this session:

- Use `codex/full-fp8-qwen36` and `codex/nki-deltanet-decode-step` only as branch
  references. Do not branch-hop into unrelated experiments for conclusions.
- Use EC2-to-EC2 `rsync` for artifact transfer.
- For compiles, always create a monitor automation.
- Follow `AGENTS.md`: every command/runtime/compile/profiling error needs
  concrete error text, context, current hypothesis, mitigation, and verification.

### Newest checkpoint — 2026-06-05 after segmented-attention CTE2048 validation

This supersedes the "Current live compile" subsection below. That compile is no
longer active.

Compiled and validated artifact:

- Compile host: `ubuntu@16.51.94.87`
- Runtime host: `ubuntu@16.51.179.180`
- Artifact:
  `/mnt/trainium_artifacts/qwen_artifacts/qwen36_27b_32768_fp8_full_lmheadfp8_kvfp8_qkvnki_tiled_segmented_cte512_nki_decode_stable_probe32k_gdnseg512_gdnrecfloat32_sampletokens_outlogits_b256_cte2048_pfx16k_slots64_async_cteargmaxsafe_20260604T224804Z_kkt_hier_scan7`
- Compile log:
  `/home/ubuntu/validation_logs/fp8_256k_decode_nki/qwen36_27b_32768_fp8_full_lmheadfp8_kvfp8_qkvnki_tiled_segmented_cte512_nki_decode_stable_probe32k_gdnseg512_gdnrecfloat32_sampletokens_outlogits_b256_cte2048_pfx16k_slots64_async_cteargmaxsafe_20260604T224804Z_kkt_hier_scan7_compile.log`
- Config: `prefix_cte_attention_backend=segmented_cte`,
  `prefix_cte_attention_segment_size=512`, external CTE bucket 2048,
  GDN fused segment 512, `QWEN36_DELTANET_SOLVE_MODE=kkt_hier`,
  `QWEN36_DELTANET_SOLVE_SCAN_STEPS=7`, fp32 recurrent/checkpoint,
  bf16 conv/checkpoint.
- Compile status: `Finished Compilation for all HLOs in
  459.30248832702637 seconds`, all 12 HLOs passed,
  `CHECKPOINT_BANK_WEIGHTS_ADDED` for tp0..tp3 with 48 recurrent
  `torch.float32` and 48 conv `torch.bfloat16`, and `COMPILE_DONE`.

Transfer and launch:

- EC2-to-EC2 transfer used agent forwarding and completed:
  `38,981,665,131 100% ... 0:02:43`.
- Runtime launch log for first validation:
  `/home/ubuntu/validation_logs/fp8_256k_decode_nki/qwen36_segcte2048_gdnseg512_serve_20260604T2320Z.log`.
- Cold 2500 restart log:
  `/home/ubuntu/validation_logs/fp8_256k_decode_nki/qwen36_segcte2048_gdnseg512_cold2500_serve_20260604T2328Z.log`.
- Boundary 2048/2049 restart log:
  `/home/ubuntu/validation_logs/fp8_256k_decode_nki/qwen36_segcte2048_gdnseg512_boundary2048_serve_20260604T2332Z.log`.

Validation results:

- `/tmp/tmp_bisect_probe3.py` on the segmented-attention CTE2048 artifact:
  prompt lengths 5, 123, 160, 485, and 1225 were coherent.
- `2500` failed with output beginning `\n\n<think>\n!!!!!!!!!`.
- Clean restart then `python /tmp/tmp_qwen36_boundary_probe.py --targets 2500
  --max-tokens 24` failed with:
  `"\n\n<think>\nHere's a!!!!!!!!!!!!!!!!!!"`.
- Clean restart then `python /tmp/tmp_qwen36_boundary_probe.py --targets
  2048,2049 --max-tokens 24`:
  `2048` was coherent; `2049` returned an API body without `choices` because
  the engine crashed.

Exact runtime evidence:

- `2048` first chunk commits cleanly:
  `finish-commit ... commit_prefix_len=2048 commit_slot=0`.
- `2500` is not a monolithic cold graph. vLLM splits it:
  first `prompt_len=2048 restore_len=0 suffix_len=2048`, then
  `apply prompt_len=2500 restore_len=2048 suffix_len=452 restore_slot=0`.
- The failing second chunk is a restored-prefix context graph:
  `prepared_shape=(1, 452) computed=tensor([[2048]]) num_queries=tensor([[452]])`,
  then padding selects `prefill_bucket=2048 prefix_bucket=2048`.
- Immediately after that suffix graph, logits are all NaN:
  `finite=0/248320 nan=248320 posinf=0 neginf=0 argmax=[0] argmax_values=[nan]`.
- `2049` is a separate scheduler/async routing bug. The scheduler emits a cached
  request with `num_computed_tokens=[2048]`, `num_scheduled_tokens=1`,
  metadata `vllm_attention_hit_len: 2048`, `request_prefix_len: 2049`,
  `active_suffix_len: 1`, and `full_input_ids`, but async execution routes to
  token generation and raises:
  `ValueError: Token generation input_ids must be non-empty`.

Conclusions that should not be re-bisected:

- Segmented attention fixes the first CTE2048 chunk: exact 2048 is coherent.
- The remaining long-context failure is after the internal 2048 boundary, when
  a restored 2048-token prefix and active suffix run together.
- There are now two distinct post-2048 issues:
  1. `2049` one-token suffix was misclassified as token generation.
  2. `2500`/suffix 452 enters context correctly but produces all-NaN logits.
- The active suspect for the all-NaN suffix is the restored-prefix composition:
  GDN checkpoint restore/carry and/or segmented attention prefix read at
  `prefix_bucket=2048`, not monolithic cold attention and not standalone GDN.

Code fix applied locally after this validation:

- File: `src/neuronx_distributed_inference/modules/async_execution.py`
- `_is_chunked_prefill_execution` now promotes cached chunked-prefill
  continuations even when the active suffix length is 1 or the sliced
  `input_ids` tensor is empty, as long as metadata says
  `request_prefix_len >= vllm_attention_hit_len + active_suffix_len` and prefill
  is not complete.
- `_is_same_request_chunked_prefill_continuation` no longer rejects
  `suffix_len == 1`; it still rejects zero/negative suffixes and true decode is
  protected by `hybrid_prefill_completion_state`.
- File: `test/unit/modules/test_async_execution.py`
- Added unit coverage for:
  - cached single-token unfinished prefill continuation routes as context;
  - completed single-token decode stays generation;
  - same-request one-token suffix prepares Hybrid APC with restore len 2048 and
    `num_queries=1`.

Verification:

- `PYTHONPATH=src python3 -m py_compile
  src/neuronx_distributed_inference/modules/async_execution.py` passed.
- `PYTHONPATH=src pytest test/unit/modules/test_async_execution.py -q` passed:
  `63 passed in 0.55s`.

Errors logged per `AGENTS.md`:

- Artifact verifier script bug: a first bounded verifier assumed top-level
  config keys and printed `backend None`, `segment None`, then failed with
  `TypeError: 'NoneType' object is not subscriptable`. Root cause: verifier
  used the wrong `neuron_config.json` shape, not an artifact problem.
  Mitigation: reran verifier against the nested config and safetensors; it
  confirmed `segmented_cte`, segment 512, CTE2048 bucket pairs, and 48/48
  checkpoint-bank dtypes in all tp shards.
- Runtime 2049 probe error: `python /tmp/tmp_qwen36_boundary_probe.py --targets
  2048,2049 --max-tokens 24` raised client-side `KeyError('choices')` because
  the server returned an error response. Server root error was
  `ValueError: Token generation input_ids must be non-empty` in
  `src/neuronx_distributed_inference/modules/async_execution.py:2080`.
  Mitigation: patched cached one-token chunked-prefill routing and added unit
  tests. Runtime verification still requires syncing code/relaunching; no new
  compile should be needed for this Python fix unless the serve image bundles
  the module into the artifact.
- Runtime 2500 coherence error: restored-prefix suffix pass at
  `prompt_len=2500`, `restore_len=2048`, `suffix_len=452`,
  `prefix_bucket=2048` produced all-NaN logits:
  `finite=0/248320 nan=248320`. Mitigation not yet applied; current best
  hypothesis is restored GDN checkpoint state versus segmented attention prefix
  read. Next useful test is a runtime debug launch toggling
  `QWEN36_DISABLE_HYBRID_GDN_RESTORE_COMMIT=1`/prefix-read controls or a
  finite-check instrumented build around restored-prefix suffix execution.
- Local test command error:
  `pytest test/unit/modules/test_async_execution.py -q` failed collection with
  `ModuleNotFoundError: No module named 'neuronx_distributed_inference'`.
  Root cause: local package path was missing. Mitigation: reran as
  `PYTHONPATH=src pytest test/unit/modules/test_async_execution.py -q`, which
  passed.

Docs/NVIDIA comparison current interpretation:

- NVIDIA/Megatron/vLLM/Qwen references agree on chunked long-context execution:
  the engine advances long prompts in chunks, attention uses paged/streamed
  prefix reads, and GDN-style state is carried explicitly in fp32. The
  comparable target on Neuron is not monolithic CTE2048 for the whole 256K
  context; it is chunked CTE with correct prefix/state handoff and fused kernels
  for speed.
- The segmented-attention CTE2048 result matches that: first chunk is coherent,
  so the remaining bug is the handoff into the next chunk.

Next recommended work, in order:

1. Sync the Python async-routing fix to the runtime repo and relaunch the current
   segmented-attention CTE2048 artifact; verify 2049 no longer crashes.
2. On a clean server, run 2500 with `QWEN36_DISABLE_HYBRID_GDN_RESTORE_COMMIT=1`.
   If all-NaN disappears, the restore/checkpoint path is the culprit. If all-NaN
   remains, turn off/limit backed segmented prefix reads next.
3. If toggles do not isolate it, add a finite-check instrumented graph around
   restored-prefix suffix execution. Full `neuron-explorer` profiling already
   proved too coarse for NaN origin without explicit finite probes.

### Follow-up runtime patch/probe results — 2026-06-05

Additional Python fixes tested locally and deployed to runtime:

- `src/neuronx_distributed_inference/modules/async_execution.py`
  - Materialize Hybrid APC scheduler metadata from base/context/token candidate
    owners before prefill-vs-generation routing.
  - Treat cached continuations as chunked prefill based on structural metadata:
    `request_prefix_len >= vllm_attention_hit_len + active_suffix_len`. The
    completion-state flag was stale/ambiguous at the 2049 route check.
  - Allow `suffix_len == 1` in same-request chunk continuation handling.
  - Add bridge-owner fallback and preserve `_hybrid_apc_last_bridge` across
    candidate owners after successful prepare.
  - Add debug-only `QWEN36_ZERO_HYBRID_GDN_RESTORE_MASK`, which zeros the model
    restore mask after plan creation without changing the Hybrid APC plan.
- `test/unit/modules/test_async_execution.py` now covers owner metadata
  materialization, wrapper bridge fallback, bridge reuse, 2049-style
  single-token continuation, and restore-mask-only debug behavior.
- Verification after latest local patch:
  `PYTHONPATH=src pytest test/unit/modules/test_async_execution.py -q` passed
  with `68 passed in 0.78s`; `PYTHONPATH=src python3 -m py_compile
  src/neuronx_distributed_inference/modules/async_execution.py` passed.

Runtime results:

- Before the structural route fix, 2049 still crashed with
  `ValueError: Token generation input_ids must be non-empty`.
- Diagnostic log
  `/home/ubuntu/validation_logs/fp8_256k_decode_nki/qwen36_segcte2048_gdnseg512_asyncdiag_boundary2049_serve_20260605T0050Z.log`
  showed scheduler metadata was present as a request record, but
  `probe_is_chunked_prefill=False` because completion state was not reliable.
- After switching to structural prefix math, 2049 no longer crashed. It returned
  a valid response with `completion_tokens=1`, but empty text.
- Logs from `asyncfix4` through `asyncfix7` show the promoted 2049 second step
  still lacks Hybrid APC prep:
  `async-prepared-return has_bridge=False has_prepared=False`, then
  `input_shape=(1, 0)` and `prefix_bucket=256`. So 2049 is non-fatal now, but
  not semantically fixed.

Errors logged per `AGENTS.md`:

- Runtime source inspection:
  `ssh ubuntu@16.51.179.180 'cd /home/ubuntu/inferentia-gdn-multihead-cte-20260531T1350Z && git rev-parse --abbrev-ref HEAD ...'`
  failed with `fatal: not a git repository (or any of the parent directories):
  .git`. Root cause: runtime tree is a copied source tree, not a Git checkout.
  Mitigation: inspected files directly.
- Boundary probe:
  `ssh ubuntu@16.51.179.180 'python /tmp/tmp_qwen36_boundary_probe.py
  --targets 2048,2049 --max-tokens 24'` failed with `bash: line 1: python:
  command not found`. Root cause: SSH PATH lacked the Neuron venv. Mitigation:
  reran after activating `/opt/aws_neuronx_venv_pytorch_inference_vllm_0_16`.
- Restore-disable isolation:
  `QWEN36_DISABLE_HYBRID_GDN_RESTORE=1` changed the plan to
  `restore_len=0`, `suffix_len=2500`, then failed bucket selection:
  `ValueError: Prefill len 2500 with prefix len 0 exceeds compiled 2D buckets
  for context_encoding_model; largest prefill bucket 2048, largest prefix bucket
  16384`.
  Log:
  `/home/ubuntu/validation_logs/fp8_256k_decode_nki/qwen36_segcte2048_gdnseg512_disable_gdn_restore_2500_serve_20260605T0205Z.log`.
- Restore-mask-only isolation:
  `QWEN36_ZERO_HYBRID_GDN_RESTORE_MASK=1` initially preserved the 2500 suffix
  plan:
  `restore_len=2048`, `suffix_len=452`, `computed=tensor([[2048]])`,
  `num_queries=tensor([[452]])`, `restore_mask=tensor([0])`.
  Then pad-time Qwen wrapper code re-entered Hybrid APC prep and raised:
  `ValueError: hybrid APC received an attention prefix hit without a matching
  GDN checkpoint; scheduler must intersect attention KV hits with GDN checkpoint
  hits or disable prefix reuse for this request`.
  Log:
  `/home/ubuntu/validation_logs/fp8_256k_decode_nki/qwen36_segcte2048_gdnseg512_zero_restore_mask_2500_serve_20260605T0220Z.log`.

Current conclusion:

- Exact 2048 first chunk remains coherent with short decode.
- 2049 route is non-fatal but still missing authoritative Hybrid APC prep on the
  promoted second step.
- 2500 all-NaN remains open. The next useful Python fix is to make same-request
  suffix continuations use one authoritative Hybrid APC preparation path and
  prevent pad-time re-preparation. Only after that are restore-mask or
  prefix-read toggles trustworthy for isolating GDN restore versus segmented
  attention prefix read.

### Current live compile

The current active compile is the expanded-attention replacement for the failed
GDN-segmented attempt:

- Automation: `monitor-qwen-gdnseg512-expattn-compile`
- Compile host: `ubuntu@16.51.94.87`
- Runtime host: `ubuntu@16.51.179.180`
- PID: `577557`
- Artifact:
  `/mnt/trainium_artifacts/qwen_artifacts/qwen36_27b_32768_fp8_full_lmheadfp8_kvfp8_qkvnki_tiled_attention_cte_nki_decode_stable_probe32k_gdnrecfloat32_sampletokens_outlogits_b256_cte2048_pfx16k_slots64_async_cteargmaxsafe_20260604T214000Z_gdnseg512_expattn_kkt_hier_scan7`
- Workdir:
  `/mnt/trainium_artifacts/qwen_artifacts/_nxd_model_workdir_32768_fp8_full_lmheadfp8_kvfp8_qkvnki_tiled_attention_cte_gdnrecfloat32_sampletokens_outlogits_b256_cte2048_cteargmaxsafe_20260604T214000Z_gdnseg512_expattn_kkt_hier_scan7`
- Compile log:
  `/home/ubuntu/validation_logs/fp8_256k_decode_nki/qwen36_27b_32768_fp8_full_lmheadfp8_kvfp8_qkvnki_tiled_attention_cte_nki_decode_stable_probe32k_gdnrecfloat32_sampletokens_outlogits_b256_cte2048_pfx16k_slots64_async_cteargmaxsafe_20260604T214000Z_gdnseg512_expattn_kkt_hier_scan7_compile.log`
- Env log:
  `/home/ubuntu/validation_logs/fp8_256k_decode_nki/qwen36_27b_32768_fp8_full_lmheadfp8_kvfp8_qkvnki_tiled_attention_cte_nki_decode_stable_probe32k_gdnrecfloat32_sampletokens_outlogits_b256_cte2048_pfx16k_slots64_async_cteargmaxsafe_20260604T214000Z_gdnseg512_expattn_kkt_hier_scan7_env.txt`
- Must-have env:
  `QWEN36_DELTANET_CTE_IMPL=current`,
  `QWEN36_DELTANET_MULTIHEAD_CTE=1`,
  `QWEN36_DELTANET_FUSED_SEGMENT_TOKENS=512`,
  `QWEN36_PREFIX_ATTENTION_IMPL=expanded`,
  `QWEN36_DELTANET_SOLVE_MODE=kkt_hier`,
  `QWEN36_DELTANET_SOLVE_SCAN_STEPS=7`,
  `NEURON_PLATFORM_TARGET_OVERRIDE=trn2`,
  `NEURON_CC_FLAGS=--target trn2 --lnc 2`,
  `NKI_LIBRARY_SRC=/home/ubuntu/nki-library-2.30/src/nkilib_src`,
  `USE_NKI_DECODE=1`.

At launch-time check, the compile had passed env/config validation, printed
`COMPILE_START`, and started generating 8 context HLOs. The command line includes
the corrected context bucket pairs through `2048:16384`.

What this compile tests:

- Keep external CTE bucket at 2048 for attention/QKV speed.
- Keep QKV-NKI tiled fast path.
- Keep `attention_cte`, not `segmented_cte`.
- Split only the GDN fused recurrence internally into 512-token staged calls with
  explicit state carry.
- Restore the working-branch expanded-KV chunked-prefill attention path with
  `QWEN36_PREFIX_ATTENTION_IMPL=expanded`, isolating the 5D grouped attention
  lowering as the remaining suspect.

Why this is the current best attempt:

- Known coherent path used smaller CTE512 chunks, but that also slowed attention
  and QKV path.
- Current CTE2048 failures happen in the full graph, not in the standalone GDN
  kernel.
- The prior GDN-segmented compile did not produce a coherence verdict because
  it failed at final checkpoint serialization with ENOSPC. The code path still
  needs a clean compile.
- If this expanded-attention build is coherent at 1225/2500, grouped
  chunked-prefill attention was the culprit. If it still NaNs cold at 1225, the
  issue is not segmented prefix attention or grouped attention alone; next step
  is layer-value instrumentation around GDN/attention/lm_head in the CTE2048
  graph.

External implementation comparison checked this session:

- Primary-source GPU references (`QwenLM/FlashQLA`, `fla-org/flash-linear-attention`,
  `NVlabs/GatedDeltaNet`, vLLM SSM state-passing docs) all point to chunked
  GDN prefill with explicit initial/final state handoff. FLA/FlashQLA optimize
  "GDN Chunked Prefill" rather than a single unbounded recurrence; Qwen
  FlashQLA reports 2-3x forward speedup over FLA by fusing the chunked GDN
  workflow, not by removing chunking.
- This supports keeping fp32 recurrent/checkpoint state and bounded GDN chunks.
  It does not prove Neuron `attention_cte` at 2048 is numerically safe, because
  the GPU attention side is normally flash/online-softmax/paged-KV rather than
  this exact Neuron full-graph lowering.
- Therefore the cheap decisive check is not another branch search: run the
  current CTE2048 build with GDN segmented to 512 and old expanded chunked
  prefill attention. If it passes, the new grouped attention lowering was the
  live NaN source. If it fails with all-NaN logits at cold 1225 again, profile
  or instrument the CTE2048 graph around attention/lm_head instead of repeating
  GDN solve/dtype experiments.

Important later evidence:

- The known-broken legacy-direct CTE2048 log proves the clean `ptok=1225`
  failure was a cold active-only request, not a backed-prefix read:
  `attention_hit_len=0`, `restore_len=0`, `computed=tensor([[0]])`,
  `num_queries=tensor([[1225]])`, `prefix_len=0`, `prefill_bucket=2048`.
- Therefore `segmented_cte`/backed-prefix assembly cannot be the primary
  explanation for the first 1225-token NaN, although it can still matter for
  true warm-prefix requests.
- Because vLLM block-KV is enabled, cold CTE still enters
  `perform_qwen_chunked_prefill`; it does not use the bare `past_key_value is
  None` prefill helper. The relevant active-only attention path is the
  chunked-prefill cache-scatter + prefix-attention path.
- Branch comparison against the two allowed references found a stronger code
  delta: `codex/full-fp8-qwen36` and `codex/nki-deltanet-decode-step` use the
  older expanded-KV attention calculation in chunked prefill, while this branch
  replaced it with `_qwen35_grouped_prefix_attention`, a 5D grouped
  matmul/softmax path. That path is exactly active in the cold `ptok=1225`
  failure. Treat this as the leading non-GDN suspect if the current GDN-segmented
  compile fails again.
- The 1225 padded call also showed `position_ids` capped at `0..1224` while
  `rotary_position_ids` extended to `0..2047`. This mismatch is real and should
  be cleaned up eventually, but it also existed in the allowed reference
  branches, so it is less likely to explain the new 485/1225 cliff by itself.

### Code changed in this session

Primary file:

- `contrib/models/Qwen3.6-27B/src/modeling_qwen35.py`

New opt-in env:

- `QWEN36_DELTANET_FUSED_SEGMENT_TOKENS`
- `QWEN36_PREFIX_ATTENTION_IMPL`

Behavior:

- If set to a positive multiple of `QWEN36_DELTANET_CHUNK_SIZE`, and the padded
  sequence is longer than that segment size, `_fused_chunked_forward` loops over
  segment slices, calls the existing fused GDN CTE kernel per segment, carries
  `final_state` into the next segment, concatenates outputs, and returns the last
  carried state.
- Existing behavior is unchanged when the env is unset or `0`.
- `QWEN36_PREFIX_ATTENTION_IMPL=expanded` restores the working-branch expanded
  KV-head chunked-prefill attention path. Default remains `grouped` so existing
  behavior is unchanged unless the env is explicitly set.

Local verification after the edit:

- `python3 -m py_compile contrib/models/Qwen3.6-27B/src/modeling_qwen35.py`
  passed.
- `pytest contrib/models/Qwen3.6-27B/test/unit/test_qwen36_model_aliases.py`
  passed: 66 tests after adding the prefix-attention selector/helper coverage.
- `pytest contrib/models/Qwen3.6-27B/test/unit/test_qwen36_compile_fp8_config.py`
  passed: 46 tests.

Harmless local tooling error logged per `AGENTS.md`:

- Command: `rg -n "QWEN36_VLLM_LOGITS_DEBUG|finite=|nan=|stage=|computed_context|context_lens|prefix" contrib/models/Qwen3.6-27B/vllm contrib/models/Qwen3.6-27B/src validation_scripts scripts . -g '*.py' -g '*.md'`
- Failure: exit code `2`, exact error `rg: scripts: No such file or directory
  (os error 2)`.
- Context: local repo search while checking debug hooks.
- Root cause: this workspace has no top-level `scripts/` directory.
- Mitigation: reran narrower searches against real paths; no investigation
  impact.

Harmless remote/tooling errors logged per `AGENTS.md`:

- Web lookup: attempted to reopen stale `web.run` search result IDs after other
  web calls had advanced the result set. Failure was an invalid/stale ref ID.
  Mitigation: kept the already-open primary-source evidence from Qwen FlashQLA
  and NVIDIA Megatron docs; no repo or EC2 state was affected.
- Runtime launcher inspection:
  `ssh ubuntu@16.51.179.180 '/home/ubuntu/inferentia-gdn-multihead-cte-20260531T1350Z/start_vllm_server.sh --help ...'`
  returned the exact stderr `Unknown argument: --help`. Context: checking the
  installed runtime launcher before serving the current artifact. Root cause:
  this script has no help handler and exits on unknown args. Mitigation:
  inspected the script body with `sed`; confirmed it accepts the needed flags
  and reads `prefix_cte_attention_backend`/segment settings from the compiled
  artifact's `neuron_config.json`.
- Compile-host process-tree helper:
  `ssh ubuntu@16.51.94.87 'P=$(cat ...pid); ... awk -v p="$P" "$2==p || $1==p {print}" ...'`
  failed because the remote shell expanded `$1`/`$2`, producing the exact awk
  error `awk: cmd. line:1: ==p || ==p {print}`. Mitigation: the same command
  also ran `pgrep -a -P "$P"` and a broader `ps` scan, which confirmed the
  compile parent was waiting on a live `neuronx-cc` layout optimization child
  and `walrus_driver` was using high CPU.

Blocking compile errors and mitigations logged per `AGENTS.md`:

- Compile `20260604T210027Z_gdnseg512_kkt_hier_scan7`, PID `568734`, failed
  after HLO/layout compilation during final sharded checkpoint serialization.
  Exact error:
  `safetensors_rust.SafetensorError: Error while serializing: I/O error: No space left on device (os error 28)`.
  Failing stage: `model.compile(...)->shard_weights(...)->save_file(...tp{rank}_sharded_checkpoint.safetensors)`.
  Artifact was incomplete: `model.pt` existed, but sharded weights were partial
  and there was no `COMPILE_DONE`. Disk was `100%` full with only `372K`
  available. Mitigation: deleted only the incomplete gdnseg artifact/workdir and
  two redundant older CTE512 debug artifacts/workdirs
  (`20260604T153000Z_schemafix_doubling_scan7` and
  `20260604T163000Z_statefix_doubling_scan7`), preserving the known coherent
  CTE512 artifact and legacy-direct CTE2048 artifact. Disk recovered to `103G`
  free.
- Cleanup command targeting error: a first `rm -rf /mnt/trainium_artifacts/...`
  was accidentally run locally, not through SSH. It returned exit 0 but removed
  nothing relevant because those EC2 paths do not exist locally. Mitigation:
  reran the same targeted cleanup through `ssh ubuntu@16.51.94.87`; verified
  free space with `df -h`.
- Replacement launch `20260604T213000Z_gdnseg512_expattn_kkt_hier_scan7`,
  PID `576592`, failed immediately with exact log error
  `env: ‘python’: No such file or directory`. Root cause: `nohup env python`
  did not activate the Neuron venv. No compile work was done.
- Replacement launch `20260604T213300Z_gdnseg512_expattn_kkt_hier_scan7`,
  PID `576754`, failed immediately with exact log error
  `FileNotFoundError: [Errno 2] No such file or directory: 'libneuronpjrt-path'`.
  Root cause: absolute venv Python was used without activation/PATH helpers.
- Replacement launch `20260604T213700Z_gdnseg512_expattn_kkt_hier_scan7`,
  PID `577097`, failed immediately with exact log error
  `RuntimeError: Unsupported Platform - r7i.24xlarge`.
  Root cause: cross-compile env was missing before Python import.
  Mitigation: created `tmp_compile_qwen32k_gdnseg512_expattn.sh`, which sources
  the Neuron venv and sets `NEURON_PLATFORM_TARGET_OVERRIDE=trn2`,
  `NEURON_CC_FLAGS=--target trn2 --lnc 2`, `NKI_LIBRARY_SRC`,
  `USE_NKI_DECODE`, DeltaNet chunk/solve envs, and
  `QWEN36_PREFIX_ATTENTION_IMPL=expanded`. Launch
  `20260604T214000Z_gdnseg512_expattn_kkt_hier_scan7` is the current active
  compile.

Runtime disk cleanup logged per `AGENTS.md`:

- Runtime host `ubuntu@16.51.179.180` had only `36G` free before transferring
  the newly compiled CTE2048 artifact; `du -sh` showed multiple stale 36-67G
  artifacts. Keeping the known coherent CTE512
  `20260604T170500Z_aliasguard_statefix_kkt_hier_scan7` reference, removed only
  stale older CTE512 runtime artifacts:
  `20260604T133807Z_doubling_scan7`, `20260604T143000Z_doubling_scan7`,
  `20260604T153000Z_schemafix_doubling_scan7`, and
  `20260604T163000Z_statefix_doubling_scan7`.
- Command context: `ssh ubuntu@16.51.179.180 'rm -rf <four artifact dirs> && df -h /mnt/trainium_artifacts'`.
- Verification: runtime free space increased to `211G`; no current compile-host
  artifact or known-good CTE512 `170500Z` artifact was removed.

Transfer error logged per `AGENTS.md`:

- First EC2-to-EC2 transfer command:
  `ssh ubuntu@16.51.94.87 'rsync -a --info=progress2 "$SRC/" ubuntu@16.51.179.180:"$DST/"'`
  failed before copying data.
- Exact error: `ubuntu@16.51.179.180: Permission denied (publickey).` followed by
  `rsync: connection unexpectedly closed (0 bytes received so far) [sender]` and
  `rsync error: unexplained error (code 255) at io.c(232) [sender=3.2.7]`.
- Root cause: compile host did not have direct key material/agent access for the
  runtime host. Mitigation: verified `ssh -A ubuntu@16.51.94.87 'ssh -o BatchMode=yes ubuntu@16.51.179.180 hostname'`
  succeeds, then retry transfer through the compile host with agent forwarding.

Serve launch error logged per `AGENTS.md`:

- First launch of transferred artifact used top-level
  `/home/ubuntu/inferentia-gdn-multihead-cte-20260531T1350Z/start_vllm_server.sh`.
  PID `42011` exited immediately before model load.
- Exact error in
  `/home/ubuntu/validation_logs/fp8_256k_decode_nki/qwen36_27b_32768_fp8_gdnseg512_expattn_serve_20260604T2208Z.log`:
  `python: can't open file '/home/ubuntu/inferentia-gdn-multihead-cte-20260531T1350Z/serve_qwen36.py': [Errno 2] No such file or directory`.
- Root cause: top-level wrapper computes `SCRIPT_DIR` as repo root, but
  `serve_qwen36.py` is under `contrib/models/Qwen3.6-27B/vllm/`.
- Mitigation: use
  `/home/ubuntu/inferentia-gdn-multihead-cte-20260531T1350Z/contrib/models/Qwen3.6-27B/vllm/start_vllm_server.sh`
  for runtime validation.
- Second launch with the contrib launcher, PID `42381`, reached model import but
  exited before serving. Exact root error in
  `/home/ubuntu/validation_logs/fp8_256k_decode_nki/qwen36_27b_32768_fp8_gdnseg512_expattn_serve_20260604T2212Z.log`:
  `ValueError: QWEN36_DELTANET_SOLVE_SCAN_STEPS must be in [1, 5] for SOLVE_BLOCK_SIZE=32; got 7`.
- Context: compile driver set `QWEN36_DELTANET_SOLVE_BLOCK_SIZE=128` and
  `QWEN36_DELTANET_CHUNK_SIZE=128`, but runtime launch initially omitted them,
  so model import used the default solve block size 32 and rejected scan7.
- Mitigation: relaunch with both `QWEN36_DELTANET_SOLVE_BLOCK_SIZE=128` and
  `QWEN36_DELTANET_CHUNK_SIZE=128` in the serving environment.

Other local changes in the dirty tree pre-existed or were from earlier parts of
this debugging run: alias guards, checkpoint dtype reassertion, logits fallback
debugging, legacy-direct kernel copy, segmented CTE work, validators. Do not
assume every dirty file belongs to the latest segmented-GDN fix.

### Artifacts and results from this session

Known coherent but slower reference:

- Artifact:
  `/mnt/trainium_artifacts/qwen_artifacts/qwen36_27b_32768_fp8_full_lmheadfp8_kvfp8_qkvnki_tiled_attention_cte_nki_decode_stable_probe32k_gdnrecfloat32_sampletokens_outlogits_b256_cte512_pfx16k_slots64_async_cteargmaxsafe_20260604T170500Z_aliasguard_statefix_kkt_hier_scan7`
- Passed `/tmp/tmp_bisect_probe3.py` at prompt lengths 5, 160, 123, 485, 1225,
  and 2500.
- Usage-accounted 16k cold-prefill max_tokens=1 measured roughly 1548-1595 tok/s.
- This is coherent but not fast enough to claim CTE2048 target speed.

Current/fast CTE2048 grouped/current path:

- Failed long prompts with all-NaN logits.
- Serve log evidence included:
  `finite=0/248320 nan=248320`,
  invalid sampled token values like `-1047477664` and sentinel `2147483647`,
  fallback argmax token `0`.
- Conclusion: sampler/aliasing is not the root cause; logits are already bad.

Single-head CTE2048 current path:

- Artifact:
  `/mnt/trainium_artifacts/qwen_artifacts/qwen36_27b_32768_fp8_full_lmheadfp8_kvfp8_qkvnki_tiled_attention_cte_nki_decode_stable_probe32k_gdnrecfloat32_sampletokens_outlogits_b256_cte2048_pfx16k_slots64_async_cteargmaxsafe_20260604T184605Z_scte0_kkt_hier_scan7`
- `QWEN36_DELTANET_MULTIHEAD_CTE=0`.
- Passed short lengths, failed at 1225 and 2500 with all-NaN logits.
- Conclusion: grouped multihead CTE is not the sole cause.

Legacy-direct CTE2048 path:

- Artifact:
  `/mnt/trainium_artifacts/qwen_artifacts/qwen36_27b_32768_fp8_full_lmheadfp8_kvfp8_qkvnki_tiled_attention_cte_nki_decode_stable_probe32k_gdnrecfloat32_sampletokens_outlogits_b256_cte2048_pfx16k_slots64_async_cteargmaxsafe_20260604T194122Z_ldir_kkt_hier_scan7`
- Env:
  `QWEN36_DELTANET_CTE_IMPL=legacy_direct`,
  `QWEN36_DELTANET_MULTIHEAD_CTE=0`.
- Compile succeeded with `CHECKPOINT_BANK_WEIGHTS_ADDED` for tp0..tp3,
  48 recurrent `torch.float32`, 48 conv `torch.bfloat16`, and `COMPILE_DONE`.
- Runtime probe result:
  - `ptok=5`: coherent
  - `ptok=160`: coherent
  - `ptok=123`: coherent
  - `ptok=485`: coherent
  - `ptok=1225`: corrupt, e.g. `' Paris is mentioned in which chapter?\n\n<think>\n!!'`
  - `ptok=2500`: corrupt, e.g. `'!!!!!!!!!!!!'`
- Serve log again showed all-NaN logits:
  `finite=0/248320 nan=248320`.
- Conclusion: reverting to the old direct triangular solve/QK-normalization path
  does not fix CTE2048.

Important caveat:

- A fine-boundary probe run after a 1225/2500 failure reported corruption even
  at `ptok=27`, but the server/cache was already poisoned by earlier long
  failures. Do not use that as a clean first-failure boundary.

### Profiling work and conclusions

Skills used:

- `neuron-nki-profiling`
- `neuron-nki-profile-querying`
- Earlier NKI docs/writing/equivalence skills were also consulted.

Full vLLM runtime profiling attempt:

- Runtime host: `ubuntu@16.51.179.180`
- Profile dir:
  `/home/ubuntu/validation_logs/fp8_256k_decode_nki/legacydirect_cte2048_profile_20260604T202953Z`
- Relaunched the legacy-direct CTE2048 artifact with:
  `NEURON_RT_INSPECT_ENABLE=1`,
  `NEURON_RT_INSPECT_DEVICE_PROFILE=1`,
  `NEURON_RT_INSPECT_SYSTEM_PROFILE=0`,
  `NEURON_RT_INSPECT_OUTPUT_DIR=.../inspect`,
  `NEURON_RT_ENABLE_DGE_NOTIFICATIONS=1`,
  `XLA_IR_DEBUG=1`,
  `XLA_HLO_DEBUG=1`,
  `NEURON_FRAMEWORK_DEBUG=1`.
- Failure: server never became healthy; bounded `/health` returned connection
  refused.
- Log evidence:
  `Timeout polling for async exec completion on nc (0..3) (waited 2 minutes).`
- Current hypothesis: full vLLM graph profiling is too invasive during model
  load/init or captured a graph that could not complete under inspect mode.
- Mitigation: killed the unhealthy server and used the emitted NEFF offline.

Offline `neuron-explorer capture` with DGE:

- NEFF emitted under the inspect dir, corresponding to
  `.../layout_opt/graph.neff`.
- Command with `--enable-dge-notifs` failed.
- Exact runtime error:
  `notification queue overflow`,
  `NRT_EXEC_SW_NQ_OVERFLOW`,
  status `1204`,
  on NC4 and NC5,
  "Too many notifications generated during execution."
- Current hypothesis: full model graph emits too many device/DMA notifications
  for DGE-level tracing.
- Mitigation: recaptured without DGE notifications.

Offline `neuron-explorer capture` without DGE:

- Succeeded.
- Produced:
  `/home/ubuntu/validation_logs/fp8_256k_decode_nki/legacydirect_cte2048_profile_20260604T202953Z/capture_nodge/profile.ntff`
- Size: about 952 MB.
- Summary JSON produced:
  `.../capture_nodge/summary.json`
- Summary highlights:
  `total_time` about 0.061 s,
  low engine utilization,
  about 20M trace events.
- Limitation: this was coarse graph timing, not source-level NaN origin.

Profile table ingestion:

- `neuron-explorer view --ingest-only` ran for more than ten minutes with
  zero-byte parquet tables.
- It was stopped.
- Error/mitigation note: one `pkill -f "neuron-explorer view -n ..."` matched
  the SSH-side shell command and SSH exited `255`. Rechecked processes and the
  stuck ingest did terminate. Use PID-based kill next time.

Standalone fused DeltaNet validator:

- Script:
  `/home/ubuntu/inferentia-gdn-multihead-cte-20260531T1350Z/contrib/models/Qwen3.6-27B/scripts/validate_deltanet_fused_nki.py`
- Ran on Trainium with CPU reference off-device.
- `seq_len=512`, `QWEN36_DELTANET_SOLVE_MODE=kkt_hier`,
  `QWEN36_DELTANET_SOLVE_SCAN_STEPS=7`:
  output/state finite, relative error around `3e-6`, passed.
- `seq_len=2048` with the same settings:
  output/state finite, relative error around `3e-6`, passed.
- Warnings seen:
  NCCL/OFI EFA warnings such as "Failed to initialize rdma protocol"; they did
  not affect correctness.
- Conclusion: the standalone fused DeltaNet kernel does not simply fail because
  CTE2048 has 16 internal 128-token chunks. The NaN must come from full graph
  composition or real activation/state values.

### Hypotheses cleared today

Do not repeat these unless a new artifact changes the inputs:

- Simple sampler bug: cleared. Bad sampled token fallback happens because logits
  are already degenerate/all-NaN.
- Token/logits output order swap: cleared by output ordering and alias checks.
- Alias output count shift: guarded; no alias-count guard fired in successful
  compiles.
- Checkpoint bank missing dtype/layout in the tested artifacts: dtypes verified
  as 48 recurrent `torch.float32` and 48 conv `torch.bfloat16` for tp0..tp3.
- fp32 recurrent cache as sole short-context cause: not supported after dtype
  checks and legacy-direct behavior.
- segmented CTE attention as the only culprit: not the active failing path in
  legacy-direct/current attention_cte tests.
- grouped multihead CTE as sole culprit: single-head CTE2048 still failed long.
- rewritten KKT/direct solve path as sole culprit: legacy-direct CTE2048 still
  failed long.
- raw fused DeltaNet NKI kernel fails at 2048 just due to `num_chunks=16`:
  standalone 2048 validator passed tightly.

### Remaining best hypothesis

The bug is in full-model CTE2048 graph composition or real-value interaction
around GDN/attention boundaries:

- CTE512 full serving is coherent but slower.
- CTE2048 full serving produces NaN logits on long prompts.
- Standalone GDN CTE2048 is numerically correct on random sane inputs.
- Therefore the next useful experiments need to change the full-graph staging or
capture real values at boundaries, not re-run standalone math checks.

Most promising current path:

- Keep attention/QKV external CTE2048.
- Stage only GDN at 512-token internal segments with explicit state carry:
  `QWEN36_DELTANET_FUSED_SEGMENT_TOKENS=512`.
- If this is coherent and speed remains near CTE2048, it becomes the practical
  fix.
- If it still fails, the next step should be value capture/logging at per-layer
  GDN input/output/final_state boundaries, because profiling has reached its
  useful limit.

### What to do when current compile finishes

If compile succeeds:

1. Verify compile log has:
   `Finished Compilation for all HLOs`,
   `CHECKPOINT_BANK_WEIGHTS_ADDED` for tp0..tp3 with 48/48 fp32/bf16,
   no alias-count guard failures,
   `COMPILE_DONE`.
2. Verify artifact files and safetensor dtypes.
3. `rsync` compile host to runtime host using EC2-to-EC2 transfer.
4. Launch runtime with the same artifact and include
   `QWEN36_DELTANET_FUSED_SEGMENT_TOKENS=512`.
5. Wait for `/health` using bounded curl; do not use unbounded curl loops.
6. Run `/tmp/tmp_bisect_probe3.py`.
7. Run clean targeted boundary probes around 1225/1265/1346/2500, but restart
   the server before boundary probing if a previous long failure may have
   poisoned cache/state.
8. Scan serve log for invalid token fallback and logits NaNs.
9. Only if coherent, run the 16k usage-accounted OpenAI chat benchmark. Use
   `stream_options: {"include_usage": true}` and compute TPOT from
   `usage.completion_tokens`.

If compile fails:

- Record exact command/log path/error text/stage.
- Current first suspicion would be traced graph blow-up or NKI compile issue from
  multiple fused GDN segment calls in one context HLO, not a model correctness
  result.

### Operational mistakes to avoid repeating

- Do not trust a prompt length failure after a prior long prompt has poisoned
  server/cache state. Restart before first-failure boundary probes.
- Use bounded `curl --max-time` in health polling. An earlier health checker hung
  because plain `curl` had no timeout.
- Do not use DGE notifications on the full graph first; it overflowed the
  notification queue. Use standalone kernel profiling or non-DGE summary first.
- Do not treat zero loose `.neff` files in model artifacts as failed compile;
  NEFFs are embedded in `model.pt`.
- Do not compare streamed chunk counts with token-level TPOT. Use usage-accounted
  completion tokens.

---

## 0. TL;DR (read this first)

We are bringing up **Qwen3.6-27B (FP8, hybrid GatedDeltaNet + full-attention)** on **AWS Trainium (trn2)** via NeuronX Distributed Inference (NxDI) + vLLM-Neuron. It serves but produces **garbage output** (`!!!!!` / `nine one!!!`) in two distinct regimes. We have **fully root-caused and fixed bug #1**. **Bug #2 (long-context cross-chunk corruption) is still open** and is the active investigation.

- **Bug #1 — SOLVED:** nkilib `qkv` kernel's `qkv_cte` path **corrupts FP8 for sequence length > 96**. Fix = ≤96 sequence-tiling in `gqa.py` (routes each sub-call to the correct `qkv_tkg`). Validated coherent at 160/198/485 tokens, decode 31 tok/s (3× over the no-kernel baseline of 10 tok/s).
- **Bug #2 — OPEN:** with a correct QKV path, output is still coherent only up to ~1024 tokens (2 CTE chunks). At ≥1280 tokens (3+ chunks) it degrades to garbage, **progressively**. Inputs (positions) are provably correct. Suspect either GDN recurrent/conv-state carry across chunk boundaries, or the flat `attention_cte` prefix-read degrading as the prefix grows.
- **NEW unconfirmed signal (today):** the "full-moat" artifact (`fulldense_tiled`, which adds mlp-cte + out-proj + quantized-mlp NKI kernels on top of the qkv fix) is **garbage at ALL lengths, even 5 tokens** — suggesting one of those extra kernels *also* corrupts FP8. **This needs confirmation** (re-anchor test in flight).

---

## 1. Infrastructure

| Role | IP | Host | Notes |
|------|-----|------|-------|
| **Runtime / serving** | `16.51.184.154` | `ip-172-31-40-137` | trn2, serves vLLM on `:8000`. Where we test coherence. |
| **Compile** | `16.51.94.87` | (trn2) | Builds NEFF artifacts. ~57–99G free disk (gets tight). |

- **SSH:** `ssh ubuntu@<ip>`. Compile→runtime works **with agent forwarding** (`ssh -A`); runtime→compile does **not** (no key). Frequent `exit 255` drops on the compile host — `nohup` long jobs and re-check.
- **venv (both hosts):** `source /opt/aws_neuronx_venv_pytorch_inference_vllm_0_16/bin/activate`
- **Model weights:** `/home/ubuntu/models/Qwen3.6-27B` (this exact path is also the vLLM model ID).
- **Artifacts dir (both hosts):** `/mnt/trainium_artifacts/qwen_artifacts/`
- **Repo:**
  - Runtime: `/home/ubuntu/inferentia-gdn-multihead-cte-20260531T1350Z`
  - Compile: `/home/ubuntu/inferentia-gdn-compile-4482016`
  - Local (laptop): `/Users/deepankarsingh1312/Downloads/career-ops/inferentia-gdn`, branch `codex/nki-deltanet-multihead-cte`

---

## 2. The model (why it's hard)

Qwen3.6-27B is a **hybrid**:
- **48 GatedDeltaNet (GDN) layers** — linear-attention/SSM-style, **fixed-size recurrent state** + conv state, carried turn-to-turn.
- **16 full-attention layers** — head_dim 256, **partial RoPE** (rope_dim 64).
- **FP8** (f8e4m3) ROW quantization on weights; **KV cache FP8**.

This combination is **not covered by AWS Autoport** (which auto-ports standard HF models to Neuron but explicitly does NOT handle hybrid SSM/attention or new NKI kernels). That gap is the whole point of this contribution.

Serving stack uses **Hybrid APC** (automatic prefix caching for the hybrid cache) + **chunked prefill** (`hybrid_apc_prefill_chunk_tokens=512`), with GDN state checkpointed every 256 tokens (`gdn_checkpoint_interval=256`, `max_gdn_checkpoint_slots=64`). Prefix CTE attention has two backends: `attention_cte` (flat) and `segmented_cte` (streams long prefixes by segment; the docs' validated 256K path).

---

## 3. Bug #1 — qkv_cte FP8 corruption at S>96  (SOLVED)

**Root cause:** nkilib's `qkv` kernel dispatches to two implementations:
- `qkv_tkg` — used when `B*S ≤ 128` **and** `S ≤ 96` (and no fused rope). **Correct.**
- `qkv_cte` — used when `S > 96` or `B*S > 128` or fused_rope. **The FP8 path corrupts.**

Threshold constant: `SEQLEN_THRESHOLD_FOR_QKV_CTE = 96`.

**Fix (validated):** tile the sequence into ≤96-token sub-calls so every QKV projection routes through the correct `qkv_tkg`. QKV projection is per-token, so tiling is exact (no cross-token dependency).

**File:** `src/neuronx_distributed_inference/modules/attention/gqa.py`, function `_kernel_qkv_forward` (~line 640). Key lines (682–728):
```python
# The nkilib `qkv` kernel routes S > SEQLEN_THRESHOLD_FOR_QKV_CTE (=96) ... to qkv_cte (FP8-broken)
def _qkv_kernel_call(_input, _mlp_prev, _attention_prev): ...
_qkv_tile = min(96, max(1, 128 // bs))
if (not fuse_rope) and (seqlen > _qkv_tile or bs * seqlen > 128):
    _qkv_parts = []
    for _ts in range(0, seqlen, _qkv_tile):
        _te = min(_ts + _qkv_tile, seqlen)
        ... _qkv_parts.append(_qkv_kernel_call(_inp, _mp, _ap))
    QKV = torch.cat(_qkv_parts, dim=(2 if self.qkv_kernel_nbsd_layout else 1))
else:
    QKV = _qkv_kernel_call(hidden_states, mlp_prev, attention_prev)
```
This is **traced into the NEFF → requires recompile** to take effect. Deployed to both hosts.

**Evidence it works:** the `qkvnki_tiled` artifact (qkv-nki tiling ON, other kernels OFF) was coherent at 160/198/485 tokens; decode 31 tok/s vs 10 tok/s without kernels (3× speedup = the moat).

---

## 4. Bug #2 — cross-CTE-chunk corruption > ~1024 tokens  (OPEN)

**Symptom (empirical, on the `attention_cte` path):**
- ≤ 485 tok (≤1 CTE chunk / group): **coherent**
- ~1024 tok (2 chunks): **partial** (`' nine one!!!'`)
- ≥ 1280 tok (3+ chunks): **garbage** (`'!!!!!!!!'`), and degradation is **progressive** with length.

**Proven correct (NOT the cause):**
- **Positions are absolute and correct** across chunks. Runtime debug (`QWEN36_HYBRID_APC_DEBUG=1`, `position_minmax` log in `model_wrapper.py:1451`) shows: chunk0 `0:511` prefix_len 0; chunk1 `512:1023` prefix_len 512; chunk2 `1024:1200` prefix_len 1024. So the GDN `reset_mask` (`position_ids==0`) does **not** spuriously fire on continuation chunks — the earlier "reset_mask" hypothesis is **REFUTED**.
- **bf16 vs fp32 state:** forcing FP32 recurrent+conv state produced **identical garbage** → not a state-precision issue. **REFUTED.**
- **DeltaNet kernel math, qk-norm-rope, multihead, sampler:** all previously exonerated.
- **Per-token QKV kernel tiling (bug #1 fix):** in place; not the cause of bug #2.

**Leading suspects (unresolved):**
1. **GDN recurrent/conv-state write at the chunk boundary** (checkpoint carry between CTE chunks).
2. **Flat `attention_cte` prefix-read** degrading as the prefix grows (vs `segmented_cte`).

**The decisive experiment (in flight):** compile the **same** config but with `prefix_cte_attention_backend=segmented_cte` (seg size 512) and re-test at 1201/1536/2048.
- **segmented_cte coherent past 1201** ⇒ bug #2 is the flat `attention_cte` prefix-read; **segcte is the fix**.
- **still garbage** ⇒ bug #2 is the **GDN state/conv carry**; pivot to the checkpoint-boundary write path.

> ⚠️ **Confound discovered today:** the segcte artifact was compiled with the **full moat** (mlp-cte + out-proj + quantized-mlp). But the full-moat `fulldense_tiled` artifact is garbage at *all* lengths (even 5 tok). If a moat kernel corrupts short sequences, the segcte test can't isolate bug #2. **Re-anchor test running now:** serve `qkvnki_tiled` (qkv-only) and confirm it's coherent short + garbage long. If so, we need a **qkv-only + segmented_cte** compile to cleanly test bug #2, not full-moat+segcte.

---

## 5. Artifact inventory (`/mnt/trainium_artifacts/qwen_artifacts/`)

Each artifact is ~34G (model.pt ~447–510M with NEFFs embedded + 4× ~8.5G sharded fp8 weights). **NEFFs are embedded in `model.pt`; there are no loose `.neff` files — this is normal.**

| Artifact (short) | kernels | prefix backend | Status / use |
|---|---|---|---|
| `qkvnki_tiled …212605Z` | qkv-nki only | attention_cte | **Known-good ≤485 tok**; cross-chunk garbage >1024. Re-anchor baseline. |
| `fulldense_tiled …044318Z` | qkv + mlp-cte + out-proj + qmlp | attention_cte | **Garbage at ALL lengths (today).** Full moat — suspect a moat kernel breaks FP8 short. |
| `fulldense_segcte …055545Z` | same full moat | **segmented_cte** seg512 | **Just compiled.** The cross-chunk test vehicle (confounded if full-moat breaks short). |
| `hybrid_apc …201758Z` | baseline | attention_cte | older reference |
| `noapc_pfxcache …172551Z` | — | — | hybrid-APC-off / prefix-caching-on probe |

---

## 6. How to reproduce / drive the test loop

**Swap which artifact is served** (helper deployed on runtime at `/home/ubuntu/swap_serve.sh`):
```bash
ssh ubuntu@16.51.184.154 'bash /home/ubuntu/swap_serve.sh <ARTIFACT_DIR_NAME> <log_basename>'
# it: kills old vllm, activates venv, launches start_vllm_server.sh (hybrid-APC + CTE512,
#     block-size 256, gdn-checkpoint 256, backend auto-read from artifact neuron_config.json)
```
Launcher: `…/contrib/models/Qwen3.6-27B/vllm/start_vllm_server.sh` — note it **auto-derives** `prefix_cte_attention_backend`/segment_size from the artifact's `neuron_config.json` (lines 535–542), and **requires `--block-size == --gdn-checkpoint-interval`** for hybrid-APC.

**Wait for ready:** `curl -s http://localhost:8000/health` returns 200 (load ~2–4 min).

**Coherence probe** (on runtime, in venv):
```bash
source /opt/aws_neuronx_venv_pytorch_inference_vllm_0_16/bin/activate
cd /tmp && python tmp_bisect_probe3.py    # P1/160/198/485/1225/2500-tok ladders
```
Model ID in any direct API call must be the full path: `/home/ubuntu/models/Qwen3.6-27B`. Do **not** pass `logprobs` (unsupported → 500).

**Context sweep validator** (auto-reads artifact neuron_config):
```bash
python validation_scripts/qwen36_hybrid_apc_context_sweep.py --lengths 1024,1280,1536,2048 --warm-mode partial
```

**Recompile** (on compile host `16.51.94.87`, driver scripts `~/tmp_compile_qwen256k_*.sh`):
```bash
FAST_COLD_PREFILL_PROBE=1 OUTPUT_LOGITS_WITH_ON_DEVICE_SAMPLING=1 \
DISABLE_CONTEXT_ENCODING_ARGMAX_KERNEL=1 ENABLE_KV_CACHE_QUANT=1 \
GDN_RECURRENT_CACHE_DTYPE=bfloat16 nohup bash ~/tmp_compile_qwen256k_<variant>.sh &
# compile ~1hr; produces artifact in /mnt/trainium_artifacts/qwen_artifacts/
# transfer compile→runtime: ssh -A compile 'rsync -a -e "ssh -A" <artifact>/ ubuntu@16.51.184.154:/mnt/trainium_artifacts/qwen_artifacts/'
```
Disk fills fast (~34G/artifact); delete stale artifacts to make room.

---

## 7. Decision tree (what to do next)

```
RE-ANCHOR: serve qkvnki_tiled, run probe3
├─ coherent short, garbage >1024   → runtime OK; moat kernels break short.
│                                     Bug #2 is real on qkv-only.
│                                     ▶ Compile qkv-only + segmented_cte → test bug #2 cleanly.
│                                     ▶ Separately bisect WHICH moat kernel breaks FP8 short
│                                       (mlp-cte vs out-proj vs quantized-mlp).
├─ coherent short AND long          → scheduler patch already fixed bug #2 on qkv-only.
│                                     ▶ Ship qkvnki_tiled. segcte moot.
└─ garbage everywhere               → serving config / runtime regression.
                                      ▶ Fix serving first (check CTE bucket, debug patches,
                                        QWEN36_HYBRID_APC_INSTALL_PATCH) before kernel conclusions.
```

If bug #2 turns out to be GDN-carry: the reference is NVIDIA/vLLM on H100 — hybrid SSM+attn APC uses **FP32 SSM state, block-boundary checkpoints, SSD-scan `initial_states`, `has_initial_state = context_lens>0`**. NVIDIA has *open* bugs on this exact model family, so upstream is not a solved baseline.

---

## 8. Key files

| File | What |
|---|---|
| `src/neuronx_distributed_inference/modules/attention/gqa.py` (~640, fix 682–728) | **Bug #1 fix** — qkv ≤96 tiling |
| `src/neuronx_distributed_inference/models/model_wrapper.py:1451` | runtime `position_minmax` debug log (gated on `QWEN36_HYBRID_APC_DEBUG=1`) |
| `contrib/models/Qwen3.6-27B/src/modeling_qwen35.py` | GDN reset (1596 conv / 1792 recurrent), chunked prefill (2997), prefix attention (182) |
| `contrib/models/Qwen3.6-27B/vllm/start_vllm_server.sh` | serve launcher (auto-derives backend from artifact) |
| `/tmp/tmp_bisect_probe3.py` (runtime) | coherence ladder probe |
| `validation_scripts/qwen36_hybrid_apc_context_sweep.py` | length sweep, warm-mode partial/exact |

---

## 9. Gotchas

- NEFFs are embedded in `model.pt`; **0 loose `.neff` files is normal** — don't mistake it for a failed compile. Verify a compile by `model.pt` size (~447–510M) + 4 sharded weight files + `COMPILE_DONE` in the driver log (the driver echoes `COMPILE_DONE` after the python proc exits; cross-check `model.pt` exists and is ~450M).
- Each server swap = full model reload (~2–4 min). Batch your tests.
- Compile-host SSH drops (`exit 255`) are common; pollers die but `nohup`'d jobs survive — re-check directly.
- `--block-size` must equal `--gdn-checkpoint-interval` (both 256) or the launcher errors.
- Model ID = full path `/home/ubuntu/models/Qwen3.6-27B`.

---

## 2026-06-05 mRoPE padding probe notes

**Runtime patch attempt:** copied local `contrib/models/Qwen3.6-27B/src/modeling_qwen35.py` to runtime host `ubuntu@16.51.179.180` to test repeating the last valid mRoPE position for masked CTE padding tokens instead of advancing fake future positions.

**Failure:** serve restart with log `/home/ubuntu/validation_logs/fp8_256k_decode_nki/qwen36_27b_32768_fp8_gdnseg512_expattn_mropefix_serve_20260604T2230Z.log` exited during import with:

```text
ModuleNotFoundError: No module named 'src.nki_kernels.nki_deltanet_fused_legacy'
```

**How we got there:** the copied local `modeling_qwen35.py` contains earlier CTE implementation selector code importing `src.nki_kernels.nki_deltanet_fused_legacy`, but that untracked helper file was not present in the runtime checkout. Artifact under test remains `/mnt/trainium_artifacts/qwen_artifacts/qwen36_27b_32768_fp8_full_lmheadfp8_kvfp8_qkvnki_tiled_attention_cte_nki_decode_stable_probe32k_gdnrecfloat32_sampletokens_outlogits_b256_cte2048_pfx16k_slots64_async_cteargmaxsafe_20260604T214000Z_gdnseg512_expattn_kkt_hier_scan7`.

**Root-cause hypothesis:** file synchronization error, not a model/runtime coherence result.

**Mitigation:** sync the missing helper `contrib/models/Qwen3.6-27B/src/nki_kernels/nki_deltanet_fused_legacy.py` to the matching runtime path, then relaunch with the same CTE2048/GDN-seg512/expanded-attention env and rerun the boundary probe.

**Follow-up check:** synced helper to runtime path and verified both `src/modeling_qwen35.py` and `src/nki_kernels/nki_deltanet_fused_legacy.py` with `python -m py_compile`.

**Non-model check failure:** a direct ad-hoc import command outside the launcher environment failed during `torch_xla`/Neuron runtime initialization:

```text
FileNotFoundError: [Errno 2] No such file or directory: 'libneuronpjrt-path'
```

**Context:** command was run over SSH with `/opt/aws_neuronx_venv_pytorch_inference_vllm_0_16/bin/python -c ...` from the contrib model directory, not through `start_vllm_server.sh`. This is best classified as missing launcher/PATH environment for a standalone import check, not evidence about the model graph or coherence. Mitigation is to use the real vLLM launcher env for the serve test.

**Launcher failure after helper sync:** restart log `/home/ubuntu/validation_logs/fp8_256k_decode_nki/qwen36_27b_32768_fp8_gdnseg512_expattn_mropefix_serve_20260604T2238Z.log` failed before model load with:

```text
./start_vllm_server.sh: line 679: exec: python: not found
```

**How we got there:** SSH launch invoked the contrib `start_vllm_server.sh` directly with the right Qwen/Neuron envs but did not source `/opt/aws_neuronx_venv_pytorch_inference_vllm_0_16/bin/activate`, so `python`/site packages were not on PATH. The earlier repeated `ModuleNotFoundError: No module named 'transformers'` sitecustomize lines were the same missing-venv symptom.

**Root-cause hypothesis:** launcher environment error only; no CTE graph was loaded and no coherence probe ran.

**Mitigation:** relaunch through the same script after sourcing the vLLM/Neuron venv.

**Relaunch after venv fix:** succeeded. Runtime log:

```text
/home/ubuntu/validation_logs/fp8_256k_decode_nki/qwen36_27b_32768_fp8_gdnseg512_expattn_mropefix_serve_20260604T2242Z.log
HEALTH_OK attempt=11
```

**Boundary probe result with mRoPE padding fix active:**

```text
nsent=90 ptok=1613 dt=1.988s bad=False text='\n\n<think>\n\n</think>\n\nBased on Fact 90, the city mentioned is'
nsent=96 ptok=1721 dt=1.634s bad=True text='\n\n<think>\n\n!!!!!!!!!!!!!'
```

**Debug evidence at failing `ptok=1721`:**

```text
attention_hit_len=0 restore_len=0 computed=tensor([[0]]) num_queries=tensor([[1721]])
prefill_bucket=2048 prefix_bucket=0
position_ids shape=(1, 2048) dtype=torch.int32 min=0 max=1720
rotary_position_ids shape=(3, 1, 2048) dtype=torch.int32 min=0 max=1720
finite=0/248320 nan=248320 posinf=0 neginf=0 argmax=[0] argmax_values=[nan]
```

**Conclusion:** mRoPE padding mismatch was real in the earlier log but is **not** the root cause. The failure still occurs on a cold active-only CTE2048 request with no APC restore/prefix hit and with aligned absolute/rotary positions. This leaves a compiled long-bucket computation issue: monolithic attention/lm-head path, packed CTE argument/layout path, or a layer-internal value blow-up in the 2048 graph. Since GDN was segmented to 512 in this artifact, the current evidence points away from cross-request/prefix state and toward the remaining monolithic CTE2048 work.

**Exact-bucket probe setup:** clean restart under log `/home/ubuntu/validation_logs/fp8_256k_decode_nki/qwen36_27b_32768_fp8_gdnseg512_expattn_exact2048_serve_20260604T2300Z.log`, health OK on attempt 12.

**Probe script failure:** the first exact-length script generated a local `target=2048 local_len=2048` prompt and sent it, but then failed while printing result due shell-quote mangling of `usage.get('prompt_tokens')`:

```text
NameError: name 'prompt_tokens' is not defined
```

**Context:** this is a probe-script bug, not a model failure. The exact-2048 request may already have executed; recover outcome from the serve log before rerunning or sending a second request.

**Recovered exact-2048 result from serve log:** the request did execute and failed with all-NaN logits even though the prompt exactly filled the CTE2048 bucket.

```text
apply prompt_len=2048 restore_len=0 suffix_len=2048 restore_slot=None commit_slot=0 input_shape=(1, 2048) output_shape=(1, 2048)
prepare request_id='cmpl-8a31bebfc13f0645-0-bf0ddf5b' attention_hit_len=0 request_prefix_len=2048 restore_len=0 commit_prefix_len=2048 restore_slot=None commit_slot=0 input_shape=(1, 2048) prepared_shape=(1, 2048) computed=tensor([[0]]) num_queries=tensor([[2048]]) restore_mask=tensor([0]) commit_mask=tensor([1])
pad-pre ... prefill_len=2048 prefix_len=0 prefill_bucket=2048 prefix_bucket=0
position_ids shape=(1, 2048) dtype=torch.int32 min=0 max=2047
rotary_position_ids shape=(3, 1, 2048) dtype=torch.int32 min=0 max=2047
finite=0/248320 nan=248320 posinf=0 neginf=0 argmax=[0] argmax_values=[nan]
```

**Artifact config recovered on runtime:**

```text
neuron_config.prefix_cte_attention_backend attention_cte
neuron_config.context_encoding_buckets [2048]
neuron_config.context_encoding_bucket_pairs [[2048, 0], [2048, 256], [2048, 512], [2048, 1024], [2048, 2048], [2048, 4096], [2048, 8192], [2048, 16384]]
use_qwen_hybrid_chunked_prefill True
use_qwen_hybrid_chunked_prefill_nki True
gdn_recurrent_cache_dtype float32
gdn_conv_cache_dtype bfloat16
```

**Code-path conclusion:** with `prefix_cte_attention_backend=attention_cte`, a cold prefix size of 0 leaves `active_block_table` as rank 1 in `model_wrapper.py`, so `cte_has_prefix_blocks=False`; attention layers receive `past_key_value=None` and route through `NeuronQwen35Attention.perform_prefill`, which calls one monolithic `_nkilib_flash_attn(..., use_causal_mask=True)` over the full 2048-token active sequence. The GDN path in this artifact is already segmented with `QWEN36_DELTANET_FUSED_SEGMENT_TOKENS=512`, so this exact-2048 failure is not evidence for GDN carry, APC restore, prefix-hit assembly, mRoPE padding, or padded-row selection.

**Allowed reference branch comparison:** `codex/full-fp8-qwen36` and `codex/nki-deltanet-decode-step` both have the same cold `perform_prefill -> _nkilib_flash_attn` shape for `attention_cte`, and both only enter the segmented path when the context-encoding block table is rank 2. In `model_wrapper.py`, the segmented backend includes `prefix_size + n_active_tokens` when sizing `active_block_table`; the attention_cte backend sizes it from `prefix_size` only. Therefore the old working branches do not prove monolithic CTE2048 attention is coherent; they prove the smaller/chunked bucket and decode paths were coherent.

**Best current root-cause hypothesis:** raw monolithic CTE2048 attention/layer graph is numerically or layout unstable for this Qwen3.6 head shape/path. This is the attention half of the NVIDIA/FlashQLA lesson: long-context speed should come from fused-but-chunked/segmented kernels, not one larger monolithic bucket.

**NVIDIA/GPU comparison checked:** current Qwen FlashQLA documentation says its speedup targets **GDN Chunked Prefill**, not monolithic long prefill, with operator fusion and algebraic reformulation preserving numerical precision. NVIDIA Megatron exposes `torch_chunk_gated_delta_rule(..., chunk_size=64)` and NeMo's Qwen3.5 CP wrapper runs causal conv + FLA gated delta on dense sequence order while preserving fp32 gate/state handling. This matches our coherent CTE512 evidence and does not support chasing larger unsegmented CTE2048 attention as the coherence fix.

**Profiling note:** Neuron profile capture can identify slow kernels/engine stalls from NEFF+NTFF, but it will not by itself prove the first tensor that becomes NaN unless the graph/kernel is instrumented with explicit value checks or `device_print`. The cheaper discriminator already ran: exact active-only CTE2048 fails all-NaN in the monolithic attention artifact. If profiling is still useful, use it for performance after coherence, or compile a debug-instrumented attention/GDN graph to print finite counts at layer boundaries.

**Local segmented-routing patch:** fixed `src/neuronx_distributed_inference/models/model_wrapper.py` so batched context-encoding with `prefix_cte_attention_backend=segmented_cte` does not collapse `active_block_table` to scalar zero when `prefix_bucket=0`. For cold CTE2048, segmented CTE needs the active block table even with no prefix; otherwise the model can fall back toward the same invalid/monolithic shape or feed a wrong block table.

**Test added:** `test_segmented_cte_cold_cte2048_keeps_active_block_table_when_batched` in `test/unit/models/test_prefix_caching_bucket_selection.py`; it asserts a cold 2048-token segmented CTE request preserves 8 active blocks per batch row.

**Local verification:** syntax check passed:

```text
python3 -m py_compile src/neuronx_distributed_inference/models/model_wrapper.py test/unit/models/test_prefix_caching_bucket_selection.py contrib/models/Qwen3.6-27B/src/modeling_qwen35.py
```

**Local test environment failures:** focused pytest cannot run on the local Mac without Neuron packages.

First command:

```text
python3 -m pytest test/unit/models/test_prefix_caching_bucket_selection.py -q
```

failed with:

```text
ModuleNotFoundError: No module named 'neuronx_distributed_inference'
```

Second command:

```text
PYTHONPATH=src python3 -m pytest test/unit/models/test_prefix_caching_bucket_selection.py -q
```

failed with:

```text
ModuleNotFoundError: No module named 'neuronx_distributed'
```

**Mitigation:** run the focused unit test on the EC2 Neuron venv after syncing the patched files to the remote checkout.

**Remote test attempt 1:** synced patched `model_wrapper.py` and `test_prefix_caching_bucket_selection.py` to compile host checkout `/home/ubuntu/inferentia-gdn-multihead-cte-20260531T1350Z`, then ran:

```text
cd /home/ubuntu/inferentia-gdn-multihead-cte-20260531T1350Z
source /opt/aws_neuronx_venv_pytorch_inference_vllm_0_16/bin/activate
PYTHONPATH=src python -m pytest test/unit/models/test_prefix_caching_bucket_selection.py -q
```

It failed during import because the compile host is CPU-only `r7i.24xlarge` and Neuron target was not specified:

```text
RuntimeError: Unsupported Platform - r7i.24xlarge
. If you want to compile on CPU, please supply a compiler target argument, with one of: trn1, inf2, trn1n, trn2, or trn3. Ex: "--target trn1"
```

**Mitigation:** rerun with `NEURON_PLATFORM_TARGET_OVERRIDE=trn2 NEURON_CC_FLAGS="--target trn2 --lnc 2"`.

**Remote test attempt 2:** full file command with target override:

```text
NEURON_PLATFORM_TARGET_OVERRIDE=trn2 NEURON_CC_FLAGS="--target trn2 --lnc 2" PYTHONPATH=src python -m pytest test/unit/models/test_prefix_caching_bucket_selection.py -q
```

Result:

```text
35 passed, 2 failed
```

The two failures are pre-existing batched Hybrid APC restore expectations, not the new segmented cold CTE2048 invariant:

```text
test_cte_batched_hybrid_apc_restore_padding_uses_full_attention_mask:
  padded_args[1].sum(dim=1) was tensor([256, 256]), expected tensor([272, 268])

test_cte_batched_hybrid_apc_restore_routes_mixed_warm_cold_to_compiled_shape:
  prefix_bucket was 256, expected 512
```

**Focused remote verification:** the new test passes by itself:

```text
NEURON_PLATFORM_TARGET_OVERRIDE=trn2 NEURON_CC_FLAGS="--target trn2 --lnc 2" PYTHONPATH=src python -m pytest test/unit/models/test_prefix_caching_bucket_selection.py::TestPrefixCachingBucketSelection::test_segmented_cte_cold_cte2048_keeps_active_block_table_when_batched -q
.
1 passed
```

**Compile driver prepared:** added `tmp_compile_qwen32k_segcte2048_gdnseg512.sh` for the next artifact:

```text
CTE bucket: 2048
prefix_cte_attention_backend: segmented_cte
prefix_cte_attention_segment_size: 512
QWEN36_DELTANET_FUSED_SEGMENT_TOKENS: 512
QWEN36_DELTANET_SOLVE_MODE: kkt_hier
QWEN36_DELTANET_SOLVE_SCAN_STEPS: 7
QKV: qkvnki_tiled
KV cache: fp8
GDN recurrent cache: float32
```

This is the first build in this sequence that chunks both attention and GDN for the CTE2048 cold path.

**Compile launched on compile host:**

```text
host: ubuntu@16.51.94.87
repo: /home/ubuntu/inferentia-gdn-multihead-cte-20260531T1350Z
driver: ./tmp_compile_qwen32k_segcte2048_gdnseg512.sh
PID: 586209
PIDFILE: /home/ubuntu/validation_logs/fp8_256k_decode_nki/qwen36_27b_32768_fp8_full_lmheadfp8_kvfp8_qkvnki_tiled_segmented_cte512_nki_decode_stable_probe32k_gdnseg512_gdnrecfloat32_sampletokens_outlogits_b256_cte2048_pfx16k_slots64_async_cteargmaxsafe_20260604T224804Z_kkt_hier_scan7_compile.pid
LOG: /home/ubuntu/validation_logs/fp8_256k_decode_nki/qwen36_27b_32768_fp8_full_lmheadfp8_kvfp8_qkvnki_tiled_segmented_cte512_nki_decode_stable_probe32k_gdnseg512_gdnrecfloat32_sampletokens_outlogits_b256_cte2048_pfx16k_slots64_async_cteargmaxsafe_20260604T224804Z_kkt_hier_scan7_compile.log
ENVLOG: /home/ubuntu/validation_logs/fp8_256k_decode_nki/qwen36_27b_32768_fp8_full_lmheadfp8_kvfp8_qkvnki_tiled_segmented_cte512_nki_decode_stable_probe32k_gdnseg512_gdnrecfloat32_sampletokens_outlogits_b256_cte2048_pfx16k_slots64_async_cteargmaxsafe_20260604T224804Z_kkt_hier_scan7_env.txt
ARTIFACT: /mnt/trainium_artifacts/qwen_artifacts/qwen36_27b_32768_fp8_full_lmheadfp8_kvfp8_qkvnki_tiled_segmented_cte512_nki_decode_stable_probe32k_gdnseg512_gdnrecfloat32_sampletokens_outlogits_b256_cte2048_pfx16k_slots64_async_cteargmaxsafe_20260604T224804Z_kkt_hier_scan7
WORKDIR: /mnt/trainium_artifacts/qwen_artifacts/_nxd_model_workdir_32768_fp8_full_lmheadfp8_kvfp8_qkvnki_tiled_segmented_cte512_gdnseg512_gdnrecfloat32_sampletokens_outlogits_b256_cte2048_cteargmaxsafe_20260604T224804Z_kkt_hier_scan7
```

**Initial compile log verification:**

```text
CONTEXT_TRACE_SHAPE ... "context_encoding_bucket_pairs": [[2048, 0], [2048, 256], [2048, 512], [2048, 1024], [2048, 2048], [2048, 4096], [2048, 8192], [2048, 16384]]
CONTEXT_TRACE_SHAPE ... "prefix_cte_attention_backend": "segmented_cte"
CONTEXT_TRACE_SHAPE ... "prefix_cte_attention_segment_size": 512
```

Env log confirms:

```text
QWEN36_DELTANET_FUSED_SEGMENT_TOKENS=512
QWEN36_DELTANET_SOLVE_MODE=kkt_hier
QWEN36_DELTANET_SOLVE_SCAN_STEPS=7
QWEN36_DELTANET_MULTIHEAD_CTE=1
GDN_RECURRENT_CACHE_DTYPE=float32
GDN_CONV_CACHE_DTYPE=bfloat16
ENABLE_QKV_NKI_KERNELS=1
```

**Automation:** existing thread heartbeat `monitor-qwen-gdnseg512-expattn-compile` was updated to monitor this new segmented CTE2048 compile every 10 minutes. After success it must verify artifact config, rsync EC2-to-EC2 compile host -> runtime host `ubuntu@16.51.179.180`, launch, run `/tmp/tmp_bisect_probe3.py`, targeted boundary probes at 146/485/1225/1721/2048/2500, scan logits logs for NaNs/fallback, then run the 16k usage-accounted OpenAI benchmark only if coherence passes.

**EC2-to-EC2 transfer precheck:** direct SSH from compile host to runtime without forwarded credentials failed:

```text
ssh ubuntu@16.51.94.87 'ssh -o BatchMode=yes -o ConnectTimeout=5 ubuntu@16.51.179.180 "echo RUNTIME_SSH_OK"'
ubuntu@16.51.179.180: Permission denied (publickey).
exit code: 255
```

**Root cause / mitigation:** compile host does not have its own runtime-host key loaded. Agent forwarding from the local session works:

```text
ssh -A ubuntu@16.51.94.87 'ssh -o BatchMode=yes -o ConnectTimeout=5 ubuntu@16.51.179.180 "echo RUNTIME_SSH_OK"'
RUNTIME_SSH_OK
```

For post-compile artifact transfer, use `ssh -A` into the compile host and then run `rsync` from compile host to runtime host. This still keeps the artifact movement EC2-to-EC2.

**Runtime validation helper prepared:** added and synced `/home/ubuntu/tmp_launch_qwen36_segcte2048.sh` on runtime host `ubuntu@16.51.179.180`.

Usage after artifact rsync:

```text
/home/ubuntu/tmp_launch_qwen36_segcte2048.sh \
  /mnt/trainium_artifacts/qwen_artifacts/<artifact_name> \
  /home/ubuntu/validation_logs/fp8_256k_decode_nki/<serve_log>.log
```

The helper:

- does not use `set -u`;
- sources `/opt/aws_neuronx_venv_pytorch_inference_vllm_0_16/bin/activate` before calling `start_vllm_server.sh`;
- kills stale `VLLM::EngineCore`, `serve_qwen36.py`, and `python.*vllm` processes;
- launches CTE2048 with `segmented_cte` artifact config, GDN segmentation env, `kkt_hier/scan7`, fp32 recurrent cache, KV FP8, qkvnki;
- waits for `/health` and prints `HEALTH_OK` or tails the serve log on failure.

**Runtime code sync:** synced patched files to runtime checkout:

```text
src/neuronx_distributed_inference/models/model_wrapper.py
src/neuronx_distributed_inference/modules/attention/nki_kernels/qwen_segcte256/attention_segmented_cte_256.py
src/neuronx_distributed_inference/modules/attention/nki_kernels/qwen_segcte256/fused_segmented_attention_256.py
contrib/models/Qwen3.6-27B/src/modeling_qwen35.py
contrib/models/Qwen3.6-27B/src/nki_kernels/nki_deltanet_fused_legacy.py
```

Runtime syntax verification passed:

```text
cd /home/ubuntu/inferentia-gdn-multihead-cte-20260531T1350Z
source /opt/aws_neuronx_venv_pytorch_inference_vllm_0_16/bin/activate
NEURON_PLATFORM_TARGET_OVERRIDE=trn2 NEURON_CC_FLAGS="--target trn2 --lnc 2" PYTHONPATH=src:contrib/models/Qwen3.6-27B python -m py_compile ...
exit code: 0
```

**Targeted boundary probe prepared:** added and synced `/tmp/tmp_qwen36_boundary_probe.py` on runtime host. It builds tokenizer-guided prompts near:

```text
146, 485, 1225, 1721, 2048, 2500
```

and calls `/v1/completions`, recording server prompt tokens, elapsed time, output snippet, and a simple degenerate-output flag.

**Probe syntax-check mistakes and fixes:**

1. Local command:

```text
python3 -m py_compile tmp_qwen36_boundary_probe.py tmp_launch_qwen36_segcte2048.sh
```

failed because `tmp_launch_qwen36_segcte2048.sh` is a shell script, not Python:

```text
File "tmp_launch_qwen36_segcte2048.sh", line 2
  set -eo pipefail
          ^^^^^^^^
SyntaxError: invalid syntax
```

Mitigation: reran only the Python probe:

```text
python3 -m py_compile tmp_qwen36_boundary_probe.py
exit code: 0
```

2. Runtime command:

```text
ssh ubuntu@16.51.179.180 'python -m py_compile /tmp/tmp_qwen36_boundary_probe.py ...'
```

failed outside the venv:

```text
bash: line 1: python: command not found
```

Mitigation: reran through the venv:

```text
source /opt/aws_neuronx_venv_pytorch_inference_vllm_0_16/bin/activate
python -m py_compile /tmp/tmp_qwen36_boundary_probe.py
exit code: 0
```

**Transient monitor poll error:** a delayed local poll command failed before reaching the compile host:

```text
ssh ubuntu@16.51.94.87 '...'
ssh: connect to host 16.51.94.87 port 22: Operation not permitted
exit code: 255
```

**Context:** the compile process was not inspected by that failed command; an immediate retry connected successfully and showed PID `586209` still running. Best hypothesis is a transient local network/sandbox permission blip, not compile-host failure. No mitigation needed beyond retrying the SSH poll.

**Repeat of same poll issue:** a second `sleep 60 && ssh ...` delayed poll failed with the same local error:

```text
ssh: connect to host 16.51.94.87 port 22: Operation not permitted
exit code: 255
```

Immediate SSH retry again succeeded. Mitigation: do not combine local `sleep` and SSH in one shell command for this monitor; use immediate SSH polls or the app heartbeat.

## 2026-06-05 Coherence Investigation Ledger

### Current target

Make the Qwen3.6 FP8 artifact coherent while keeping the fast cold-prefill path:

```text
qkvnki_tiled + segmented_cte attention + CTE2048 external bucket
GDN fused segment 512 + fp32 recurrent/checkpoint state
kkt_hier/scan7 + sample-token/output-logits contract
```

Current runtime host:

```text
ubuntu@16.51.179.180
/home/ubuntu/inferentia-gdn-multihead-cte-20260531T1350Z
```

Current artifact under test:

```text
/mnt/trainium_artifacts/qwen_artifacts/qwen36_27b_32768_fp8_full_lmheadfp8_kvfp8_qkvnki_tiled_segmented_cte512_nki_decode_stable_probe32k_gdnseg512_gdnrecfloat32_sampletokens_outlogits_b256_cte2048_pfx16k_slots64_async_cteargmaxsafe_20260604T224804Z_kkt_hier_scan7
```

### Things cleared so far

Do not re-open these unless new evidence contradicts them:

- Output tuple order is not the primary issue. The compiled forward outputs are still `[sample_token(int32), logits(fp32), KV..., deltanet_state..., checkpoint...]`, and alias guards did not fire in the successful compile.
- The old token/logits swap hypothesis is cleared by runtime evidence: when failure occurs, logits are all-NaN, not merely clean logits with a corrupted sampled-token slot.
- BF16 recurrent precision alone is not the root cause. CPU gates with BF16 carry across chunk boundaries passed; the hard failure is a cliff to NaN/sentinel token IDs, not gradual drift.
- `SCAN_STEPS=2` truncation and `SCAN_STEPS=7` repeated-squaring were useful hypotheses, but the currently tested build uses `kkt_hier/scan7`, and the remaining failure is gated by GDN checkpoint restore.
- `segmented_cte` attention kernel alone is not the hard NaN site. With segmented attention prefix still active, zeroing only `hybrid_restore_mask` made the 2500-token boundary coherent.
- Cold inactive-row leakage is fixed locally: `restore_to_active_rows(..., zero_inactive=True)` now zeroes inactive prefill rows instead of leaving stale `seq_id=0` state in the active buffers.
- Empty 2049 suffix prep is fixed locally by preserving and tensorizing `full_input_ids` in Qwen metadata.
- Re-entrant pad-time prep after debug-zeroing restore mask is fixed locally by tracking `_qwen36_hybrid_apc_controls_materialized`.
- Async routing for single-token cached prefill continuations now routes to CTE when candidate ownership/bridge metadata says the request is still context encoding.

### Verified local tests after patches

```text
PYTHONPATH=src pytest test/unit/modules/test_async_execution.py -q
68 passed

PYTHONPATH=src pytest contrib/models/Qwen3.6-27B/test/unit/test_qwen36_model_aliases.py -q
70 passed

python3 -m py_compile \
  contrib/models/Qwen3.6-27B/src/modeling_qwen35.py \
  src/neuronx_distributed_inference/modules/async_execution.py
passed
```

### Runtime evidence that isolates the remaining failure

Normal state-zero launch:

```text
/home/ubuntu/validation_logs/fp8_256k_decode_nki/qwen36_segcte2048_gdnseg512_statezero_serve_20260605T0350Z.log
```

The 2048 prefix is committed, then the 2500 request restores that checkpoint:

```text
prepare ... request_prefix_len=2048 restore_len=0 commit_prefix_len=2048 restore_slot=None commit_slot=3 ... restore_mask=tensor([0]) commit_mask=tensor([1])
prepare ... attention_hit_len=2048 request_prefix_len=2500 restore_len=2048 ... restore_slot=3 ... restore_mask=tensor([1])
pad-pre ... input_shape=(1,452) position_minmax=2048:2499 ... prefill_len=452 prefix_len=2048
```

Immediately after restore, logits become all NaN:

```text
Replacing invalid completed-prefill sampled token ... token_id=2147483647 (0x7fffffff) fallback_token_id=0
Qwen3.6 fallback logits summary ... logits_dtype=torch.float32 finite=0/248320 nan=248320 argmax=[0] argmax_values=[nan]
```

Zero-GDN-restore isolation launch:

```text
/home/ubuntu/validation_logs/fp8_256k_decode_nki/qwen36_segcte2048_gdnseg512_zero_restore_mask_statezero_serve_20260605T0405Z.log
```

This run keeps the attention prefix hit but zeros only `hybrid_restore_mask`:

```text
prepare ... attention_hit_len=2048 request_prefix_len=2500 restore_len=2048 restore_slot=0 ... restore_mask=tensor([0])
pad-pre ... input_shape=(1,452) position_minmax=2048:2499 ... prefill_len=452 prefix_len=2048
```

Observed 2500-token output was coherent:

```text
{"bad": false, "text": "\n\n<think>\nHere's a thinking process"}
```

This is the decisive split: segmented attention prefix read is still active, but GDN checkpoint restore is disabled. Therefore the remaining hard NaN is in GDN checkpoint commit/restore state, not attention CTE itself.

### Remaining live suspects

1. The checkpoint bank slot committed at prefix 2048 contains non-finite recurrent or conv state, even though the 2048 logits looked coherent enough to sample.
2. The checkpoint bank slot is finite but restored with the wrong layer order, dtype layout, or active row mapping.
3. The checkpoint bank slot is overwritten or reused incorrectly between the 2048 commit and 2500 restore.
4. The restored fp32 recurrent state and bf16 conv state are individually finite but their combination trips the first restored suffix CTE/GDN forward into NaN.

### 2026-06-05 update: zero-GDN-restore is not sufficient

Runtime candidate:

```text
QWEN36_ZERO_HYBRID_GDN_RESTORE_MASK=1
/home/ubuntu/validation_logs/fp8_256k_decode_nki/qwen36_segcte2048_gdnseg512_zero_gdn_restore_candidate_20260605T0500Z.log
```

This mode keeps segmented attention prefix reads active but forces GDN checkpoint restore mask to zero. It passed the older bisect probe:

```text
P1 short: coherent
160: coherent
123: coherent
485: coherent
1225: coherent
2500: coherent
```

It failed the exact boundary probe:

```text
1225: coherent
1265: coherent
1346: coherent
2048: "\n\n<think>\n!!!!!"       bad=true
2049: "!!!!!!!!"                  bad=true
2500: "!!!!!!!!"                  bad=true
4096: "!!!!!!!!"                  bad=true
```

The log proves the 2048 failure is a cold CTE failure, not GDN restore:

```text
prepare ... request_prefix_len=2048 restore_len=0 commit_prefix_len=2048 restore_slot=None commit_slot=1 ... restore_mask=tensor([0]) commit_mask=tensor([1])
Qwen3.6 fallback logits summary ... finite=0/248320 nan=248320 ... argmax_values=[nan]
```

For 2500 under zero-GDN-restore, attention prefix hit remains active but restore is zeroed:

```text
prepare ... attention_hit_len=2048 request_prefix_len=2500 restore_len=2048 restore_slot=3 ... restore_mask=tensor([0])
Qwen3.6 fallback logits summary ... finite=0/248320 nan=248320
```

Conclusion: there are two NaN surfaces:

- restored GDN checkpoint state can trigger NaN on some 2048-prefix suffixes;
- the CTE2048 graph itself can produce all-NaN logits for exact cold prompts even when GDN checkpoint restore is disabled.

Do not ship `QWEN36_ZERO_HYBRID_GDN_RESTORE_MASK=1` as the coherence fix. It is only an isolation tool.

### 2026-06-05 checkpoint state debug attempt

Added a local debug hook gated by `QWEN36_HYBRID_GDN_STATE_DEBUG=1` around `HybridGDNCheckpointCache.commit_from_active_rows` and `restore_to_active_rows`, then synced `modeling_qwen35.py` to runtime.

Verification:

```text
PYTHONPATH=src pytest contrib/models/Qwen3.6-27B/test/unit/test_qwen36_model_aliases.py -q
70 passed

python3 -m py_compile contrib/models/Qwen3.6-27B/src/modeling_qwen35.py
passed

runtime py_compile under Neuron venv:
PYCOMPILE_OK
```

First launch did not propagate `QWEN36_HYBRID_GDN_STATE_DEBUG` through the `nohup env` launcher. Fixed `tmp_launch_qwen36_segcte2048.sh` to pass:

```text
QWEN36_HYBRID_GDN_STATE_DEBUG="${QWEN36_HYBRID_GDN_STATE_DEBUG:-0}"
QWEN36_ZERO_HYBRID_GDN_RESTORE_MASK="${QWEN36_ZERO_HYBRID_GDN_RESTORE_MASK:-0}"
```

After relaunch, the failing 2500 probe reproduced:

```text
{"bad": true, "text": "\n\n<think>\nHere!!!!"}
```

But no `[qwen36_hybrid_gdn_state]` lines were emitted:

```text
grep -c "qwen36_hybrid_gdn_state" ...stateprobe2_20260605T0445Z.log
0
```

Best current explanation: the checkpoint bank update/restore for the compiled artifact is inside the compiled execution surface; Python-side print hooks around the source function are not reachable for row-value inspection at request time. A graph-visible debug output or a recompile is needed to inspect checkpoint tensor values directly.

### Next action

Do not start another compile just to test attention vs GDN restore. The runtime toggle already separated that.

The next useful compile must isolate the full-FP8 moat from the CTE2048 structure:

- qkv-only + segmented attention + GDN segment 512 + CTE2048, no full FP8 MLP/out-proj/lm-head moat;
- or qkv-only + CTE512 as the coherence baseline with the fastest currently safe chunk size.

Expected interpretation:

- qkv-only CTE2048 coherent means one of full-FP8 MLP/out-proj/lm-head/KV moats is corrupting the 2048 graph;
- qkv-only CTE2048 still NaN means the large CTE2048 structure is unsafe and the coherent path is chunked CTE512-style execution;
- qkv-only CTE512 coherent but slower gives the fallback target to optimize with profiling.

Only after that split should we compile again.

### Errors observed and mitigations

- `apply_patch` failed once while adding Qwen regression tests because the expected context around `test_request_scoped_vllm_metadata_is_added_for_hybrid_apc` had moved. Mitigation: reopened exact regions and applied smaller patches. Tests passed afterward.
- Boundary probe command failed once with an invalid integer:

```text
python /tmp/tmp_qwen36_boundary_probe.py --targets 1225 --max-tokens 1,2,3,4,5,6,7,8
argument --max-tokens: invalid int value: '1,2,3,4,5,6,7,8'
exit code: 2
```

Mitigation: reran with a single integer `--max-tokens` value per invocation.

- Running Python on the runtime host outside the Neuron venv failed:

```text
bash: line 1: python: command not found
```

Mitigation: source `/opt/aws_neuronx_venv_pytorch_inference_vllm_0_16/bin/activate` for runtime probes.

- Direct compile-host-to-runtime SSH initially failed:

```text
ubuntu@16.51.179.180: Permission denied (publickey)
exit code: 255
```

Mitigation: use agent forwarding from local to compile host, then EC2-to-EC2 rsync from compile host to runtime host:

```text
ssh -A ubuntu@16.51.94.87 'rsync ... ubuntu@16.51.179.180:...'
```

## 2026-06-05 BF16-Control CTE2048 Isolation Compile

Purpose: isolate whether the CTE2048 all-NaN cold-boundary failure is caused by the full-FP8 weight/cache moat or by the CTE2048 structure itself.

Compile host:

```text
ubuntu@16.51.94.87
```

Before launch, disk was too tight for another artifact:

```text
df -h /mnt/trainium_artifacts /home/ubuntu
/dev/root 484G 460G 24G 96%
```

Large stale failed artifacts were inspected with:

```text
du -h -d 1 /mnt/trainium_artifacts/qwen_artifacts | sort -h | tail -40
```

Cleanup correction/error:

- First attempted cleanup used the local `exec_command` `shell` parameter incorrectly with `shell="ssh ubuntu@16.51.94.87"`. It returned exit code 0 but did not free remote disk.
- Corrected command:

```text
ssh ubuntu@16.51.94.87 'rm -rf \
  /mnt/trainium_artifacts/qwen_artifacts/qwen36_27b_32768_fp8_full_lmheadfp8_kvfp8_hybrid_apc_nki_decode_stable_probe32k_gdnrecbfloat16_sampletokens_outlogits_b256_cte512_pfx16k_slots64_async_cteargmaxsafe_20260603T201758Z \
  /mnt/trainium_artifacts/qwen_artifacts/qwen36_27b_32768_fp8_full_lmheadfp8_kvfp8_qkvnki_tiled_attention_cte_nki_decode_stable_probe32k_gdnrecfloat32_sampletokens_outlogits_b256_cte2048_pfx16k_slots64_async_cteargmaxsafe_20260604T194122Z_ldir_kkt_hier_scan7'
```

After cleanup:

```text
/dev/root 484G 393G 91G 82%
```

Synced compile-relevant files to compile host:

```text
contrib/models/Qwen3.6-27B/test/integration/qwen36_27b_compile_fp8.py
contrib/models/Qwen3.6-27B/src/modeling_qwen35.py
contrib/models/Qwen3.6-27B/src/nki_kernels/nki_deltanet_fused_legacy.py
src/neuronx_distributed_inference/modules/async_execution.py
src/neuronx_distributed_inference/models/model_wrapper.py
src/neuronx_distributed_inference/modules/attention/gqa.py
src/neuronx_distributed_inference/modules/attention/nki_kernels/qwen_segcte256/fused_segmented_attention_256.py
src/neuronx_distributed_inference/modules/attention/nki_kernels/qwen_segcte256/attention_segmented_cte_256.py
```

Remote verification passed:

```text
bash -n /home/ubuntu/tmp_compile_qwen32k_bf16_qkvnki_segcte2048_gdnseg512.sh
SCRIPT_SYNTAX_OK

python -m py_compile ...synced files...
PYCOMPILE_OK
```

Compile driver:

```text
/home/ubuntu/tmp_compile_qwen32k_bf16_qkvnki_segcte2048_gdnseg512.sh
```

Launched:

```text
PID=595791
BASE=qwen36_27b_32768_bf16control_qkvnki_tiled_segmented_cte512_nki_decode_stable_probe32k_gdnseg512_gdnrecfloat32_sampletokens_outlogits_b256_cte2048_pfx16k_slots64_async_cteargmaxsafe_20260605T004451Z_kkt_hier_scan7
ARTIFACT=/mnt/trainium_artifacts/qwen_artifacts/qwen36_27b_32768_bf16control_qkvnki_tiled_segmented_cte512_nki_decode_stable_probe32k_gdnseg512_gdnrecfloat32_sampletokens_outlogits_b256_cte2048_pfx16k_slots64_async_cteargmaxsafe_20260605T004451Z_kkt_hier_scan7
WORKDIR=/mnt/trainium_artifacts/qwen_artifacts/_nxd_model_workdir_32768_bf16control_qkvnki_tiled_segmented_cte512_gdnseg512_gdnrecfloat32_sampletokens_outlogits_b256_cte2048_cteargmaxsafe_20260605T004451Z_kkt_hier_scan7
LOG=/home/ubuntu/validation_logs/fp8_256k_decode_nki/qwen36_27b_32768_bf16control_qkvnki_tiled_segmented_cte512_nki_decode_stable_probe32k_gdnseg512_gdnrecfloat32_sampletokens_outlogits_b256_cte2048_pfx16k_slots64_async_cteargmaxsafe_20260605T004451Z_kkt_hier_scan7_compile.log
ENVLOG=/home/ubuntu/validation_logs/fp8_256k_decode_nki/qwen36_27b_32768_bf16control_qkvnki_tiled_segmented_cte512_nki_decode_stable_probe32k_gdnseg512_gdnrecfloat32_sampletokens_outlogits_b256_cte2048_pfx16k_slots64_async_cteargmaxsafe_20260605T004451Z_kkt_hier_scan7_env.txt
PIDFILE=/home/ubuntu/validation_logs/fp8_256k_decode_nki/qwen36_27b_32768_bf16control_qkvnki_tiled_segmented_cte512_nki_decode_stable_probe32k_gdnseg512_gdnrecfloat32_sampletokens_outlogits_b256_cte2048_pfx16k_slots64_async_cteargmaxsafe_20260605T004451Z_kkt_hier_scan7_compile.pid
```

Confirmed intended isolation env:

```text
WEIGHT_DTYPE=bf16_control
ENABLE_KV_CACHE_QUANT=0
MEMORY_FLAGS=bf16_weights,bf16_lm_head,bf16_kv_cache,no_tkg_checkpoint_commit,gdn_recurrent_float32,gdn_conv_bfloat16
KERNELS=decode_deltanet,qkvnki_tiled,segmented_attention_cte
PREFIX_CTE_ATTENTION_BACKEND=segmented_cte
PREFIX_CTE_ATTENTION_SEGMENT_SIZE=512
QWEN36_DELTANET_FUSED_SEGMENT_TOKENS=512
CTE_BUCKETS=2048
```

Initial log showed clean startup and HLO generation:

```text
WEIGHT_DTYPE_MODE bf16_control
FP8_MODE disabled_bf16_control
QUANTIZE_LM_HEAD False
QUANTIZE_SKIP bf16_control
COMPILE_START
Generating 8 hlos for key: context_encoding_model
```

Automation:

```text
monitor-qwen-gdnseg512-expattn-compile
```

was updated to monitor this BF16-control compile every 10 minutes. Do not create a duplicate heartbeat; this thread supports only one active heartbeat.

## 2026-06-05 BF16-Control Runtime Verdict and Active-Carry Fix

Successful replacement compile after raw HF shards were restored:

```text
BASE=qwen36_27b_32768_bf16control_qkvnki_tiled_segmented_cte512_nki_decode_stable_probe32k_gdnseg512_gdnrecfloat32_sampletokens_outlogits_b256_cte2048_pfx16k_slots64_async_cteargmaxsafe_20260605T011731Z_kkt_hier_scan7
ARTIFACT=/mnt/trainium_artifacts/qwen_artifacts/qwen36_27b_32768_bf16control_qkvnki_tiled_segmented_cte512_nki_decode_stable_probe32k_gdnseg512_gdnrecfloat32_sampletokens_outlogits_b256_cte2048_pfx16k_slots64_async_cteargmaxsafe_20260605T011731Z_kkt_hier_scan7
COMPILE_LOG=/home/ubuntu/validation_logs/fp8_256k_decode_nki/qwen36_27b_32768_bf16control_qkvnki_tiled_segmented_cte512_nki_decode_stable_probe32k_gdnseg512_gdnrecfloat32_sampletokens_outlogits_b256_cte2048_pfx16k_slots64_async_cteargmaxsafe_20260605T011731Z_kkt_hier_scan7_compile.log
```

Compile success evidence:

```text
Finished Compilation for all HLOs in 419.2114279270172 seconds
CHECKPOINT_BANK_WEIGHTS_ADDED tp0..tp3 48 recurrent torch.float32 / 48 conv torch.bfloat16
COMPILE_DONE
```

Transferred EC2-to-EC2 from compile host to runtime host with agent-forwarded rsync. Runtime artifact verification matched compile host file sizes; runtime config had `quantized=false`, `kv_cache_quant=false`, `qkv_nki_kernel_enabled=true`, `prefix_cte_attention_backend=segmented_cte`, `prefix_cte_attention_segment_size=512`, `context_encoding_buckets=[2048]`.

Short/boundary coherence passed:

```text
/tmp/tmp_bisect_probe3.py: coherent through 2500
/tmp/tmp_qwen36_boundary_probe.py --targets 146,485,1225,2048,2049,2500,4096 --max-tokens 24: all bad=false
/tmp/tmp_qwen36_multiturn_probe.py --targets 160,1225,2500 --max-tokens 128: all bad=false, contains_paris=true
serve log scan: no fallback, no finite=0, no NaN summaries
```

Benchmark invocation error encountered:

```text
command: python /tmp/qwen36_chat_completion_context_bench.py --model-path /home/ubuntu/models/Qwen3.6-27B --lengths 16384 ...
failure: HTTP 404, "The model `Qwen3.6-27B` does not exist."
context: runtime server exposes model id `/home/ubuntu/models/Qwen3.6-27B`
mitigation: reran with --model /home/ubuntu/models/Qwen3.6-27B
```

16k benchmark then exposed the remaining bug:

```text
output: "!!!!!!!?????????????????????????????????????????????????????????"
usage-accounted prompt_tokens=16378 completion_tokens=64
cold prefill: group_effective_prompt_tokens_per_second=1426.879
decode: decode_tokens_per_second=13.871
serve log:
  Replacing invalid completed-prefill sampled token ... negative token_id=-2147483648 fallback_token_id=0
  Qwen3.6 fallback logits summary ... finite=0/248320 nan=248320
```

Important isolate:

- Simple long prompts at 4096, 6144, 8192, 10240, 12288, 14336, and 16384 were finite/coherent after restart.
- Benchmark-shaped repeated filler prompts NaN already at 4096.
- With raw output debug on the 4096 benchmark trigger, the same request split as:

```text
chunk 1: prompt_len=2048 restore_len=0 suffix_len=2048 commit_slot=0 -> finite logits
chunk 2: prompt_len=4092 restore_len=2048 suffix_len=2044 restore_slot=0 commit_slot=None -> all-NaN logits
```

Exact failing log:

```text
/home/ubuntu/validation_logs/fp8_256k_decode_nki/qwen36_bf16control_segcte2048_debug_4096_20260605T0220.log
raw_output[1] chunk 2 finite=0/248320 nan=248320 token=-2147483648
```

Shape-only was not sufficient: a simple 4092-token prompt used the same `2048 + 2044 + pad 4` shape and returned finite logits. So the failure is data-dependent in the restored GDN state path, not a deterministic padding-only attention failure.

Decisive runtime split:

```text
QWEN36_ZERO_HYBRID_GDN_RESTORE_MASK=1
same 4096 benchmark trigger -> finite token "ack"
log: chunk 2 restore_mask=tensor([0]), logits finite=248320/248320
```

Best current root cause:

```text
Same-request chunked prefill incorrectly round-trips GDN state through the Hybrid APC checkpoint bank.

For chunk N -> chunk N+1 in the same request, GDN should use active recurrent state carry, matching FLA/NVIDIA-style initial_state=previous_final_state. The checkpoint bank should be used for cross-request prefix hits/refill, not for immediate same-request continuation. The failing path commits slot 0 at 2048, then immediately restores slot 0 for the next suffix; that restored checkpoint state is data-dependent corrupt/explosive and can produce all-NaN logits. Zeroing restore avoids the bad slot and proves attention/segmented CTE is not the NaN source.
```

Code mitigation applied locally:

```text
src/neuronx_distributed_inference/modules/async_execution.py
  - same-request suffix continuation still uses attention prefix reads and can keep commit controls,
    but zeros only hybrid_restore_mask after suffix prep so GDN is carried as active state instead
    of restored from checkpoint bank.

contrib/models/Qwen3.6-27B/src/modeling_qwen35.py
  - context restore preserves inactive active GDN rows when hybrid_restore_prefix_lens is active
    but hybrid_restore_mask is zero; cold/no-prefix context still zeros inactive rows.

test/unit/modules/test_async_execution.py
  - same-request suffix tests now assert restore_mask=0 while keeping restore_prefix_lens.
```

Verification:

```text
python3 -m py_compile src/neuronx_distributed_inference/modules/async_execution.py contrib/models/Qwen3.6-27B/src/modeling_qwen35.py test/unit/modules/test_async_execution.py
PYTHONPATH=src python3 -m unittest test.unit.modules.test_async_execution.TestHybridAPCAsyncBridge
Ran 56 tests in 0.016s OK
```

Compile still required: the model-side `zero_inactive` change is trace/compile baked. Next build should be the same BF16-control CTE2048 artifact plus this active-carry fix. If it passes 4096 benchmark trigger, 16k benchmark, and boundary/multi-turn, re-enable the FP8 moat one component at a time.

Active-carry compile launched:

```text
HOST=ubuntu@16.51.94.87
PID=619240
BASE=qwen36_27b_32768_bf16control_qkvnki_tiled_segmented_cte512_nki_decode_stable_probe32k_gdnseg512_gdnrecfloat32_sampletokens_outlogits_b256_cte2048_pfx16k_slots64_async_cteargmaxsafe_20260605T023210Z_activecarry_kkt_hier_scan7
ARTIFACT=/mnt/trainium_artifacts/qwen_artifacts/qwen36_27b_32768_bf16control_qkvnki_tiled_segmented_cte512_nki_decode_stable_probe32k_gdnseg512_gdnrecfloat32_sampletokens_outlogits_b256_cte2048_pfx16k_slots64_async_cteargmaxsafe_20260605T023210Z_activecarry_kkt_hier_scan7
WORKDIR=/mnt/trainium_artifacts/qwen_artifacts/_nxd_model_workdir_32768_bf16control_qkvnki_tiled_segmented_cte512_gdnseg512_gdnrecfloat32_sampletokens_outlogits_b256_cte2048_cteargmaxsafe_20260605T023210Z_activecarry_kkt_hier_scan7
LOG=/home/ubuntu/validation_logs/fp8_256k_decode_nki/qwen36_27b_32768_bf16control_qkvnki_tiled_segmented_cte512_nki_decode_stable_probe32k_gdnseg512_gdnrecfloat32_sampletokens_outlogits_b256_cte2048_pfx16k_slots64_async_cteargmaxsafe_20260605T023210Z_activecarry_kkt_hier_scan7_compile.log
ENVLOG=/home/ubuntu/validation_logs/fp8_256k_decode_nki/qwen36_27b_32768_bf16control_qkvnki_tiled_segmented_cte512_nki_decode_stable_probe32k_gdnseg512_gdnrecfloat32_sampletokens_outlogits_b256_cte2048_pfx16k_slots64_async_cteargmaxsafe_20260605T023210Z_activecarry_kkt_hier_scan7_env.txt
PIDFILE=/home/ubuntu/validation_logs/fp8_256k_decode_nki/qwen36_27b_32768_bf16control_qkvnki_tiled_segmented_cte512_nki_decode_stable_probe32k_gdnseg512_gdnrecfloat32_sampletokens_outlogits_b256_cte2048_pfx16k_slots64_async_cteargmaxsafe_20260605T023210Z_activecarry_kkt_hier_scan7_compile.pid
```

Pre-launch cleanup:

```text
Deleted previous compile-host BF16-control artifact/workdir after runtime copy had been rsynced and verified.
Before cleanup: 46G free.
After cleanup: 107G free.
```

Remote verification before launch:

```text
PYTHONPATH=src python -m py_compile src/neuronx_distributed_inference/modules/async_execution.py contrib/models/Qwen3.6-27B/src/modeling_qwen35.py test/unit/modules/test_async_execution.py
PYTHONPATH=src python -m unittest test.unit.modules.test_async_execution.TestHybridAPCAsyncBridge
Ran 56 tests OK
bash -n /home/ubuntu/tmp_compile_qwen32k_bf16_qkvnki_segcte2048_gdnseg512.sh
```

Immediate compile poll:

```text
RUNNING pid=619240
WEIGHT_DTYPE_MODE bf16_control
FP8_MODE disabled_bf16_control
COMPILE_START
Generating 8 hlos for key: context_encoding_model
Finished generating HLO for context_encoding_model in 21.787s for first bucket
```

Automation updated:

```text
id=monitor-qwen-gdnseg512-expattn-compile
name=monitor-qwen-activecarry-compile
interval=10 minutes
```

## 2026-06-05 Active-Carry Compile Verdict

**Compile result:** successful.

```text
compile host: ubuntu@16.51.94.87
PID: 619240
artifact:
/mnt/trainium_artifacts/qwen_artifacts/qwen36_27b_32768_bf16control_qkvnki_tiled_segmented_cte512_nki_decode_stable_probe32k_gdnseg512_gdnrecfloat32_sampletokens_outlogits_b256_cte2048_pfx16k_slots64_async_cteargmaxsafe_20260605T023210Z_activecarry_kkt_hier_scan7
log:
/home/ubuntu/validation_logs/fp8_256k_decode_nki/qwen36_27b_32768_bf16control_qkvnki_tiled_segmented_cte512_nki_decode_stable_probe32k_gdnseg512_gdnrecfloat32_sampletokens_outlogits_b256_cte2048_pfx16k_slots64_async_cteargmaxsafe_20260605T023210Z_activecarry_kkt_hier_scan7_compile.log

Finished Compilation for all HLOs in 417.2448556423187 seconds
CHECKPOINT_BANK_WEIGHTS_ADDED tp0_sharded_checkpoint.safetensors 48 48 torch.float32 torch.bfloat16
CHECKPOINT_BANK_WEIGHTS_ADDED tp1_sharded_checkpoint.safetensors 48 48 torch.float32 torch.bfloat16
CHECKPOINT_BANK_WEIGHTS_ADDED tp2_sharded_checkpoint.safetensors 48 48 torch.float32 torch.bfloat16
CHECKPOINT_BANK_WEIGHTS_ADDED tp3_sharded_checkpoint.safetensors 48 48 torch.float32 torch.bfloat16
COMPILE_DONE
```

Safetensors metadata check on compile and runtime hosts:

```text
paths 4
tp0..tp3 stateish dtypes: {'torch.bfloat16': 96, 'torch.float32': 48}
```

Transfer:

```text
rsync EC2-to-EC2 compile host -> runtime host completed.
64,535,374,536 bytes transferred in ~5m50s.
Runtime artifact size: 61G.
Runtime disk after transfer: 18G free on /dev/root.
```

**Errors encountered and mitigations:**

1. Safetensors metadata probe command quoting failed:

```text
command:
ssh ubuntu@16.51.94.87 'source /opt/aws_neuronx_venv_pytorch_inference_vllm_0_16/bin/activate; ART=...; python -c "... \nfor p in paths: ..."'
error:
SyntaxError: unexpected character after line continuation character
exit code: 1
context: compile-host artifact metadata validation after COMPILE_DONE
hypothesis: shell-escaped newlines inside python -c were parsed literally.
mitigation: reran as a heredoc Python script.
verification: dtype metadata printed correctly for all four shards.
```

2. Initial EC2-to-EC2 rsync failed:

```text
command:
ssh ubuntu@16.51.94.87 'rsync -aH --partial --info=progress2 "$ART" ubuntu@16.51.179.180:/mnt/trainium_artifacts/qwen_artifacts/'
error:
ubuntu@16.51.179.180: Permission denied (publickey).
rsync error: unexplained error (code 255)
context: compile host did not have direct runtime SSH key.
hypothesis: local SSH agent key was not forwarded to compile host.
mitigation: verified `ssh -A ubuntu@16.51.94.87 'ssh ubuntu@16.51.179.180 true'` works; reran rsync through `ssh -A`.
verification: rsync completed successfully.
```

3. First runtime validation after compile still failed because only the artifact had been transferred:

```text
serve log:
prepare ... request_prefix_len=4092 restore_len=2048 restore_slot=0 ... restore_mask=tensor([1])
Replacing invalid completed-prefill sampled token ... negative token_id=-2147483648
Qwen3.6 fallback logits summary ... finite=0/248320 nan=248320
context: runtime host was still using old serve-time Python `async_execution.py`.
hypothesis: active-carry patch was compile-host/local only; scheduler input prep is runtime Python, not embedded solely in model.pt.
mitigation: scp patched `async_execution.py`, `modeling_qwen35.py`, and tests to runtime; ran py_compile and 56 async bridge tests.
verification: runtime tests passed, but validation still showed restore_mask=1 because the predicate missed the live final-chunk path.
```

4. Async predicate was too narrow:

```text
symptom:
final same-request chunk had `hybrid_prefill_completion_state=True`, so the "not complete" guard prevented active-carry override.
mitigation attempt:
added bridge-level same-request detection in `prepare_suffix_only_request`.
verification:
unit tests passed, but live logs still showed no `suffix-active-carry`.
root cause:
the live path was not `prepare_suffix_only_request`; it was `prepare_request` with full prompt metadata, and `apply_hybrid_apc_prefill_plan` sliced to the suffix internally.
```

5. Bridge request record disappeared between chunks:

```text
evidence:
finish_hybrid_apc_request calls `bridge.finish_request(prepared_request.request_id)` after every prefill chunk.
effect:
the same-request committed checkpoint key was removed before chunk N+1 prepared.
mitigation:
added a bounded `HybridAPCSchedulerBridge._same_request_committed_keys` map.
commit_prefill records keys committed by request id; both normal prefill and suffix-only prep check that map.
verification:
new unit tests cover:
  - same-request suffix-only path: restore_mask=0, restore_prefix_lens kept.
  - same-request full-prompt slice path: restore_mask=0, restore_prefix_lens kept.
  - different request using the same checkpoint still restore_mask=1.
local, runtime, and compile host tests passed:
  Hybrid APC manager: 56 tests OK
  TestHybridAPCAsyncBridge: 56 tests OK
```

**Final code fix shape:**

```text
contrib/models/Qwen3.6-27B/src/hybrid_apc.py
  - apply_hybrid_apc_prefill_plan(..., gdn_active_carry=False)
  - apply_hybrid_apc_suffix_prefill_plan(..., gdn_active_carry=False)
  - when checkpoint key was committed by the same request, keep attention prefix reads and restore_prefix_lens, but set hybrid_restore_mask=0.
  - cross-request prefix hits still use restore_mask=1.

src/neuronx_distributed_inference/modules/async_execution.py
  - still contains the generic same-request suffix active-carry override, but the live path is now fixed in the bridge.

contrib/models/Qwen3.6-27B/src/modeling_qwen35.py
  - preserves active GDN rows when restore_prefix_lens is active but restore_mask=0.
```

**Runtime validation after final bridge fix:**

Serve log:

```text
/home/ubuntu/validation_logs/fp8_256k_decode_nki/qwen36_activecarry_fullpath_segcte2048_serve_20260605T0348.log
```

4096 benchmark-shaped failure trigger:

```text
command:
python /tmp/qwen36_chat_completion_context_bench.py --lengths 4096 --max-tokens 1 --unique-per-request ...
result:
status=200
prompt_tokens=4092
content_text="ack"
group_effective_prompt_tokens_per_second=2050.20

log:
prefill-active-carry request_id=... prefix_len=2048 slot=0
apply prompt_len=4092 restore_len=2048 suffix_len=2044 restore_slot=0 commit_slot=None gdn_active_carry=True
prepare ... restore_mask=tensor([0]) hybrid_restore_prefix_lens=2048
invalid-token/fallback count: 0
```

Short/boundary:

```text
/tmp/tmp_bisect_probe3.py:
  coherent at 5, 160, 123, 485, 1225; 2500 returned empty in that loose script.

/tmp/tmp_qwen36_boundary_probe.py --targets 146,485,1225,2048,2049,2500,4092,4096,16384 --max-tokens 24
  146 bad=false
  485 bad=false
  1225 bad=false
  2048 bad=false
  2049 bad=false
  2500 bad=false
  4092 bad=false
  4096 bad=false
  16384 bad=true with empty/EOS one-token output, but serve log had no invalid token fallback and no finite=0 logits.
```

Multi-turn:

```text
/tmp/tmp_qwen36_multiturn_probe.py --targets 160,1225,2500 --max-tokens 128
  160 bad=false contains_paris=true
  1225 bad=false contains_paris=true
  2500 bad=false contains_paris=true
```

16k usage-accounted benchmark:

```text
command:
python /tmp/qwen36_chat_completion_context_bench.py --lengths 16384 --max-tokens 64 --unique-per-request --output-json /tmp/qwen36_activecarry_16k_bench.json

result:
status=200
prompt_tokens=16378
completion_token_source=usage
completion_tokens=4
content_text="ack 7"
group_effective_prompt_tokens_per_second=2445.2418
decode_tokens_per_second=14.5753
token_tpot_seconds=0.068609
```

Final serve-log scan:

```text
grep -Ec "fallback logits summary|Replacing invalid|finite=0|negative token_id|2147483647" qwen36_activecarry_fullpath_segcte2048_serve_20260605T0348.log
0
```

**Current conclusion:**

- The same-request GDN checkpoint-bank roundtrip was the proven all-NaN/coherence bug for the fast CTE2048 BF16-control artifact.
- The corrected behavior is NVIDIA/FLA-style active state carry inside one request, with checkpoint-bank restore reserved for cross-request prefix reuse.
- BF16-control + QKV-NKI + segmented attention CTE2048 + GDN segment 512 is coherent through short, chunk boundary, multi-turn, and the benchmark-shaped 16k trigger, with no invalid-token fallback.
- This is not yet the final full-FP8 performance artifact. Next step is to re-enable the FP8 moat incrementally on top of this code fix and keep the same validation sequence.

Automation:

```text
Deleted stale heartbeat automation:
id=monitor-qwen-gdnseg512-expattn-compile
name snapshot=monitor-qwen-activecarry-compile
```

## 2026-06-05 Full-FP8 Active-Carry Validation Verdict

Compiled artifact:

```text
/mnt/trainium_artifacts/qwen_artifacts/qwen36_27b_32768_fp8_full_lmheadfp8_kvfp8_qkvnki_tiled_segmented_cte512_nki_decode_stable_probe32k_gdnseg512_gdnrecfloat32_sampletokens_outlogits_b256_cte2048_pfx16k_slots64_async_cteargmaxsafe_20260605T034039Z_activecarryfp8_kkt_hier_scan7
```

Compile host:

```text
ubuntu@16.51.94.87
compile log:
/home/ubuntu/validation_logs/fp8_256k_decode_nki/qwen36_27b_32768_fp8_full_lmheadfp8_kvfp8_qkvnki_tiled_segmented_cte512_nki_decode_stable_probe32k_gdnseg512_gdnrecfloat32_sampletokens_outlogits_b256_cte2048_pfx16k_slots64_async_cteargmaxsafe_20260605T034039Z_activecarryfp8_kkt_hier_scan7_compile.log
```

Compile result:

```text
INFO:Neuron:Generated all HLOs in 192.3236801624298 seconds
INFO:Neuron:Finished Compilation for all HLOs in 463.4065647125244 seconds
CHECKPOINT_BANK_WEIGHTS_ADDED tp0_sharded_checkpoint.safetensors 48 48 torch.float32 torch.bfloat16
CHECKPOINT_BANK_WEIGHTS_ADDED tp1_sharded_checkpoint.safetensors 48 48 torch.float32 torch.bfloat16
CHECKPOINT_BANK_WEIGHTS_ADDED tp2_sharded_checkpoint.safetensors 48 48 torch.float32 torch.bfloat16
CHECKPOINT_BANK_WEIGHTS_ADDED tp3_sharded_checkpoint.safetensors 48 48 torch.float32 torch.bfloat16
COMPILE_DONE
```

Safetensors dtype verification:

```text
tp0..tp3 each:
  {'torch.bfloat16': 402, 'torch.float32': 529, 'torch.float8_e4m3fn': 481, 'torch.int32': 17}
  stateish_fp32=48
  stateish_bf16=48
```

Transfer:

```text
rsync source host: ubuntu@16.51.94.87
destination host: ubuntu@16.51.179.180
command shape:
ssh -A ubuntu@16.51.94.87 'rsync -aH --numeric-ids --info=progress2 ARTIFACT ubuntu@16.51.179.180:/mnt/trainium_artifacts/qwen_artifacts/'
result: success, 38,979,975,275 bytes transferred, runtime copy size 37G.
```

Runtime launch:

```text
serve log:
/home/ubuntu/validation_logs/fp8_256k_decode_nki/qwen36_activecarryfp8_segcte2048_serve_20260605T0415.log
health: HTTP 200
```

4096 one-token trigger passed:

```text
command:
python /tmp/qwen36_chat_completion_context_bench.py --lengths 4096 --repeats 1 --concurrency 1 --max-tokens 1 --timeout 600 --unique-per-request --output-json /tmp/qwen36_fp8_activecarry_bench4096_one.json

result:
status=200
prompt_tokens=4092
completion_tokens=1
content_text="ack"
group_effective_prompt_tokens_per_second=2145.3980

log:
prefill-active-carry request_id=... prefix_len=2048 slot=0
prepare ... request_prefix_len=4092 restore_len=2048 ... restore_mask=tensor([0]) ... hybrid_restore_prefix_lens=2048
invalid-token/fallback count before later probes: 0
```

Full-FP8 coherence failure:

```text
/tmp/tmp_bisect_probe3.py:
  5 / 160 / 123 / 485: coherent
  1225: " Paris is mentioned!!!!!!!!!"
  2500: empty

/tmp/tmp_qwen36_boundary_probe.py --targets 146,160,485,505,526,1225,2048,2049,2500,4092,4096 --max-tokens 32:
  146 bad=false
  160 bad=false
  485 bad=false
  505 bad=false
  526 bad=true  text starts coherent then "!!!!!!!!!!!!!!!!!!"
  1225 bad=true text starts coherent then "!!!!!!!!!!!!!!!!!!!!!!!!"
  2048 bad=true
  2049 bad=true
  2500 bad=true
  4092 bad=false in this ordered run, likely contaminated by prefix/cache sequence.
  4096 bad=false in this ordered run, likely contaminated by prefix/cache sequence.
```

Clean unique-prompt decode matrix after restart:

```text
serve log:
/home/ubuntu/validation_logs/fp8_256k_decode_nki/qwen36_activecarryfp8_decode_matrix_serve_20260605T0425.log

505 prompt:
  max_tokens=1 returned only whitespace, marked bad by probe heuristic.
  max_tokens=2/4/8/12/16 mostly coherent.
  max_tokens=24 bad: "<think>...!!!!!!!!!!!!!!!!"
  max_tokens=32 coherent/early EOS in one salt.

526 prompt:
  max_tokens=1 whitespace only, marked bad by probe heuristic.
  max_tokens=2/4/8 coherent or early EOS.
  max_tokens=12 bad: "<think>\n!!!!!!!!!"
  max_tokens=16/24/32 bad with repeated "!"; collapse occurs during decode.

1225 prompt:
  max_tokens=1 whitespace only, marked bad by probe heuristic.
  max_tokens=4/8/16 coherent or early EOS in sampled salts.
```

Clean decode-matrix log scan:

```text
grep -Ec "fallback logits summary|Replacing invalid|finite=0|negative token_id|2147483647" qwen36_activecarryfp8_decode_matrix_serve_20260605T0425.log
150

Representative log:
Replacing invalid completed-prefill sampled token with logits argmax before vLLM output update:
  stage=sample_on_device row=0 negative token_id=-1047477664 fallback_token_id=0 prefill_completion_state=[True]
Qwen3.6 fallback logits summary:
  logits_shape=(1, 1, 248320) logits_dtype=torch.float32 finite=0/248320 nan=248320
```

Important interpretation:

```text
- The fast-path active-carry bug is fixed for BF16-control; full-FP8 still fails.
- The full-FP8 failure is not segmented attention CTE2048 or same-request checkpoint restore:
  4096 one-token prefill trigger passes, and failures occur during token-generation sampling after several generated tokens.
- The failure is not a CTE2048 chunk-boundary-only bug:
  526-token prompts fail with max_tokens>=12, below a 2048 CTE boundary but just past the 512 GDN segment/commit boundary.
- Since BF16-control with the same active-carry/GDNseg512/segmented-attention shape was coherent, the remaining culprit is FP8-specific:
  likely FP8 KV-cache/token-generation interaction, FP8 qkv/lm_head, or FP8 quantized linear weights feeding decode state.
```

Errors encountered during this validation:

```text
1. Runtime cleanup command exited 255 with no stdout:
   command:
   ssh ubuntu@16.51.179.180 'pkill -f serve_qwen36.py || true; pkill -f "VLLM::EngineCore" || true; sleep 5; rm -rf BF16_ARTIFACTS; df -h /; ...'
   context:
   runtime host 16.51.179.180, old BF16-control server was running, disk had 18G free.
   best hypothesis:
   pkill -f serve_qwen36.py matched the remote ssh shell command line and killed the shell before rm/df could run.
   mitigation:
   checked state, confirmed server stopped but artifacts remained, then ran deletion only:
   ssh ubuntu@16.51.179.180 'rm -rf BF16_ARTIFACT_1 BF16_ARTIFACT_2; df -h /'
   result:
   success, free space increased to 138G.

2. Runtime stop helper warning:
   command:
   ssh ubuntu@16.51.179.180 'kill $(cat PIDFILE) ...; pgrep -x "VLLM::EngineCore" | xargs -r kill ...'
   warning:
   pgrep: pattern that searches for process name longer than 15 characters will result in zero matches
   best hypothesis:
   Linux process-name matching with -x truncates names over 15 chars; VLLM::EngineCore must be found by full command line via ps filtering.
   mitigation:
   later used:
   ps -axo pid,command | awk '/VLLM::EngineCore/ && !/awk/ {print $1}' | xargs -r kill -9

3. Older comparison artifact launch stuck:
   artifact:
   /mnt/trainium_artifacts/qwen_artifacts/qwen36_27b_32768_fp8_full_lmheadfp8_kvfp8_qkvnki_tiled_attention_cte_nki_decode_stable_probe32k_gdnrecfloat32_sampletokens_outlogits_b256_cte2048_pfx16k_slots64_async_cteargmaxsafe_20260604T174349Z_aliasguard_statefix_kkt_hier_scan7
   command:
   bash /home/ubuntu/tmp_launch_qwen36_segcte2048.sh ARTIFACT /home/ubuntu/validation_logs/fp8_256k_decode_nki/qwen36_compare_fp8_cte2048_nogdnseg_serve_20260605T0432.log
   failure:
   did not reach /health after about 4 minutes; log remained at "Loading presharded checkpoints"; API pid 80207 and EngineCore pid 80335 remained resident.
   best hypothesis:
   stale artifact/runtime-code mismatch or loader hang; not worth using as evidence for coherence.
   mitigation:
   killed API/EngineCore/resource_tracker with explicit SIGKILL via ps/awk PID filtering.
   verification:
   no vLLM/EngineCore/resource_tracker processes remained.
```

Next recommended isolation:

```text
Do not compile another attention_cte/segmented_cte variant for this specific failure.

The next compile should split the FP8 moat on top of the proven active-carry code:
  1. Prefer full FP8 weights + lm_headfp8 but KV cache BF16 (kv_cache_quant=false) with same QKV-NKI tiled + GDNseg512 + segmented CTE2048.
     Reason: the observed collapse is in token generation after prefill, and KV FP8 is the highest-leverage decode-only difference from BF16-control.
  2. If KV-BF16 still fails, disable lm_head FP8 next while keeping FP8 body weights.
  3. If that still fails, isolate QKV-NKI tiled vs standard_qkv under the same active-carry code.

For every compile: create/update a heartbeat automation before launch; after compile, rsync EC2-to-EC2, run 4096 one-token, unique-prompt decode matrix at 505/526, boundary, multi-turn, then only run 16k usage-accounted benchmark if invalid-token/logit fallback count stays zero.
```

## 2026-06-05 NVIDIA/AWS FP8 policy check

User asked to check NVIDIA/Qwen and AWS docs because the production FP8 stack is
not "everything FP8." That is correct.

Observed external policy:

```text
Qwen/Qwen3-Next-80B-A3B-Instruct-FP8 config:
  quantization_config fmt=e4m3, quant_method=fp8, activation_scheme=dynamic,
  weight_block_size=[128,128].
  modules_to_not_convert explicitly includes:
    lm_head
    model.embed_tokens / model norm and layer norms
    self_attn.q_norm / self_attn.k_norm
    linear_attn.A_log, conv1d, dt_bias, in_proj_ba, norm
    MLP router/gate-style modules such as mlp.gate and shared_expert_gate.

Qwen/Qwen3.5-27B-FP8 config:
  text config keeps mamba_ssm_dtype=float32.
  modules_to_not_convert includes lm_head and linear_attn conv/gate pieces
  such as in_proj_a/in_proj_b.

NVIDIA TensorRT-LLM / ModelOpt:
  ModelOpt disables *lm_head* quantization by default; quantizing lm_head is
  an opt-in path.
  TRT-LLM docs say KV cache is not quantized by default; FP8 KV cache is an
  additional feature with explicit quality-risk caveat.

vLLM quantized KV docs:
  FP8 KV without calibration uses default scale=1.0.
  Recommended highest-quality path uses calibration; FP8 KV attention may
  quantize Q/K/V in the attention computation too.

AWS Trainium2/NKI docs:
  NeuronCore-v3 TensorEngine supports FP8_E4/FP8_E5 matmul inputs at double
  BF16/FP16 throughput with FP32 accumulation. This documents FP8 as a fast
  matmul operand mode, not a blanket requirement to store logits, KV, recurrent
  state, or numerically sensitive control/state tensors in FP8.
```

Local code comparison:

```text
contrib/models/Qwen3.6-27B/test/integration/qwen36_27b_compile_fp8.py already
matches that policy by default:
  --quantize-lm-head is opt-in.
  help text says default keeps lm_head BF16, matching common NVIDIA/vLLM policy.
  --enable-kv-cache-quant is opt-in direct-cast KV FP8.
  GDN recurrent cache is float32 in the active coherent path.

The temporary compile driver was more aggressive than the harness default:
  it hardcoded --quantize-lm-head and originally defaulted KV FP8.
```

Driver fix applied:

```text
file:
  tmp_compile_qwen32k_segcte2048_gdnseg512.sh

changes:
  Added QUANTIZE_LM_HEAD, default 0.
  Artifacts now tag lmheadbf16 vs lmheadfp8.
  Env log MEMORY_FLAGS now says lm_head_bf16 or lm_head_fp8 truthfully.
  Python compile command now passes --quantize-lm-head only when
  QUANTIZE_LM_HEAD=1.

verification:
  local: bash -n tmp_compile_qwen32k_segcte2048_gdnseg512.sh
  local: QUANTIZE_LM_HEAD=0 ENABLE_KV_CACHE_QUANT=0 TS=DRYRUN bash -n ...
  remote scp to ubuntu@16.51.94.87 succeeded.
  remote: bash -n /home/ubuntu/tmp_compile_qwen32k_segcte2048_gdnseg512.sh
  remote grep verified QUANTIZE_LM_HEAD default 0 and artifact/env tags.
```

Compile harness FP8 policy fix:

```text
file:
  contrib/models/Qwen3.6-27B/test/integration/qwen36_27b_compile_fp8.py

change:
  Full-FP8 manual checkpoint conversion no longer quantizes
  linear_attn.in_proj_a.weight or linear_attn.in_proj_b.weight.
  modules_to_not_convert also now includes:
    linear_attn.in_proj_a
    linear_attn.in_proj_b
    linear_attn.in_proj_ba

reason:
  Official Qwen3.5/Qwen3-Next FP8 configs keep these DeltaNet/gate-control
  projections out of FP8 (Qwen3.5 uses in_proj_a/in_proj_b; Qwen3-Next uses
  combined in_proj_ba). They are tiny compared with in_proj_qkv/in_proj_z/out_proj,
  so the speed cost should be negligible while removing a precision-sensitive
  decode path.

tests:
  local direct file:
  PYTHONPATH=src python3 contrib/models/Qwen3.6-27B/test/unit/test_qwen36_compile_fp8_config.py
  result: 47 tests OK.

  remote compile host:
  cd /home/ubuntu/inferentia-gdn-multihead-cte-20260531T1350Z
  source /opt/aws_neuronx_venv_pytorch_inference_vllm_0_16/bin/activate
  PYTHONPATH=src python contrib/models/Qwen3.6-27B/test/unit/test_qwen36_compile_fp8_config.py
  result: 47 tests OK.
```

Current in-flight compile remains useful but is not the final NVIDIA-style
replica:

```text
PID: 635224 on ubuntu@16.51.94.87
artifact:
/mnt/trainium_artifacts/qwen_artifacts/qwen36_27b_32768_fp8_full_lmheadfp8_kvbf16_qkvnki_tiled_segmented_cte512_nki_decode_stable_probe32k_gdnseg512_gdnrecfloat32_sampletokens_outlogits_b256_cte2048_pfx16k_slots64_async_cteargmaxsafe_20260605T043000Z_kkt_hier_scan7

purpose:
  isolates KV FP8 by using KV BF16, but still quantizes lm_head FP8.

next if it fails:
  run QUANTIZE_LM_HEAD=0 ENABLE_KV_CACHE_QUANT=0 with same active-carry,
  qkvnki_tiled, segmented_cte512, GDNseg512 settings.

next if that also fails:
  exclude Qwen3.5/3.6 linear-attention gate projections in_proj_a/in_proj_b
  from full FP8, because official Qwen FP8 configs keep those gate pieces
  higher precision.
```

Intermediate lmhead-FP8/KV-BF16 validation:

```text
artifact:
/mnt/trainium_artifacts/qwen_artifacts/qwen36_27b_32768_fp8_full_lmheadfp8_kvbf16_qkvnki_tiled_segmented_cte512_nki_decode_stable_probe32k_gdnseg512_gdnrecfloat32_sampletokens_outlogits_b256_cte2048_pfx16k_slots64_async_cteargmaxsafe_20260605T043000Z_kkt_hier_scan7

compile result:
  Finished Compilation for all HLOs.
  CHECKPOINT_BANK_WEIGHTS_ADDED tp0..tp3 48 recurrent torch.float32 and
  48 conv torch.bfloat16.
  COMPILE_DONE.

served on runtime:
  /home/ubuntu/validation_logs/fp8_256k_decode_nki/qwen36_fp8_lmheadfp8_kvbf16_segcte2048_serve_20260605T0515.log

4096 one-token:
  content_text="ack"
  prompt_tokens=4092
  group_effective_prompt_tokens_per_second=2120.38
  invalid-token/logit fallback count after trigger: 0

boundary max_tokens=32:
  146/160/485/505/526/1225/2048/2049/4092/4096 coherent.
  2500 returned empty/EOS and was marked bad by the probe.

multi-turn:
  160/1225/2500 all coherent, contains Paris, no bad marker.

log scan:
  invalid-token/logit fallback count remained 0.

interpretation:
  KV BF16 removes the catastrophic NaN/"!" decode corruption seen in the
  full-FP8 KV artifact. Remaining 2500 empty/EOS is not a NaN path. Because
  this artifact still quantizes lm_head and was built before the in_proj_a/b
  BF16 policy change, it is an intermediate result, not the final replica.
```

Final NVIDIA/Qwen-style compile launch:

```text
cleanup:
  Removed the intermediate lmheadfp8_kvbf16 compile-host artifact and workdir
  after it was copied to runtime, freeing compile host space from 65G to 103G.

bad launch caught and aborted:
  TS=20260605T050732Z ENABLE_KV_CACHE_QUANT=0 QUANTIZE_LM_HEAD=0 ...
  failure:
    log printed QUANTIZE_SKIP existing checkpoint found.
  context:
    _quantized/qwen36_27b_fp8_full_lmheadbf16 already existed from before the
    in_proj_a/in_proj_b BF16 policy change, so this would have reused stale
    quantized gate weights.
  mitigation:
    killed PID 645354 before full compile.

driver fix:
  tmp_compile_qwen32k_segcte2048_gdnseg512.sh now supports FORCE_QUANTIZE=1
  and passes --force-quantize to the compile harness.
  local syntax checks passed.
  remote script syntax check passed.

correct relaunch:
  TS=20260605T050957Z ENABLE_KV_CACHE_QUANT=0 QUANTIZE_LM_HEAD=0 FORCE_QUANTIZE=1 bash /home/ubuntu/tmp_compile_qwen32k_segcte2048_gdnseg512.sh
  PID=646654
  artifact:
  /mnt/trainium_artifacts/qwen_artifacts/qwen36_27b_32768_fp8_full_lmheadbf16_kvbf16_qkvnki_tiled_segmented_cte512_nki_decode_stable_probe32k_gdnseg512_gdnrecfloat32_sampletokens_outlogits_b256_cte2048_pfx16k_slots64_async_cteargmaxsafe_20260605T050957Z_kkt_hier_scan7
  early log:
    QUANTIZE_LM_HEAD False
    enable_kv_cache_quant=false
    QUANTIZE_START manual_fp8
```

Errors encountered during the docs/driver pass:

```text
1. Lost local exec session after compaction:
   command/tool:
   write_stdin session_id=85242
   failure:
   Unknown process id 85242
   context:
   The compile was started before context compaction; local PTY session id did
   not survive, but the remote compile process itself was independent.
   mitigation:
   switched to direct SSH PID/log checks:
   ssh ubuntu@16.51.94.87 'ps -p 635224 -o pid,etime,stat,cmd'
   ssh ubuntu@16.51.94.87 'tail -120 COMPILE_LOG'
   verification:
   remote PID 635224 still running; compile log showed passing HLO compilation.

2. Remote rg missing:
   command:
   ssh ubuntu@16.51.94.87 'rg -n "QUANTIZE_LM_HEAD|BASE=|MEMORY_FLAGS|--quantize-lm-head" /home/ubuntu/tmp_compile_qwen32k_segcte2048_gdnseg512.sh'
   failure:
   bash: line 1: rg: command not found
   exit code: 127
   context:
   compile host does not have ripgrep installed.
   mitigation:
   used grep instead.
   verification:
   grep output showed QUANTIZE_LM_HEAD default 0, lmhead artifact tags,
   truthful MEMORY_FLAGS, and optional --quantize-lm-head flag array.

3. Local unittest import path issue:
   command:
   PYTHONPATH=src python3 -m unittest contrib/models/Qwen3.6-27B/test/unit/test_qwen36_compile_fp8_config.py
   failure:
   ModuleNotFoundError: No module named 'contrib.models.Qwen3'
   exit code: 1
   context:
   unittest module-name loading treats the path component Qwen3.6-27B as a
   dotted Python package path, which breaks on the dot/hyphen model directory.
   mitigation:
   reran by direct file path:
   PYTHONPATH=src python3 contrib/models/Qwen3.6-27B/test/unit/test_qwen36_compile_fp8_config.py
   verification:
   47 tests OK.

4. Web raw Hugging Face config open blocked:
   command/tool:
   web open of https://huggingface.co/.../raw/main/config.json URLs.
   failure:
   "URL ... is not safe to open; You can only use the exact same URL from the
   previous search results or the user's message"
   context:
   documentation check attempted raw config URLs after opening the HF blob pages.
   mitigation:
   used the already-opened HF blob config pages from search results and their
   visible config lines instead.
  verification:
  extracted the needed quantization_config/modules_to_not_convert details from
  the opened HF config pages.
```

2026-06-05 follow-up: NVIDIA/Qwen FP8 policy and AWS/NKI check
----------------------------------------------------------------

User asked to verify the NVIDIA/Qwen implementation because not everything in
their served FP8 stack is actually FP8, then replicate the same policy on
Trainium.

Primary source findings:

* Qwen/Qwen3.5-27B-FP8 `config.json`:
  * `quantization_config.quant_method = fp8`
  * `activation_scheme = dynamic`
  * `weight_per_tensor = false`, `act_per_tensor = false`
  * `weight_block_size = [128, 128]`
  * `modules_to_not_convert` includes `lm_head`, embeddings, and every
    `linear_attn.conv1d`, `linear_attn.in_proj_a`, and
    `linear_attn.in_proj_b` control/gate projection.
  * `text_config.mamba_ssm_dtype = float32`.

* Qwen/Qwen3-Next-80B-A3B-Instruct-FP8 `config.json`:
  * `modules_to_not_convert` includes `lm_head`, layer norms, Q/K norms,
    `linear_attn.A_log`, `linear_attn.conv1d`, `linear_attn.dt_bias`,
    `linear_attn.in_proj_ba`, `linear_attn.norm`, and MoE/router gates such as
    `mlp.gate` / `mlp.shared_expert_gate`.

* NVIDIA TensorRT-LLM / ModelOpt:
  * FP8 quantization must be quality checked; it is not guaranteed.
  * KV cache is not quantized by default. FP8 KV is an explicit extra risk.
  * ModelOpt code disables `*lm_head*` by default for FP8 quantization.
  * vLLM FP8 dynamic quantization similarly skips final `lm_head` by default.

* AWS/NKI:
  * Trainium2 NeuronCore-v3 TensorE supports FP8_E4/FP8_E5 input matmuls at
    double BF16/FP16 throughput.
  * `nki.isa.nc_matmul` uses FP32 internal accumulation and FP32 PSUM output on
    NeuronCore-v3.
  * This means FP8 should be used for large TensorE matmul operands; recurrent
    state, checkpoint banks, KV cache, lm_head, normalization and control/gate
    tensors should stay BF16/FP32 unless specifically calibrated and proven.

What is now replicated in our final tested artifact:

artifact:
  /mnt/trainium_artifacts/qwen_artifacts/qwen36_27b_32768_fp8_full_lmheadbf16_kvbf16_qkvnki_tiled_segmented_cte512_nki_decode_stable_probe32k_gdnseg512_gdnrecfloat32_sampletokens_outlogits_b256_cte2048_pfx16k_slots64_async_cteargmaxsafe_20260605T050957Z_kkt_hier_scan7

runtime dtype/config audit:

```text
kv_cache_quant False
prefix_cte_attention_backend segmented_cte
prefix_cte_attention_segment_size 512
excluded:
  lm_head: true
  layers.0.linear_attn.in_proj_a: true
  layers.0.linear_attn.in_proj_b: true
  layers.0.linear_attn.in_proj_ba: true
  layers.0.linear_attn.conv1d: true
  layers.0.linear_attn.A_log: true
  layers.0.linear_attn.dt_bias: true
  layers.0.linear_attn.norm: true
dtypes:
  lm_head.weight: torch.bfloat16
  layers.0.linear_attn.in_proj_a.weight: torch.bfloat16
  layers.0.linear_attn.in_proj_b.weight: torch.bfloat16
  layers.0.linear_attn.in_proj_qkv.weight: torch.float8_e4m3fn
  layers.0.linear_attn.in_proj_z.weight: torch.float8_e4m3fn
  layers.0.linear_attn.out_proj.weight: torch.float8_e4m3fn
```

Conclusion:

The original "everything FP8" mistake is fixed. Our active artifact now follows
the same component-level policy as Qwen/NVIDIA: FP8 only for eligible large
linear/matmul weights, with KV cache, lm_head, DeltaNet gates/control, norms and
checkpoint state kept higher precision.

Remaining mismatch and hypothesis:

Qwen's public FP8 metadata is block-scaled FP8 (`weight_block_size=[128,128]`),
but our manual Neuron checkpoint converter currently uses
`quantize_fp8_per_channel(..., channel_axis=0)` because NxDI's FP8 linear
preprocess path expects per-channel `.scale` tensors and broadcasts them to
`[128, width]` for the NKI kernels. This may still be valid for Neuron, but it
is not identical to the public Qwen/NVIDIA checkpoint format.

The remaining all-NaN logits are therefore not explained by lm_head/KV/control
being FP8 anymore. Current best candidates are:

1. A remaining FP8 matmul weight path on Neuron: `linear_attn.in_proj_qkv`,
   `linear_attn.in_proj_z`, `linear_attn.out_proj`, attention `Wqkv/o_proj`, or
   MLP projections.
2. Scale-layout or scale-granularity mismatch in our per-channel FP8 Neuron
   path versus Qwen's block-scaled public FP8 metadata.
3. A prompt/data-dependent Neuron FP8 kernel issue, because the same actual
   4092-token shape can pass or fail depending on prompt contents.

Do not circle back to KV FP8 or lm_head FP8 as the leading cause; they are
already BF16 in the audited artifact.

Additional scale-layout probe:

First command failed:

```text
command:
  ssh ubuntu@16.51.179.180 '/opt/aws_neuronx_venv_pytorch_inference_vllm_0_16/bin/python -c "...; print(...);\\nfor k in keys[:80]:\\n ..."'
failure:
  SyntaxError: unexpected character after line continuation character
exit code:
  1
context:
  Remote one-line Python scale-shape probe for the final audited artifact. The
  shell quoting passed literal `\n` sequences into `python -c`.
mitigation:
  Reran as a one-line Python expression with a list-comprehension print.
verification:
  Command succeeded; no runtime/model state was changed.
```

Corrected scale-shape result on runtime host, tp0 final artifact:

```text
layers.0.linear_attn.in_proj_qkv.scale        (2560, 1)   fp32
layers.0.linear_attn.in_proj_z.scale          (1536, 1)   fp32
layers.0.linear_attn.out_proj.scale           (5120, 1)   fp32
layers.0.mlp.down_proj.scale                  (5120, 1)   fp32
layers.0.mlp.gate_proj.scale                  (4352, 1)   fp32
layers.0.mlp.up_proj.scale                    (4352, 1)   fp32
layers.3.self_attn.o_proj.o_proj.scale        (5120, 1)   fp32
layers.3.self_attn.output_gate_proj.scale     (1536, 1)   fp32
layers.3.self_attn.qkv_proj.Wqkv.scale        (128, 2048) fp32
```

Interpretation:

No obvious simple scale-orientation bug showed up. Most scales are Neuron
per-channel `[out, 1]`; the fused attention Wqkv scale is already in the NKI
broadcast layout `[128, width]`. Keep scale granularity/layout on the suspect
list, but the next direct isolation should target remaining FP8 matmul groups
instead of repeating the KV/lm_head/control checks.

2026-06-05 targeted FP8 group isolation
---------------------------------------

Added a compile-time ablation knob instead of hard-coding another branch:

```text
contrib/models/Qwen3.6-27B/test/integration/qwen36_27b_compile_fp8.py
  --fp8-exclude-groups
    choices:
      linear_attn
      linear_attn_qkv
      linear_attn_z
      linear_attn_out_proj
      mlp
      self_attn
      self_attn_qkv
      self_attn_o_proj

tmp_compile_qwen32k_segcte2048_gdnseg512.sh
  FP8_EXCLUDE_GROUPS="..."
```

The shell driver now encodes the ablation in BASE, WORKDIR,
QUANTIZED_CHECKPOINTS, ENVLOG, and the compile command. This prevents repeating
the stale quantized-checkpoint bug that previously confounded the lm_head/KV
policy compile.

Local validation:

```text
PYTHONPATH=src python3 contrib/models/Qwen3.6-27B/test/unit/test_qwen36_compile_fp8_config.py
  49 tests OK

bash -n tmp_compile_qwen32k_segcte2048_gdnseg512.sh
  OK

python3 -m py_compile contrib/models/Qwen3.6-27B/test/integration/qwen36_27b_compile_fp8.py
  OK
```

Remote validation on compile host after scp:

```text
cd /home/ubuntu/inferentia-gdn-multihead-cte-20260531T1350Z
source /opt/aws_neuronx_venv_pytorch_inference_vllm_0_16/bin/activate
PYTHONPATH=src python contrib/models/Qwen3.6-27B/test/unit/test_qwen36_compile_fp8_config.py
  49 tests OK

bash -n /home/ubuntu/tmp_compile_qwen32k_segcte2048_gdnseg512.sh
/opt/aws_neuronx_venv_pytorch_inference_vllm_0_16/bin/python -m py_compile contrib/models/Qwen3.6-27B/test/integration/qwen36_27b_compile_fp8.py
  OK
```

Errors encountered:

```text
1. Local search path typo:
   command:
     rg -n "modules_to_not_convert|get_modules_to_not_convert|post_create_quantized_module_hook" src/neuronx_distributed_inference src/neuronx_distributed | head -240
   failure:
     rg: src/neuronx_distributed: No such file or directory (os error 2)
   context:
     Investigating how modules_to_not_convert is consumed. The repo has
     src/neuronx_distributed_inference but not src/neuronx_distributed.
   mitigation:
     reran with `rg ... src`, which found the relevant NxDI wrapper and model
     conversion paths.
   verification:
     confirmed modules_to_not_convert is passed to quantize_pytorch_model_* and
     that the current Qwen3.6 path needs excluded groups reflected both in the
     manual checkpoint conversion and NeuronConfig.

2. Automation id mismatch:
   attempted tool call:
     automation_update mode=update id=monitor-qwen-nvidia-fp8-compile
   failure:
     This thread already has an active heartbeat automation. Only one automation
     can be attached to this thread.
   context:
     The automation name was monitor-qwen-nvidia-fp8-compile, but the actual id
     on disk was monitor-qwen-fp8-kvbf16-compile.
   mitigation:
     inspected ~/.codex/automations with rg, found the active id, and updated
     monitor-qwen-fp8-kvbf16-compile instead.
   verification:
     automation_update returned:
       Updated automation in the app.
       automationId=monitor-qwen-fp8-kvbf16-compile

3. Compile/runtime disk almost full:
   command:
     ssh ubuntu@16.51.94.87 'df -h /mnt/trainium_artifacts /home/ubuntu | tail -n +2'
     ssh ubuntu@16.51.179.180 'df -h /mnt/trainium_artifacts /home/ubuntu | tail -n +2'
   failure/risk:
     compile host: /dev/root 484G used 481G free 2.6G, 100%
     runtime host: /dev/root 484G used 457G free 28G, 95%
   context:
     Active fp8xlinear_attn compile had already generated the new 34G
     quantized checkpoint and still needed space for the final compiled
     artifact. Runtime also needed enough space for the later EC2-to-EC2 rsync.
   mitigation:
     Requested explicit approval for exact destructive cleanup and deleted only
     obsolete paths:
       compile host:
         /mnt/trainium_artifacts/qwen_artifacts/_quantized/qwen36_27b_fp8_full_lmheadfp8
         /mnt/trainium_artifacts/qwen_artifacts/_quantized/qwen36_27b_fp8_mlp_edgebf16
       runtime host:
         /mnt/trainium_artifacts/qwen_artifacts/qwen36_27b_32768_fp8_full_lmheadfp8_kvfp8_qkvnki_tiled_segmented_cte512_nki_decode_stable_probe32k_gdnrecfloat32_sampletokens_outlogits_b256_cte512_pfx16k_slots64_async_cteargmaxsafe_20260604T114155Z_doubling_scan7
   verification:
     compile host free space after cleanup:
       /dev/root 484G used 417G free 67G, 87%
     runtime free space after cleanup:
       /dev/root 484G used 390G free 95G, 81%
     active compile PID 657991 was still running after cleanup.

4. Delayed compound SSH sample failed:
   command:
     sleep 60 && ssh ubuntu@16.51.94.87 'ps -p 657991 ...; grep ...compile.log'
   failure:
     ssh: connect to host 16.51.94.87 port 22: Operation not permitted
     exit code 255
   context:
     A delayed local compound command was used to sample the compile after one
     minute. This looked like a local sandbox/network wrapper issue around the
     delayed command, not a remote host failure.
   mitigation:
     Reran as a direct `ssh ubuntu@16.51.94.87 ...` command.
   verification:
     Direct SSH succeeded. PID 657991 was still alive at about 7 minutes, all
     context_encoding_model and token_generation_model HLO generation lines had
     appeared, and no compile error/traceback had appeared.
```

First isolation compile launched:

```text
command:
  ssh ubuntu@16.51.94.87 'TS=$(date -u +%Y%m%dT%H%M%SZ) ENABLE_KV_CACHE_QUANT=0 QUANTIZE_LM_HEAD=0 FORCE_QUANTIZE=1 FP8_EXCLUDE_GROUPS=linear_attn bash /home/ubuntu/tmp_compile_qwen32k_segcte2048_gdnseg512.sh'

PID:
  657991

BASE:
  qwen36_27b_32768_fp8_full_fp8xlinear_attn_lmheadbf16_kvbf16_qkvnki_tiled_segmented_cte512_nki_decode_stable_probe32k_gdnseg512_gdnrecfloat32_sampletokens_outlogits_b256_cte2048_pfx16k_slots64_async_cteargmaxsafe_20260605T060539Z_kkt_hier_scan7

artifact:
  /mnt/trainium_artifacts/qwen_artifacts/qwen36_27b_32768_fp8_full_fp8xlinear_attn_lmheadbf16_kvbf16_qkvnki_tiled_segmented_cte512_nki_decode_stable_probe32k_gdnseg512_gdnrecfloat32_sampletokens_outlogits_b256_cte2048_pfx16k_slots64_async_cteargmaxsafe_20260605T060539Z_kkt_hier_scan7

workdir:
  /mnt/trainium_artifacts/qwen_artifacts/_nxd_model_workdir_32768_fp8_full_fp8xlinear_attn_lmheadbf16_kvbf16_qkvnki_tiled_segmented_cte512_gdnseg512_gdnrecfloat32_sampletokens_outlogits_b256_cte2048_cteargmaxsafe_20260605T060539Z_kkt_hier_scan7

quantized checkpoint:
  /mnt/trainium_artifacts/qwen_artifacts/_quantized/qwen36_27b_fp8_full_fp8xlinear_attn_lmheadbf16

compile log:
  /home/ubuntu/validation_logs/fp8_256k_decode_nki/qwen36_27b_32768_fp8_full_fp8xlinear_attn_lmheadbf16_kvbf16_qkvnki_tiled_segmented_cte512_nki_decode_stable_probe32k_gdnseg512_gdnrecfloat32_sampletokens_outlogits_b256_cte2048_pfx16k_slots64_async_cteargmaxsafe_20260605T060539Z_kkt_hier_scan7_compile.log

env log:
  /home/ubuntu/validation_logs/fp8_256k_decode_nki/qwen36_27b_32768_fp8_full_fp8xlinear_attn_lmheadbf16_kvbf16_qkvnki_tiled_segmented_cte512_nki_decode_stable_probe32k_gdnseg512_gdnrecfloat32_sampletokens_outlogits_b256_cte2048_pfx16k_slots64_async_cteargmaxsafe_20260605T060539Z_kkt_hier_scan7_env.txt

early verification:
  ps shows PID 657991 running.
  command line includes:
    --fp8-exclude-groups linear_attn
    --force-quantize
    no --enable-kv-cache-quant
    no --quantize-lm-head
  env log confirms:
    ENABLE_KV_CACHE_QUANT=0
    QUANTIZE_LM_HEAD=0
    FP8_EXCLUDE_GROUPS=linear_attn
    FORCE_QUANTIZE=1
  compile log confirms:
    FP8_EXCLUDE_GROUPS linear_attn
    MODULES_TO_NOT_CONVERT_COUNT 2702
    QUANTIZE_START manual_fp8
  later compile log confirms:
    MANUAL_FP8_WEIGHT_COUNT 263
  previous NVIDIA-style lmheadbf16/kvbf16 policy count:
    MANUAL_FP8_WEIGHT_COUNT 407
  interpretation:
    This moved 144 linear-attention matmul weights from FP8 back to BF16 while
    leaving attention/MLP FP8 enabled for the first isolation run.
```

Monitor automation:

```text
id:
  monitor-qwen-fp8-kvbf16-compile
name:
  Monitor Qwen FP8 linear-attn isolation compile
cadence:
  every 10 minutes
```

Expected validation if compile succeeds:

1. Verify `COMPILE_DONE`, `Finished Compilation for all HLOs`, no assert/guard
   failures, and checkpoint-bank weights for tp0..tp3 with recurrent fp32 and
   conv bf16.
2. Audit safetensors:
   * `linear_attn.in_proj_qkv`, `linear_attn.in_proj_z`, `linear_attn.out_proj`
     must be BF16.
   * `lm_head` must remain BF16.
   * attention and MLP FP8 should remain FP8 where expected.
3. rsync artifact EC2-to-EC2 to runtime host `ubuntu@16.51.179.180`.
4. Launch with `/home/ubuntu/tmp_launch_qwen36_segcte2048.sh`.
5. Run:
   * `/tmp/tmp_bisect_probe3.py`
   * boundary probe 146/160/485/505/526/1225/2048/2049/2500/4092/4096
   * unique 4k prompt sweep, because previous failure was data-dependent
   * `/tmp/tmp_qwen36_multiturn_probe.py`
   * 16k usage-accounted OpenAI chat context benchmark only if coherence passes.

Validation result for fp8xlinear_attn
------------------------------------

Compile:

```text
All 12 HLOs passed.
INFO:Neuron:Finished Compilation for all HLOs in 444.4395794868469 seconds
CHECKPOINT_BANK_WEIGHTS_ADDED tp0_sharded_checkpoint.safetensors 48 48 torch.float32 torch.bfloat16
CHECKPOINT_BANK_WEIGHTS_ADDED tp1_sharded_checkpoint.safetensors 48 48 torch.float32 torch.bfloat16
CHECKPOINT_BANK_WEIGHTS_ADDED tp2_sharded_checkpoint.safetensors 48 48 torch.float32 torch.bfloat16
CHECKPOINT_BANK_WEIGHTS_ADDED tp3_sharded_checkpoint.safetensors 48 48 torch.float32 torch.bfloat16
COMPILE_DONE
```

Artifact files:

```text
model.pt 866517824
neuron_config.json 128168
tp0_sharded_checkpoint.safetensors 11235372236
tp1_sharded_checkpoint.safetensors 11235372236
tp2_sharded_checkpoint.safetensors 11235372236
tp3_sharded_checkpoint.safetensors 11235372236
artifact size 43G
```

Dtype audit:

```text
kv_cache_quant False
prefix_cte_attention_backend segmented_cte 512
lm_head.weight                                      torch.bfloat16
layers.0.linear_attn.in_proj_qkv.weight            torch.bfloat16
layers.0.linear_attn.in_proj_z.weight              torch.bfloat16
layers.0.linear_attn.out_proj.weight               torch.bfloat16
layers.0.mlp.up_proj.weight                        torch.float8_e4m3fn
layers.3.self_attn.qkv_proj.Wqkv.weight            torch.float8_e4m3fn
layers.3.self_attn.o_proj.o_proj.weight            torch.float8_e4m3fn
hybrid_gdn_checkpoint_cache.recurrent_slots.0      torch.float32
hybrid_gdn_checkpoint_cache.conv_slots.0           torch.bfloat16
```

Transfer and launch:

```text
rsync EC2-to-EC2 transferred 45,808,134,936 bytes successfully.
Runtime artifact verified at 43G.
Launch log:
  /home/ubuntu/validation_logs/fp8_256k_decode_nki/qwen36_fp8xlinear_attn_lmheadbf16_kvbf16_segcte2048_serve_20260605T0636.log
HEALTH_OK attempt=79
```

Probe results:

```text
4096 one-token smoke:
  output: ack
  prompt_tokens: 4092
  effective prefill: ~2026 tok/s
  invalid fallback count after smoke: 0

/tmp/tmp_bisect_probe3.py:
  short: coherent
  160: coherent
  123: coherent
  485: coherent
  1225: coherent
  2500: coherent

boundary probe max_tokens=32:
  146/160/485/505/526/1225/2048/2049 coherent
  2500 returned empty/EOS once, then repeated 2500 three times coherent
  4092 coherent
  4096 coherent
  invalid fallback count before unique 4k sweep: 0

unique 4k prompt sweep:
  4088 actual 4079: ack 8
  4090 actual 4079: ack 8
  4092 actual 4092: ack 8
  4094 actual 4092: ack 8
  4096 actual 4092: !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  4098 actual 4092: ack 8
  4100 actual 4092: ack 8
  4102 actual 4092: ack 8
  4104 actual 4092: ack 8
```

Failure evidence:

```text
serve log invalid/fallback count after unique 4k sweep:
  64

example:
  Replacing invalid completed-prefill sampled token ... token_id=2143289344 (0x7fc00000) fallback_token_id=0
  Qwen3.6 fallback logits summary ... finite=0/248320 nan=248320 ... argmax=[0] argmax_values=[nan]
```

Conclusion:

Excluding all remaining `linear_attn` FP8 matmuls did **not** fix the
data-dependent all-NaN logits failure. `linear_attn.in_proj_qkv`,
`linear_attn.in_proj_z`, and `linear_attn.out_proj` are exonerated as the sole
cause. The next isolation should use `FP8_EXCLUDE_GROUPS=mlp`, which keeps
QKV/attention FP8 for speed while moving MLP matmuls to BF16. If that still
fails, the remaining suspect is attention FP8 (`self_attn`, likely Wqkv/o_proj
or the NKI attention projection path).

2026-06-05 MLP isolation compile
--------------------------------

The active goal now requires final validation up to 256k context. We are still
using 32k compiles for isolation because the data-dependent all-NaN failure
reproduces at a 4k prompt shape; once the FP8 culprit is isolated, the final
candidate must be compiled and validated at 256k.

Cleanup before launch:

```text
Deleted failed compile-host-only fp8xlinear_attn outputs:
  /mnt/trainium_artifacts/qwen_artifacts/qwen36_27b_32768_fp8_full_fp8xlinear_attn_lmheadbf16_kvbf16_qkvnki_tiled_segmented_cte512_nki_decode_stable_probe32k_gdnseg512_gdnrecfloat32_sampletokens_outlogits_b256_cte2048_pfx16k_slots64_async_cteargmaxsafe_20260605T060539Z_kkt_hier_scan7
  /mnt/trainium_artifacts/qwen_artifacts/_quantized/qwen36_27b_fp8_full_fp8xlinear_attn_lmheadbf16

Compile-host free space after cleanup:
  /dev/root 484G used 386G free 98G, 80%
```

Launch:

```text
command:
  ssh ubuntu@16.51.94.87 'TS=$(date -u +%Y%m%dT%H%M%SZ) ENABLE_KV_CACHE_QUANT=0 QUANTIZE_LM_HEAD=0 FORCE_QUANTIZE=1 FP8_EXCLUDE_GROUPS=mlp bash /home/ubuntu/tmp_compile_qwen32k_segcte2048_gdnseg512.sh'

PID:
  667111

BASE:
  qwen36_27b_32768_fp8_full_fp8xmlp_lmheadbf16_kvbf16_qkvnki_tiled_segmented_cte512_nki_decode_stable_probe32k_gdnseg512_gdnrecfloat32_sampletokens_outlogits_b256_cte2048_pfx16k_slots64_async_cteargmaxsafe_20260605T065004Z_kkt_hier_scan7

artifact:
  /mnt/trainium_artifacts/qwen_artifacts/qwen36_27b_32768_fp8_full_fp8xmlp_lmheadbf16_kvbf16_qkvnki_tiled_segmented_cte512_nki_decode_stable_probe32k_gdnseg512_gdnrecfloat32_sampletokens_outlogits_b256_cte2048_pfx16k_slots64_async_cteargmaxsafe_20260605T065004Z_kkt_hier_scan7

workdir:
  /mnt/trainium_artifacts/qwen_artifacts/_nxd_model_workdir_32768_fp8_full_fp8xmlp_lmheadbf16_kvbf16_qkvnki_tiled_segmented_cte512_gdnseg512_gdnrecfloat32_sampletokens_outlogits_b256_cte2048_cteargmaxsafe_20260605T065004Z_kkt_hier_scan7

quantized checkpoint:
  /mnt/trainium_artifacts/qwen_artifacts/_quantized/qwen36_27b_fp8_full_fp8xmlp_lmheadbf16

compile log:
  /home/ubuntu/validation_logs/fp8_256k_decode_nki/qwen36_27b_32768_fp8_full_fp8xmlp_lmheadbf16_kvbf16_qkvnki_tiled_segmented_cte512_nki_decode_stable_probe32k_gdnseg512_gdnrecfloat32_sampletokens_outlogits_b256_cte2048_pfx16k_slots64_async_cteargmaxsafe_20260605T065004Z_kkt_hier_scan7_compile.log

env log:
  /home/ubuntu/validation_logs/fp8_256k_decode_nki/qwen36_27b_32768_fp8_full_fp8xmlp_lmheadbf16_kvbf16_qkvnki_tiled_segmented_cte512_nki_decode_stable_probe32k_gdnseg512_gdnrecfloat32_sampletokens_outlogits_b256_cte2048_pfx16k_slots64_async_cteargmaxsafe_20260605T065004Z_kkt_hier_scan7_env.txt
```

Early verification:

```text
ps shows PID 667111 running.
command line includes:
  --fp8-exclude-groups mlp
  --force-quantize
  no --enable-kv-cache-quant
  no --quantize-lm-head
env log confirms:
  ENABLE_KV_CACHE_QUANT=0
  QUANTIZE_LM_HEAD=0
  FP8_EXCLUDE_GROUPS=mlp
  FORCE_QUANTIZE=1
compile log confirms:
  FP8_EXCLUDE_GROUPS mlp
  MODULES_TO_NOT_CONVERT_COUNT 2702
  QUANTIZE_START manual_fp8
  MANUAL_FP8_WEIGHT_COUNT 212

2026-06-05 follow-up: exact Qwen3.6/NVIDIA/AWS FP8 policy check
----------------------------------------------------------------

User asked to verify the NVIDIA/Qwen FP8 implementation because their shipped
FP8 models are not "everything FP8", then replicate that boundary on Trainium.

Primary source findings:

* Qwen/Qwen3.6-27B-FP8 official `config.json` uses:
  * `quantization_config.quant_method = fp8`
  * `activation_scheme = dynamic`
  * `fmt = e4m3`
  * `weight_block_size = [128, 128]`
  * `text_config.dtype = bfloat16`
  * `text_config.mamba_ssm_dtype = float32`
  * `modules_to_not_convert` includes `lm_head`, `model.embed_tokens`, every
    text layer `input_layernorm` and `post_attention_layernorm`, full-attention
    `q_norm` / `k_norm`, and linear-attention control tensors:
    `A_log`, `conv1d`, `dt_bias`, `in_proj_ba`, `in_proj_b`, `in_proj_a`,
    and `norm`.
  * It does not exclude large text matmuls such as DeltaNet `in_proj_qkv`,
    `in_proj_z`, `out_proj`, dense MLP `gate_proj/up_proj/down_proj`, or
    full-attention QKV/O matmuls.
* Qwen/Qwen3-Next-80B-A3B-Instruct-FP8 has the same broad policy:
  `lm_head`, norms, linear-attention control/gate pieces, and routing/gating
  controls stay higher precision while large eligible matmuls are FP8.
* NVIDIA TensorRT-LLM ModelOpt docs explicitly show skipping `*lm_head*` by
  appending a disable rule, and ModelOpt auto-quantize may choose "do not
  quantize" per layer.
* NVIDIA TensorRT-LLM benchmark docs show FP8 compute checkpoints can default
  to no KV-cache quantization; FP8 KV cache is a separate knob.
* AWS/NKI docs for Trainium2 say FP8_E4/FP8_E5 TensorE matmuls get double
  BF16/FP16 throughput while using FP32 accumulation. `nki.isa.nc_matmul`
  supports FP8 inputs, but the internal accumulation is FP32 and on
  NeuronCore-v3 the destination PSUM tile must be FP32.

Replication status in this repo:

* Already replicated:
  * `lm_head` BF16 by default via `QUANTIZE_LM_HEAD=0`.
  * KV cache BF16 by default via `ENABLE_KV_CACHE_QUANT=0`.
  * checkpoint-bank recurrent FP32 and conv BF16.
  * `linear_attn` control tensors are excluded from FP8 conversion by default:
    `conv1d`, `conv1d_weight`, `A_log`, `A_log_weight`, `dt_bias`,
    `dt_bias_weight`, `in_proj_a`, `in_proj_b`, `in_proj_ba`, `norm`,
    recurrent/conv buffers.
  * layer norms, q/k norms, embeddings and rotary state are excluded.
* Diagnostic-only, not final policy:
  * Current compile uses `FP8_EXCLUDE_GROUPS=mlp` to isolate whether dense MLP
    FP8 scale/layout is the last coherency culprit. If it passes, do not ship
    all-MLP BF16 as the final speed-preserving answer; fix the MLP FP8
    scale/layout path and re-enable MLP matmuls.
* Important upstream clue:
  * SGLang issue #23687 reports Qwen3.6-27B-FP8 garbage output when dense MLP
    FP8 scale tensors are dropped/misregistered, causing default scale use.
    That matches the current MLP-isolation strategy.

Errors encountered in this pass:

1. Local rg path typo:
   command:
   `rg ... src/neuronx_distributed ...`
   failure:
   `rg: src/neuronx_distributed: No such file or directory (os error 2)`
   context:
   while searching FP8 scale/load paths. This repo has
   `src/neuronx_distributed_inference`, not `src/neuronx_distributed`.
   mitigation:
   reran searches against `src/neuronx_distributed_inference`.
   verification:
   found loader normalization in
   `src/neuronx_distributed_inference/models/application_base.py`, which
   rewrites `.weight_scale` checkpoint keys to `.scale` before Qwen conversion.

2. Remote Neuron utility import on CPU compile host:
   command:
   `ssh ubuntu@16.51.94.87 'source ...; python - <<PY ... import quantize_fp8_per_channel ... PY'`
   failure:
   `RuntimeError: Unsupported Platform - r7i.24xlarge. If you want to compile
   on CPU, please supply a compiler target argument, with one of: trn1, inf2,
   trn1n, trn2, or trn3. Ex: "--target trn1"`
   context:
   direct import of `neuronx_distributed` initializes platform detection on
   the CPU compile host.
   mitigation:
   read installed source directly with `grep`/`sed` instead of importing.
   verification:
   confirmed `quantize_fp8_per_channel` returns FP8 weights and FP32 scales
   shaped with the selected channel axis.

3. Monitoring artifact-presence check returned nonzero during packaging:
   command:
   `ssh ubuntu@16.51.94.87 '...; test -f /mnt/trainium_artifacts/qwen_artifacts/qwen36_27b_32768_fp8_full_fp8xmlp_lmheadbf16_kvbf16_qkvnki_tiled_segmented_cte512_nki_decode_stable_probe32k_gdnseg512_gdnrecfloat32_sampletokens_outlogits_b256_cte2048_pfx16k_slots64_async_cteargmaxsafe_20260605T065004Z_kkt_hier_scan7/model.pt && echo ARTIFACT_MODEL_PT_PRESENT'`
   failure:
   exit code 1 from `test -f`; no error text. The compile process was still
   alive and the compile log was still active.
   context:
   MLP-isolation compile PID 667111 had printed `Finished Compilation for all
   HLOs` and `Saving the neuron_config`, but had not yet written `model.pt`.
   hypothesis:
   packaging/saving was still in progress, not a compiler failure.
   mitigation:
   switched to later checks using `ls ... 2>/dev/null || true` plus PID/log
   status so missing `model.pt` does not make the monitor look failed.
   verification:
   later poll showed PID 667111 still running and artifact directory containing
   `neuron_config.json`.

Direct source refresh for NVIDIA/Qwen + AWS policy:

* Qwen/Qwen3.6-27B-FP8 config uses FP8 E4M3 dynamic/block quantization with
  `text_config.dtype=bfloat16` and `mamba_ssm_dtype=float32`. Its
  `modules_to_not_convert` excludes `lm_head`, embeddings, layer norms, q/k
  norms, and the DeltaNet control/state-style modules (`A_log`, `conv1d`,
  `dt_bias`, `in_proj_a`, `in_proj_b`, `in_proj_ba`, `norm`). The big matmul
  weights remain eligible for FP8.
* TensorRT-LLM docs treat FP8 KV cache as a separate opt-in knob; by default KV
  cache is not quantized, and NVIDIA warns aggressive cache quantization can
  degrade output quality.
* TensorRT-LLM checkpoint docs make FP8 scaling factors explicit checkpoint
  tensors, and note linear checkpoint weights use `(out_feature, in_feature)`
  while plugin implementations may use transposed layouts. This is directly
  relevant to the remaining suspected MLP scale/layout issue.
* AWS/NKI `nc_matmul` docs say FP8 inputs are supported, but Tensor Engine
  accumulation is FP32; on NeuronCore-v3 the PSUM destination is FP32. Therefore
  the speed-preserving policy should keep FP8 on large matmul operands and keep
  logits/lm_head/cache/state/control/norm paths in BF16/FP32.
interpretation:
  Compared with the 407-weight NVIDIA-style FP8 policy, this leaves MLP
  matmuls BF16 while keeping attention and linear_attn FP8. It is the direct
  follow-up after fp8xlinear_attn failed.
```

Monitor automation updated:

```text
id:
  monitor-qwen-fp8-kvbf16-compile
name:
  Monitor Qwen FP8 MLP isolation compile
cadence:
  every 10 minutes
```

## 2026-06-05 MLP-BF16 isolation verdict

Artifact:

```text
/mnt/trainium_artifacts/qwen_artifacts/qwen36_27b_32768_fp8_full_fp8xmlp_lmheadbf16_kvbf16_qkvnki_tiled_segmented_cte512_nki_decode_stable_probe32k_gdnseg512_gdnrecfloat32_sampletokens_outlogits_b256_cte2048_pfx16k_slots64_async_cteargmaxsafe_20260605T065004Z_kkt_hier_scan7
```

What it tested:

* `FP8_EXCLUDE_GROUPS=mlp`
* `lm_head` BF16
* KV cache BF16
* MLP `gate_proj/up_proj/down_proj` BF16
* full-attention QKV/O still FP8
* linear-attention `in_proj_qkv/in_proj_z/out_proj` still FP8

Packaging and recovery:

* HLO compilation succeeded, but initial postcompile checkpoint-bank insertion
  failed with disk full:
  `safetensors_rust.SafetensorError: Error while serializing: I/O error: No space left on device (os error 28)`.
* Added `--postprocess-only` to
  `contrib/models/Qwen3.6-27B/test/integration/qwen36_27b_compile_fp8.py`
  so `_ensure_hybrid_checkpoint_weights(...)` and
  `_sanitize_reloadable_neuron_config(...)` can be rerun on an existing
  artifact without `model.compile(...)`.
* Freed disk by deleting the generated current FP8 quantized checkpoint and
  the partial `.tmp` shard, then reran postprocess with
  `NEURON_PLATFORM_TARGET_OVERRIDE=trn2` and
  `NEURON_CC_FLAGS="--target trn2 --lnc 2"`.
* Postprocess printed `CHECKPOINT_BANK_WEIGHTS_ADDED` for tp0..tp3 with
  `48 48 torch.float32 torch.bfloat16` and `COMPILE_DONE`.

Verification before serve:

* Runtime sharded checkpoint dtypes:
  * checkpoint recurrent slots: 48 `torch.float32` per TP shard
  * checkpoint conv slots: 48 `torch.bfloat16` per TP shard
  * `layers.0.mlp.{gate_proj,up_proj,down_proj}.weight`: `torch.bfloat16`
  * `layers.3.self_attn.qkv_proj.Wqkv.weight`: `torch.float8_e4m3fn`
  * `layers.0.linear_attn.in_proj_qkv.weight`: `torch.float8_e4m3fn`
  * `lm_head.weight`: `torch.bfloat16`
* Runtime launched successfully from:
  `/home/ubuntu/validation_logs/fp8_256k_decode_nki/qwen36_fp8xmlp_mlpbf16_segcte2048_serve_20260605T0735Z.log`

Coherence results:

* `tmp_bisect_probe3.py`:
  * short/160/123/485/1225 mostly coherent text
  * 2500 returned empty text once, no invalid logits at that point
* unique 4k shape sweep:
  * 4088/4090/4092/4094/4096/4098/4100/4102/4104 all returned `ack`/`ack 8`
  * this fixes the earlier all-NaN 4096 synthetic shape
* corrected boundary probe:
  * 146/160/485/505/526/1225/2048/2049/4092/4096 were not flagged bad
  * 2500 returned empty one-token completion and was flagged bad
* mid-shape usage-accounted `ack` sweep:
  * 2400/2480/2490/2500/2510/2520/2600/3000/3500 all returned `ack`/`ack 8`
* multi-turn probe:
  * 160 and 1225 were not flagged bad
  * 2500 produced `!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!`
  * serve log showed repeated invalid sampled tokens and all-NaN logits:
    `finite=0/248320 nan=248320`, fallback token 0
* 16k usage-accounted `ack` benchmark passed:
  * prompt_tokens=15988
  * effective prefill about 2244 tok/s
  * decode about 14 tok/s

Interpretation after rechecking older FP8-MLP artifacts:

* Do **not** treat MLP FP8 as inherently corrupt. The allowed old branches
  `codex/full-fp8-qwen36` and `codex/nki-deltanet-decode-step` both include
  `mlp.gate_proj`, `mlp.up_proj`, and `mlp.down_proj` in `fp8_full`.
* The known older runtime artifact
  `...20260605T043000Z_kkt_hier_scan7` also used aggressive FP8 policy:
  `lm_head` FP8, KV BF16, and no `layers.*.mlp.*` exclusion in
  `modules_to_not_convert`.
* Direct dtype verification on runtime host `ubuntu@16.51.179.180` showed:
  * `layers.0.mlp.gate_proj.weight`: `torch.float8_e4m3fn`
  * `layers.0.mlp.up_proj.weight`: `torch.float8_e4m3fn`
  * `layers.0.mlp.down_proj.weight`: `torch.float8_e4m3fn`
  * `layers.0.linear_attn.in_proj_a.weight`: `torch.float8_e4m3fn`
  * `layers.0.linear_attn.in_proj_b.weight`: `torch.float8_e4m3fn`
  * `lm_head.weight`: `torch.float8_e4m3fn`
* The same known-good artifact had the separate MLP NKI path disabled:
  `mlp_kernel_enabled=False`, `mlp_tkg_nki_kernel_enabled=False`, and
  `quantized_mlp_kernel_enabled=False`. So FP8 MLP **weights** are cleared, but
  the later quantized-MLP NKI/full-moat kernel path is still a separate suspect.
* Re-serving that older artifact under the current runtime code passed the
  probes that caught the MLP-BF16 failure:
  * multi-turn 160/1225/2500: all `bad=false`
  * boundary 146/160/485/505/526/1225/2048/2049/4092/4096: coherent
  * boundary 2500: empty/EOS only, with no invalid-token or NaN log lines
  * unique 4k sweep 4088..4104: all returned `ack 8`
  * serve log scan: no `negative token_id`, no `out-of-vocab token_id`, no
    `fallback argmax`, no `finite=0`, and no NaN summaries
* Therefore `FP8_EXCLUDE_GROUPS=mlp` changed one failing synthetic shape, but it
  did **not** prove MLP FP8 is the root cause. The remaining failure is more
  likely a later policy/loader/layout interaction introduced after the older
  `lmheadfp8_kvbf16` artifact, especially around the transition to
  lm_head/control BF16 exclusions or stale/rebuilt scale metadata.
* Error logged while checking this: the first dtype-inspection command used a
  quoted heredoc and remote Python received literal
  `$ART/weights/tp0_sharded_checkpoint.safetensors`, raising
  `FileNotFoundError: No such file or directory:
  $ART/weights/tp0_sharded_checkpoint.safetensors`. Mitigation was to export
  `ART` and read it via `os.environ["ART"]`; the rerun succeeded and produced
  the dtype evidence above.
* Next diagnostic should re-anchor on the working `lmheadfp8_kvbf16` policy and
  flip one variable at a time (`lm_head` BF16 versus `linear_attn.in_proj_a/b`
  BF16) before disabling MLP or self-attention FP8 wholesale.

2026-06-05 continuation: old-policy FP8 gate fix and 256k compile launch
------------------------------------------------------------------------

Current corrected anchor:

* The coherent old artifact uses the old full-FP8 body policy:
  `lm_head` FP8, KV BF16, MLP FP8, self-attention FP8, and
  `linear_attn.in_proj_a/in_proj_b` FP8.
* Its `modules_to_not_convert` count is 2188 and does **not** exclude
  `layers.0.linear_attn.in_proj_a`, `in_proj_b`, or `in_proj_ba`.
* Re-serving it under current runtime passed:
  * multi-turn 160/1225/2500: `bad=false`
  * boundary 146/160/485/505/526/1225/2048/2049/4092/4096 coherent
  * unique 4k sweep 4088..4104: all `ack 8`
  * no invalid-token fallback, no `finite=0`, and no NaN summaries
* 16k usage-accounted benchmark passed:
  * prompt_tokens=16378
  * content `ack 7`
  * effective cold prefill about 2345 tok/s
  * usage-counted decode about 16 tok/s

Near-32k failure classification:

* Command:
  `/tmp/qwen36_chat_completion_context_bench.py --lengths 28672,32760
  --max-tokens 32 --unique-per-request`
* Runtime artifact:
  `...20260605T043000Z_kkt_hier_scan7`
* Result:
  script exit code 1; 28672 returned HTTP 200 with empty content/no usage,
  then 32760 returned immediately with no chunks/usage.
* Exact serve error:
  `ValueError: Prefill len 2048 with prefix len 18432 exceeds compiled 2D
  buckets for context_encoding_model; largest prefill bucket 2048, largest
  prefix bucket 16384`.
* Root cause:
  not coherence/NaN. The old 32k artifact was compiled with prefix buckets only
  through 16k (`pfx16k`), so vLLM chunked prefill crashes once same-request
  prefix exceeds 16384.
* Mitigation:
  final long-context compile must include context-encoding bucket pairs with
  prefix buckets beyond 16k; 256k compile includes
  `2048:{256,512,1024,2048,4096,8192,16384,32768,65536,131072,262144}`.

Code changes made locally and synced to compile host:

* `contrib/models/Qwen3.6-27B/test/integration/qwen36_27b_compile_fp8.py`
  now has `--fp8-quantize-linear-attn-gates`.
  * When off, current Qwen/NVIDIA-style policy keeps `in_proj_a/b/ba` in
    `modules_to_not_convert`.
  * When on, it replicates the old coherent config by not excluding
    `in_proj_a/b/ba`; manual FP8 conversion quantizes only `in_proj_a/b`,
    matching the old branches.
  * Compile manifest/log prints `FP8_QUANTIZE_LINEAR_ATTN_GATES`.
* `tmp_compile_qwen32k_segcte2048_gdnseg512.sh`
  now supports env-controlled `SEQ_LEN`, `MAX_CONTEXT_LENGTH`,
  `PA_NUM_BLOCKS`, `MAX_GDN_CHECKPOINT_SLOTS`,
  `FP8_QUANTIZE_LINEAR_ATTN_GATES`, dynamic prefix buckets, and dynamic token
  generation buckets up to 256k.
* Local verification:
  * `PYTHONPATH=src python3
    contrib/models/Qwen3.6-27B/test/unit/test_qwen36_compile_fp8_config.py`
    -> 51 tests OK.
  * `python3 -m py_compile
    contrib/models/Qwen3.6-27B/test/integration/qwen36_27b_compile_fp8.py`
    -> OK.
  * `bash -n tmp_compile_qwen32k_segcte2048_gdnseg512.sh` -> OK.
* Remote verification on compile host:
  `REMOTE_DRIVER_SYNTAX_OK`.

Errors encountered and logged:

1. Direct `torch.load` dtype audit on compiled `model.pt` failed:
   `RuntimeError: Cannot use weights_only=True with TorchScript archives passed
   to torch.load`.
   Mitigation: used config/serve behavior instead.
2. `torch.jit.load` dtype audit also failed:
   `Unknown type name '__torch__.torch.classes.neuron.LayoutTransformation'`.
   Mitigation: avoid direct CPU JIT loading of Neuron compiled archive.
3. Quantized sidecar for the old coherent artifact was missing on runtime:
   `/mnt/trainium_artifacts/qwen_artifacts/_quantized/qwen36_27b_fp8_full_lmheadfp8:
   No such file or directory`.
   Mitigation: used compiled artifact config and runtime validation.
4. First local `apply_patch` for the gate-policy flag failed because the patch
   context did not match the edited file. Mitigation: repatched in smaller
   hunks; tests passed.
5. First slot-count patch failed for the same reason after the script's
   `PA_NUM_BLOCKS` block had changed. Mitigation: repatched against current
   context.
6. First 256k compile launch failed before Python started because generated log
   and PID filenames were too long:
   `File name too long` for `_compile.pid` and `_compile.log`.
   Mitigation: shortened the artifact/log basename to
   `qwen36_256k_fp8_...`.

Remote cleanup:

* Deleted the known-failed compile-host MLP-BF16 diagnostic artifact and workdir:
  * `...20260605T065004Z_kkt_hier_scan7`
  * `_nxd_model_workdir_32768_fp8_full_fp8xmlp...20260605T065004Z...`
* Disk after cleanup: about 97G free on `/mnt/trainium_artifacts`.

256k compile in flight:

```text
PID=677548
ARTIFACT=/mnt/trainium_artifacts/qwen_artifacts/qwen36_256k_fp8_fp8all_lmheadfp8_gatesfp8_kvbf16_qkvnki_tiled_segmented_cte512_gdnseg512_cte2048_pfx256k_slots64_20260605T080513Z_kkt_hier_scan7
WORKDIR=/mnt/trainium_artifacts/qwen_artifacts/_nxd_work_256k_fp8_fp8all_lmheadfp8_gatesfp8_kvbf16_qkvnki_tiled_segmented_cte512_gdnseg512_cte2048_20260605T080513Z_kkt_hier_scan7
LOG=/home/ubuntu/validation_logs/fp8_256k_decode_nki/qwen36_256k_fp8_fp8all_lmheadfp8_gatesfp8_kvbf16_qkvnki_tiled_segmented_cte512_gdnseg512_cte2048_pfx256k_slots64_20260605T080513Z_kkt_hier_scan7_compile.log
ENVLOG=/home/ubuntu/validation_logs/fp8_256k_decode_nki/qwen36_256k_fp8_fp8all_lmheadfp8_gatesfp8_kvbf16_qkvnki_tiled_segmented_cte512_gdnseg512_cte2048_pfx256k_slots64_20260605T080513Z_kkt_hier_scan7_env.txt
```

Compile command policy:

```text
SEQ_LEN=262144
MAX_CONTEXT_LENGTH=262144
ENABLE_KV_CACHE_QUANT=0
QUANTIZE_LM_HEAD=1
FP8_QUANTIZE_LINEAR_ATTN_GATES=1
FORCE_QUANTIZE=1
```

Immediate log status:

* Process alive at elapsed 01:03.
* `MODULES_TO_NOT_CONVERT_COUNT 2188`.
* `FP8_QUANTIZE_LINEAR_ATTN_GATES True`.
* `CONTEXT_TRACE_SHAPE` includes prefix bucket 262144 and TKG bucket 262144.
* Current stage: `QUANTIZE_START manual_fp8`.
* Disk during quantization: about 77G free.

Automation:

* Existing heartbeat `monitor-qwen-fp8-kvbf16-compile` was updated to monitor
  this 256k compile every 10 minutes.

Additional errors encountered:

1. Postprocess-only first retry missed the compile-target env:
   command:
   direct `python ... --postprocess-only` on compile host without target env.
   failure:
   `RuntimeError: Unsupported Platform - r7i.24xlarge. If you want to compile
   on CPU, please supply a compiler target argument, with one of: trn1, inf2,
   trn1n, trn2, or trn3. Ex: "--target trn1"`
   mitigation:
   reran with `NEURON_PLATFORM_TARGET_OVERRIDE=trn2` and
   `NEURON_CC_FLAGS="--target trn2 --lnc 2"`.
   verification:
   postprocess completed with checkpoint-bank additions and `COMPILE_DONE`.

2. Postprocess retry log filename was too long:
   command:
   postprocess retry piped to `tee` with the full artifact basename as the log
   filename.
   failure:
   `tee: ..._postprocess_retry.log: File name too long`
   effect:
   pipeline exit code 1 even though Python printed `COMPILE_DONE`.
   mitigation:
   relied on captured terminal output and direct safetensors verification.

3. Runtime stop/delete command killed its own SSH session:
   command:
   `pkill -f "serve_qwen36.py"` inside a remote shell whose command line also
   contained `serve_qwen36.py`.
   failure:
   local SSH exited 255 with no output.
   mitigation:
   verified server processes were stopped, then reran only the artifact delete.

4. First EC2-to-EC2 rsync failed due compile-host auth:
   command:
   `ssh ubuntu@16.51.94.87 'rsync ... ubuntu@16.51.179.180:...'`
   failure:
   `ubuntu@16.51.179.180: Permission denied (publickey).`
   mitigation:
   reran with local SSH agent forwarding:
   `ssh -A ubuntu@16.51.94.87 'rsync ... ubuntu@16.51.179.180:...'`
   verification:
   rsync completed, transferring about 57.4 GB.

5. Boundary probe CLI mismatch:
   command:
   `python /tmp/tmp_qwen36_boundary_probe.py --lengths ... --output-json ...`
   failure:
   `error: unrecognized arguments: --lengths ... --output-json ...`
   mitigation:
   reran with `--targets ...` and no `--output-json`.

## 2026-06-05 FP8 MLP Recheck

User concern:
older coherent branches had FP8 MLP working, so MLP should not be treated as
the corruption source.

Checked branches:

* `codex/full-fp8-qwen36`
* `codex/nki-deltanet-decode-step`

Evidence:

* Both branches' `fp8_full` policy includes MLP linear projections in the
  supported FP8 weight set:
  `mlp.{gate_proj,up_proj,down_proj}.weight`.
* Both branches' full-FP8 `modules_to_not_convert` exclude norms, rotary,
  convolution/state/control tensors, but do not exclude `layers.*.mlp`.
* `codex/full-fp8-qwen36` hard-disables the separate quantized MLP kernel in
  full-FP8 mode:
  `quantized_mlp_kernel_enabled=False`.
* `codex/nki-deltanet-decode-step` only enables MLP/quantized-MLP kernels when
  explicit CLI flags are set.

Artifact checked:

`/mnt/trainium_artifacts/qwen_artifacts/qwen36_27b_32768_fp8_full_lmheadfp8_kvbf16_qkvnki_tiled_segmented_cte512_nki_decode_stable_probe32k_gdnseg512_gdnrecfloat32_sampletokens_outlogits_b256_cte2048_pfx16k_slots64_async_cteargmaxsafe_20260605T043000Z_kkt_hier_scan7`

Remote dtype inspection on runtime host confirmed:

```text
modules_count 2188
mlp_exclusions_count 0
has_lm_head_exclusion False
quantized True
quantization_dtype f8e4m3
kv_cache_quant False
mlp_kernel_enabled False
mlp_tkg_nki_kernel_enabled False
quantized_mlp_kernel_enabled False
prefix_cte_attention_backend segmented_cte
prefix_cte_attention_segment_size 512

layers.0.mlp.gate_proj.weight torch.float8_e4m3fn
layers.0.mlp.up_proj.weight   torch.float8_e4m3fn
layers.0.mlp.down_proj.weight torch.float8_e4m3fn
layers.0.linear_attn.in_proj_a.weight torch.float8_e4m3fn
layers.0.linear_attn.in_proj_b.weight torch.float8_e4m3fn
lm_head.weight torch.float8_e4m3fn
```

Conclusion:

FP8 MLP weights through the normal matmul path are cleared. Do not spend more
time disabling MLP weights as a coherence fix. The only MLP-related suspect that
remains distinct is the separate MLP NKI / `quantized_mlp_kernel_enabled` path,
which was not enabled in the coherent artifact above.

Errors encountered during this recheck:

1. Local `rg` scan included missing path
   `contrib/models/Qwen3.6-27B/configs`.
   Failure:
   `rg: contrib/models/Qwen3.6-27B/configs: No such file or directory (os error 2)`.
   Mitigation:
   reran using concrete files and `git show` for the two target branches.

2. Remote artifact dtype script used a non-exported shell variable.
   Failure:
   Python raised `KeyError: 'ART'`.
   Mitigation:
   reran with `export ART=...`.

3. Remote safetensors script called `safe_open.close()`.
   Failure:
   `AttributeError: 'builtins.safe_open' object has no attribute 'close'`.
   Mitigation:
   reran without explicit close.

4. Remote one-line Python used `with safe_open(...)` after semicolon-separated
   statements.
   Failure:
   `SyntaxError: invalid syntax`.
   Mitigation:
   reran with simple one-line `safe_open(...)` assignment and key lookup.

## 2026-06-05 Coherent 256k Artifact / Runtime Fit Findings

Compile artifact:

`/mnt/trainium_artifacts/qwen_artifacts/qwen36_256k_fp8_fp8all_lmheadfp8_gatesfp8_kvbf16_qkvnki_tiled_segmented_cte512_gdnseg512_cte2048_pfx256k_slots64_20260605T080513Z_kkt_hier_scan7`

What passed:

* Compile completed with `Finished Compilation for all HLOs`,
  `CHECKPOINT_BANK_WEIGHTS_ADDED` for tp0..tp3, recurrent checkpoint-bank
  tensors in `torch.float32`, conv checkpoint-bank tensors in
  `torch.bfloat16`, and `COMPILE_DONE`.
* A staged launch with `MAX_MODEL_LEN=65536`, compiled `SEQ_LEN=262144`, and
  `NUM_GPU_BLOCKS_OVERRIDE=256` reached `/health`.
* Short/boundary/multi-turn probes were coherent at 146, 160, 485, 505, 526,
  1225, 2048, 2049, 4092, and 4096 tokens. The 2500-token synthetic ladder
  case returned an empty 1-token completion, but no mojibake, invalid-token
  fallback, or logits NaN was observed.
* Serve log scan after probes found no `negative token_id`, no
  `out-of-vocab token_id`, no `fallback argmax`, and no `finite=0`.

What failed:

1. Full 256k admission with BF16 KV.
   Log:
   `/home/ubuntu/validation_logs/fp8_256k_decode_nki/qwen36_256k_fp8_oldpolicy_serve_20260605T084630Z.log`.
   Error:
   `ValueError: To serve at least one request with the models's max seq len
   (262144), (4.0 GiB KV cache is needed, which is larger than the available
   KV cache memory (1.47 GiB). Based on the available memory, the estimated
   maximum model length is 96512.`
   Context:
   the log also said `Using Qwen hybrid KV-cache spec for 16/64 attention
   layers with 1 local KV heads`, so this was not accidental 48-layer KV
   allocation. The 4.0 GiB estimate matches 16 attention layers * K/V *
   262144 tokens * 1 local KV head * 256 head dim * BF16.
   Mitigation:
   added launcher passthrough for `--kv-cache-dtype` and
   `--kv-cache-memory-bytes` so FP8 KV accounting can be tested before another
   compile.

2. Full 256k admission with `GPU_MEMORY_UTILIZATION=0.99`.
   Log:
   `/home/ubuntu/validation_logs/fp8_256k_decode_nki/qwen36_256k_fp8_oldpolicy_serve_gmem099_20260605T085650Z.log`.
   Error:
   same `4.0 GiB KV cache` vs `1.47 GiB` failure.
   Hypothesis:
   on this Neuron/vLLM path, `gpu_memory_utilization` does not increase the
   usable KV pool after the large 256k/CTE2048 artifact is loaded.

3. Attempted staged lower launch with both `MAX_MODEL_LEN=65536` and
   `SEQ_LEN=65536`.
   Log:
   `/home/ubuntu/validation_logs/fp8_256k_decode_nki/qwen36_256k_fp8_oldpolicy_serve_max65k_20260605T085845Z.log`.
   Error:
   `AssertionError: max_context_length cannot be more than max_length`.
   Mitigation:
   tested a local clamp of `runtime_max_prompt` to `seq_len`, but that caused
   the next failure and was reverted before commit.

4. Staged lower launch with the `runtime_max_prompt` clamp.
   Log:
   `/home/ubuntu/validation_logs/fp8_256k_decode_nki/qwen36_256k_fp8_oldpolicy_serve_max65k_clamped_20260605T090024Z.log`.
   Error:
   `RuntimeError: Configuration mismatch: max_prompt_length in
   --additional-config (65536) does not match the Neuron model's compiled max
   prompt length (262144).`
   Mitigation:
   reverted the clamp; a lower `--max-model-len` can be used only while keeping
   `seq_len/max_prompt_length` at the compiled 262144.

5. 16k/32k/64k benchmark on the staged server hit Neuron runtime resources.
   Command:
   `qwen36_chat_completion_context_bench.py --lengths 16384,32768,64000
   --turns 1 --repeats 1 --concurrency 1 --max-tokens 64 --timeout 900
   --unique-per-request`.
   Serve log:
   `/home/ubuntu/validation_logs/fp8_256k_decode_nki/qwen36_256k_fp8_oldpolicy_serve_max65k_neuron262k_blocks256_20260605T090236Z.log`.
   Error:
   `Failed to allocate 1.234MB (alignment: none, usage: dma rings io)`,
   `Failed to create descriptors pring for dynamic ring`,
   `Failed to allocate memory for dma ring for queue qSPIO0_0`, and
   `Failure: NRT_RESOURCE in nrt_execute()`.
   Mitigation:
   killed only the stuck benchmark PID. Root hypothesis is HBM/runtime-resource
   pressure from the 256k CTE2048 artifact plus BF16 KV, not a coherence bug.

Local checks before committing these fixes:

* `bash -n` passed for `start_vllm_server.sh`,
  `tmp_launch_qwen36_segcte2048.sh`, and
  `tmp_compile_qwen32k_segcte2048_gdnseg512.sh`.
* `python3 -m py_compile` passed for `qwen36_27b_compile_fp8.py`,
  `validate_qwen_segcte_attention.py`, and
  `nki_deltanet_fused_legacy.py`.
* `PYTHONPATH=src:contrib/models/Qwen3.6-27B python3 -m pytest
  contrib/models/Qwen3.6-27B/test/unit/test_qwen36_compile_fp8_config.py
  contrib/models/Qwen3.6-27B/test/unit/test_qwen36_model_aliases.py
  contrib/models/Qwen3.6-27B/test/unit/test_hybrid_apc_manager.py -q`
  passed: 177 tests and 7 subtests.
* `PYTHONPATH=src python3 -m pytest test/unit/modules/test_async_execution.py
  -q` passed: 68 tests.

Local test limitation:

* `PYTHONPATH=src python3 -m pytest
  test/unit/modules/test_async_execution.py
  test/unit/models/test_prefix_caching_bucket_selection.py -q` failed during
  collection of `test_prefix_caching_bucket_selection.py` with
  `ModuleNotFoundError: No module named 'neuronx_distributed'`.
* `PYTHONPATH=src:contrib/models/Qwen3.6-27B python3 -m pytest
  contrib/models/Qwen3.6-27B/test/unit/test_vllm_serving_config.py
  contrib/models/Qwen3.6-27B/test/unit/test_config.py
  contrib/models/Qwen3.6-27B/test/unit/test_deltanet_decay.py -q` failed
  during collection of `test_config.py` with
  `ModuleNotFoundError: No module named 'neuronx_distributed'`.
* Mitigation:
  those NxD-runtime-dependent tests need the Neuron/NxD environment on the EC2
  host. They were not used as local commit gates.

## 2026-06-05 Decode-Step Baseline Reproduction Anchor

Current live baseline for the next multi-head branch comparison:

* Runtime host: `ubuntu@16.26.184.190`.
* Backend: `http://127.0.0.1:8001`.
* Proxy: `http://127.0.0.1:8000`.
* Backend log:
  `/home/ubuntu/validation_logs/fp8_256k_decode_nki/head_norowscale_baseline_backend_20260605T1228Z.log`.
* Proxy log:
  `/home/ubuntu/validation_logs/fp8_256k_decode_nki/head_norowscale_baseline_proxy_nothink_20260605T1239Z.log`.
* Artifact:
  `/mnt/trainium_artifacts/qwen_artifacts/qwen36_27b_256k_fp8_full_lmheadfp8_kvfp8_hybrid_apc_nki_decode_splitqkv_gdnrecbfloat16_sampletokens_b256_cte256_512_pfx16k_slots64_async_20260605T113222Z_head_norowscale`.

Compile host/source:

* Host: `ubuntu@16.26.135.243`.
* Source: `/home/ubuntu/inferentia-gdn-decode-step-baseline-20260605`.
* Base commit: `834543e`.
* Local source delta:
  split-QKV TKG row scales defaulted off via
  `QWEN36_SPLIT_QKV_TKG_ROW_SCALES=0`, restoring the May 28 kernel contract
  with `QuantizationType.NONE` and `qkv_w_scales=None`.

Compile command:

```bash
TS=$(date -u +%Y%m%dT%H%M%SZ)_head_norowscale \
LOGDIR=/home/ubuntu/validation_logs/fp8_256k_decode_nki \
LOAD_AFTER_COMPILE=0 \
QWEN36_SPLIT_QKV_TKG_ROW_SCALES=0 \
bash tmp_compile_qwen256k_fp8_full_decode_splitqkv_sampletokens.sh
```

Compile result:

* `Finished Compilation for all HLOs`.
* `CHECKPOINT_BANK_WEIGHTS_ADDED` for tp0..tp3.
* Checkpoint banks: `48 48 torch.bfloat16` per TP shard.
* `COMPILE_DONE`.

Backend launch flags:

```bash
--model-path /home/ubuntu/models/Qwen3.6-27B
--compiled-artifacts /mnt/trainium_artifacts/qwen_artifacts/qwen36_27b_256k_fp8_full_lmheadfp8_kvfp8_hybrid_apc_nki_decode_splitqkv_gdnrecbfloat16_sampletokens_b256_cte256_512_pfx16k_slots64_async_20260605T113222Z_head_norowscale
--max-model-len 262144
--seq-len 262144
--cte-buckets "256 512"
--context-encoding-bucket-pairs "256:256 512:256 256:512 512:512 256:1024 512:1024 256:2048 512:2048 256:4096 512:4096 256:8192 512:8192 512:16384"
--token-generation-buckets "512 768 1024 1280 2048 2304 4096 4352 8192 8448 16384 16640 24576 24832 32768 33024 65536 65792 131072 131328 262144"
--tensor-parallel-size 4
--logical-nc-config 2
--max-num-seqs 1
--ctx-batch-size 1
--async-mode
--enable-prefix-caching
--enable-hybrid-apc
--enable-vllm-chunked-prefill
--block-size 256
--gdn-checkpoint-interval 256
--max-gdn-checkpoint-slots 64
--gdn-recurrent-cache-dtype bfloat16
--gdn-conv-cache-dtype bfloat16
--hybrid-cache-mode all
--hybrid-apc-require-vllm-metadata
--hybrid-apc-enable-backed-prefix-reads
--num-gpu-blocks-override 1024
--host 127.0.0.1
--port 8001
```

Proxy launch:

```bash
python3 contrib/models/Qwen3.6-27B/vllm/qwen36_chat_proxy.py \
  --host 0.0.0.0 \
  --port 8000 \
  --backend-url http://127.0.0.1:8001
```

Validation result:

* Runtime health passed for backend and proxy.
* Cold chat HTTP 200 at 146, 160, 485, 505, 526, 1225, 2048, 2049, and
  2500 tokens.
* No serve-log evidence of `negative token_id`, `out-of-vocab token_id`,
  `fallback argmax`, logits NaN, `NRT_EXEC`, or `NRT_RESOURCE`.
* Natural-language smoke is coherent and matches the old May 28 "usable
  artifact" class.
* Strict semantic quality is not clean: exact marker-copy and structured recall
  fail. Treat this as a runtime-stable slow-prefix baseline, not final output
  coherence.

Speed baseline:

* 512 target: 511 prompt tokens, TTFT 0.9689s, effective prompt tok/s 527.2.
* 16k target: 16373 prompt tokens, TTFT 46.4619s, effective prompt tok/s
  352.4.

Dedicated detailed note:

`profile_artifacts/qwen36_head_norowscale_baseline_20260605/README.md`.
