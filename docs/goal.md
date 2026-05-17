# Qwen3.6 Cold-Prefill Perf Goal

## Working Rule

Work stays on `qwen36-cold-prefill-perf`.

The experimental-branch compiled artifacts are not compatible with this branch.
Compile branch-specific artifacts from:

```bash
/home/ubuntu/inferentia-gdn-cold-prefill
```

Do not use or overwrite the experimental checkout at:

```bash
/home/ubuntu/inferentia-gdn
```

Trainium instance:

```bash
ssh -i /Users/deepankarsingh1312/Downloads/trainium.pem ubuntu@16.26.90.15
```

Remote environment:

```bash
cd /home/ubuntu/inferentia-gdn-cold-prefill
export PATH=/opt/aws_neuronx_venv_pytorch_inference_vllm_0_16/bin:$PATH
export PYTHONPATH=/home/ubuntu/inferentia-gdn-cold-prefill/src:/home/ubuntu/inferentia-gdn-cold-prefill:/home/ubuntu/inferentia-gdn-cold-prefill/contrib/models/Qwen3.6-27B
export PYTHONUNBUFFERED=1
```

Common paths:

```bash
export QWEN36_MODEL=/home/ubuntu/models/Qwen3.6-27B
export QWEN36_QUANT=/home/ubuntu/models/Qwen3.6-27B-fp8-mlp-only
export QWEN36_OUT=/home/ubuntu/validation_logs/qwen36_cold_prefill_evidence
mkdir -p "$QWEN36_OUT"
```

## Current Branch Artifact

Branch-specific artifacts compiled from the cold checkout as of
2026-05-17 09:17 IST:

| length | artifact | compile log | status |
| ---: | --- | --- | --- |
| 2048 | `/home/ubuntu/qwen_artifacts_cold/qwen36_cold_prefill_branch_2048_fp8_mlp_only_hybrid_apc_b128_host_sampling_v6` | `/home/ubuntu/validation_logs/cold_prefill_compile/compile_2k_host_sampling_v6.log` | compiled, smoke-tested |
| 8192 | `/home/ubuntu/qwen_artifacts_cold/qwen36_cold_prefill_branch_8192_fp8_mlp_only_hybrid_apc_b128_host_sampling_v1` | `/home/ubuntu/validation_logs/cold_prefill_compile/compile_8k_host_sampling_v1.log` | compiled, smoke-tested |
| 32768 | `/home/ubuntu/qwen_artifacts_cold/qwen36_cold_prefill_branch_32768_fp8_mlp_only_hybrid_apc_b128_host_sampling_v1` | `/home/ubuntu/validation_logs/cold_prefill_compile/compile_32k_host_sampling_v1.log` | compiled, smoke-tested |
| 131072 | `/home/ubuntu/qwen_artifacts_cold/qwen36_cold_prefill_branch_131072_fp8_mlp_only_hybrid_apc_b128_host_sampling_v1` | `/home/ubuntu/validation_logs/cold_prefill_compile/compile_128k_host_sampling_v1.log` | compiled, smoke-tested |
| 262144 | `/home/ubuntu/qwen_artifacts_cold/qwen36_cold_prefill_branch_262144_fp8_mlp_only_kvfp8_hybrid_apc_b128_host_sampling_v1` | `/home/ubuntu/validation_logs/cold_prefill_compile/compile_262k_kvfp8_host_sampling_v1.log` | compiled, smoke-tested; KV cache uses FP8 direct-cast |

262K branch-specific compile attempts from the cold checkout did not produce a
usable artifact on the current `trn2.3xlarge` instance:

| length | attempted artifact | compile log | status |
| ---: | --- | --- | --- |
| 262144 | `/home/ubuntu/qwen_artifacts_cold/qwen36_cold_prefill_branch_262144_fp8_mlp_only_hybrid_apc_b128_host_sampling_v1` | `/home/ubuntu/validation_logs/cold_prefill_compile/compile_262k_host_sampling_v1.log` | failed during TKG Neuron compile with Trainium2 HBM verifier |
| 262144 | `/home/ubuntu/qwen_artifacts_cold/qwen36_cold_prefill_branch_262144_fp8_mlp_only_hybrid_apc_b256_host_sampling_v1` | `/home/ubuntu/validation_logs/cold_prefill_compile/compile_262k_b256_host_sampling_v1.log` | failed during TKG Neuron compile with Trainium2 HBM verifier |

Shared shape/config:

```text
max_context_length: 1024
context_encoding_buckets: [128, 256, 512, 1024]
prefix_buckets: [128, 256, 512, 1024]
block_size: 128
tp_degree: 4
logical_nc_config: 2
prefix_caching: enabled
hybrid_apc: enabled
gdn_checkpoint_interval: 128
max_gdn_checkpoint_slots: 8
sampling: host/vLLM CPU sampling, not Neuron on-device sampling
KV cache: BF16 for the 2K/8K/32K/128K artifacts; FP8 direct-cast for the
262K recovery artifact because the BF16-KV TKG graph exceeds Trainium2 HBM on
this instance.
```

Length-specific PA blocks:

| seq_len | pa_num_blocks | runtime override |
| ---: | ---: | ---: |
| 2048 | 16 | `--num-gpu-blocks-override 16` |
| 8192 | 64 | `--num-gpu-blocks-override 64` |
| 32768 | 256 | `--num-gpu-blocks-override 256` |
| 131072 | 1024 | `--num-gpu-blocks-override 1024` |
| 262144 | 2048 | `--num-gpu-blocks-override 2048` |

The older branch artifact:

```bash
/home/ubuntu/qwen_artifacts_cold/qwen36_cold_prefill_branch_2048_fp8_mlp_only_hybrid_apc_b128_v5
```

was deleted to recover disk. It loaded and ran pure prefill, but was not valid
for decode validation because it was compiled with on-device sampling and
emitted invalid huge first-token IDs such as `2143289344`, which then caused TKG
out-of-bounds failures.

## Reproduce 2K Compile

```bash
cd /home/ubuntu/inferentia-gdn-cold-prefill

PATH=/opt/aws_neuronx_venv_pytorch_inference_vllm_0_16/bin:$PATH \
PYTHONPATH=/home/ubuntu/inferentia-gdn-cold-prefill/src:/home/ubuntu/inferentia-gdn-cold-prefill:/home/ubuntu/inferentia-gdn-cold-prefill/contrib/models/Qwen3.6-27B \
PYTHONUNBUFFERED=1 \
/opt/aws_neuronx_venv_pytorch_inference_vllm_0_16/bin/python \
  contrib/models/Qwen3.6-27B/test/integration/qwen36_27b_compile_fp8.py \
  --repo-root /home/ubuntu/inferentia-gdn-cold-prefill \
  --model-path /home/ubuntu/models/Qwen3.6-27B \
  --compiled-path /home/ubuntu/qwen_artifacts_cold/qwen36_cold_prefill_branch_2048_fp8_mlp_only_hybrid_apc_b128_host_sampling_v6 \
  --quantized-checkpoints-path /home/ubuntu/models/Qwen3.6-27B-fp8-mlp-only \
  --seq-len 2048 \
  --cte-buckets 128 256 512 1024 \
  --prefix-buckets 128 256 512 1024 \
  --block-size 128 \
  --pa-num-blocks 16 \
  --tp-degree 4 \
  --logical-nc-config 2 \
  --enable-prefix-caching \
  --enable-hybrid-apc \
  --disable-on-device-sampling \
  --kernel-q-tile-size 128 \
  --kernel-kv-tile-size 1024 \
  --gdn-checkpoint-interval 128 \
  --max-gdn-checkpoint-slots 8 \
  --hybrid-apc-require-vllm-metadata
```

Do not compile this artifact with `--enable-vllm-chunked-prefill`; current NxDI
compile rejects that mode with `NotImplementedError: Chunked Prefill is not
available in NxDI for now`. Runtime vLLM chunked prefill is still used by the
benchmark command.

## Runtime Flags That Matter

Always pass the compiled PA block count for the artifact being tested:

```bash
--num-gpu-blocks-override <compiled_pa_num_blocks>
```

Without that override, vLLM can allocate a much larger physical block table than
the compiled Neuron cache supports.

For the current F-path smoke, use:

```bash
--enable-prefix-caching \
--enable-hybrid-apc \
--hybrid-apc-require-vllm-metadata \
--gdn-checkpoint-interval 128 \
--max-gdn-checkpoint-slots 8 \
--hybrid-cache-mode all \
--block-size 128
```

The runner and minimal OpenAI-compatible server now read HBM usage through
`neuron-monitor` instead of allocating a second XLA device after vLLM has loaded
the model. This keeps successful rows from emitting misleading NRT logical-core
allocation errors. Set `QWEN36_HBM_PROBE=xla` only when explicitly debugging the
older XLA memory-info path, or `QWEN36_HBM_PROBE=none` to disable HBM
collection.

The Neuron config serializer/loader also normalizes serialized
`kv_quant_config` values back to `QuantizationType` enums and torch dtypes.
This is required for loading artifacts compiled with KV-cache quantization; the
first 262K load attempt failed before model load until this was fixed.

Always pass the compiled context-model prompt shape when loading these artifacts:

```bash
--compiled-max-prompt-length 1024
```

The first strict-final collection attempt with the branch-specific artifacts
failed on the first A row because the runtime additional config advertised
`max_prompt_length=512` while the artifact was compiled with
`max_prompt_length=1024`. The runner now applies the compiled prompt length to
both the top-level vLLM additional config and
`override_neuron_config.max_context_length`.

## Evidence Collected

Direct TKG sanity with host-sampling v6 passed:

```bash
/home/ubuntu/validation_logs/qwen36_cold_prefill_evidence/qwen36_cold_direct_hybrid_vllm_chunk_probe_2k_max_tokens2_host_sampling_v6.log
```

Result:

```text
returncode: 0
prompt tokens: 16
generated tokens: 2
token IDs: [0, 0]
selected CTE bucket: 128
```

HBM-probe patch smoke:

```bash
/home/ubuntu/validation_logs/qwen36_cold_prefill_evidence/qwen36_cold_smoke_2k_F_hbm_probe_patch_128_mt1.json
/home/ubuntu/validation_logs/qwen36_cold_prefill_evidence/qwen36_cold_smoke_2k_F_hbm_probe_patch_128_mt1.log
```

Result:

```text
returncode: 0
artifact_load_success: true
prompt tokens: 127
token IDs: [0]
selected CTE bucket: 128
hbm_usage.source: neuron-monitor
hbm_usage.bytes_used: 50,608,832,912
output tail has no NRT logical-core allocation infodump
```

All-length `max_tokens=1` smoke:

```bash
/home/ubuntu/validation_logs/qwen36_cold_prefill_evidence/qwen36_cold_smoke_2k_F_host_sampling_all_lengths_mt1.json
/home/ubuntu/validation_logs/qwen36_cold_prefill_evidence/qwen36_cold_smoke_2k_F_host_sampling_all_lengths_mt1.log
```

| target | prompt tokens | returncode | token IDs | selected bucket | chunks | prefill ms | actual tok/s |
| --- | ---: | ---: | --- | ---: | ---: | ---: | ---: |
| 128 | 127 | 0 | `[0]` | 128 | 1 | 114.185 | 1112.231 |
| 256 | 256 | 0 | `[0]` | 256 | 1 | 144.259 | 1774.587 |
| 512 | 509 | 0 | `[0]` | 512 | 1 | 232.890 | 2185.584 |
| 1024 | 1021 | 0 | `[0]` | 1024 | 1 | 453.145 | 2253.140 |
| 2048 | 2045 | 0 | `[0]` | 1024 | 2 | 454.612 | 4498.343 |

Longer-artifact `max_tokens=1` smoke:

```bash
/home/ubuntu/validation_logs/qwen36_cold_prefill_evidence/qwen36_cold_smoke_8k_F_host_sampling_mt1.json
/home/ubuntu/validation_logs/qwen36_cold_prefill_evidence/qwen36_cold_smoke_32k_F_host_sampling_mt1.json
/home/ubuntu/validation_logs/qwen36_cold_prefill_evidence/qwen36_cold_smoke_128k_F_host_sampling_mt1.json
```

| target | prompt tokens | returncode | token IDs | selected bucket | chunks | prefill ms | actual tok/s |
| --- | ---: | ---: | --- | ---: | ---: | ---: | ---: |
| 8192 | 8189 | 0 | `[0]` | 1024 | 8 | 463.339 | 17673.892 |
| 32768 | 32765 | 0 | `[0]` | 1024 | 32 | 499.072 | 65651.797 |
| 131072 | 131071 | 0 | `[0]` | 1024 | 128 | 611.900 | 214203.265 |

262K KV-FP8 recovery compile:

```bash
/home/ubuntu/validation_logs/cold_prefill_compile/compile_262k_kvfp8_host_sampling_v1.log
```

Result:

```text
KV_CACHE_QUANT {"direct_cast": true, "enabled": true, "quant_dtype": "float8_e4m3fn"}
token_generation_model priority HLO compiled in 658.774 s
all HLOs compiled in 360.748 s
COMPILE_DONE
```

262K KV-FP8 `max_tokens=1` smoke:

```bash
/home/ubuntu/validation_logs/qwen36_cold_prefill_evidence/qwen36_cold_smoke_262k_F_kvfp8_host_sampling_mt1.json
/home/ubuntu/validation_logs/qwen36_cold_prefill_evidence/qwen36_cold_smoke_262k_F_kvfp8_host_sampling_mt1.log
```

| target | prompt tokens | returncode | token IDs | selected bucket | chunks | prefill ms | actual tok/s | HBM source |
| --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | --- |
| 262144 | 262143 | 0 | `[0]` | 1024 | 256 | 807.770 | 324526.660 | `neuron-monitor` |

262K compile failure evidence:

```bash
/home/ubuntu/validation_logs/cold_prefill_compile/compile_262k_host_sampling_v1.log
/home/ubuntu/validation_logs/cold_prefill_compile/compile_262k_b256_host_sampling_v1.log
```

Both 262K attempts reached HLO generation and failed when compiling
`token_generation_model` with:

```text
[NCC_EVRF009] Size of total input and output tensors exceeds HBM limit of Trainium2.
```

The block-128 attempt needed `26,541,173,020` bytes vs. `25,769,803,776`
available. The block-256 attempt needed `26,549,557,532` bytes vs.
`25,769,803,776` available. The block-256 top tensors were:

```text
input329 of shape bf16[248320,1280]
input1386 of shape bf16[62080,5120]
input105 of shape bf16[1025,256,1,256]
```

Short-length `max_tokens=32` TKG smoke:

```bash
/home/ubuntu/validation_logs/qwen36_cold_prefill_evidence/qwen36_cold_smoke_2k_F_host_sampling_short_lengths_mt32.json
/home/ubuntu/validation_logs/qwen36_cold_prefill_evidence/qwen36_cold_smoke_2k_F_host_sampling_short_lengths_mt32.log
```

| target | prompt tokens | returncode | generated IDs | unique IDs | selected bucket | chunks | generated | decode tokens | request ms |
| --- | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 128 | 127 | 0 | 32 | `[0]` | 128 | 1 | 32 | 31 | 1707.120 |
| 256 | 256 | 0 | 32 | `[0]` | 256 | 1 | 32 | 31 | 1727.281 |
| 512 | 509 | 0 | 32 | `[0]` | 512 | 1 | 32 | 31 | 1825.483 |
| 1024 | 1021 | 0 | 32 | `[0]` | 1024 | 1 | 32 | 31 | 2042.913 |

The standalone smoke intentionally skipped the 2048 prompt for `max_tokens=32`
because approximately `2045 + 32` tokens exceeds the compiled
`max_length=2048`. The strict-final benchmark runner now reserves decode budget
for generation rows at an artifact's max length by setting
`effective_prompt_target_tokens = seq_len - max_tokens` while keeping the
nominal target prompt length for acceptance accounting.

Strict-final orchestration preflight and dry-run evidence:

```bash
/home/ubuntu/validation_logs/qwen36_cold_prefill_strict_final/qwen36_cold_prefill_cold_branch_preflight.json
/home/ubuntu/validation_logs/qwen36_cold_prefill_strict_final/qwen36_cold_prefill_cold_branch_dryrun_g_tiles.json
/home/ubuntu/validation_logs/qwen36_cold_prefill_strict_final/qwen36_cold_prefill_cold_branch_dryrun_long_hj.json
```

Result:

```text
preflight branch: qwen36-cold-prefill-perf
all five compiled artifact paths: present and non-empty
long-context variants: A_single512_old_chunked, H_128k_candidate, J_262k_recovery_block128
G tile dry-run block sizes: 128 only
H/J CTE args: --cte-bucket-profile short
H effective prompt target for max_tokens=32: 131040 of 131072
J effective prompt target for max_tokens=32: 262112 of 262144
```

Strict-final first launch evidence:

```bash
/home/ubuntu/validation_logs/qwen36_cold_prefill_strict_final/qwen36_cold_prefill_cold_branch_strict_full_v1_matrix.log
/home/ubuntu/validation_logs/qwen36_cold_prefill_strict_final/qwen36_cold_prefill_cold_branch_strict_full_v1_console.log
/home/ubuntu/validation_logs/qwen36_cold_prefill_strict_final/qwen36_cold_prefill_cold_branch_strict_full_v1_manifest.json
```

Result:

```text
first A_single512_old_chunked row failed before the compiled prompt fix
artifact_load_success: false
root cause: additional_config max_prompt_length/max_context_length was 512,
but the compiled artifact expects 1024
```

Post-fix strict canary evidence:

```bash
/home/ubuntu/validation_logs/qwen36_cold_prefill_strict_final/qwen36_cold_prefill_failed_A_retry_compiled_prompt_context_1024.json
/home/ubuntu/validation_logs/qwen36_cold_prefill_strict_final/qwen36_cold_prefill_canary_A_to_G_prompt128_mt1_mt32.json
/home/ubuntu/validation_logs/qwen36_cold_prefill_strict_final/qwen36_cold_prefill_long_canary_H_J_mt1.json
```

Result:

```text
A retry: returncode=0, artifact_load_success=true
A-G short canary: 20 rows, all returncode=0 and artifact_load_success=true
G tile cases: (q,kv,block) = (128,512,128), (128,1024,128),
  (128,2048,128), (256,1024,128)
H long canary: 131071 prompt tokens, bucket 1024, 128 chunks,
  prefill 609.962 ms, token IDs [0]
J long canary: 262143 prompt tokens, bucket 1024, 256 chunks,
  prefill 815.319 ms, token IDs [0]
```

Strict-final v2 launch:

```bash
/home/ubuntu/validation_logs/qwen36_cold_prefill_strict_final/qwen36_cold_prefill_cold_branch_strict_full_v2_preflight.json
/home/ubuntu/validation_logs/qwen36_cold_prefill_strict_final/qwen36_cold_prefill_cold_branch_strict_full_v2_dryrun.txt
/home/ubuntu/validation_logs/qwen36_cold_prefill_strict_final/qwen36_cold_prefill_cold_branch_strict_full_v2_console.log
/home/ubuntu/validation_logs/qwen36_cold_prefill_strict_final/qwen36_cold_prefill_cold_branch_strict_full_v2_matrix.log
/home/ubuntu/validation_logs/qwen36_cold_prefill_strict_final/qwen36_cold_prefill_cold_branch_strict_full_v2_run_manifest.json
```

Result:

```text
preflight: passed
dry-run: includes all five branch-specific artifacts, PA block overrides,
  --compiled-max-prompt-length 1024, H/J long variants, and block128 G sweep
collection: stopped intentionally during matrix phase after 12 completed rows
matrix rows: artifact_load_success=true, returncode=0
blocker: every completed row had gdn_state_diff=null
```

The strict-final orchestrator now aborts remaining phases when `--fail-fast` is
set and a phase returns nonzero. This prevents repeating the earlier behavior
where a failed matrix launch continued into long-context collection.

The branch now has an opt-in host-side `GDN_STATE_DIFF` emission path in
`Qwen35ModelWrapper._copy_past_key_values`, enabled automatically by
`run_offline_inference.py` for `--enable-hybrid-apc` runs. The payload measures
the context trace output versus the TKG/CTE recurrent and conv state buffers
after host copy. This is intended to satisfy the existing strict gate with
measured state-copy evidence instead of a fabricated sidecar.

## Commands Used For Current Smoke

All-length prefill smoke:

```bash
python validation_scripts/qwen36_cold_prefill_benchmark.py \
  --model-path /home/ubuntu/models/Qwen3.6-27B \
  --compiled-artifacts /home/ubuntu/qwen_artifacts_cold/qwen36_cold_prefill_branch_2048_fp8_mlp_only_hybrid_apc_b128_host_sampling_v6 \
  --prompt-lengths 128 256 512 1024 2048 \
  --variants F_short_text_compact_fused_cold_zero \
  --max-tokens-values 1 \
  --repetitions 1 \
  --output-json /home/ubuntu/validation_logs/qwen36_cold_prefill_evidence/qwen36_cold_smoke_2k_F_host_sampling_all_lengths_mt1.json \
  --tensor-parallel-size 4 \
  --logical-nc-config 2 \
  --max-num-seqs 1 \
  --ctx-batch-size 1 \
  --block-size 128 \
  --num-gpu-blocks-override 16 \
  --enable-prefix-caching \
  --enable-hybrid-apc \
  --hybrid-apc-require-vllm-metadata \
  --gdn-checkpoint-interval 128 \
  --max-gdn-checkpoint-slots 8 \
  --hybrid-cache-mode all \
  --fail-fast
```

Decode smoke:

```bash
python validation_scripts/qwen36_cold_prefill_benchmark.py \
  --model-path /home/ubuntu/models/Qwen3.6-27B \
  --compiled-artifacts /home/ubuntu/qwen_artifacts_cold/qwen36_cold_prefill_branch_2048_fp8_mlp_only_hybrid_apc_b128_host_sampling_v6 \
  --prompt-lengths 128 256 512 1024 \
  --variants F_short_text_compact_fused_cold_zero \
  --max-tokens-values 32 \
  --repetitions 1 \
  --output-json /home/ubuntu/validation_logs/qwen36_cold_prefill_evidence/qwen36_cold_smoke_2k_F_host_sampling_short_lengths_mt32.json \
  --tensor-parallel-size 4 \
  --logical-nc-config 2 \
  --max-num-seqs 1 \
  --ctx-batch-size 1 \
  --block-size 128 \
  --num-gpu-blocks-override 16 \
  --enable-prefix-caching \
  --enable-hybrid-apc \
  --hybrid-apc-require-vllm-metadata \
  --gdn-checkpoint-interval 128 \
  --max-gdn-checkpoint-slots 8 \
  --hybrid-cache-mode all \
  --fail-fast
```

## Next Work

1. Sync the state-diff emission patch to the Trainium cold checkout and run a
   short A-G canary to confirm rows now carry `gdn_state_diff`.
2. Relaunch strict-final only after the canary proves state-diff rows are
   present; keep the stopped v2 evidence as the pre-patch failure record.
3. Watch the full run for performance regressions, DMA spill evidence, and any
   vLLM block-index mismatch against the compiled PA cache. Short A-G and long
   H/J canaries have already passed artifact-load compatibility.
4. If strict acceptance still fails only because GDN recurrent/conv state-diff
   fields are missing, collect an alternate measured sidecar and pass it through
   `--gdn-state-diff-json`; do not fabricate a sidecar.
5. If strict acceptance fails on performance rather than launch shape, keep the
   row evidence and tune from the current block128/short-bucket baseline.

Suggested PA block counts for block size 128:

| seq_len | pa_num_blocks |
| ---: | ---: |
| 2048 | 16 |
| 8192 | 64 |
| 32768 | 256 |
| 131072 | 1024 |
| 262144 | 2048 |

262K block-size 128 and block-size 256 have both been tried. Block size did not
remove the TKG HBM compile limit because the largest sequence-sized PA tensor
remains effectively `seq_len`-scaled.

Current disk state after retaining the 2K, 8K, 32K, 128K, and 262K successful
artifacts leaves roughly 37G free on `/home/ubuntu`.

## Stop Conditions

Stop and debug if any of these occur:

```text
artifact_load_success is false
returncode is nonzero
token IDs are outside the tokenizer vocabulary
selected CTE bucket does not match the prompt length
2048 max_tokens=1 does not report two 1024-token CTE chunks
output_tail contains DMA spill evidence
vLLM receives block indexes outside the compiled PA cache
```

## Current Status

The 2K, 8K, 32K, 128K, and 262K cold-branch host-sampling artifacts are compiled
and smoke-tested on the Trainium instance. The 262K artifact is a KV-FP8
recovery variant; the earlier BF16-KV 262K attempts still document the TKG HBM
limit. The 2K artifact fixes the invalid on-device-sampling token IDs and the
follow-on TKG out-of-bounds failure seen with the first branch artifact. The HBM
probe now uses `neuron-monitor` in both runtime entrypoints, and the smoke rows
confirm successful metrics collection without the previous post-run NRT
allocation infodump.

The full long-context goal is not complete yet. The remaining blocker is no
longer branch-specific artifact compilation or launch-shape compatibility. The
full strict-final v2 run was stopped after proving the matrix loads and runs
with the agreed 262K KV-FP8 recovery shape but does not emit measured GDN
recurrent/conv state-diff evidence. The next gate is a short Trainium canary
with the new runtime state-diff emission patch, followed by a fresh strict-final
run if the canary rows include `gdn_state_diff`.
