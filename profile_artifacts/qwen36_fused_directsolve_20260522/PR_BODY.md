# Qwen3.6: Fused DeltaNet Direct-Solve Follow-Up

## Summary

This branch is a reviewer-friendly presentation of the fused DeltaNet direct-solve result for Qwen3.6. It is intentionally based on PR 164, `contrib/qwen36-27b-vllm-apc-pr` at `ac7df71`, so reviewers can see the direct-solve change and validation artifacts without also reviewing the full experimental branch stack.

The important result is that the fused DeltaNet CTE path is now coherent with realistic Qwen gate values when the Neumann power-doubling solve is replaced by a direct triangular RHS solve.

## Branch Lineage

The actual development history was:

```text
PR 164 / vLLM APC baseline, building on PR #140
  -> experimental
      -> qwen-fused-neumann-stable-decay
```

PR 164 is the original Qwen3.6 vLLM APC baseline for this branch, and PR 164
itself builds on Jim Burtoft's Qwen3.6-27B contrib work in PR #140. After that,
the `experimental` branch accumulated the runtime and validation work needed to
make Hybrid APC usable and measurable. The final
`qwen-fused-neumann-stable-decay` branch was created from `experimental` and
added the fused DeltaNet stability work.

This clean branch extracts the direct-solve fused DeltaNet work and its result artifacts onto PR 164 for review. It does not include the entire `experimental` branch history.

## Why This Exists

The original fused DeltaNet path used a Neumann power-doubling solve for the recurrence. That approach is mathematically convenient, but it is fragile for realistic Qwen gate scales because it repeatedly forms full matrix powers and can amplify numerical error. In practice, the fused path could produce unstable or incoherent tokens.

The chunked DeltaNet path already used a more stable direct triangular solve. This branch ports that idea to the fused path: compute the causal recurrence through a direct triangular RHS solve instead of Neumann power-doubling.

The goal is not to claim the fused path is now the final production baseline by itself. The goal is to make the fused-kernel stability fix reviewable and to preserve the validation evidence that it produces coherent output inside the full experimental runtime lineage.

## Major Changes From PR 164 To The Tested Branch

The full tested branch differs from PR 164 by roughly 105 source/result files. The main work streams from `experimental` were:

### Hybrid APC Runtime

- Added Hybrid APC request records and cache metadata.
- Added checkpoint-slot lifecycle management.
- Added restore and commit masks for GDN/recurrent state reuse.
- Added backed prefix reads and stricter unbacked-read guards.
- Added explicit metadata contracts so runtime decisions are scheduler-authorized instead of inferred locally.

### vLLM / NxDI Scheduler Bridge

- Added Qwen-specific vLLM scheduler patching for Hybrid APC.
- Propagated request IDs into the Neuron model runner path.
- Recognized cached chunked-prefill continuations.
- Tracked active scheduled suffix lengths.
- Added no-prefix fallback handling.
- Authorized backed prefix continuations through scheduler metadata.

### Qwen Model Execution

- Extended Qwen chunked prefill for Hybrid APC.
- Added GDN checkpoint commit/restore handling.
- Added text-only CTE input handling.
- Added compact CTE masks.
- Fixed prefix/suffix boundary handling.
- Guarded decode rows from unnecessary APC restore handling.
- Added chat-template thinking controls for validation and serving.

### NxDI Prefix-Cache Plumbing

- Updated prefix-cache model wrapper paths for vectorized APC args.
- Fixed prefix-cache bucket selection and padded-row safety.
- Added cached decode row handling.
- Added async checkpoint lifecycle cleanup.
- Added unit tests around bucket selection, async execution, and Hybrid APC prefix cache behavior.

### DeltaNet NKI Kernels

- Added DeltaNet backend compile controls.
- Added chunked and fused DeltaNet validator paths.
- Tested masked Neumann variants.
- Stabilized the fused DeltaNet kernel.
- Replaced the fused Neumann power-doubling solve with the direct triangular RHS solve.

### FP8 / Artifact Compile Path

- Added FP8 MLP-only compile configuration coverage.
- Added artifact config audit guardrails.
- Aligned validation max prompt length with compiled artifacts.
- Added `pa_num_blocks` and bucket sanity checks.
- Added 128K/TKG bucket validation support.

### Serving And API Compatibility

- Updated OpenAI-compatible proxy/server behavior.
- Normalized chat-template `enable_thinking=false`.
- Fixed stop-sequence handling.
- Added server startup/offline inference helpers.
- Added OpenAI API probe scripts and results.

### Validation And Results

- Added exactness validation.
- Added OpenAI chat APC validation.
- Added boundary APC probes.
- Added context sweeps.
- Added offline decode benchmarking.
- Added memory/perf capture.
- Recorded 4K/128K FP8 Hybrid APC results, decode results, and fused direct-solve results.

## What This Clean Branch Adds

This branch extracts only the fused DeltaNet follow-up commits:

- `Stabilize Qwen fused DeltaNet decay`
- `Add isolated fused DeltaNet NKI validation`
- `Load fused NKI kernel directly in validator`
- `Fix fused DeltaNet solve stability`
- Validation/result documentation commits

Concretely, it changes:

- `contrib/models/Qwen3.6-27B/src/nki_kernels/nki_deltanet_fused.py`
- `contrib/models/Qwen3.6-27B/scripts/validate_deltanet_fused_nki.py`
- `contrib/models/Qwen3.6-27B/test/unit/test_deltanet_decay.py`
- `profile_artifacts/qwen36_fused_directsolve_20260522/*`

## Implementation Details

The fused kernel previously used Neumann power-doubling to solve the recurrent DeltaNet update. The direct-solve version computes the lower-triangular causal recurrence explicitly in the RHS solve path. This avoids the repeated full-matrix power operations that were unstable under realistic gate values.

The validator was also made standalone enough to load the fused NKI kernel directly. This matters because review and debug runs should not depend on package import side effects.

The CPU regression test was updated to cover realistic decay/gate scales and to catch the class of instability that showed up in the fused branch.

## Validation Results

Validation artifacts are stored under:

```text
profile_artifacts/qwen36_fused_directsolve_20260522/
```

### Local Checks

```bash
python3 -m py_compile \
  contrib/models/Qwen3.6-27B/src/nki_kernels/nki_deltanet_fused.py \
  contrib/models/Qwen3.6-27B/scripts/validate_deltanet_fused_nki.py \
  contrib/models/Qwen3.6-27B/test/unit/test_deltanet_decay.py

python3 -m pytest contrib/models/Qwen3.6-27B/test/unit/test_deltanet_decay.py -q
```

Result:

```text
2 passed
```

### Artifact Under Test

```text
qwen36_27b_128k_fp8_mlp_edgebf16_hybrid_apc_nki_fusedstable_directsolve_retry_b256_cte256_512_pfx16k_slots64_tkg8192_32768_131072_async_20260522T130050Z
```

Runtime:

- Instance: `trn2.3xlarge`
- Runtime path: offline vLLM/NxDI
- Sampling: on-device greedy
- `output_logits=false`
- TKG buckets: `[8192, 32768, 131072]`
- Prefix buckets in this artifact: up to `16384`

### Coherence

File:

```text
qwen36_directsolve_chat_coherence_20260522T1332Z.json
```

Result:

- Overall pass: `true`
- Chat template: `enable_thinking=false`
- Fact, code, and prefix-cache prompts produced real non-repetitive output.
- Smoke decode throughput: about `20.5 tok/s`

### Decode

File:

```text
qwen36_directsolve_decode_bench_20260522T1348Z.json
```

Result:

- Average decode throughput: `21.63 tok/s`
- TPOT: `46.2 ms/token`
- 128-token decode latency: `5.92 s`

### Cold / Warm Prefill

File:

```text
context_sweep_partial_20260522T1348Z.json
```

| Prompt tokens | Cold TTFT | Cold prefill | Warm TTFT | Warm prefill |
| ---: | ---: | ---: | ---: | ---: |
| 512 | 1.31 s | 390 tok/s | 0.42 s | 1.2k tok/s |
| 4096 | 7.03 s | 582 tok/s | 0.42 s | 9.8k tok/s |
| 8192 | 13.61 s | 602 tok/s | 0.43 s | 18.9k tok/s |
| 16384 | 27.84 s | 589 tok/s | 0.45 s | 36.3k tok/s |

### Memory

File:

```text
qwen36_directsolve_perf_capture_20260522T1348Z.json
```

Result:

- Neuron HBM peak sum: `60.1 GiB`
- Host process RSS peak: `46.3 GiB`
- Main logical cores peaked around `14.57 GiB` each on cores `0`, `2`, `4`, and `6`

Memory caveat: this is a Neuron high-water peak from the Hybrid APC artifact,
not a prompt-length-only 16K allocation. The artifact was compiled with
`pa_num_blocks=512`, `max_gdn_checkpoint_slots=64`, and TKG buckets
`[8192, 32768, 131072]`. PR 164 reports `~53.25 GB` decimal during a 64K
vLLM/APC eval, but this clean branch does not include that run's raw memory log
or artifact config. A strict memory regression claim needs like-for-like A/B
measurement with the same capture script and comparable cache/bucket settings.

## Known Limitations

The compiled artifact used for this validation has `prefix_buckets` only through `16384`. The 32K row failed with:

```text
Prefix len 16640 exceeds largest bucket 16384 for context_encoding_model
```

That is an artifact bucket coverage limitation, not a direct-solve correctness failure. A long-context follow-up compile should include prefix buckets beyond 16K, ideally through the target 64K/128K range.

The artifact results were produced from `qwen-fused-neumann-stable-decay`, which includes the full `experimental` runtime lineage. This clean branch shows the fused direct-solve extraction and result evidence, but it should not be read as proof that PR 164 plus only these extracted commits reproduces every Hybrid APC runtime behavior from `experimental`.

## What Is Intentionally Not Included

This clean branch does not include the full 80+ commit `experimental` stack. It also does not include large raw logs, obsolete investigation branches, or temporary scripts. Those were useful during development but would make this review branch hard to inspect.

## Recommended Next Step

Use this branch as the reviewer-facing result branch for the fused direct-solve change. If reviewers require source-level reproducibility for the full artifact behavior, stack or merge a curated `experimental` runtime-stabilization branch beneath this direct-solve work.
