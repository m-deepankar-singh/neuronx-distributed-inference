# Qwen3.6 Fused Direct-Solve Validation

This note records the reviewer-facing validation for the fused DeltaNet CTE
direct-solve branch. The raw JSON captures are kept in
`profile_artifacts/qwen36_fused_directsolve_20260522/`.

## Artifact

| Field | Value |
|-------|-------|
| Source branch | `qwen-fused-neumann-stable-decay` |
| Source commit | `ae2613d` |
| Clean PR branch | `qwen36-pr164-directsolve-baseline` |
| Artifact | `qwen36_27b_128k_fp8_mlp_edgebf16_hybrid_apc_nki_fusedstable_directsolve_retry_b256_cte256_512_pfx16k_slots64_tkg8192_32768_131072_async_20260522T130050Z` |
| Runtime | trn2.3xlarge, TP=4, LNC=2 |
| Serving path | Offline vLLM/NxDI, on-device greedy sampling |

The Trn2 validation host was rechecked on 2026-05-22 before publishing these
notes: it was reachable, reported instance type `trn2.3xlarge`, logical Neuron
core config `2`, and contained the artifact directory at 36G.

The clean PR branch is based on PR 164 and extracts the fused DeltaNet
direct-solve change. The measured artifact was compiled from the full
experimental runtime lineage:

```text
PR 164 / vLLM APC baseline, building on PR #140
  -> experimental
      -> qwen-fused-neumann-stable-decay
```

PR 164 itself builds on Jim Burtoft's Qwen3.6-27B contrib work in PR #140.
That lineage matters because the artifact also uses the Hybrid APC runtime,
FP8 128K compile settings, and vLLM/NxDI serving fixes developed after the
original PR 164 baseline.

## What Changed

The fused DeltaNet CTE kernel previously used Neumann power-doubling for the
intra-chunk correction. That approximation was fast, but it was not stable for
realistic Qwen3.6 gate/decay values and produced invalid output in fused BF16
experiments.

The direct-solve version replaces that correction with the lower-triangular
solve strategy used by the stable chunked CTE path. This keeps the fused
single-kernel structure while avoiding the observed Neumann instability.

## Coherence

Raw file:
`../../../../profile_artifacts/qwen36_fused_directsolve_20260522/qwen36_directsolve_chat_coherence_20260522T1332Z.json`

| Check | Result |
|-------|--------|
| Overall pass | PASS |
| Chat template | `enable_thinking=false` |
| Fact prompt | Coherent |
| Code prompt | Coherent |
| Prefix-cache prompt | Coherent |
| Smoke decode | ~20.5 tok/s |

The sampled outputs were non-repetitive, real text. This addresses the earlier
failure mode where the fused path could emit invalid repeated tokens.

## Decode

Raw file:
`../../../../profile_artifacts/qwen36_fused_directsolve_20260522/qwen36_directsolve_decode_bench_20260522T1348Z.json`

| Metric | Result |
|--------|--------|
| Average decode throughput | 21.63 tok/s |
| TPOT | 46.2 ms/token |
| 128-token decode average latency | 5.92s |
| Sampling | On-device greedy, `output_logits=false` |

## Cold And Warm Prefill

Raw file:
`../../../../profile_artifacts/qwen36_fused_directsolve_20260522/context_sweep_partial_20260522T1348Z.json`

| Prompt tokens | Cold TTFT | Cold prefill | Warm TTFT | Warm prefill |
|--------------:|----------:|-------------:|----------:|-------------:|
| 512 | 1.31s | 390 tok/s | 0.42s | 1.2K tok/s |
| 4,096 | 7.03s | 582 tok/s | 0.42s | 9.8K tok/s |
| 8,192 | 13.61s | 602 tok/s | 0.43s | 18.9K tok/s |
| 16,384 | 27.84s | 589 tok/s | 0.45s | 36.3K tok/s |

The 32K row did not complete because this artifact was compiled with
`prefix_buckets` only through 16K:

```text
Prefix len 16640 exceeds largest bucket 16384 for context_encoding_model
```

That is a bucket-coverage limitation of this artifact, not a fused direct-solve
correctness failure. Longer-context warm APC validation requires recompiling
with larger prefix buckets.

## Memory

Raw file:
`../../../../profile_artifacts/qwen36_fused_directsolve_20260522/qwen36_directsolve_perf_capture_20260522T1348Z.json`

| Metric | Result |
|--------|--------|
| Neuron HBM peak sum | 60.1 GiB |
| Host process RSS peak | 46.3 GiB |
| Main logical-core peaks | ~14.57 GiB on cores 0, 2, 4, and 6 |

## Observed Package Versions

These versions were read from the running validation host and the active NxDI
venv on 2026-05-22.

| Component | Version |
|-----------|---------|
| neuronx-distributed-inference | 0.9.17334+ced6ae4e |
| neuronx-distributed | 0.18.27753+1cafd54f |
| neuronx-cc | 2.24.8799.0+6f62ff7c |
| nki | 0.3.0+23928721754.g18aa1271 |
| torch | 2.9.1 |
| torch-neuronx | 2.9.0.2.13.26312+8e870898 |
| torch-xla | 2.9.0 |
| transformers | 4.57.6 |
| aws-neuronx-runtime-lib | 2.31.24.0-0b044f4ce |
| aws-neuronx-tools | 2.29.22.0-b486b0ade |
| NXDI venv | `/opt/aws_neuronx_venv_pytorch_2_9_nxd_inference/` |

## Validation Commands

Local CPU checks used for this branch:

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

Neuron validation used the compiled artifact above on trn2.3xlarge.
