# Qwen3.6-27B vLLM Prefix Caching Recheck

Validation date: 2026-05-11

## Scope

This recheck validates the existing vLLM/APC baseline path, not the new MTP
standalone server path.

- Branch/tag on remote: `qwen36-27b-vllm-apc-baseline-v3`
- Artifact: `/opt/dlami/nvme/qwen_artifacts/qwen36_27b_128k_fp8_mlp_only_vllm_statereset_run1`
- Backend: vLLM on `127.0.0.1:8001`
- Proxy: guarded OpenAI-compatible chat proxy on `:8000`
- Flags: `--enable-vllm-chunked-prefill --enable-prefix-caching --mamba-cache-mode align`
- Context length: 131072
- CTE bucket: 512
- Block size: 256

The current native MTP artifact is validated through the standalone NxDI server.
MTP plus vLLM APC is a separate integration step and was not claimed by this
test.

## Setup Notes

The fresh/switch-recovered vLLM environment initially failed with:

`Model Qwen3_5ForConditionalGeneration is not supported on Neuron for now`

Running `contrib/models/Qwen3.6-27B/vllm/install_qwen36_vllm.sh` against the
active vLLM/Neuron venv restored the contrib model registry patch. After that,
the vLLM APC backend loaded the precompiled artifact successfully.

## APC Hardening Result

The saved passing run is:

`contrib/models/Qwen3.6-27B/docs/results/vllm_apc_hardening_20260511.json`

| Case | Prompt tokens | Wall time | Result |
|---|---:|---:|---|
| Exact repeat cold | 10601 | 26.58s | `TOKYO_2020_OK` |
| Exact repeat warm | 10601 | 3.07s | exact output match |
| Cross-prefix A cold | 10376 | 26.60s | `CROSS_A_OK` |
| Cross-prefix B cold | 10376 | 26.55s | `CROSS_B_OK` |
| Cross-prefix A warm after B | 10376 | 2.66s | exact output match |

Exact-repeat speedup: **8.66x**.

Cross-prefix reuse stayed isolated and output-stable.

## Shared-Prefix Concurrency

Requests are still effectively queued because the artifact is compiled with
`max_num_seqs=1`, but APC remains correct under concurrent arrival.

| Concurrency | Success | Total wall | Aggregate prompt tok/s |
|---:|---:|---:|---:|
| 1 | 1/1 | 2.87s | 3620.5 |
| 2 | 2/2 | 4.37s | 4748.4 |
| 4 | 4/4 | 8.84s | 4695.4 |

## Longer Prefix Spot Check

The saved long exact-repeat run is:

`contrib/models/Qwen3.6-27B/docs/results/vllm_apc_long_exact_repeat_20260511.json`

| Prompt tokens | Cold | Warm | Speedup | Result |
|---:|---:|---:|---:|---|
| 25997 | 62.23s | 2.92s | 21.28x | `LONG_APC_OK` both runs |

## Verdict

vLLM prefix caching with `mamba-cache-mode align` is working on the existing
baseline-v3 vLLM artifact:

- exact-repeat APC gives a clear warm-hit speedup;
- cross-prefix reuse is isolated;
- concurrent shared-prefix requests return the expected markers;
- a longer 26K-token prefix shows a 21x cold-to-warm latency reduction.

This remains the correct production path for APC today. Combining APC with the
new native MTP fused-spec artifact is not validated yet and should be treated as
future work.
