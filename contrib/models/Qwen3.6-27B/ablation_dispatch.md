# Qwen3.6-27B DeltaNet Dispatch Ablation

## Summary

Phase A did not produce a valid timing measurement. The 2x fused-DeltaNet
stress artifact compiled, but it failed to load on `trn2.3xlarge` because the
context model could not allocate the required 1 GB shared scratchpad on multiple
NeuronCores.

Result: do not proceed to the head-batched DeltaNet rewrite from this ablation
alone. The dispatch fraction is not computable from the requested 2x-call test.

## Setup

- Branch: `chunked-prefill-deltanet-head-batched`
- Base commit: `e9118a2`
- Stress commit: `cfe1958`
- Instance: `trn2.3xlarge`
- TP degree: `4`
- Sequence length: `65536`
- CTE bucket: `512`
- Internal microchunk: `128`
- Fused GDN path: `USE_QWEN_FUSED_GDN_PREFILL=1`

Important correction: the normal 150 tok/s artifact uses
`_nki_chunked_forward`. The `_fused_chunked_forward` path is only active when
the model is compiled with `USE_QWEN_FUSED_GDN_PREFILL=1`.

## Baseline Fused-GDN Timing

Existing fused-GDN artifact:

`/opt/dlami/nvme/qwen_artifacts/qwen36_27b_hybrid_fusedgdn_cte512_micro128_tkg65536_run1`

Profile results:

| Prompt tokens | Chunks | Prefill seconds | Tok/s | Avg chunk seconds | Decode tok/s | Generated tokens |
|---:|---:|---:|---:|---:|---:|---|
| 762 | 2 | 6.860222 | 111.074 | 3.430111 | 18.21 | `[271, 248068, 271, 248069, 271]` |
| 1024 | 2 | 6.821633 | 150.111 | 3.410816 | 18.21 | `[561, 25358, 220, 17, 15]` |

This confirms the fused-GDN path is comparable to the validated baseline before
the stress modification.

## Stress Variant

Temporary modification:

- In `_fused_chunked_forward`, each per-head `_deltanet_fused_kernel` call was
  executed twice.
- The two outputs/states were averaged so the second call could not be trivially
  optimized away.
- This measures extra fused-kernel call cost, not pure dispatch overhead,
  because the second call also performs useful kernel compute.

Stress artifact:

`/home/ubuntu/qwen_artifacts/qwen36_27b_hybrid_fusedgdn_stress2x_cte512_tkg65536_run1`

Compile result:

- Compile succeeded.
- Compile time: 26.53 minutes.
- Artifact size: 82 GB.

Load result:

- Load failed.
- Runtime error: `NRT_RESOURCE in nrt_load_util`.
- Failure mode: `Failed to allocate 1.000GB (alignment: 4.000MB, usage: shared scratchpad)`.
- The failure reproduced after killing the first stuck load and retrying.

Representative memory table from the retry:

| HBM group | Total | Model code | Tensors | Shared scratchpad |
|---|---:|---:|---:|---:|
| ND 0 HBM 1 | 22.500 GB | 134.317 MB | 21.369 GB | 1.000 GB |
| ND 0 HBM 2 | 22.500 GB | 134.317 MB | 21.369 GB | 1.000 GB |
| ND 0 HBM 3 | 22.500 GB | 134.317 MB | 21.369 GB | 1.000 GB |

The load failure occurs before any prompt is executed, so there is no valid
stress chunk time `T`.

## Decision Gate

Requested formula:

`dispatch_fraction = ((T - 3.42s) / 48 / num_layers / added_dispatches) * 2304 / 3.42s`

This cannot be evaluated because the stress artifact does not load and `T` is
unavailable.

Phase A verdict: **STOP**.

Do not proceed to Phase B head-batched kernel implementation based on this
ablation. A lower-footprint measurement is needed first if we still want to
test the dispatch-overhead hypothesis.

## Recommended Follow-Up Measurement

If continuing, use a narrower stress that does not double every GDN kernel call
in the graph:

1. Stress only every 8th or every 16th GDN layer, then extrapolate.
2. Or compile a variant that stresses only the first N DeltaNet layers.
3. Or instrument Neuron trace/kernel invocation counts without changing graph
   size.

These are less direct than the requested 2x stress, but they are more likely to
load within the current `trn2.3xlarge` HBM/scratchpad budget.
