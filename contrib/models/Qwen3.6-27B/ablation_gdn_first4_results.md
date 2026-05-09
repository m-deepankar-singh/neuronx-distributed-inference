# Qwen3.6-27B GDN First-4 Stress Ablation

Branch: `codex/qwen36-27b-gdn-ablation`  
Commit: `a6d0d24`  
Baseline tag: `qwen36-27b-baseline-v1`  
Instance: `trn2.3xlarge`, TP=4, CTE bucket=512, seq_len=65536

## Setup

The ablation adds `QWEN_GDN_STRESS_MODE=first4`, which duplicates the DeltaNet
NKI kernel call for the first four linear-attention layers and averages the two
identical outputs/states. With the env var unset, baseline behavior is unchanged.

Artifact:

`/opt/dlami/nvme/qwen_artifacts/qwen36_27b_gdnstress_first4_cte512_tkg65536_run1`

Compile/load result:

- Compile succeeded.
- CTE HLO traced at `torch.Size([1, 512])`.
- Artifact size: 82 GB.
- Load-after-compile succeeded.

## Measurements

| Prompt tokens | Artifact | CTE seconds | Avg chunk s | Ingest tok/s | Decode tok/s | Generated tokens |
|---:|---|---:|---:|---:|---:|---|
| 1024 | baseline | 6.843246 | 3.421623 | 149.64 | 18.10 | `[271, 248068, 271, 248069, 271]` |
| 1024 | first4 stress | 6.870492 | 3.435246 | 149.04 | 18.15 | `[271, 248068, 271, 248069, 271]` |
| 762 | baseline | 6.861703 | 3.430852 | 111.05 | 18.05 | `[271, 248068, 271, 248069, 271]` |
| 762 | first4 stress | 6.888741 | 3.444371 | 110.62 | 18.05 | `[271, 248068, 271, 248069, 271]` |

## Decision

First-four stress adds about `0.0135s` per 512-token chunk. If this scales
linearly across all 48 DeltaNet layers, the full DeltaNet chunk-kernel envelope
is about `0.16s` per chunk:

`0.0135s * (48 / 4) = 0.162s`

Against a `3.42s` baseline chunk, this is roughly `4.7%` of the total CTE
time. That is far below the 30% gate for head-batched DeltaNet work.

Verdict: **do not proceed with head-batched GDN as the next optimization**.
The measured first-order impact is too small to justify a 1-2 week kernel
rewrite for this artifact.

Recommended next candidates:

- FP8 weight/MLP quantization, if accuracy gates are acceptable.
- vLLM integration work, because production scheduling/APC remains valuable.
- Deeper source-level profiling only if a better Neuron trace can attribute the
  remaining ~3.25s/chunk device execution envelope.
