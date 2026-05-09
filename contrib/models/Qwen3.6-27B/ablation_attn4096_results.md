# Qwen3.6 27B Attention Cache-Limit Ablation Results

Branch: `codex/qwen36-27b-attn-cache-ablation`  
Commit: `08ed674 Add Qwen3.6 attention cache-limit ablation`  
Artifact: `/home/ubuntu/qwen_artifacts/qwen36_27b_hybrid_chunked_nki_cte512_attn4096_tkg65536_run1`  
Baseline artifact: `/opt/dlami/nvme/qwen_artifacts/qwen36_27b_hybrid_chunked_nki_cte512_micro128_tkg65536_run1`

## Build

- Compile succeeded.
- Compile time: 1596.57s / 26.61 min.
- CPU sanity check: max_abs_diff `2.384185791015625e-07`, pass `<1e-5`.

## Ablation Performance

| Prompt tokens | Chunks | CTE seconds | Tok/s | Avg chunk s | Decode tok/s | Generated tokens |
|---:|---:|---:|---:|---:|---:|---|
| 63 | 1 | 3.320955 | 18.97 | 3.320955 | 17.67 | `[561, 25358, 220, 17, 15]` |
| 762 | 2 | 6.586539 | 115.69 | 3.293269 | 18.21 | `[561, 25358, 220, 17, 15]` |
| 1024 | 2 | 6.588158 | 155.43 | 3.294079 | 18.21 | `[561, 25358, 220, 17, 15]` |
| 4096 | 8 | 26.352029 | 155.43 | 3.294004 | 18.21 | `[561, 25358, 220, 17, 15]` |

Baseline 762-token run:

| Prompt tokens | Chunks | CTE seconds | Tok/s | Avg chunk s | Decode tok/s | Generated tokens |
|---:|---:|---:|---:|---:|---:|---|
| 762 | 2 | 6.860178 | 111.08 | 3.430089 | 18.21 | `[271, 248068, 271, 248069, 271]` |

## Decision

The ablation does **not** pass the decision gate.

- Speedup at 762 tokens: `6.860178 / 6.586539 = 1.0415x`.
- Average chunk time improved from `3.430s` to `3.293s`, about `4.0%`.
- Exact-token match failed on the 762-token greedy comparison.

Conclusion: static attention-cache limiting is not a viable next optimization path as implemented. Do not proceed to bucketed cache-length artifacts. Revisit DeltaNet 256, MLP/weight quantization, or deeper attention kernel work only if a stronger ablation identifies a larger bottleneck.
