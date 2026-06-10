# Qwen3.6 4K Hybrid APC Perf Results

Captured on Trn2 target `16.26.82.174` on 2026-05-21 using the 4K BF16 Hybrid APC NKI chunked artifact:

`/mnt/trainium_artifacts/qwen_artifacts/qwen36_27b_4096_bf16_hybrid_apc_nki_chunked_seqid_stale_evict_prefix4096_ctx2_tkg2_r7i_20260521T052205Z`

## Cold TTFT

Unique prompts, streaming, `max_tokens=1`.

| Prompt tokens | TTFT avg (s) | Prompt tok/s avg |
| ---: | ---: | ---: |
| 256 | 0.893 | 289 |
| 503 | 1.552 | 324 |
| 1024 | 2.870 | 357 |
| 2038 | 5.640 | 361 |
| 3572 | 11.427 | 313 |

## Warm APC Boundary Partial TTFT

Each case warms an exact checkpoint-boundary prefix, then measures `prefix + 16 suffix tokens`.

| Boundary | Cold partial TTFT (s) | Warm APC TTFT (s) | Speedup |
| ---: | ---: | ---: | ---: |
| 256 | 1.534 | 0.938 | 1.64x |
| 512 | 2.821 | 0.833 | 3.39x |
| 1024 | 5.578 | 0.836 | 6.68x |
| 2048 | 11.344 | 0.837 | 13.55x |

## Decode TPOT

Boundary partial prompts with `max_tokens=32`.

| Boundary | Cold TPOT (ms/token) | Warm APC TPOT (ms/token) |
| ---: | ---: | ---: |
| 256 | 64.6 | 62.3 |
| 512 | 62.5 | 62.5 |
| 1024 | 62.4 | 62.4 |
| 2048 | 62.3 | 62.6 |

## Concurrency 2 Cold Prefill

Unique prompts, streaming, `max_tokens=1`, two concurrent requests.

| Target tokens | Group prompt tok/s avg | TTFT p50 (s) |
| ---: | ---: | ---: |
| 512 | 485 | 1.58 |
| 1024 | 723 | 2.83 |
| 2048 | 728 | 5.60 |

## Files

- `qwen36_4k_cold_unique_ttft_prefill.json`
- `qwen36_4k_warm_repeated_ttft_prefill.json`
- `qwen36_4k_apc_boundary_ttft_tpot.json`
- `qwen36_4k_cold_unique_concurrency2_ttft_prefill.json`
