# Qwen3.6 4K Hybrid APC Perf + Memory Rerun

Captured on Trn2 target `16.26.82.174` on 2026-05-21 using:

`/mnt/trainium_artifacts/qwen_artifacts/qwen36_27b_4096_bf16_hybrid_apc_nki_chunked_seqid_stale_evict_prefix4096_ctx2_tkg2_r7i_20260521T052205Z`

## Memory

`neuron-monitor` ran for the full server load and benchmark window.

| Metric | Peak |
| --- | ---: |
| Runtime Neuron device HBM | 81.10 GB / 75.53 GiB |
| Trn2 device HBM capacity | 103.08 GB / 96.00 GiB |
| Device HBM utilization | 78.7% |
| Neuron runtime host allocation | 65.36 GB / 60.87 GiB |
| Linux host memory used | 18.96 GB / 17.66 GiB |
| Server RSS | 1.42 GB / 1.32 GiB |

## Cold TTFT

Unique chat prompts, streaming, `max_tokens=1`, concurrency 1.

| Prompt tokens | TTFT avg (s) | Prompt tok/s avg |
| ---: | ---: | ---: |
| 256 | 0.895 | 288 |
| 503 | 1.555 | 324 |
| 1024 | 2.874 | 357 |
| 2038 | 5.648 | 361 |
| 3572 | 11.432 | 312 |

## Decode TPOT

Boundary partial raw completions with `max_tokens=32`.

| Boundary | Cold TPOT (ms/token) | Warm APC TPOT (ms/token) |
| ---: | ---: | ---: |
| 256 | 64.5 | 62.4 |
| 512 | 62.5 | 62.4 |
| 1024 | 62.5 | 62.5 |
| 2048 | 62.1 | 62.4 |

## Warm APC Boundary Partial TTFT

Each case warms an exact checkpoint-boundary prefix, then measures `prefix + 16 suffix tokens`.

| Boundary | Cold partial TTFT (s) | Warm APC TTFT (s) | Speedup |
| ---: | ---: | ---: | ---: |
| 256 | 1.510 | 0.830 | 1.82x |
| 512 | 2.819 | 0.829 | 3.40x |
| 1024 | 5.581 | 0.830 | 6.72x |
| 2048 | 11.350 | 0.832 | 13.65x |

## Concurrency 2 Cold Prefill

Unique chat prompts, streaming, `max_tokens=1`, two concurrent requests.

| Target tokens | Group prompt tok/s avg | TTFT p50 (s) |
| ---: | ---: | ---: |
| 512 | 545 | 1.57 |
| 1024 | 483 | 2.83 |
| 2048 | 728 | 5.60 |

## Files

- `memory_summary.json`
- `neuron_monitor.jsonl`
- `host_memory.csv`
- `events.jsonl`
- `qwen36_4k_cold_unique_ttft_prefill.json`
- `qwen36_4k_cold_unique_concurrency2_ttft_prefill.json`
- `qwen36_4k_apc_boundary_ttft_tpot.json`
- `server.log`
