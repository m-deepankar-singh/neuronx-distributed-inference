# Qwen3.6 27B FP8 MLP CTE512 OpenAI API Validation

Artifact:
`/mnt/trainium_artifacts/qwen_artifacts/qwen36_27b_128k_fp8_mlp_edgebf16_hybrid_apc_nki_chunked_b256_cte256_512_pfx16k_slots64_20260521T092332Z`

Trn2 host: `16.51.6.234`

Run directory:
`/home/ubuntu/validation_logs/fp8_128k/openai_api_64k_cte512_patchedscheduler16k_20260521T154444Z`

## Runtime Fixes Applied Before Final Run

- `async_execution.py`: suffix-only Hybrid APC prep now uses the active chunk boundary for vLLM chunked prefill instead of the final full prompt length.
- `qwen36_hybrid_apc_scheduler_patch.py`: production/reject-unbacked mode now disables unbacked vLLM prefix-cache reads unless a matching GDN-backed prefix is registered.

## Supported-Range Results

The artifact is compiled with `context_encoding_buckets=[256,512]` and
`prefix_buckets=[256,512,1024,2048,4096,8192,16384]`, so the clean pass is up to
the 16K prefix bucket.

| Case | Prompt tokens | TTFT s | Prefill tok/s | Decode tok/s | TPOT s | Output |
|---|---:|---:|---:|---:|---:|---|
| semantic smoke | 34 | 1.111 | n/a | n/a | n/a | `391` |
| cold 1K | 1007 | 1.881 | 535.33 | n/a | n/a | real token |
| cold 4K | 4085 | 6.942 | 588.45 | n/a | n/a | real token |
| cold 8K | 8189 | 13.489 | 607.11 | n/a | n/a | real token |
| cold 16K | 16381 | 27.539 | 594.82 | n/a | n/a | real token |
| warm 8K initial | 8192 | 13.385 | 612.05 | n/a | n/a | real token |
| warm 8K repeat avg | 8192 | 13.379 | 612.30 | n/a | n/a | real token |
| warm 16K initial | 16384 | 27.428 | 597.34 | n/a | n/a | real token |
| warm 16K repeat avg | 16384 | 27.422 | 597.48 | n/a | n/a | real token |
| decode 8K avg, 64 tokens | 8184 | 13.378 | 611.75 | 6.31 | 0.1585 | coherent refusal-style text |

Summary:
- Cold prefill average: `581.43 tok/s`
- Warm full prefill average: `604.82 tok/s`
- Decode average: `6.31 tok/s`
- Warm repeats are real responses after the scheduler fix, but not faster because unbacked prefix reads are now safely skipped.

## Memory

- Peak runtime HBM: `63.82 GB` decimal, `59.44 GiB`
- Device HBM utilization: `61.9%`
- Peak Neuron runtime host memory: `50.39 GB`
- Peak server RSS: `1.30 GB`

## 32K/64K Limit Check

The explicit 32K/64K check fails at 32K:

`AssertionError: Prefix len 16896 exceeds largest bucket 16384 for context_encoding_model`

This artifact cannot validate 32K/64K context in the current chunked prefix-cache
mode because it was compiled with largest `prefix_bucket=16384`.

Files:
- `qwen36_128k_fp8_cte512_openai_16k_perf_patchedscheduler.json`
- `openai_perf_16k_patchedscheduler.log`
- `memory_summary_16k_patchedscheduler.json`
- `openai_64k_limit_check.log`
- `run_env.txt`
