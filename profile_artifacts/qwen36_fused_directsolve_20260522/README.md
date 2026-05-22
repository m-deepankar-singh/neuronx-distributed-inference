# Qwen3.6 Fused Direct-Solve Validation

Validation target:

- Branch commit: `ae2613d` source, replayed onto PR 164 clean branch
- Artifact: `qwen36_27b_128k_fp8_mlp_edgebf16_hybrid_apc_nki_fusedstable_directsolve_retry_b256_cte256_512_pfx16k_slots64_tkg8192_32768_131072_async_20260522T130050Z`
- Runtime host: `trn2.3xlarge`
- Runtime path: offline vLLM/NxDI, on-device greedy sampling

## Summary

The fused DeltaNet CTE path now uses a direct triangular RHS solve instead of Neumann power-doubling. This fixes the fused-kernel instability observed with realistic Qwen gates while keeping the fused CTE path available for validation.

## Coherence

`qwen36_directsolve_chat_coherence_20260522T1332Z.json`

- Overall pass: `true`
- Chat template used `enable_thinking=false`
- Fact, code, and prefix-cache prompts produced non-repetitive real text
- Smoke decode throughput: about `20.5 tok/s`

## Decode

`qwen36_directsolve_decode_bench_20260522T1348Z.json`

- Average decode throughput: `21.63 tok/s`
- TPOT: `46.2 ms/token`
- 128-token decode average latency: `5.92 s`
- Artifact uses on-device greedy sampling with `output_logits=false`

## Cold And Warm Prefill

`context_sweep_partial_20260522T1348Z.json`

| Prompt tokens | Cold TTFT | Cold prefill | Warm TTFT | Warm prefill |
| ---: | ---: | ---: | ---: | ---: |
| 512 | 1.31 s | 390 tok/s | 0.42 s | 1.2k tok/s |
| 4096 | 7.03 s | 582 tok/s | 0.42 s | 9.8k tok/s |
| 8192 | 13.61 s | 602 tok/s | 0.43 s | 18.9k tok/s |
| 16384 | 27.84 s | 589 tok/s | 0.45 s | 36.3k tok/s |

The 32K row did not complete because this artifact was compiled with `prefix_buckets` only through `16384`:

```text
Prefix len 16640 exceeds largest bucket 16384 for context_encoding_model
```

That is an artifact bucket coverage limitation, not a direct-solve correctness failure.

## Memory

`qwen36_directsolve_perf_capture_20260522T1348Z.json`

- Neuron HBM peak sum: `60.1 GiB`
- Host process RSS peak: `46.3 GiB`
- Main logical cores peaked around `14.57 GiB` each on cores `0`, `2`, `4`, and `6`

## Follow-Up

For long-context validation, recompile the same branch with prefix buckets beyond `16384`, ideally through the intended 64K/128K validation range.
