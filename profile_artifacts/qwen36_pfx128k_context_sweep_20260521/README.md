# Qwen3.6 pfx128k Context Sweep

Artifact tested on Trn2 `16.51.6.234`:

`/mnt/trainium_artifacts/qwen_artifacts/LATEST_FP8_128K_PFX128K`

Resolved artifact:

`qwen36_27b_128k_fp8_mlp_edgebf16_hybrid_apc_nki_chunked_b256_cte256_512_pfx128k_slots64_20260521T160116Z`

The artifact config includes `prefix_buckets` through `131072`, `pa_num_blocks=512`,
`ctx_batch_size=1`, `tkg_batch_size=1`, `output_logits=false`, and on-device
greedy sampling.

## Results

| Target prompt tokens | Max model len | Cold TTFT (s) | Cold prefill tok/s | Warm TTFT (s) | Warm effective tok/s | Exact repeat | Real token |
| ---: | ---: | ---: | ---: | ---: | ---: | :---: | :---: |
| 16,384 | 32,768 | 27.003 | 606.8 | 0.427 | 38,403.7 | yes | yes |
| 32,760 | 32,768 | 63.230 | 518.1 | 0.591 | 55,465.0 | yes | yes |

64K and 128K did not reach generation because vLLM rejected the pfx128k artifact
at engine initialization due to KV-cache HBM headroom:

| Attempt | Error |
| --- | --- |
| `max_model_len=65,536` | Needs `16.0 GiB` KV cache; available KV cache memory was `8.33 GiB`. Estimated max model length: `34,048`. |
| `max_model_len=131,072` | Needs `32.0 GiB` KV cache; available KV cache memory was `8.33 GiB`. Estimated max model length: `34,048`. |

## Peak Memory

| Run | Peak Neuron runtime device GB | Peak Neuron tensor GB | Peak Neuron runtime host GB | Peak Linux host GB |
| --- | ---: | ---: | ---: | ---: |
| 16K/32K pass | 94.17 | 50.94 | 54.31 | 22.30 |
| 64K failed load | 80.79 | 50.94 | 54.15 | 23.36 |
| 128K failed load | 83.42 | 50.94 | 45.40 | 19.71 |

## Interpretation

The pfx128k compile is valid enough to load and run through the 32K bucket, but
it is not usable for 64K or 128K serving on this Trn2 shape as currently built.
The limiting factor is static/runtime HBM consumed before KV allocation, not
HTTP serving and not sampling. vLLM reports only `8.33 GiB` KV-cache memory
available after loading this pfx128k artifact.

This is materially worse than the earlier pfx16k/older artifacts: the expanded
prefix bucket coverage through 128K appears to leave too little KV-cache
headroom for long-context serving.

## Files

- `context_sweep_pfx128k_32k_20260521T171439Z/context_sweep.json`
- `context_sweep_pfx128k_32k_20260521T171439Z/context_sweep.log`
- `context_sweep_pfx128k_32k_20260521T171439Z/neuron_monitor.jsonl`
- `context_sweep_pfx128k_64k_20260521T171943Z/context_sweep.log`
- `context_sweep_pfx128k_20260521T171036Z/context_sweep.log`
