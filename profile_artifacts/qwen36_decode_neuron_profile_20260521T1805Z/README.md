# Qwen3.6 27B Decode Neuron Profile

This captures the short Neuron runtime-inspection run used to investigate the decode regression on the 128K FP8 MLP-only Hybrid APC artifact.

## Artifact

`/mnt/trainium_artifacts/qwen_artifacts/qwen36_27b_128k_fp8_mlp_edgebf16_hybrid_apc_nki_chunked_b256_cte256_512_pfx16k_slots64_20260521T092332Z`

Key config from `offline_decode_profiled.json`:

- `seq_len=max_length=max_context_length=131072`
- `context_encoding_buckets=[256,512]`
- `token_generation_buckets=[131072]`
- `tkg_batch_size=1`
- `pa_block_size=256`
- artifact `pa_num_blocks=512`
- profiled runner override `pa_num_blocks=256`, `max_model_len=65536`
- `async_mode=false`
- `output_logits=false`
- on-device greedy sampling enabled

The first profiler attempt at `max_model_len=131072` failed because runtime inspection reduced available KV memory; vLLM reported that the 128K allocation needed 32.0 GiB but only 17.59 GiB was available. The successful profile therefore used `max_model_len=65536` / `pa_num_blocks=256`.

## Benchmark Result

`offline_decode_profiled.json` is a profiler-overhead run, not a clean speed number:

- warmup: 2 tokens in 7.495 s
- measured: 8 tokens in 1.495 s
- measured throughput: 5.35 tok/s

The clean offline/OpenAI non-streaming/streaming measurements on the same artifact remained around 6.2 tok/s, so HTTP streaming was not the bottleneck.

## Matched NEFF Summary

`summary_692913983793260_vnc_2.json` summarizes one matched decode NEFF/NTFF pair from the runtime-inspection trace:

- `total_time`: 0.055418640075 s
- `total_active_time`: 0.054943919405 s
- `hbm_read_bytes`: 13,255,389,184
- `hbm_write_bytes`: 13,255,077,888
- `tensor_engine_active_time_percent`: 47.9%
- `dma_active_time_percent`: 92.6%
- `instance_type`: trn2.3xlarge

## Interpretation

The matched TKG graph itself is already in the 55-62 ms/token band, which explains the observed TPOT regression from the older vLLM APC PR branch at roughly 37 ms/token to this artifact around 62 ms/token. The full 128K serving wall-clock is worse, around 6.2 tok/s, which indicates additional vLLM/NxDI runner overhead on top of the device graph.

The current artifact has a single full-length TKG bucket (`131072`), `tkg_batch_size=1`, and synchronous runtime decode. The next performance artifact should be compiled with smaller/multiple TKG buckets, async decode enabled if supported, and the single-token DeltaNet conv fast path included in the graph.
