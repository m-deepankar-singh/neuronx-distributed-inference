# Qwen3.6 27B Decode-Fast Artifact Smoke Result

Artifact:

`/mnt/trainium_artifacts/qwen_artifacts/qwen36_27b_128k_fp8_mlp_edgebf16_hybrid_apc_nki_chunked_b256_cte256_512_pfx16k_slots64_tkg8192_32768_131072_async_20260521T181759Z`

This is the 128K FP8 MLP-only Hybrid APC artifact compiled from `c4c1662` with decode-performance controls restored:

- `token_generation_buckets=[8192,32768,131072]`
- `async_mode=true` in the compiled artifact
- `tkg_batch_size=1`
- `pa_num_blocks=512`
- on-device greedy sampling enabled
- Qwen Hybrid APC NKI chunked prefill enabled

The artifact was copied to Trn2 `16.51.6.234` and `LATEST_DECODEFAST_128K` points to it.

## Offline Decode Smoke

Command output is in `offline_decode_32.json`.

- max model length: 65,536
- prompt: `Explain software benchmarking in two concise paragraphs.`
- measured completion: 32 tokens
- elapsed: 1.785 s
- throughput: 17.93 tok/s
- warmup: 2 tokens in 0.825 s

This is a material improvement over the previous 128K FP8 artifact at roughly 6.2 tok/s, but still below the older vLLM APC PR branch target around 27 tok/s.

The generated text was real model text and did not show the prior repeated-exclamation failure pattern, but this is only a short smoke test rather than a full exactness/coherence gate.
