# Qwen3.6 Decode Runtime Fastpath Check

Artifact:

`/mnt/trainium_artifacts/qwen_artifacts/qwen36_27b_128k_fp8_mlp_edgebf16_hybrid_apc_nki_chunked_b256_cte256_512_pfx16k_slots64_20260521T092332Z`

Test host: Trn2 `16.51.6.234`

This run applied the runtime-only Hybrid APC decode shortcut in
`async_execution.py` to the existing precompiled artifact, then reran the
offline vLLM/NxDI decode benchmark with the full 128K runner shape.

Result:

- Average decode throughput stayed at `6.20 tok/s`.
- The artifact still reports `token_generation_buckets=[131072]`,
  `tkg_batch_size=1`, `async_mode=false`, `output_logits=false`, and on-device
  greedy sampling enabled.

Conclusion:

The 6 tok/s bottleneck is not Python-side Hybrid APC restore/commit argument
construction during decode. The remaining gap to the older 26-27 tok/s control
is in artifact/runtime shape: single full-length TKG bucket, sync Neuron runtime
decode, and single-sequence TKG batch.
