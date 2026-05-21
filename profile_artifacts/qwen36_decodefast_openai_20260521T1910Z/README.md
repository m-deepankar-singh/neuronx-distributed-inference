# Qwen3.6 Decode-Fast OpenAI Server Probe

Artifact:

`/mnt/trainium_artifacts/qwen_artifacts/qwen36_27b_128k_fp8_mlp_edgebf16_hybrid_apc_nki_chunked_b256_cte256_512_pfx16k_slots64_tkg8192_32768_131072_async_20260521T181759Z`

Server:

- host: `0.0.0.0`
- port: `8000`
- base URL: `http://16.51.6.234:8000/v1`
- max model length: `65536`
- PA blocks override: `512`
- CTE buckets: `[256,512]`
- TKG buckets: `[8192,32768,131072]`

## Results

`openai_decode_probe.json` records a warmup, non-streaming decode, and streaming decode through `/v1/chat/completions`.

- warmup: 2 completion tokens in 0.850 s
- non-streaming: 32 completion tokens in 1.769 s, 18.09 tok/s
- streaming: 32 completion tokens in 1.772 s, 18.06 tok/s
- streaming TTFT: 401 ms to first content
- streaming TPOT excluding first content: 44.2 ms/token

The OpenAI server path matches the offline decode smoke result (~17.93 tok/s), so this artifact no longer shows the prior ~6.2 tok/s OpenAI/vLLM decode bottleneck.

The server log still reports that vLLM async scheduling is disabled with prefix caching for Mamba models, even though the compiled Neuron artifact has `async_mode=true`. This means the gain is coming from the compiled TKG bucket/control changes rather than vLLM async scheduling.

The initial server launch failed because `start_vllm_server.sh` did not include the repo `src/` directory on `PYTHONPATH`, causing the child process to miss the Qwen3.6 NxDI registry patch. Commit `d8cb825` fixes that wrapper path before this successful run.
