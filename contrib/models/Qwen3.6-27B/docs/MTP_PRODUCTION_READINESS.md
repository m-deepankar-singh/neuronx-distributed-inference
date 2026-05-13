# Qwen3.6-27B MTP Production Readiness

Validation date: 2026-05-11

## Target

- Branch: `codex/qwen36-mtp-cpu-reference`
- Artifact: `/opt/dlami/nvme/qwen_artifacts/qwen36_27b_128k_fp8_mtp_run2`
- Model id: `qwen3.6-27b-128k-fp8-mtp`
- Server: `validation_scripts/qwen36_27b_openai_server.py`
- Context length: 131072
- CTE bucket: 512
- Speculation length: 2
- Prefix cache: disabled for this MTP isolation pass

## API Hardening

| Test | Result |
|---|---|
| `GET /health` | pass |
| `GET /v1/models` | pass |
| System-first chat completion | pass |
| Misplaced system message | pass, returns HTTP 400 `BadRequestError` |
| Stop string handling | pass |
| Stop string removed from final response | pass |
| Streaming chat completion | pass, emits chunks and `[DONE]` |
| Deterministic greedy repeat x3 | pass |
| 3 concurrent requests | pass, serialized correctly |

The server intentionally serializes requests because one NxDI model instance
owns one mutable KV/GDN cache. This is correct for the current single-worker
debug/production harness; higher concurrency should be handled by vLLM or
multiple worker processes.

## Long-Context Results

| Prompt tokens | Chunks | Prefill seconds | Prefill tok/s | Decode tok/s | Result |
|---:|---:|---:|---:|---:|---|
| 4664 | 10 | 11.937 | 390.7 | 48.3 | pass |
| 18587 | 37 | 44.181 | 420.7 | 48.2 | pass |
| 74279 | 146 | 174.389 | 425.9 | 48.4 | pass |

The 74K-token validation proves the 128K artifact is exercising multi-chunk
prefill well beyond demo-sized prompts.

## Decode Results

Prior MTP smoke tests through the OpenAI-compatible server measured:

| Prompt tokens | Completion tokens | Decode tok/s |
|---:|---:|---:|
| 32 | 78 | 41.6 |
| 3959 | 128 | 45.2 |
| 28 | 256 | 44.3 |

The production-readiness run measured short-tail decode around 48 tok/s for
32-token completions. The stable expected range for this server path is
approximately 44-48 tok/s.

## Memory Snapshot

`neuron-ls` confirms the artifact is loaded on one `trn2.3xlarge` with
logical NeuronCore config 2.

`neuron-monitor` idle snapshot after validation:

- Total Neuron device memory used: 59.45 GB decimal / 55.36 GiB
- Host runtime memory used: 4.04 GB decimal / 3.76 GiB
- Per logical NeuronCore tensor memory: about 12.96 GB decimal
- Runtime error counters: all zero
- ECC counters: all zero

## Verdict

This is ready to promote to the next baseline for the standalone
OpenAI-compatible NxDI server:

- Long-context chunked prefill is stable through at least 74K prompt tokens.
- Decode improves from the previous ~27 tok/s baseline to ~44-48 tok/s with
  native Qwen MTP.
- API behavior is sane for health, models, normal chat, streaming, stop
  strings, deterministic greedy generation, concurrent serialized access, and
  malformed system-message order.

Raw JSON and logs are saved locally under:

`local_runs/qwen36_mtp_20260511/`
