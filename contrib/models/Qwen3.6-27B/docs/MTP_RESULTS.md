# Qwen3.6-27B MTP Results

Validation date: 2026-05-11

## Artifact

- Branch: `codex/qwen36-mtp-cpu-reference`
- Artifact: `/opt/dlami/nvme/qwen_artifacts/qwen36_27b_128k_fp8_mtp_run2`
- Model weights: `/opt/dlami/nvme/models/Qwen3.6-27B`
- Target precision: MLP-only FP8 target model, BF16 MTP draft model
- Context: 131072 tokens
- Context encoding bucket: 512
- Speculation length: 2
- Artifact size: 40 GB

## Compile And Load

- Context trace shape confirmed: `[1, 512]`
- Fused speculation trace shape confirmed: `[1, 1]`
- Fused-spec NEFF compile: pass
- Context NEFF compile: pass
- Presharded checkpoint save/load: pass
- Load-after-compile: pass

The draft model must remain BF16. Quantizing the draft tried to load missing
`draft_model.mtp.*.scale` tensors from the MLP-only FP8 checkpoint.

## Quality

The quality smoke used the compiled MTP artifact through the NxDI adapter.

- `chat_math_no_thinking`: generated `3151`
- `chat_math_thinking`: produced a coherent reasoning trace
- Raw prompt path: answered `3151`

This validates that the earlier fused-spec cache-state bug that produced
repetitive bad output is fixed.

## OpenAI-Compatible Server

Server command used the compiled MTP artifact with prefix caching disabled for
this isolated MTP validation:

- Port: `8000`
- Model id: `qwen3.6-27b-128k-fp8-mtp`
- `--seq-len 131072`
- `--chunk-size 512`
- `--prefix-cache-entries 0`

### Smoke: 32-token prompt, 128-token limit

- Prompt tokens: 32
- Completion tokens: 78
- Prefill: 1.191 s, 26.9 tok/s effective
- Decode: 1.852 s, 41.6 tok/s
- Output: coherent 2020 Olympics summary

The prefill rate is low here because the prompt fills only a small fraction of
one 512-token chunk.

### Chunked Prefill: 3959-token prompt, 128-token limit

- Prompt tokens: 3959
- Completion tokens: 128
- Chunks: 8
- Prefill: 9.556 s, 414.3 tok/s
- Decode: 2.808 s, 45.2 tok/s
- Output: coherent summary and answer

### Longer Decode: 28-token prompt, 256-token limit

- Prompt tokens: 28
- Completion tokens: 256
- Prefill: 1.189 s, 23.6 tok/s effective
- Decode: 5.752 s, 44.3 tok/s
- Output: coherent technical explanation of speculative decoding

## Summary

Compared with the previous 128K FP8 baseline, the MTP path preserves long-context
chunked prefill throughput and improves server decode from roughly 27 tok/s to
about 44-45 tok/s in the OpenAI-compatible path.

The observed decode gain is about 1.6x. It is below the ideal 2-2.5x NVIDIA
number, likely because this artifact uses speculation length 2 and the current
Python server still performs one host loop per accepted fused-spec step.

Raw logs are saved locally under:

`local_runs/qwen36_mtp_20260511/`
