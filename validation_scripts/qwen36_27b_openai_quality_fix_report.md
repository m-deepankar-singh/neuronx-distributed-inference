# Qwen3.6-27B OpenAI Server Quality Fix

Date: 2026-05-10

## Root Cause

Two wrapper issues caused quality failures in the OpenAI-compatible server:

1. OpenAI `temperature=0` was passed directly into the NxDI on-device sampler. The Qwen3.6 artifact is traced with `do_sample=True`, so literal temperature zero divides logits by zero. The validated greedy path is `top_k=1, top_p=1.0, temperature=1.0`.
2. The server manually padded the final context chunk to 512 tokens before calling the model. The validated HF adapter passes the true final-chunk length and lets NxDI bucket-padding handle the static graph. Manual padding advanced hybrid DeltaNet state through padding positions and caused repetitive `Answer:` / `</think>` failures.

## Fix

- Normalize OpenAI greedy requests to NxDI greedy sampling:
  - user `temperature <= 0` -> sampler `temperature=1.0`, `top_k=1`, `top_p=1.0`
- Pass final partial prefill chunks at real length instead of manually padding to 512.
- Report `finish_reason="stop"` when EOS is reached instead of always returning `"length"`.

## Validation

Live server: `http://127.0.0.1:8000` on `16.51.5.28`

Artifact: `/opt/dlami/nvme/qwen_artifacts/qwen36_27b_128k_fp8_mlp_only_run1`

Probe results after fix:

- Chat math, `temperature=0`, `enable_thinking=false`: `3151`, `finish_reason=stop`
- Olympics summary: coherent Tokyo/COVID postponement answer
- JSON prompt: coherent JSON containing `cat` and `3`
- Metal Gear Solid prompt: coherent one-sentence summary
- Safe SSH key prompt: coherent safe-storage answer
- 772-token Olympics smoke with 128 output tokens: coherent, no invalid IDs, `x_latency_seconds=7.16`

Remaining note: raw `/v1/completions` prompts are not instruction-templated. They now produce sane tokens, but chat completions are the expected production path for this chat model.
