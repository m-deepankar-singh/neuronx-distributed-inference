# Qwen3.6-27B vLLM APC Baseline v3

## Baseline Identity

- Tag: `qwen36-27b-vllm-apc-baseline-v3`
- Source branch: `codex/qwen36-vllm-prefix-cache`
- Source commit before this card: `410076d`
- Artifact: `/opt/dlami/nvme/qwen_artifacts/qwen36_27b_128k_fp8_mlp_only_vllm_statereset_run1`
- Model path: `/opt/dlami/nvme/models/Qwen3.6-27B`
- Serving layout: vLLM backend on `127.0.0.1:8001`, guarded OpenAI-compatible proxy on `:8000`

## Runtime Configuration

- Max context: `131072`
- CTE bucket: `512`
- Block size: `256`
- Tensor parallel degree: `4`
- vLLM chunked prefill: enabled
- vLLM prefix caching: enabled
- Mamba/GDN cache mode: `align`
- Proxy behavior: chat endpoint only, raw completions disabled, thinking disabled by default

## Measured Performance

| Path | Result |
|---|---:|
| Cold prefill | `404-428 tok/s` |
| Decode | `26.3-26.6 tok/s` |
| Exact-repeat APC | `25.38s -> 1.55s`, `16.35x` |
| Partial-prefix APC | `25.52s -> 1.70s`, `15.00x` |
| Cross-prefix APC | `25.17s -> 1.36s`, exact text match |

## Validation Passed

- Exact-repeat APC: exact output match.
- Offline exact-repeat APC: exact token-ID match.
- Offline partial-prefix APC: exact token-ID match.
- Cross-prefix reuse after unrelated prefix: exact output match.
- Shared-prefix concurrency at `1`, `2`, and `4`: all markers returned correctly.
- Long-context quality:
  - short math: passed;
  - Olympics factual prompt: passed;
  - 32K needle: passed;
  - 64K needle: passed.

## Known Caveat

The artifact is compiled with `max_num_seqs=1`, so concurrent requests are
correct but queued. This baseline is production-usable for single active
sequence plus APC acceleration, not yet throughput-scaled continuous batching.

## Source Results

- `validation_scripts/qwen36_vllm_apc_hardening_report.md`
- `validation_scripts/qwen36_vllm_apc_hardening_eval_20260510T101345Z.json`
- `validation_scripts/qwen36_27b_vllm_long_chat_eval_20260510T101703Z.json`

## Restore

- Runbook: `validation_scripts/qwen36_vllm_apc_baseline_v3_restore_runbook.md`
- One-command launcher: `contrib/models/Qwen3.6-27B/vllm/start_baseline_v3.sh`
