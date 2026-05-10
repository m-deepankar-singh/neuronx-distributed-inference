# Qwen3.6-27B vLLM Quality-Fix Report

## Configuration
- branch: `codex/qwen36-vllm-support`
- source fix commit: `eede8db`
- model: `/opt/dlami/nvme/models/Qwen3.6-27B`
- artifact: `/opt/dlami/nvme/qwen_artifacts/qwen36_27b_128k_fp8_mlp_only_vllm_qualityfix_run1`
- seq_len: `131072`
- CTE bucket: `512`
- vLLM: native chunked prefill, `--generation-config vllm`

## Fix
The previous vLLM artifact repeated prompt suffixes or `</think>` because padded
CTE chunks were treated as real DeltaNet/context positions. The fix makes both
padding-sensitive paths derive the real token count from `attention_mask`:

- DeltaNet padding mask for context encoding
- context logits gather for model-local chunked prefill

The fallback remains `input_ids != pad_token_id` for non-vLLM paths.

## Chat Quality
Focused `/v1/chat/completions` eval uses deterministic greedy settings plus
`chat_template_kwargs={"enable_thinking": false}`.

| case | pass | prompt tokens | completion tokens | output prefix |
|---|---:|---:|---:|---|
| math | yes | 41 | 14 | `137 * 23 = 3151` |
| olympics | yes | 43 | 46 | `The 2020 Summer Olympics were held in **Tokyo, Japan**...` |
| json | yes | 37 | 24 | ```json {"animal": "cat", "count": 3} ``` |
| mgs_summary | yes | 30 | 21 | `Metal Gear Solid is a stealth-action video game series...` |
| refusal_safe | yes | 35 | 27 | `To store an SSH private key securely...` |

Raw result: `qwen36_27b_chat_quality_eval_20260510T081217Z.json`.

## Throughput Regression

| target prompt | measured prompt | prefill tok/s | decode tok/s | latency s |
|---:|---:|---:|---:|---:|
| 512 | 495 | 411.82 | 26.56 | 6.02 |
| 2048 | 1976 | 413.72 | 26.54 | 9.60 |
| 6000 | 5787 | 404.39 | 26.50 | 19.14 |
| 16000 | 15429 | 417.50 | 26.49 | 39.37 |
| 32000 | 30858 | 424.32 | 26.36 | 75.15 |
| 64000 | 61715 | 427.79 | 26.35 | 145.48 |

- peak Neuron device memory: `53.25 GB` decimal (`49.59 GiB`)
- peak host memory: `3.60 GB`
- peak NeuronCore utilization: `100%`

Raw result: `qwen36_27b_vllm_qualityfix_eval_20260510T081237Z.json`.

## Notes
- `/v1/chat/completions` is the production path for this chat model.
- Raw `/v1/completions` prompts are not instruction-templated and still repeat
  prompt text on some instruction-style prompts. That is expected unless callers
  pass a fully rendered chat prompt.
