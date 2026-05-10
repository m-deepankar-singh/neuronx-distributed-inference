# Qwen3.6-27B 128K FP8 CTE=512 Evaluation Report

## Configuration
- model: `qwen3.6-27b-neuron-128k-fp8-mlp`
- artifact: `/opt/dlami/nvme/qwen_artifacts/qwen36_27b_128k_fp8_mlp_only_run1`
- seq_len: `131072`
- cte_chunk: `512`
- api: `minimal OpenAI-compatible non-streaming server`
- server health: `{'status': 'ok', 'model': 'qwen3.6-27b-neuron-128k-fp8-mlp'}`
- raw throughput JSON: `validation_scripts/qwen36_27b_128k_fp8_cte512_eval_20260509T210633Z.json`
- raw chat-quality JSON: `validation_scripts/qwen36_27b_chat_quality_eval_20260509T213734Z.json`

## Throughput

| target prompt | measured prompt | TTFT proxy s | prefill tok/s | decode tok/s | TPOT ms | request latency s | total goodput tok/s | completion goodput tok/s |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 512 | 495 | 1.25 | 396.92 | 26.87 | 37.21 | 6.01 | 103.82 | 21.462 |
| 2048 | 1976 | 4.81 | 410.97 | 26.88 | 37.20 | 9.57 | 219.97 | 13.480 |
| 6000 | 5787 | 14.34 | 403.69 | 26.88 | 37.21 | 19.10 | 309.77 | 6.755 |
| 16000 | 15429 | 36.96 | 417.48 | 26.80 | 37.31 | 39.35 | 393.79 | 1.652 |
| 32000 | 30858 | 72.68 | 424.60 | 26.86 | 37.23 | 75.06 | 411.99 | 0.866 |
| 64000 | 61715 | 144.11 | 428.25 | 26.50 | 37.74 | 145.32 | 424.92 | 0.227 |
| 120000 | 115715 | 270.57 | 427.67 | 22.64 | 44.18 | 271.28 | 426.62 | 0.063 |

- prefill range: 396.92-428.25 tok/s
- decode range: 22.64-26.88 tok/s
- decode median: 26.86 tok/s
- TTFT proxy is measured as non-streaming `max_tokens=1` request latency; this includes prefill plus one generated token.
- TPOT is paired: `(latency(max_tokens=N)-latency(max_tokens=1))/(N-1)`.

## Memory And Utilization
- peak Neuron device memory: 53.25 GB decimal (49.59 GiB)
- peak host memory reported by neuron-monitor: 3.66 GB
- peak NeuronCore utilization: 100.0%
- neuron-monitor samples: 294

## Completion/Chat Quality

Raw `/v1/completions` instruction checks failed: the model repeated labels/filler rather than answering. `/v1/chat/completions` also failed short instruction checks, mostly producing repeated `</think>` tokens. This is a model/serving quality issue, not a throughput failure.

| endpoint | case | prompt tokens | completion tokens | latency s | contains expected? | output prefix |
|---|---|---:|---:|---:|---|---|
| completions | arithmetic_exact | 22 | 32 | 2.40 | False | `\nAnswer:\nAnswer:\nAnswer:\nAnswer:\nAnswer:\nAnswer:\nAnswer:\nAnswer:\nAnswer:\nAnswer:\nAnswer` |
| completions | factual_olympics_2020 | 24 | 96 | 4.78 | False | `\nAnswer:\nAnswer:\nAnswer:\nAnswer:\nAnswer:\nAnswer:\nAnswer:\nAnswer:\nAnswer:\nAnswer:\nAnswer:\nAnswer:\nAnswer:\nAnswer:\nAnswer:\nAns` |
| completions | instruction_json | 28 | 80 | 4.20 | False | `\nJSON:\nJSON:\nJSON:\nJSON:\nJSON:\nJSON:\nJSON:\nJSON:\nJSON:\nJSON:\nJSON:\nJSON:\nJSON:\nJSON:\nJSON:\nJSON:\nJSON:\nJSON:\nJSON:\nJSON:` |
| completions | needle_32k | 30402 | 80 | 74.48 | False | ` The archive discusses logistics, sports history, character motivation, mission planning, and ordinary background details. The archive discu` |
| completions | needle_120k | 114002 | 80 | 268.90 | False | ` The archive discusses logistics, sports history, character motivation, mission planning, and ordinary background details. The archive discu` |
| chat | math | 41 | 32 | 2.40 | False | `</think>\n\n</think>\n\n</think>\n\n</think>\n\n</think>\n\n</think>\n\n</think>\n\n</think>\n\n</think>\n\n</think>\n\n</think>\n\n</think>` |
| chat | olympics | 43 | 96 | 4.78 | False | `</think>\n\n</think>\n\n</think>\n\n</think>\n\n</think>\n\n</think>\n\n</think>\n\n</think>\n\n</think>\n\n</think>\n\n</think>\n\n</think>` |
| chat | json | 37 | 80 | 4.18 | False | `</think>\n\n</think>\n\n</think>\n\n</think>\n\n</think>\n\n</think>\n\n</think>\n\n</think>\n\n</think>\n\n</think>\n\n</think>\n\n</think>` |
| chat | mgs_summary | 30 | 80 | 4.18 | False | `</think>\n\n</think>\n\n</think>\n\n</think>\n\n</think>\n\n</think>\n\n</think>\n\n</think>\n\n</think>\n\n</think>\n\n</think>\n\n</think>` |
| chat | refusal_safe | 35 | 96 | 4.77 | False | `</think>\n\n</think>\n\n</think>\n\n</think>\n\n</think>\n\n</think>\n\n</think>\n\n</think>\n\n</think>\n\n</think>\n\n</think>\n\n</think>` |

## Findings

- Runtime stability: PASS. No HTTP/runtime errors in the full eval; 120K prompt completed.
- Throughput: PASS. Prefill is stable around ~400-428 tok/s from 512 to 120K prompt tokens; decode is ~26.8 tok/s through 64K and ~22.6 tok/s near 120K.
- Memory: PASS for this single-stream setup. Peak device memory was ~53.25 GB decimal on trn2.3xlarge.
- Goodput: for long-prefill requests, total-token goodput is dominated by prefill and stays ~400+ tok/s; completion-token goodput falls with long prompts because TTFT dominates non-streaming latency.
- Intelligence/quality: FAIL in this server/artifact configuration. Short instruction and needle prompts do not answer correctly and often repeat prompt labels, filler, or `</think>`. This needs separate investigation before treating the server as production-quality for user-facing QA/chat.

