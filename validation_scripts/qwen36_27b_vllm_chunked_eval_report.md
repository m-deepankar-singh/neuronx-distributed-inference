# Qwen3.6-27B vLLM Chunked-Prefill Eval Report

## Configuration
- model: `/opt/dlami/nvme/models/Qwen3.6-27B`
- artifact: `/opt/dlami/nvme/qwen_artifacts/qwen36_27b_128k_fp8_mlp_only_run1`
- seq_len: `131072`
- cte_chunk: `512`
- server: vLLM OpenAI-compatible API with native chunked prefill enabled
- launch: `--generation-config vllm`, `--enable-chunked-prefill`, `--max-num-batched-tokens 512`, `--block-size 256`
- raw JSON: `qwen36_27b_vllm_chunked_eval_20260510T070700Z.json`
- monitor JSONL: `qwen36_27b_vllm_chunked_eval_20260510T070700Z_neuron_monitor.jsonl`

## Throughput

| target prompt | measured prompt | TTFT proxy s | prefill tok/s | decode tok/s | TPOT ms | request latency s | total goodput tok/s |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 512 | 495 | 1.22 | 406.38 | 26.59 | 37.61 | 6.03 | 103.45 |
| 2048 | 1976 | 4.77 | 413.93 | 26.50 | 37.73 | 9.60 | 219.19 |
| 6000 | 5787 | 14.31 | 404.48 | 26.47 | 37.79 | 19.14 | 309.03 |
| 16000 | 15429 | 36.95 | 417.53 | 26.36 | 37.94 | 39.38 | 393.43 |
| 32000 | 30858 | 72.72 | 424.34 | 26.40 | 37.88 | 75.14 | 411.52 |
| 64000 | 61715 | 144.25 | 427.84 | 26.31 | 38.01 | 145.47 | 424.49 |
| 120000 | 115715 | 270.88 | 427.19 | 26.60 | 37.59 | 271.48 | 426.31 |

- prefill range: `404.48-427.84 tok/s`
- prefill median-ish: `417.53 tok/s`
- decode range: `26.31-26.60 tok/s`
- decode median-ish: `26.47 tok/s`
- paired decode uses `(latency(max_tokens=N) - latency(max_tokens=1)) / (N-1)`; this matches the baseline methodology.

## Memory And Utilization

- peak Neuron device memory: `53.25 GB` decimal (`49.59 GiB`)
- peak host memory: `3.75 GB`
- peak NeuronCore utilization: `100.0%`
- neuron-monitor samples: `294`

## Quality Checks

| endpoint | case | prompt tokens | completion tokens | expected found? | output prefix |
|---|---|---:|---:|---|---|
| completions | arithmetic_exact | 22 | 32 | False | ` Answer: Answer: Answer: Answer: Answer: Answer: Answer: Answer: Answer: Answer: Answer` |
| completions | factual_olympics_2020 | 24 | 96 | False | ` Answer: Answer: Answer: Answer: Answer: Answer: Answer: Answer: Answer: Answer: Answer: Answer: Answer: Answer: Answer:` |
| completions | instruction_json | 28 | 80 | True | ` animal must be cat and count must be 3. No extra text. JSON: animal must be cat and count must be 3. No extra text. JSO` |
| completions | needle_32k | 30402 | 80 | False | ` The archive discusses logistics, sports history, character motivation, mission planning, and ordinary background detail` |
| completions | needle_120k | 114002 | 1 | False | `` |
| chat | chat_smoke | 28 | 128 | n/a | `Here's a thinking process:  1.  **Analyze the Request:**     *   **Topic:** AWS Trainium.     *   **Format:** Two bullet` |

## Findings

- Runtime stability: PASS. No API/runtime errors in the full eval, including the 120K prompt.
- Throughput: PASS. vLLM chunked-prefill serving matches the FP8 baseline when measured with paired TTFT/TPOT methodology.
- Memory: PASS for single-stream 128K serving on trn2.3xlarge; peak device memory was about 53.25 GB decimal.
- Quality: NOT YET PRODUCTION-READY. The server often repeats prompt text or emits thinking-process text, and needle/factual checks fail. This is a prompt/template/sampling/model-quality track, not a throughput regression.
