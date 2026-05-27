# Agent Operating Notes

## Error Logging Requirement

For every error encountered while working in this repository, log it in the
conversation or the relevant project note before moving on. Include:

- What failed: command, compile, test, runtime path, or automation.
- How it failed: exact error text, exit code, failing bucket/shape, PID/log path,
  or other concrete evidence.
- How we got there: inputs, flags, artifact, branch, host, and relevant runtime
  context.
- Root cause or best current hypothesis, clearly marked if not proven.
- Fix or mitigation applied, including commands/files changed.
- Verification result after the fix, or the remaining blocker if not fixed.

Do not summarize an error as only "failed" or "compiler issue". Keep enough
detail that a later agent can reconstruct the failure and avoid repeating it.

## Qwen3.6 FP8 Performance Measurement Notes

Do not compare decode TPOT from streamed content chunks with token-level TPOT.
For OpenAI/vLLM streaming benchmarks, request
`stream_options: {"include_usage": true}` and compute TPOT from
`usage.completion_tokens`. If usage is unavailable, explicitly label any
tokenizer-derived fallback, and keep content-chunk timing separate as
`content_chunk_tpot_seconds`.

Do not call an exact repeated prompt "warm refill". Repeating the identical
prompt measures a full-prompt cache hit. For Hybrid APC/GDN refill, warm a
shared prefix with one suffix, then measure the same prefix with a different
suffix, and record the shared prefix length, suffix length, artifact, CTE
buckets, prefix buckets, and whether backed prefix reads were enabled.

The 2026-05-26 correction that exposed this:

- Wrong comparison: `qwen36_chat_completion_context_bench.py` reported
  `tpot_seconds` from `content_chunk_count`, which made a 64-token 16k run look
  like roughly 80-110 ms/chunk instead of token TPOT.
- Fix: request streaming usage and compute token TPOT from completion tokens.
  The corrected 16k pfx16k FP8 run on TRN2 measured about 50-52 ms/token and
  19-20 decode tok/s.
- Wrong warm interpretation: `qwen36_hybrid_apc_context_sweep.py` generated the
  exact same prompt twice, so sub-second "warm" could be mistaken for refill.
- Fix: default warm mode now uses partial refill: shared prefix + suffix A for
  warmup, then shared prefix + suffix B for measurement. The corrected 16k
  partial refill used a 16,368-token shared prefix and measured about 0.91s.
