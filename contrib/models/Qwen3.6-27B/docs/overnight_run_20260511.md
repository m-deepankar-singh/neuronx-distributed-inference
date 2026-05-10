# Qwen3.6-27B Overnight Run - 2026-05-11

## Objective

Validate direct-RHS DeltaNet solve changes against baseline v3. Ship only if the
required gates pass:

- cosine/logprob comparison average >= 0.999
- MMLU-Lite delta < 1 percentage point
- GSM8K subset delta < 1 percentage point
- compile/load/smoke/perf gates for the candidate artifact

## Phase 1: Existing Direct-RHS v1

Result: blocked.

- Baseline v3 was healthy on ports `8001` backend and `8000` proxy.
- A parallel TP=4 direct-RHS server cannot run on `trn2.3xlarge`; baseline v3
  already occupies all 4 NeuronCores.
- vLLM/OpenAI logprobs are unavailable for this Qwen3.6 proxy path:
  `logprobs=True` returns `list index out of range`.

Direct-RHS v1 was not marked shippable because the cosine gate was not covered.

## Phase 2: Direct-RHS Solve v2

Branch: `codex/qwen36-gdn-direct-rhs-solve-v2`  
Commit: `756e684`  
Artifact: `/opt/dlami/nvme/qwen_artifacts/qwen36_27b_128k_fp8_direct_rhs_v2_run1`

Implemented a v2 kernel that preserves the baseline-v3 broadcast-input NKI
layout and materialized triangular inverse, but applies the RHS reordering:

```text
v_new = N @ (v_beta - ((k_beta * exp_gc) @ state))
```

Checks passed:

- Local `py_compile`
- Local `test_deltanet_direct_rhs_solve.py`
- Remote `py_compile`
- Remote `test_deltanet_direct_rhs_solve.py`
- Remote `nki.simulate` vs baseline-v3 chunk kernel:
  output max abs `2.33e-10`, state max abs `9.31e-10`
- Hardware compile attempt 1
- Hardware load/warmup
- Smoke prompt through proxy: `17 * 23 -> 391`

Compile evidence:

```text
CONTEXT_TRACE_SHAPE {"context_encoding_buckets": [512], "max_context_length": 512, "seq_len": 131072}
Finished generating HLO for context_encoding_model in 14.312s
Finished generating HLO for token_generation_model in 2.430s
COMPILE_DONE
LOAD_AFTER_COMPILE_OK
```

Validation blocker:

- `/v1/chat/completions` with `logprobs=True, top_logprobs=20` returns
  `{"error":{"message":"list index out of range", ...}}`.
- Raw `/v1/completions` is disabled by the Qwen3.6 proxy, so completion
  logprob routes are not usable.
- Therefore the required cosine gate cannot be computed through the current
  serving path.

MMLU-Lite, GSM8K, and perf gates were not run because the validation order says
to stop when an earlier correctness gate is missing or fails.

## Artifact State

- Baseline v3 artifact remains preserved:
  `/opt/dlami/nvme/qwen_artifacts/qwen36_27b_128k_fp8_mlp_only_vllm_statereset_run1`
- Direct-RHS v1 artifact remains preserved:
  `/opt/dlami/nvme/qwen_artifacts/qwen36_27b_128k_fp8_direct_rhs_run2`
- Direct-RHS v2 artifact was created:
  `/opt/dlami/nvme/qwen_artifacts/qwen36_27b_128k_fp8_direct_rhs_v2_run1`
- Deleted only the known failed blocked-solve artifact to free disk before
  compile:
  `/opt/dlami/nvme/qwen_artifacts/qwen36_27b_128k_fp8_blocked_solve_run1`

## Server State

Baseline v3 was restored at the end:

- backend: `127.0.0.1:8001`
- proxy: `0.0.0.0:8000`
- smoke prompt returned `391`

## Recommended Next Action

Do not tag or ship direct-RHS v2 yet. The implementation compiles and smokes,
but the required cosine gate is not covered. The next useful step is to add a
logits-capable validation path that bypasses the OpenAI proxy logprob bug, or
compile a dedicated logits-enabled baseline/candidate pair for direct tensor
comparison.
