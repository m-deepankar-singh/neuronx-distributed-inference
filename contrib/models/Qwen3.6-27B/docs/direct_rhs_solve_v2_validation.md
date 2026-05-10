# Qwen3.6-27B Direct-RHS Phase 1 Validation

Date: 2026-05-11  
Branch: `codex/qwen36-gdn-direct-rhs-solve-v2`  
Baseline control: `qwen36-27b-vllm-apc-baseline-v3`  
Candidate artifact: `/opt/dlami/nvme/qwen_artifacts/qwen36_27b_128k_fp8_direct_rhs_run2`

## Objective

Validate the existing direct-RHS v1 artifact using the new acceptance rule:

- top-logprob cosine comparison, average cosine >= 0.999 across 5 prompts
- MMLU-Lite accuracy delta < 1 percentage point
- GSM8K subset accuracy delta < 1 percentage point

If those gates pass, ship v1. If a gate is not covered or fails, do not mark
v1 shippable.

## Step 1: Parallel Server Bring-Up

Baseline v3 was running and healthy:

```text
curl http://127.0.0.1:8000/v1/models  # OK
curl http://127.0.0.1:8001/v1/models  # OK
neuron-ls: VLLM::EngineCore PID 51875 owns Neuron cores 0-3
```

Attempting to start the direct-RHS v1 TP=4 backend in parallel on port 8003
failed during model load:

```text
RuntimeError: The PyTorch Neuron Runtime could not be initialized.
Engine core initialization failed.
```

This is expected on `trn2.3xlarge`: there is one Neuron device with 4 cores,
and baseline v3 already occupies all 4 cores for TP=4. Parallel TP=4 baseline
and candidate servers are not possible on this host. Any comparison must be run
sequentially unless a larger instance is used.

## Step 2: Logprob/Cosine Gate

The OpenAI-compatible vLLM Neuron server does not currently return usable
logprobs for this model. Baseline v3 responses:

```text
/v1/chat/completions    logprobs=True, top_logprobs=20 -> {"error":{"message":"list index out of range", ...}}
/v1/completions         logprobs=20                    -> {"error":{"message":"list index out of range", ...}}
/v1/completions         prompt_logprobs=20             -> HTTP 200, but "logprobs": null and prompt_logprobs [null]
```

Because the API does not expose the required top-logprob vectors, the cosine
gate is **not covered**. Per the validation rules, this means direct-RHS v1
cannot be marked shippable from Phase 1 evidence.

## Result

Phase 1 is blocked before MMLU/GSM8K:

- parallel baseline/candidate serving is not possible on the current host
- the required cosine gate cannot be computed through vLLM/OpenAI logprobs

No correctness gate was weakened. Direct-RHS v1 remains promising for speed, but
is unverified under the required acceptance criteria.

## Next Action

Proceed to Phase 2 only with a validation path that does not depend on vLLM
logprobs, or compile a logits-enabled artifact for both baseline and candidate.
For the current objective, the next implementation branch should keep the
direct-RHS algebraic reordering but preserve baseline NKI layout and masking
more strictly.

## Phase 2 Update: Direct-RHS Solve v2

Commit: `756e684` (`codex/qwen36-gdn-direct-rhs-solve-v2`)
Artifact: `/opt/dlami/nvme/qwen_artifacts/qwen36_27b_128k_fp8_direct_rhs_v2_run1`

Implemented a v2 kernel as
`src/nki_kernels/nki_deltanet_chunked_direct_rhs_v2.py`. The file starts from
the baseline-v3 broadcast-input chunk kernel and only reorders the RHS solve:

```text
baseline: v_new = (N @ v_beta) - (N @ (k_beta * exp_gc)) @ state
v2:       v_new = N @ (v_beta - ((k_beta * exp_gc) @ state))
```

Local and remote static checks passed:

```text
python -m py_compile modeling_qwen35.py nki_deltanet_chunked_direct_rhs_v2.py qwen36_27b_compile_fp8.py
pytest contrib/models/Qwen3.6-27B/test/unit/test_deltanet_direct_rhs_solve.py -q
```

The required NKI simulator gate also passed on the Trainium host. The test
loaded the baseline-v3 chunk kernel and the v2 chunk kernel through
`nki.simulate`, using identical random fp32 inputs:

```text
output max_abs 2.3283064365386963e-10 mean_abs 1.1270234745452967e-11
state  max_abs 9.313225746154785e-10  mean_abs 7.640044152879e-11
SIMULATE_GATE_PASS
```

Hardware compile attempt 1 succeeded:

```text
CONTEXT_TRACE_SHAPE {"context_encoding_buckets": [512], "max_context_length": 512, "seq_len": 131072}
Finished generating HLO for context_encoding_model in 14.312s, input example shape = torch.Size([1, 512])
Finished generating HLO for token_generation_model in 2.430s, input example shape = torch.Size([1, 1])
Done Sharding weights in 51.424s
COMPILE_DONE
Finished weights loading in 14.031s
Warmup completed in 1.397s
LOAD_AFTER_COMPILE_OK
```

Smoke validation passed through the OpenAI-compatible proxy:

```text
Prompt: What is 17 * 23? Answer with only the number.
Response: 391
Latency: 1.37s
```

The required cosine/logprob gate remains blocked on this serving path:

```text
/v1/chat/completions logprobs=True, top_logprobs=20 -> {"error":{"message":"list index out of range", ...}}
/v1/completions logprobs=20 -> HTTP 400, raw completions disabled by proxy
/v1/completions prompt_logprobs=20 -> HTTP 400, raw completions disabled by proxy
```

Because the cosine gate cannot be computed, v2 is **compiled and smoke-tested
but not shippable under the stated validation rules**. MMLU/GSM8K were not run,
because the validation sequence requires stopping when an earlier gate is not
covered.
