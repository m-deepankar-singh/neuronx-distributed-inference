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

