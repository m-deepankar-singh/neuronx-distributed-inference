Context (read first)

You are working in ~/inferentia-gdn on a trn2.3xlarge instance. The repo is the AWS Neuron NxDI fork with Qwen3.6-27B hybrid (GDN+attention) contributed at contrib/models/Qwen3.6-27B/.

Validated baselines you must not break:
- qwen36-27b-vllm-apc-baseline-v3 @ commit abc011e — current production, 418 tok/s prefill, 27 tok/s decode, vLLM APC working
- qwen36-27b-baseline-v2 @ 4b73970 — pre-vLLM, 461 tok/s prefill
- qwen36-27b-baseline-v1 @ 99e24fc — original 64K hybrid

Branches with prior data you should reference:
- codex/qwen36-gdn-direct-rhs-solve (v1) — gave 1.13× prefill, failed bit-e /opt/dlami/nvme/qwen_artifacts/qwen36_27b_128k_fp8_direct_rhs_run2. You will reuse this artifact.
- codex/qwen36-gdn-blocked-solve — failed 4× slower (proved VE row-shuffle on Trainium2)
- codex/qwen36-gdn-materialized-n-cleanup — failed greedy with low-risk refactors (proved bit-exact gate is too strict)

Hard lessons from prior attempts:
1. Bit-exact greedy match against baseline v3 is too strict. Even logicallyspose vs nc_matmul(eye), removing redundant masks) produce different bits.Use cosine ≥ 0.999 + quality benchmarks within 1pp as the correctness gate, not bit-exact greedy match. This matches what baseline-v1 was originally validated against vs HF
reference.
2. Replacing TE matmul with VE row-shuffle is disastrously slow on Trainium2. Do not rewrite the matmul pattern.
3. Neumann power-doubling failed at BF16 cosine 0.9156. Do not retry Neumantion.

Primary objective

Validate the existing direct-RHS solve artifact (v1) with cosine + quality  it and ship.

If v1 passes, that is the night's win. Do not start new kernel work until vunshippable.

Phase 1: Validate existing direct-RHS v1 artifact (target 2-3 hours)

The artifact /opt/dlami/nvme/qwen_artifacts/qwen36_27b_128k_fp8_direct_rhs_v1 branch. Do not recompile.

Steps:

1. Bring up two servers in parallel:
  - Baseline v3 on port 8001 (vLLM backend) + 8000 (chat proxy) — should already be running, verify with curl http://localhost:8000/v1/models
  - Direct-RHS v1 artifact on port 8003 (vLLM backend) + 8002 (chat proxy) _baseline_v3.sh script with --compiled-artifacts/opt/dlami/nvme/qwen_artifacts/qwen36_27b_128k_fp8_direct_rhs_run2 and ports overridden
2. Cosine logit comparison (5 prompts, 5 minutes total):
  - Use vllm/qwen36_chat_proxy.py or a small standalone script that calls /v1/completions with max_tokens=1, logprobs=20 on both servers
  - For each of 5 fixed prompts (short, medium, long, code, multilingual),
  - Compute cosine similarity between baseline v3 and direct-RHS top-20 logit vectors
  - Gate: average cosine ≥ 0.999. If < 0.999, mark v1 unshippable and proce
3. MMLU-Lite (200 questions, target ~30 min):
  - Use validation_scripts/qwen36_27b_chat_quality_eval.py if it supports M
  - Run 200 MMLU questions on both servers, greedy decode (top_k=1, temperature=0)
  - Record accuracy on each
  - Gate: |Δaccuracy| < 1.0 percentage point absolute. If exceeded, mark v1 unshippable and proceed to Phase 2.
4. GSM8K subset (50 problems, target ~45 min):
  - Same harness, 50 GSM8K problems
  - Record accuracy on each
  - Gate: |Δaccuracy| < 1.0 percentage point absolute.
5. If all three gates pass:
  - Create branch codex/qwen36-gdn-direct-rhs-solve-v2 from codex/qwen36-gdn-direct-rhs-solve
  - Add a results document at contrib/models/Qwen3.6-27B/docs/direct_rhs_so measurements
  - Tag the commit as qwen36-27b-rhs-solve-v1 (semver: this is "rhs-solve" version 1, not "v1" of the model)
  - Push branch and tag
  - Mark Phase 1 complete in the final report
6. If any gate fails:
  - Document the failure mode in direct_rhs_solve_v2_validation.md
  - Commit the validation harness and the failure report
  - STOP Phase 1 and proceed to Phase 2.

Phase 2: Direct-RHS solve v2 implementation (only if Phase 1 v1 fails)

Only enter Phase 2 if v1 failed at least one gate in Phase 1.

Goal: rewrite the direct-RHS solve to be correctness-clean (pass cosine ≥ 0.999) while preserving the 1.13× speed of v1.

The mathematical change is unchanged from v1:
v_new = N @ v_beta - (N @ k_beta_exp_gc) @ state
      = N @ (v_beta - k_beta_exp_gc @ state)
Compute k_beta_exp_gc @ state first (one matmul), subtract from v_beta, the

The implementation discipline (different from v1):
- Do not modify any nc_transpose / nc_matmul(stationary, moving=eye) patterns. Keep all transpose ops verbatim from nki_deltanet_chunked.py.
- Do not remove "redundant" masks. They affect bits.
- Do not change SBUF tile layouts or partition assignments.
- The only change is the order of three operations: form v_beta - kbg @ sta of after.

Create new kernel file contrib/models/Qwen3.6-27B/src/nki_kernels/nki_deltaas a copy of nki_deltanet_chunked.py with that single algebraic reordering.Wire via config flag use_direct_rhs_solve_v2 (default False) in modeling_qwen35.py.

Validation gates (in order — STOP if any fails):
1. Local CPU nki.simulate against the existing kernel: max_abs_diff < 1e-5
2. Compile new artifact at /opt/dlami/nvme/qwen_artifacts/qwen36_27b_128k_fp8_direct_rhs_v2_run1
3. Load on hardware
4. Smoke prompt returns coherent output
5. Cosine ≥ 0.999 on 5-prompt logit comparison
6. MMLU-Lite Δaccuracy < 1pp
7. GSM8K subset Δaccuracy < 1pp
8. 16K vLLM perf ≥ 460 tok/s (i.e., at least matches v1's 1.10× over baseline)
9. 64K vLLM perf coherent + ≥ 460 tok/s

If gates 1-9 pass: tag qwen36-27b-rhs-solve-v2, push, document.

If gate 1 fails (simulator cosine too low): there is a bug in the algebraic compile.

If gates 2-4 fail: compile/load issue. Document and stop.

If gates 5-7 fail: quality regression. Mark unshippable, document, stop. Do

If gate 8 fails: the rewrite compiled but doesn't speed up. Document — possdependent. Stop.

Phase 3: FP8 KV cache prep (only if both Phase 1 and Phase 2 are blocked)

Only enter Phase 3 if you have remaining time after Phase 1+2 are both unsh

Goal: scout NxDI's FP8 KV cache support and produce an actionable design do

Steps:
1. Search src/neuronx_distributed_inference/ for FP8 KV cache support:
grep -rn "fp8_kv\|FP8KV\|kv_cache_quant\|cache_quant" src/neuronx_distribut
2. Identify the config flag and its requirements (calibration data, dtype, etc.)
3. Check whether HybridCacheManager (contrib/models/Qwen3.6-27B/src/modelin support FP8 KV alongside fp32 GDN state
4. Write contrib/models/Qwen3.6-27B/docs/fp8_kv_cache_plan.md with:
  - Found API and requirements
  - Whether GDN recurrent state can stay fp32 with FP8 KV (it must)
  - Estimated decode speedup (1.1-1.3×)
  - Estimated implementation effort (days)
  - Whether calibration is needed
  - Compile config delta from baseline v3
5. Commit and push to codex/qwen36-fp8-kv-cache-prep

Do not implement FP8 KV in this phase. Only document.

Hard rules

1. Never modify the live baseline v3 artifact at /opt/dlami/nvme/qwen_artifseline*. Treat it as immutable.
2. Never overwrite the v1 direct-RHS artifact at qwen36_27b_128k_fp8_direct_rhs_run2. Reuse for validation.
3. Commit and push after every phase milestone. No "I'll commit at the end. its own commit.
4. Never weaken a correctness gate to make a branch ship. If something fails the gate, document and stop. Do not relax the cosine threshold below 0.999 or the quality threshold
above 1pp.
5. If you hit a NRT_RESOURCE or compile failure, do not retry blindly. Document what failed, commit the log, stop.
6. Restore baseline v3 server on port 8000/8001 at the end of every phase. k up before you move to the next phase.
7. Maximum compile attempts per night: 3. Each compile is 17-22 min. Don't burn the night on compile loops.
8. No new kernel algorithms. Do not try Neumann, do not try blocked solve, se are settled negative results.

Final report

When you stop work (either gates passed, gates failed, or out of time), wrib/models/Qwen3.6-27B/docs/overnight_run_$(date +%Y%m%d).md with:
- Which phases ran
- Each gate result
- Final state of artifacts (which are tagged, which are archived)
- Final state of running servers
- Recommended next action

Push this report to whichever branch was active when you stopped.

Expected outcomes (in priority order)

┌─────────────────────────────────────────────────────────────┬─────────────┬──────────────────────────────────────────────────┐
│                           Outcome                           │ Probabilityeans                   │
├─────────────────────────────────────────────────────────────┼─────────────┼──────────────────────────────────────────────────┤
│ Phase 1 passes — v1 ships at 1.13×                          │ ~60%       , real win banked      │
├─────────────────────────────────────────────────────────────┼─────────────┼──────────────────────────────────────────────────┤
│ Phase 1 fails, Phase 2 passes — v2 ships at 1.10-1.20×      │ ~20%       e rewrite              │
├─────────────────────────────────────────────────────────────┼─────────────┼──────────────────────────────────────────────────┤
│ Phase 1 fails, Phase 2 fails — Phase 3 produces FP8 KV plan │ ~15%       code roadmap unblocked │
├─────────────────────────────────────────────────────────────┼─────────────┼──────────────────────────────────────────────────┤
│ Hardware/compile blocker                                    │ ~5%                               │
└─────────────────────────────────────────────────────────────┴─────────────┴──────────────────────────────────────────────────┘

Total scope: 6-9 hours of focused work, fits in one overnight session.

Begin

Start with Phase 1, step 1. Verify baseline v3 is up and responding. Bring up the direct-RHS v1 artifact on parallel ports. Run the cosine comparison. Report after Phase 1.

Do not chain phases without reporting. Do not skip the validation gates.