# Codex Prompt — EAGLE V1/V3 Speculative Decoding for Qwen3.6-27B Hybrid

## Context

Repo: `~/inferentia-gdn` on trn2.3xlarge.
Target: `contrib/models/Qwen3.6-27B`.
Baseline: `qwen36-27b-vllm-apc-baseline-v3` @ `abc011e`. 418 tok/s prefill, 27 tok/s decode.

Kernel-internal prefill optimization is closed (see `docs/OPTIMIZATION_ARC.md`,
7 negative results). Next real win is **decode tok/s via EAGLE speculative
decoding**. NxDI's vLLM-Neuron stack supports EAGLE V1 and V3.

## Goal

Decode tok/s 27 → 40–55 (1.5–2×) via EAGLE V1 or V3.

Do NOT touch the CTE kernel. Do NOT modify production baseline v3 artifact.

## Phase A: Scouting (target 0.5 day)

A.1 Locate EAGLE support in the local NxDI tree:
```bash
grep -rn "EAGLE\|eagle\|speculative\|draft_model\|spec_decode" \
  src/neuronx_distributed_inference/
grep -rn "EAGLE\|eagle" examples/ 2>/dev/null | head
```

A.2 Identify:
- The config keys (likely `speculation_config`, `draft_model_path`, `eagle_version`)
- Whether EAGLE V1 (one-token draft heads) or V3 (multi-token tree) is supported
- Whether the draft model must be a separately-compiled NxDI model or just HF weights
- Whether hybrid models (GDN+attention) are supported by the spec decode scheduler

A.3 Check for a published EAGLE head for Qwen3.6-27B or Qwen3.5-27B:
- HuggingFace: search `EAGLE-Qwen3.6`, `EAGLE-Qwen3.5`, `Qwen3.6-EAGLE`
- AEON-7 ships `Qwen3.6-27B-Multimodal-NVFP4-MTP` with MTP head — different
  speculative scheme (Multi-Token Prediction), worth checking if NxDI supports MTP
- z-lab/`Qwen3.6-27B-DFlash` is a CUDA-only draft (not portable to Neuron)

A.4 Decide on draft strategy:
- **If a pre-trained EAGLE head exists**: use it. Skip training entirely.
- **If MTP is supported by NxDI**: use AEON-7's MTP head (BF16, bit-exact verified).
- **If neither**: defer EAGLE — training an EAGLE head on Qwen3.6-27B is a
  separate multi-week project, out of scope for this branch.

A.5 Output: `docs/eagle_scouting.md` with:
- NxDI EAGLE API location and required config
- Whether hybrid models are supported by the spec scheduler
- Draft model availability (or absence)
- Go/no-go decision

HARD STOP after Phase A. Report and wait for confirmation.

## Phase B: Integration (target 1 week, only if Phase A green)

B.1 Branch: `codex/qwen36-eagle-spec-decode`. Start from baseline v3.

B.2 Modify `Qwen35InferenceConfig` to accept speculation config:
- `enable_speculation = True`
- `speculation_length = 4` (or whatever NxDI default is)
- `draft_model_path = <path>`

B.3 Compile the spec-decode artifact:
- Target should compile both target (Qwen3.6-27B) and draft model
- Artifact path: `/opt/dlami/nvme/qwen_artifacts/qwen36_27b_eagle_run1`

B.4 Validation gates (in order, STOP if any fails):
- Compile + load on hardware
- Short prompt smoke: coherent output, no invalid IDs
- 762-token prompt: greedy decode matches baseline v3 within cosine ≥0.999
- 16K prompt: coherent output
- Acceptance rate measurement on 100 diverse prompts (target ≥60% for 1.5×, ≥75% for 2×)

B.5 Performance gates:
- Decode tok/s ≥ 40 (1.48× baseline) → minimum acceptable
- Decode tok/s ≥ 50 (1.85× baseline) → good
- Decode tok/s ≥ 55 (2.04× baseline) → stretch

B.6 Quality regression check:
- MMLU subset (200 questions), Δaccuracy < 1pp vs baseline v3
- GSM8K subset (50 problems), Δaccuracy < 1pp vs baseline v3

B.7 Output: `docs/eagle_results.md`. Tag `qwen36-27b-eagle-v1` if all gates pass.

## Hard constraints

1. Do not modify `qwen36-27b-vllm-apc-baseline-v3` artifact or its server.
2. Do not retry kernel optimization. The optimization arc is settled.
3. Commit + push after each step. Branch: `codex/qwen36-eagle-spec-decode`.
4. If draft model is unavailable AND MTP path is unsupported, stop and report.
   Do not start training an EAGLE head without explicit authorization.
5. Maximum compile attempts per phase: 3. Each compile is ~17 min.

## Expected outcomes

| Outcome | Probability | Meaning |
|---|---|---|
| EAGLE head available, integration succeeds | 30% | Best case, 1.5–2× decode banked |
| MTP path on NxDI supports Qwen3.6, succeeds | 25% | Alternative spec path, similar gain |
| Spec decode supported but no draft → defer | 30% | Documented path, no shipping artifact |
| Spec decode not supported on hybrid models | 15% | Pivot to FP8 KV cache instead |

Begin Phase A. Report after Phase A. Do not chain phases.
