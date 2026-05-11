# Codex Prompt — MTP Speculative Decoding for Qwen3.6-27B Hybrid

## Why MTP (not EAGLE)

Qwen3.6-27B **ships with native MTP heads baked into the safetensors**. No draft model
download, no training. vLLM on NVIDIA invokes this via:

```
--speculative-config '{"method":"qwen3_next_mtp","num_speculative_tokens":2}'
```

This achieves ~90% acceptance and **2.5× decode** on Qwen3-Next / Qwen3.6 family
(Google B200 benchmark, vLLM Qwen3-Next blog post). It is the highest-ROI decode
optimization NVIDIA uses for this model class.

## Context

Repo: `~/inferentia-gdn` on trn2.3xlarge.
Target: `contrib/models/Qwen3.6-27B`.
Baseline: `qwen36-27b-vllm-apc-baseline-v3` @ `abc011e`. 418 tok/s prefill, 27 tok/s decode.

Kernel-internal prefill optimization is closed (see `OPTIMIZATION_ARC.md`, 7 negative
results). Decode is the next real win, and MTP is how NVIDIA gets 2.5× decode on this
exact model.

## Goal

Decode tok/s 27 → 55-70 (2-2.5×) via Qwen3.6-27B's native MTP heads,
served through NxDI's speculative decoding path on vLLM-Neuron.

Do NOT touch the CTE kernel. Do NOT modify baseline v3 artifact.

## Phase A: Scouting (target 0.5 day)

A.1 Confirm NxDI's spec decode supports the MTP scheme:
```bash
grep -rn "mtp\|MTP\|multi_token\|nextn\|NextN\|qwen3_next_mtp\|speculative" \
  src/neuronx_distributed_inference/
```

A.2 Identify which speculation methods NxDI supports:
- EAGLE V1, V3 (most likely supported)
- MTP / NextN (the one we want — verify)
- DFlash (probably not — CUDA-only)

A.3 Inspect the Qwen3.6-27B HF config:
```bash
cat /opt/dlami/nvme/models/Qwen3.6-27B/config.json | grep -i "mtp\|nextn\|num_predict"
```
The MTP layers should appear as `num_nextn_predict_layers` or similar.

A.4 Check whether NxDI's NeuronConfig accepts speculative config:
```bash
grep -rn "SpeculationConfig\|speculation_config\|enable_speculation" \
  src/neuronx_distributed_inference/models/config.py
```

A.5 If NxDI supports speculation but NOT the qwen3_next_mtp method specifically,
identify the API gap:
- Is it the spec scheduler (vLLM side)?
- Is it the draft execution path (NxDI side)?
- Is it both?

A.6 Output `docs/mtp_scouting.md` with:
- NxDI spec-decode API location and methods supported
- Whether qwen3_next_mtp / nextn is supported, or what gap exists
- Whether Qwen3.6-27B's MTP heads are accessible from the loaded weights
- Required NxDI/vLLM changes (if any)
- Go/no-go for Phase B

HARD STOP after Phase A. Report and wait.

## Phase B: Integration (target 1-2 weeks, only if Phase A green)

B.1 Branch: `codex/qwen36-mtp-spec-decode`.

B.2 Modify Qwen3.6-27B contrib to expose MTP heads:
- Locate MTP layer definitions in `contrib/models/Qwen3.6-27B/src/modeling_qwen35.py`
- If MTP layers are not loaded, add loading path (check HF reference impl)

B.3 Wire NxDI's speculative path:
- Modify NeuronConfig to accept speculation_config
- Connect MTP heads to NxDI's draft execution
- Wire vLLM's speculative scheduler to invoke the path

B.4 Compile artifact: `/opt/dlami/nvme/qwen_artifacts/qwen36_27b_mtp_run1`

B.5 Validation gates (in order, STOP if any fails):
- Compile + load on hardware
- Short prompt smoke: coherent output
- 762-token greedy: matches baseline within cosine ≥0.999
- Acceptance rate on 100 diverse prompts: target ≥70% (NVIDIA reports ~90%, lower
  acceptance on Neuron acceptable as long as wall-clock improves)
- 16K prompt: coherent

B.6 Performance gates:
- Decode tok/s ≥ 40 (1.5× baseline) → minimum
- Decode tok/s ≥ 50 (1.85× baseline) → good
- Decode tok/s ≥ 55 (2.04× baseline) → matches NVIDIA's claim
- 16K prefill within 5% of baseline (prefill shouldn't regress)

B.7 Quality regression: MMLU-200, GSM8K-50, Δaccuracy < 1pp.

B.8 Output `docs/mtp_results.md`. Tag `qwen36-27b-mtp-v1` if all gates pass.

## Phase C: Fallback to EAGLE (only if MTP path is structurally unsupported)

If Phase A reveals MTP is not supported by NxDI and the gap is too large to bridge
in a single PR, fall back to EAGLE V1/V3 which NxDI supports more directly. This
requires either:
- A pre-trained EAGLE head for Qwen3.6-27B (probably doesn't exist)
- Training one (multi-week, separate scope)

Document the decision and stop.

## Hard constraints

1. Baseline v3 artifact and server stay untouched.
2. No kernel optimization. Done.
3. Commit + push after each step.
4. Max compile attempts per phase: 3.
5. If acceptance rate is < 50% on hardware, do not ship even if wall-clock improves —
   indicates a wiring bug.

## Expected outcomes

| Outcome | Probability | What it means |
|---|---|---|
| NxDI supports qwen3_next_mtp natively; integrates clean | 25% | 2-2.5× decode, ~3 days actual work |
| NxDI supports spec decode but needs MTP method adapter | 40% | Real integration work, 1-2 weeks |
| MTP method must be added to NxDI core | 25% | 3-4 weeks, larger upstream effort |
| MTP path structurally blocked by NxDI design | 10% | Fall back to FP8 KV cache work |

## Why this is right per NVIDIA's own playbook

From the vLLM Qwen3-Next blog: vLLM achieves 2.5× decode on Qwen3-Next using
MTP at 90% acceptance. Google's 1M tok/s on 96 B200 benchmark uses MTP-1 spec
decoding plus FP8 KV cache as the two key optimizations beyond batched serving.

The MTP heads in Qwen3.6-27B are the SAME approach. Using them on Trainium is
literal parity with NVIDIA's published optimization technique, not a Neuron-specific
hack. Reviewers will recognize this immediately.

Begin Phase A.
