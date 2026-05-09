# Codex Prompt — FP8 Weight Quantization for Qwen3.6-27B Hybrid

```
═══════════════════════════════════════════════════════════════════════
FP8 QUANTIZATION TASK — Qwen3.6-27B Hybrid Weights
═══════════════════════════════════════════════════════════════════════

CONTEXT

Branch: codex/qwen36-27b-64k-internal (or current working branch)
Repo: ~/inferentia-gdn
Target: contrib/models/Qwen3.6-27B
Hardware: trn2.3xlarge, TP=4
Current baseline: 149.6 tok/s prefill BF16, 18 tok/s decode.

NxDI 2.29 has GA FP8 quantization support for:
- QKV projection kernel (with fused FP8 KV cache option)
- MLP kernel (gate/up/down with MXFP4/MXFP8 paths)
- Output projection kernel

These are the standard NxDI quantized paths used by Llama, Mixtral, etc.
For Qwen3.6 hybrid, the attention layers (16 of 64) and MLP layers
(64 of 64) can use these paths directly. The GDN/DeltaNet kernel
custom NKI path stays BF16 in this phase (FP8 GDN is separate work).

Reference docs:
- AWS Neuron NxDI feature guide on quantization
- NKI 0.3.0 release notes (FP8 KV cache in QKV kernel)

GOAL: configure FP8 weight quantization for the standard layers
(attention QKV/O, all MLPs), validate correctness, measure speedup.
Keep DeltaNet kernel BF16 for now.

OUT OF SCOPE — do NOT do any of these in this phase:
- FP8 in custom DeltaNet NKI kernel (separate phase)
- FP8 KV cache (separate phase, after weight FP8 validates)
- Head-batched GDN kernel (separate phase)
- Speculative decoding (separate phase)
- Flash attention head_dim=256 (skipping — ablation shows low ROI)

═══════════════════════════════════════════════════════════════════════
PHASE A — ENVIRONMENT VERIFICATION (target: 1 hour)
═══════════════════════════════════════════════════════════════════════

A.1 Confirm NxDI version supports FP8 weight quantization for the
  layer types you have. Check:
    - QKV kernel FP8 (attention layers)
    - MLP kernel FP8 (gate/up/down)
    - Output projection FP8

A.2 Locate the quantization config in NxDI:
    grep -rn "quantization\|FP8\|fp8\|MXFP8" \
      src/neuronx_distributed_inference/models/config.py
    grep -rn "QuantizationConfig\|QuantConfig" \
      src/neuronx_distributed_inference/

  Identify the API for enabling FP8 on specific layer types.

A.3 Check whether GDN/DeltaNet layers can be EXCLUDED from
  quantization (we want them in BF16 for now).
  Look for layer-type filtering in the quantization config.

A.4 Output env_check.md with:
    - NxDI quantization API location
    - Whether layer-type filtering is supported
    - List of layer types that get FP8 in this config

HARD STOP if:
- NxDI quantization doesn't support hybrid models with mixed layer types
- GDN layers can't be excluded (would require BF16 fallback path that
  doesn't exist)

═══════════════════════════════════════════════════════════════════════
PHASE B — CALIBRATION DATA (target: 0.5 day)
═══════════════════════════════════════════════════════════════════════

B.1 FP8 weight quantization typically needs activation statistics
  (per-tensor or per-channel scales). Check if NxDI's FP8 path
  needs explicit calibration data.

B.2 If calibration is needed:
  - Gather 16-32 representative prompts (mix of short/long)
  - Run BF16 model on these to capture activation statistics
  - Save scales/zero-points for FP8 conversion

B.3 If post-training quantization without calibration is supported:
  - Skip B.2, proceed directly to Phase C

B.4 Output calibration.md with:
    - Whether calibration is needed
    - If yes: prompts used, statistics captured

═══════════════════════════════════════════════════════════════════════
PHASE C — FP8 ARTIFACT COMPILE (target: 0.5 day)
═══════════════════════════════════════════════════════════════════════

C.1 Modify Qwen35InferenceConfig to add FP8 quantization for:
  - All attention QKV projections (16 layers)
  - All attention output projections (16 layers)
  - All MLP gate/up/down projections (64 layers)
  - GDN layer projections: TRY excluding from FP8 first
    (in_proj_qkv, in_proj_z, in_proj_a, in_proj_b, conv1d, out_proj)

C.2 Compile the FP8 artifact alongside the existing BF16 baseline.
  Expected: faster compile (smaller weights) and smaller artifact
  (~50-60 GB instead of 82 GB).

C.3 Verify artifact loads on hardware.
  HARD STOP if NRT_RESOURCE: report which tensor blew HBM. May need
  to back off FP8 from some layer types.

═══════════════════════════════════════════════════════════════════════
PHASE D — CORRECTNESS + PERF VALIDATION (target: 1 day)
═══════════════════════════════════════════════════════════════════════

D.1 Correctness gates (in order):
  a. Short 63-token prompt: coherent output, no invalid IDs
  b. Capital-of-France test: "The capital of France is" → contains "Paris"
  c. 762-token prompt: cosine logit similarity vs BF16 baseline >= 0.99
     at first decode step
  d. 16K prompt: top-1 token agreement with BF16 baseline for at least
     8/10 first generations (FP8 may differ slightly, exact match not
     required, but high agreement is)

  HARD STOP if any correctness gate fails. Quantization broke math.
  Most likely cause: GDN layers being affected indirectly (e.g., FP8
  output of the GDN's input projections producing degraded numerics
  that cascade through the recurrence).

D.2 Performance gates:
  a. 4K prompt: tok/s
  b. 16K prompt: tok/s, target >= 1.3x baseline (>= 195 tok/s)
  c. 64K prompt: tok/s, target >= 1.25x baseline (>= 187 tok/s)
  d. Decode (TKG): tok/s, target >= 1.4x baseline (>= 25 tok/s)

  Decode improvement should be larger than prefill because decode is
  more weight-load-bound (FP8 halves weight bandwidth).

D.3 HBM characterization:
  - Capture peak HBM with neuron-monitor during 64K run
  - Should drop substantially from 98 GB BF16 baseline
  - More HBM headroom = more room for FP8 KV cache later

D.4 Output fp8_results.md with:
    - All measurements
    - Speedup vs BF16 baseline at each prompt size
    - Cosine similarity at first decode (correctness signal)
    - HBM peak comparison
    - Compile time and artifact size

═══════════════════════════════════════════════════════════════════════
HARD CONSTRAINTS
═══════════════════════════════════════════════════════════════════════

1. GDN/DeltaNet kernel stays BF16. Do not attempt FP8 in custom NKI
   kernel in this phase.
2. If correctness fails, do not weaken the gates. Investigate why.
3. Stop after each phase. Report. Wait for confirmation.
4. Commit + push after each step. Branch: chunked-prefill-fp8-weights.
5. Keep BF16 baseline artifact untouched for comparison.

═══════════════════════════════════════════════════════════════════════
EXPECTED OUTCOME
═══════════════════════════════════════════════════════════════════════

Best case:
- 16K prefill: 195-220 tok/s (1.3-1.45x BF16)
- Decode: 25-30 tok/s (1.4-1.7x BF16)
- HBM peak: 60-70 GB (down from 98)
- Correctness: cosine >= 0.99, top-1 agreement >= 8/10

Acceptable:
- 16K prefill: 175-195 tok/s (1.17-1.30x)
- Decode: 22-25 tok/s (1.22-1.40x)
- Correctness gates pass

Failure modes:
- Compile fails: layer-type filtering doesn't work, need different
  quantization API
- HBM blows: unexpected — FP8 should reduce HBM, not increase
- Correctness fails: GDN cascade issue, exclude more layer types
- Negligible speedup: NxDI FP8 path may not be efficient on hybrid

Begin Phase A. Report after Phase A. Do not chain.
```
