# Qwen3.6-27B on Trainium — Status Summary

**For:** AWS / NxDI review · Monday 5/25 sync
**PR:** [aws-neuron/neuronx-distributed-inference#164](https://github.com/aws-neuron/neuronx-distributed-inference/pull/164)
**Author:** M Deepankar Singh · singh.deepankar39@gmail.com · github.com/m-deepankar-singh
**Last updated:** 2026-05-22

---

## TL;DR

Qwen3.6-27B hybrid (DeltaNet + GQA) is serving on Trainium via a contrib path against AWS Neuron NxDI:

- **9,800-LOC** contribution merged-ready in PR #164; **1,200 additional LOC** in follow-up branch covering Hybrid APC runtime + fused DeltaNet stability fix
- **128K-context artifact** compiled and loaded on a single `trn2.3xlarge` (TP=4, LNC=2, SDK 2.29)
- **Hybrid APC validated end-to-end** through 16K with exact-match correctness; **13.5–18.5× warm-cache speedup** depending on scenario
- **Fused DeltaNet kernel** had a numerical stability issue under realistic Qwen gate values (Neumann power-doubling) — proactively diagnosed and replaced with a direct triangular solve. Validation pushed.
- **Two real production bugs** fixed mid-validation: active chunk-boundary handling for suffix-only Hybrid APC prep; scheduler safety gate for unbacked vLLM prefix-cache reads
- **Scope boundaries explicit and intentional:** continuous batching past `max_num_seqs=1`, 32K/64K APC bucket recompile, and native MTP integration are follow-up work

---

## PR Scope (as it stands)

### In-scope (this PR / follow-up clean branch)

- Qwen3.6-27B text + VL model code (`modeling_qwen35*.py`)
- Three NKI DeltaNet kernel variants: basic, chunked, **fused (direct-solve, stable)**
- Hybrid cache manager (DeltaNet recurrent/conv state + GQA KV)
- MLP-only FP8 quantization path enabling 128K context
- vLLM Neuron registry + launcher + OpenAI-compatible serving helpers
- Native vLLM APC integration with exact-match validation
- 57 CPU unit tests (config, weight-conversion, hybrid cache, DeltaNet decay)

### Explicitly out-of-scope (intentional, follow-up work)

- Continuous batching past `max_num_seqs=1` (artifact compile boundary, not correctness)
- 32K / 64K APC bucket coverage (current artifact `pfx16k` has prefix buckets ≤ 16384)
- Native Qwen MTP speculative decoding integration (separate work; ~2× decode improvement validated separately)
- Hybrid kernel selector (Neumann fast path + direct-solve stable fallback)

---

## Hardware Validation — `trn2.3xlarge`, TP=4, LNC=2, SDK 2.29

### Long-context artifact compile + load

| Scenario | Result |
|---|---|
| 128K artifact compile/load | PASS |
| 32K & 64K short-after-long state reset | PASS |
| 32K & 64K needle-retrieval | PASS |

### Cold prefill (PR #164 baseline + direct-solve follow-up)

| Prompt | Cold TTFT | Cold prefill |
|---:|---:|---:|
| 512 | 1.31 s | 390 tok/s |
| 1024 | 1.88 s | 535 tok/s |
| 4096 | 6.94–7.03 s | 582–588 tok/s |
| 8192 | 13.49–13.61 s | 602–607 tok/s |
| 16384 | 27.54–27.84 s | 589–595 tok/s |
| 32768 | ~76.6–81.1 s | — |
| 65536 | ~153–162 s | — |

### Warm prefill (APC, direct-solve artifact)

| Prompt | Warm TTFT | Warm prefill |
|---:|---:|---:|
| 512 | 0.42 s | 1.2 k tok/s |
| 4096 | 0.42 s | 9.8 k tok/s |
| 8192 | 0.43 s | 18.9 k tok/s |
| 16384 | 0.45 s | 36.3 k tok/s |

### Decode

| Metric | PR #164 baseline | Direct-solve variant |
|---|---:|---:|
| Decode throughput | 26.3–26.6 tok/s | 21.63 tok/s |
| TPOT | ~37.6–38.0 ms | 46.2 ms |
| 64-token decode | — | 5.92 s (128-tok) |

**Note on decode regression:** the direct-solve variant trades ~18% decode TPOT for numerical stability under realistic Qwen gate scales. The original Neumann power-doubling fused path was faster but unstable in production-realistic conditions. Direct triangular RHS solve is the correct production default; the regression is the cost of correctness. A hybrid kernel selector (Neumann fast path + direct-solve fallback) is a viable follow-up if the gap matters.

### APC / Prefix Cache (PR #164 baseline, max_num_seqs=1)

| Scenario | Cold | Warm | Speedup | Correctness |
|---|---:|---:|---:|---|
| Server exact-repeat, ~10.8K prompt | 26.68 s | 1.67 s | **16.0×** | exact text match |
| Offline exact-repeat | 26.19 s | 2.38 s | **11.0×** | exact token-ID match |
| Offline partial-prefix reuse | 25.52 s | 1.70 s | **15.0×** | exact token-ID match |
| Server cross-prefix reuse | 25.17 s | 1.36 s | **18.5×** | exact text match |

Shared-prefix concurrency at 1/2/4 requests returned all requested markers exactly. Current artifact is compiled for `max_num_seqs=1`, so requests queue rather than true multi-sequence batching — addressed in follow-up.

### Memory

| | PR #164 baseline (64K eval) | Direct-solve variant |
|---|---:|---:|
| Peak Neuron HBM | ~53.25 GB | 60.1 GiB / 64.54 GB |
| HBM utilization | — | 61.9% |
| Peak runtime host memory | — | 50.39 GB |
| Server RSS | — | 1.30 GB |

**Memory delta explained:** the +9.85 GB direct-solve overhead vs PR #164 README is reserved by the GDN checkpoint slot bank (`max_gdn_checkpoint_slots=64` across TP=4) plus larger TKG bucket coverage (`[8192, 32768, 131072]`). Not a regression — different artifact configuration. Like-for-like A/B requires matched compile settings.

---

## Fused DeltaNet Stability Fix (May 22 follow-up branch)

Discovered during validation: the fused DeltaNet path's Neumann power-doubling solve was numerically unstable under realistic Qwen gate scales — repeated full matrix powers amplified error, sometimes producing incoherent tokens. The chunked DeltaNet path already used a stable direct triangular solve.

**Fix:** Ported the direct-solve approach to the fused kernel — compute the lower-triangular causal recurrence explicitly in the RHS solve path instead of via Neumann power-doubling.

**Validation:**
- Standalone validator script (`scripts/validate_deltanet_fused_nki.py`) loads the fused NKI kernel directly without package import side effects
- CPU regression test (`test/unit/test_deltanet_decay.py`) updated to cover realistic decay/gate scales
- 2 unit tests pass
- Hardware coherence run: chat-template prompts produce real non-repetitive output, smoke decode at ~20.5 tok/s

**Trade-off captured above:** ~18% decode TPOT regression vs Neumann fast path, accepted as the cost of production stability.

---

## Production Hardening (mid-validation bug fixes)

Two real issues caught and fixed during the validation run on `experimental`:

1. **Active chunk-boundary handling for suffix-only Hybrid APC prep** — was producing edge-case errors when prefix continuation crossed a chunk boundary mid-sequence
2. **Scheduler safety gate for unbacked vLLM prefix-cache reads** — was causing empty streams or engine death in reject/production mode; now skipped cleanly

Both fixes pushed to `experimental` branch; covered by 4 new scheduler tests + 8 async tests.

---

## Path Forward — Questions for the Maintainer

We'd like to align on:

1. **Merge strategy** — does the maintainer want the fused direct-solve stability fix in PR #164, or as a follow-up PR after baseline merge?
2. **Continuous batching scope** — `max_num_seqs > 1` requires a recompile and additional scheduler integration. Block PR #164 on it, or follow-up?
3. **32K / 64K APC bucket coverage** — recompile-only change. In-scope or follow-up?
4. **Hybrid kernel selector** (Neumann fast path + direct-solve fallback) to recover the ~18% decode regression when gate scales are within stable range — worth pursuing?
5. **MTP integration** — the standalone MTP work delivers ~2× decode on Qwen 3.x. Integrate into NxDI spec-decode config in a follow-up PR?

---

## Repos / Artifacts

- **Main PR:** [#164](https://github.com/aws-neuron/neuronx-distributed-inference/pull/164)
- **Fused direct-solve clean branch:** see PR comment from 2026-05-22
- **Validation artifacts:**
  - `profile_artifacts/qwen36_cte512_openai_20260521/SUMMARY.md` — APC end-to-end run
  - `profile_artifacts/qwen36_fused_directsolve_20260522/` — direct-solve coherence, decode, context sweep, memory
  - `profile_artifacts/qwen36_128k_fp8_mlp_hybrid_apc_20260521/` — 128K artifact validation

---

*This is a living document. Update as the review progresses.*
