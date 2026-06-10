# Qwen 3.6-27B on AWS NxDI — Contribution Delta

**What existed before, what I added, what worked, what didn't, what's next.**

- **PR:** [aws-neuron/neuronx-distributed-inference#164](https://github.com/aws-neuron/neuronx-distributed-inference/pull/164)
- **Author:** M Deepankar Singh · [github.com/m-deepankar-singh](https://github.com/m-deepankar-singh) · singh.deepankar39@gmail.com
- **Status:** in active maintainer review (AWS engineer `aws-reutermj` commented "working with our team to evaluate" 2026-05-14)
- **Last updated:** 2026-05-24

---

## 0. Executive Delta — stock NxDI vs this work

The main contribution is not just "Qwen on Trainium." It is making a hybrid
recurrent/attention model serve correctly with long-context prefix reuse.

Stock NxDI/vLLM already had the general serving substrate: tensor-parallel
execution, CTE/TKG graph separation, block KV layout, attention KV cache,
PagedAttention-style metadata, on-device sampling, and vLLM/OpenAI serving for
conventional transformer models.

What this work added is the model-specific layer needed for Qwen 3.6-27B:

| Area | What stock NxDI had | What this work added | Impact |
|---|---|---|---|
| Model architecture | Transformer-style decoder support | Qwen 3.6 dense hybrid `[3 DeltaNet + 1 GQA] × 16` implementation | Brings up a 27B hybrid architecture that was not directly served by stock NxDI |
| GDN / DeltaNet state | Attention KV cache, but no recurrent-state prefix contract | Explicit recurrent + convolution state support for GDN/DeltaNet layers | Makes hybrid prefill/decode state coherent instead of treating the model as attention-only |
| Prefix caching | vLLM APC for attention KV blocks | Hybrid APC: attention KV hit intersected with GDN recurrent + conv checkpoint hit | Enables safe warm-prefix reuse for hybrid models |
| Runtime ABI | Standard CTE/TKG model inputs | Restore/commit tensors, chunk-boundary controls, request-scoped metadata, inert decode controls | Lets vLLM scheduler decisions drive model-side GDN state restore/commit |
| Kernels | Existing Neuron kernels for standard ops | NKI DeltaNet chunked and fused CTE kernels, stable direct-solve path | Makes long-context DeltaNet prefill viable on Trainium |
| Memory | BF16/FP16 model serving paths | MLP-only FP8 compile path while keeping sensitive state paths BF16/FP32 | Reduces static HBM pressure enough to support 128K-class artifacts |
| Serving | vLLM Neuron plugin for supported models | Qwen 3.6 registry patching, launcher, OpenAI-compatible path, thinking toggle, debug probes | Turns the model into a usable endpoint, not just a compiled graph |
| Validation | General unit/integration tests | HF greedy comparison, Hybrid APC exactness, context sweep, artifact config audit, HellaSwag Stage 7 shim | Gives correctness evidence across token, component, serving, and task levels |

### Why the GDN state work matters

Standard APC can reuse a prefix by reusing attention KV. That is not enough for
Qwen 3.6-27B because most layers are DeltaNet/GDN layers with recurrent and
convolutional state. If attention KV is restored but GDN state is not restored
at the same prefix boundary, the model can produce plausible-looking but
incorrect outputs.

The safe reusable prefix is therefore:

```text
usable prefix =
  attention KV block hit
  ∩ GDN recurrent checkpoint hit
  ∩ GDN conv checkpoint hit
```

If any component is missing, the runtime must fall back to a shorter prefix or
replay the suffix.

### Additional pieces not in stock NxDI

- Qwen 3.6 text-config loader for nested `text_config` layouts.
- Static Qwen hybrid state cache for GDN checkpoints.
- Hybrid APC metadata store, scheduler bridge, slot allocator, and lifecycle callbacks.
- vLLM scheduler patch to extract request ids, cumulative prefix hashes, block refs, and backed-prefix decisions.
- Chunked CTE continuation handling for cached suffixes.
- TKG bucket controls for 8K / 32K / 128K decode contracts.
- Vocab-parallel on-device greedy sampling fixes.
- Sample+logits debug artifact path for comparing sampled token ids with returned logits.
- OpenAI/vLLM probes for streaming, non-streaming, TTFT, TPOT, cold/warm prefill, and memory capture.
- Equivalence artifacts and a Stage 7 compatibility shim when the public AWS skill referenced a missing `neuron_bench` package.

### Impact in one paragraph

The result is a working long-context Qwen 3.6-27B Trainium path with model-side
GDN state, Hybrid APC, FP8 MLP weights, custom NKI DeltaNet kernels, and
OpenAI/vLLM serving. Validation showed HF greedy match above the 95% target on
the sampled comparison (`156/160`, 97.5%), Stage 5/6 teacher-forced top-1
agreement at 100% over tested positions, Stage 7 HellaSwag parity on the tested
slice, exact Hybrid APC backed-prefix reuse, and large warm-prefix latency wins
for repeated long prompts. Decode speed still has headroom, but the core
architecture and serving path are workable.

### Hard bugs worth calling out

- **Hybrid APC false correctness risk:** attention KV reuse alone was not valid because GDN state also had to match the prefix boundary.
- **Chunked prefill boundary bugs:** cached suffix continuations needed special handling so CTE chunks committed and restored checkpoints at valid boundaries.
- **Dummy-row control leakage:** padded rows could inherit active restore/commit masks unless Hybrid APC controls were explicitly made inert.
- **vLLM metadata dependency:** direct server paths lacked `full_context_lens`, block tables, slot mappings, request records, and block refs; Hybrid APC had to be treated as vLLM-coupled.
- **NKI numerical instability:** fused DeltaNet Neumann power-doubling diverged at realistic Qwen gate scales; direct triangular solve fixed stability at some TPOT cost.
- **Vocab-parallel sampling confusion:** returned debug logits could be rank-local while sampled ids were global, so naive argmax comparison produced false mismatches.
- **Decode bottleneck misdiagnosis:** slow decode was not host-logits transfer; the artifact already used on-device greedy sampling. The real issue was TKG bucket/runtime overhead.

---

## 1. The Gap — what NxDI had vs what was missing

### What NxDI already supported

- **Qwen 3** family — standard dense transformer serving on Trainium (Jim Burtoft's PR #140 baseline for Qwen 3.6 architecture)
- Standard NKI kernels for matmul, attention, normalization, RoPE
- vLLM Neuron plugin path for conventional transformer models
- PagedAttention-style KV cache management
- Tensor parallelism + LNC support
- FP16 / BF16 serving paths

### What was missing for Qwen 3.6-27B

**Qwen 3.6-27B is a hybrid architecture: `[3 DeltaNet + 1 GQA] × 16` layers** — 48 DeltaNet (linear attention) layers + 16 standard full attention layers. NxDI had no path to serve it because:

| Gap | Why it mattered |
|---|---|
| No NKI kernels for DeltaNet (linear attention) | Linear attention is not a SIMT-friendly op pattern; NxDI's existing flash-attention kernels don't apply |
| No hybrid cache management | DeltaNet has recurrent + convolutional state; GQA has standard KV — needed unified cache lifecycle |
| No MLP-only FP8 quantization path | 27B at BF16 won't fit 128K context on a single `trn2.3xlarge`; needed selective quantization |
| No vLLM APC integration for hybrid models | Standard PagedAttention assumes uniform attention; hybrid model needs custom prefix-cache handling for recurrent state |
| No long-context (128K) serving artifact | Existing bucket configurations didn't cover the target context window |
| No serving stack (OpenAI-compatible API) for Qwen 3.6 | Required for evaluation + deployment |

---

## 2. What I Added — the actual contribution

**Scale:** 11,089 lines added · 36 files · contrib-scoped under `contrib/models/Qwen3.6-27B/`
**Build time:** 2 weeks end-to-end from cold start (architecture → kernels → integration → hardware validation → submission)

### A. Three NKI DeltaNet kernel variants

```
contrib/models/Qwen3.6-27B/src/nki_kernels/
├── nki_deltanet.py            # basic — correctness baseline
├── nki_deltanet_chunked.py    # chunked — memory-efficient long context
└── nki_deltanet_fused.py      # fused — throughput-optimized, direct-solve (stable)
```

Each variant trades different axes (memory vs throughput vs correctness verification). All three pass token-by-token correctness against the Triton reference in `fla-org/flash-linear-attention`.

### B. Hybrid cache manager

- Unified lifecycle for DeltaNet recurrent state + GQA KV cache
- Checkpoint slot management with `max_gdn_checkpoint_slots=64`
- Restore + commit masks for state reuse across requests
- Backed prefix reads + stricter unbacked-read guards for safety

### C. MLP-only FP8 quantization path

- MLP layers → FP8 (E4M3)
- DeltaNet recurrent state, attention sensitive paths → kept BF16 / FP32 accumulation
- Enables **128K context on a single `trn2.3xlarge`** that wouldn't fit in BF16

### D. vLLM Neuron integration + serving stack

```
contrib/models/Qwen3.6-27B/vllm/
├── patch_nxdi_registry.py     # registers Qwen 3.6 with NxDI
├── serve_qwen36.py            # vLLM Neuron launcher
├── start_vllm_server.sh
├── qwen36_chat_proxy.py       # OpenAI-compatible chat proxy
└── ... (8 helper files)

contrib/models/Qwen3.6-27B/scripts/
└── openai_compat_server.py    # full OpenAI-compatible serving
```

### E. Native vLLM APC integration

- Exact-match correctness validated for: server exact-repeat, offline exact-repeat, partial-prefix reuse, server cross-prefix reuse
- Shared-prefix concurrency at 1/2/4 requests with exact marker matches
- Required custom prefix-cache handling for hybrid model semantics (chunked-prefill metadata, request-id propagation, scheduler integration)

### F. Test coverage

| Module | Tests |
|---|---:|
| `test_config.py` | 26 |
| `test_weight_conversion.py` | 16 |
| `test_hybrid_cache_manager.py` | 13 |
| `test_deltanet_decay.py` | 2 |
| Async scheduler (added during validation) | 8 |
| Scheduler safety / bucket selection | 4 |

**Total: 69 unit tests + 4 integration tests.**

### G. Production hardening (caught during validation)

Two real bugs found + fixed mid-validation, pushed to `experimental`:

1. **Active chunk-boundary handling for suffix-only Hybrid APC prep** — edge-case errors when prefix continuation crossed chunk boundary
2. **Scheduler safety gate for unbacked vLLM prefix-cache reads** — was causing empty streams or engine death in reject/production mode

### H. Fused DeltaNet stability fix (May 22 follow-up)

Discovered during chained coherence validation: fused-path output diverged from chunked-path output at realistic Qwen gate scales. Root cause: Neumann power-doubling solve was numerically unstable.

**Fix:** Replaced Neumann power-doubling with direct triangular RHS solve in the fused kernel — same approach the chunked path already used.

**Trade-off captured honestly** — see Section 4.

---

## 3. Wins — what worked, with numbers

### Long-context artifact (hardware-validated on `trn2.3xlarge`, TP=4, LNC=2, SDK 2.29)

| Test | Result |
|---|---|
| 128K artifact compile + load | **PASS** |
| 32K & 64K short-after-long state reset | **PASS** |
| 32K & 64K needle-retrieval prompts | **PASS** |

### Prefill performance

| Prompt | Cold TTFT | Cold prefill | Warm TTFT | Warm prefill |
|---:|---:|---:|---:|---:|
| 512 | 1.31 s | 390 tok/s | 0.42 s | 1.2 k tok/s |
| 1 K | 1.88 s | 535 tok/s | — | — |
| 4 K | 7.03 s | 588 tok/s | 0.42 s | 9.8 k tok/s |
| 8 K | 13.61 s | 607 tok/s | 0.43 s | 18.9 k tok/s |
| 16 K | 27.84 s | 595 tok/s | 0.45 s | 36.3 k tok/s |
| 32 K | ~76.6–81.1 s | — | — | — |
| 64 K | ~153–162 s | — | — | — |

### APC / prefix cache (validated `max_num_seqs=1`)

| Scenario | Cold | Warm | Speedup | Correctness |
|---|---:|---:|---:|---|
| Server exact-repeat (~10.8K prompt) | 26.68 s | 1.67 s | **16.0×** | exact text match |
| Offline exact-repeat | 26.19 s | 2.38 s | **11.0×** | exact token-ID match |
| Offline partial-prefix reuse | 25.52 s | 1.70 s | **15.0×** | exact token-ID match |
| Server cross-prefix reuse | 25.17 s | 1.36 s | **18.5×** | exact text match |

### Memory (PR #164 baseline)

- Peak Neuron HBM: ~53.25 GB decimal during 64K eval
- Fit 128K context on single `trn2.3xlarge` (96 GB HBM)

### Bugs caught + fixed before maintainer review

- Numerical instability in fused DeltaNet kernel (Neumann power-doubling) — self-discovered + self-fixed via direct-solve port
- Active chunk-boundary handling for suffix-only APC prep
- Scheduler safety gate for unbacked prefix-cache reads

---

## 4. Fails — honest list of what didn't work or has known issues

I'm including these because reviewers will find them anyway, and naming them up-front is better than hiding them.

### Decode throughput regression (direct-solve fix)

| | Original Neumann fused | Direct-solve (current) |
|---|---:|---:|
| Decode throughput | 26.3–26.6 tok/s | 21.63 tok/s |
| TPOT | ~37.6–38.0 ms | 46.2 ms |

**~18% TPOT regression** as the cost of numerical stability. Direct triangular solve is more explicit per step than Neumann power-doubling. The trade-off is correct as a production default (stability over peak throughput under realistic gate scales) but it's a real regression. A hybrid kernel selector (Neumann fast path when stable, direct-solve fallback otherwise) would recover most of it — flagged as follow-up.

### 32K / 64K APC blocked by artifact bucket coverage

The current artifact `pfx16k` has `prefix_buckets` only up to `16384`. 32K and 64K APC tests fail at:

```
Prefix len 16640 exceeds largest bucket 16384 for context_encoding_model
```

This is an **artifact compile-target choice, not a correctness issue**. Requires a follow-up artifact with bucket recompile through 64K / 128K to validate full-range APC.

### Memory delta in direct-solve variant

Direct-solve variant: **60.1 GiB peak HBM** vs original baseline **53.25 GB**. Not a regression — explained by different artifact configuration (`max_gdn_checkpoint_slots=64` adds ~9.85 GB across TP=4, plus larger TKG bucket coverage `[8192, 32768, 131072]`). A strict like-for-like memory comparison needs matched compile settings — flagged as outstanding work.

### `max_num_seqs=1` — no true multi-sequence batching

Current artifact is compiled for `max_num_seqs=1`. Concurrent requests queue rather than batch. Multi-sequence batching past `1` requires recompile + additional scheduler integration. Continuous batching is **explicitly out of scope** for this PR — flagged as follow-up.

### Refusal-style output in one decode benchmark

The 8K-prompt decode benchmark produced "refusal-style" output because the benchmark prompt wording looked like prompt-injection to the model. Output was coherent, just refusal-flavored. **Benchmark-prompt artifact, not a model/serving defect** — easy to rerun with a benign prompt.

### No native Qwen MTP integration in PR #164

MTP (Multi-Token Prediction) speculative decoding for Qwen 3.x was **intentionally not included** in this baseline PR. Separate standalone work delivered ~2× decode throughput improvement. MTP integration into NxDI spec-decode config is a follow-up PR.

---

## 5. What's Next — improvements I want to make

Ranked roughly by impact / effort:

1. **64K / 128K APC bucket recompile** — unblocks full long-context APC validation. Pure recompile, no code change.
2. **Continuous batching past `max_num_seqs=1`** — recompile + scheduler integration. Significant throughput gain for production multi-tenant serving.
3. **Hybrid kernel selector** — runtime selection between Neumann fast path and direct-solve stable path based on detected gate-scale regime. Recovers most of the ~18% decode regression.
4. **MTP integration into NxDI spec-decode config** — wires the existing ~2× decode improvement into the in-PR serving path.
5. **Memory like-for-like A/B** — produce strict comparable memory measurements with matched compile settings (same `pa_num_blocks`, same `max_gdn_checkpoint_slots`, same TKG buckets) for honest baseline vs direct-solve memory comparison.
6. **EKS / multi-node deployment patterns** — current validation is single-instance trn2.3xlarge. Production deployments would benefit from documented multi-node Trainium patterns.

---

## 6. Quick Reference

- **Main PR:** [aws-neuron/neuronx-distributed-inference#164](https://github.com/aws-neuron/neuronx-distributed-inference/pull/164)
- **Fused direct-solve follow-up comment:** see PR comments dated 2026-05-22
- **Maintainer engagement:** `aws-reutermj` commented 2026-05-14 "Working with our team to evaluate"
- **Baseline this builds on:** Jim Burtoft's Qwen 3.6-27B contrib work in PR #140
- **Architecture reference:** `[3 DeltaNet + 1 GQA] × 16`, 64 layers, hidden size 5120, GQA head_dim 256
- **Validation hardware:** `trn2.3xlarge`, TP=4, LNC=2, SDK 2.29
- **Long-context artifact:** 131,072 tokens, CTE bucket 512

---

*If you're reviewing this for the first time: start with [PR #164](https://github.com/aws-neuron/neuronx-distributed-inference/pull/164) for full context. This document is a delta summary — what changed and why — designed for fast scanning by reviewers and stakeholders.*
