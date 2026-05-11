# Qwen3.6-27B Hybrid on Trainium2 — Optimization Arc

This document captures the full optimization arc for the Qwen3.6-27B hybrid
(GatedDeltaNet + GQA) model on AWS Trainium2 (trn2.3xlarge, TP=4, LNC=2).

It exists to:
1. Document the validated baselines and the gains that produced them.
2. Capture the negative results that closed off optimization directions.
3. Identify the remaining headroom and the next real win.

## Architecture

- **Model:** Qwen3.6-27B hybrid
- **Layers:** 64 total — 48 GatedDeltaNet (linear attention) + 16 GQA (full attention)
- **head_dim:** 256 (full attention); 128 (DeltaNet head_k/head_v)
- **MLP:** dense SwiGLU, intermediate_size=17408, 64 layers
- **Hardware:** trn2.3xlarge, 1 trn2 chip, 8 NeuronCore-v3, TP=4 + LNC=2
- **SDK:** Neuron 2.29, NKI 0.3.0
- **Precision:** BF16 weights + fp32 GDN recurrent state + fp32 conv state; FP8 for MLP weights (MLP-only)

## Baselines

| Tag | Commit | Prefill (tok/s @ 16K) | Decode (tok/s) | What changed |
|---|---|---:|---:|---|
| `qwen36-27b-baseline-v1` | `99e24fc` | 150 | 18 | Original 64K chunked prefill, validated cosine ≥0.999 vs HF reference |
| `qwen36-27b-baseline-v2` | `4b73970` | 461 | 27 | Sharding (TP=4) + chunked prefill in OpenAI server + active-length attention mask fix |
| `qwen36-27b-vllm-apc-baseline-v3` | `abc011e` | 418 | 27 | vLLM integration with chunked prefill + Automatic Prefix Caching (44–65% hit rate on real traffic) |

**v1 → v3 progression: 2.8× prefill, 1.5× decode.**

The v3 artifact is the current production target. Prefill regresses slightly
vs v2 because vLLM's chunked-prefill scheduler adds modest overhead, but
gains continuous batching, APC, and a production-grade OpenAI-compatible
server in exchange.

## What worked

| Optimization | Effect | Notes |
|---|---|---|
| DeltaNet projection sharding (TP=4) | **3× prefill** (150 → 461 tok/s) | Biggest single win. ColumnParallelLinear/RowParallelLinear replacing nn.Linear for in_proj_qkv, in_proj_z, in_proj_a, in_proj_b, out_proj, conv1d_weight, A_log_weight, dt_bias_weight |
| Active-length attention mask | **1.88× decode** (14.3 → 27 tok/s) | Server bug: full 131K mask forced largest TKG bucket every step. Pass `attention_mask[:, :active_len]` instead |
| vLLM chunked prefill on Neuron | Enabled production serving | AWS docs listed chunked prefill as unsupported on Neuron vLLM. Wired NxDI's chunked CTE path to vLLM's chunked-prefill scheduler via `is_block_kv_layout=True` + `chunked_prefill_config` |
| vLLM APC with `mamba-cache-mode align` | 5–10× TTFT on warm hits | 44–65% real-traffic hit rate, burst throughput 1000–4600 tok/s |
| FP8 MLP-only (NxDI native) | ~16 GB HBM savings, 1.06× server latency | FP8 is bandwidth-relevant for decode, not compute-relevant for prefill. Kept GDN + attention BF16 |

## What didn't work (closed-off directions)

Seven kernel-level optimization attempts. All produced either no gain, marginal
gains that fail strict gates, or active regression. The data is conclusive:
**the current chunked DeltaNet kernel is near its local optimum on Trainium2.**

| Attempt | Branch | Result | What it ruled out |
|---|---|---|---|
| Compact inputs | `codex/qwen36-gdn-core-rewrite` | 7× compile speedup, **no runtime gain** | Input staging is not the bottleneck; compiler already elides broadcasts |
| Direct RHS solve v1 | `codex/qwen36-gdn-direct-rhs-solve` | 1.13× prefill, **fails bit-exact greedy** | Algebraic rewrite changes float accumulation order; compiler optimizes both forms to ~identical execution |
| Materialized-N cleanup | `codex/qwen36-gdn-materialized-n-cleanup` | **No gain, fails greedy** | `nc_transpose` ≠ `nc_matmul(eye)` bit-wise; redundant masks contribute to bits |
| Blocked triangular solve (16-block VE) | `codex/qwen36-gdn-blocked-solve` | **4× slower than baseline** | Vector Engine row-shuffle is dramatically worse than 128 TE matmuls on Trainium2 |
| Direct RHS solve v2 | `codex/qwen36-logits-validation` | **Performance-neutral**, cosine + MMLU + GSM8K all pass | Compiler optimizes algebraic reorderings to identical execution |
| Stable fused GDN prefill | `codex/qwen36-gdn-stable-fused-prefill` | **Performance-neutral**, correctness clean | Fused-vs-chunked structure is not the bottleneck |
| Head-grouped GDN megakernel | `codex/qwen36-gdn-headgroup-megakernel` | **No gain** | Dispatch count is not the bottleneck (confirmed by no-op ablation) |
| MLP NKI kernel | `codex/qwen36-mlp-nki-kernel` | **No gain** | MLP is only 4% of TTFT (confirmed by ablation) |

### Ablation evidence

Two no-op ablation studies decomposed where the 16K TTFT time actually goes:

| Component | Share of 16K TTFT | Source |
|---|---:|---|
| **GDN recurrent solve** | **~60%** | `codex/qwen36-gdn-solve-ablation` |
| GDN surrounding (projections, conv1d, l2norm, gate, out_proj) | ~6% | `codex/qwen36-cte-bottleneck-ablation` |
| Attention (16 layers, head_dim=256 non-flash) | ~26% | same |
| MLP (64 SwiGLU layers) | ~4% | same |
| Other | ~4% | residual |

The solve dominates. The launches inside the solve dominate the solve. The TE
pipeline on Trainium2 amortizes those launches well — better than any
restructuring we attempted.

## Hardware utilization on baseline v3

At batch=1 single-stream:
- **Tensor Engine compute**: ~2.5% of peak (prefill)
- **HBM bandwidth**: ~22% of peak (decode)
- **HBM capacity**: ~98 GB peak / 96 GB available with FP8 MLP

The low compute utilization is structural for batch=1 — the hardware was
designed for batched serving. At `max-num-seqs=8`+ in vLLM, aggregate
throughput reaches 1000–4600 tok/s burst (confirmed in APC measurements).

## Remaining headroom — informed by NVIDIA's published playbook

Kernel-internal prefill optimization is settled. The optimizations NVIDIA uses to
get 2.5× decode on this exact model class on B200 are documented in the vLLM
Qwen3-Next blog and the Google GKE 1M tok/s benchmark. Most are portable.

| Direction | Target | Realistic gain | Effort | Portable? |
|---|---|---|---|---|
| **MTP speculative decoding** (`qwen3_next_mtp`, 2-3 tokens, ~90% accept on NVIDIA) | Decode tok/s | **2-2.5× (27 → 55-70 tok/s)** | 1-2 weeks | **Yes — MTP heads ship in Qwen3.6-27B weights, NxDI supports spec decode** |
| **FP8 KV cache** | Decode bandwidth | 1.1-1.3× decode | 1 week | Yes — NxDI native |
| **FP8 attention QKV/O** | Decode bandwidth | 1.1× decode | 1 week | Yes — NxDI native |
| **APC hit-rate tuning** (`mamba-cache-mode all` vs `align`) | Warm prefill | Push 44-65% → 80-95% hit rate | 3-5 days | Yes — plumbing exists |
| **Multi-stream throughput** (`max-num-seqs=8/16`) | Aggregate req/s | 2-5× requests/sec | 2-3 days | Yes — already wired |
| **Data parallelism DP=2** (replicate model, reduce TP) | Aggregate throughput | 2× requests/sec at higher TP cost | 1 week | Yes — NeuronX supports DP |
| **NKI flash attention for head_dim=256** | Attention layers (26% of TTFT) | ~1.1-1.2× prefill | 2-3 weeks, medium risk | No equivalent in NKI 0.3.0 — would be custom |
| **CUDA graphs equivalent** | Launch overhead | Marginal on Trainium | — | No direct equivalent on NeuronX |

**The biggest single accessible win is MTP speculative decoding.** This is what
NVIDIA uses to get their headline decode numbers on this model. The MTP heads
ship in the model weights — no separate draft model needed.

## How this compares to NVIDIA

Honest gap analysis vs published H100/B200 numbers for Qwen3.6-27B / Qwen3-Next:

| Metric | Our trn2.3xlarge (v3) | NVIDIA published |
|---|---|---|
| **Single-stream prefill, 16K** | 418 tok/s | Not published; estimated 1500-3000 tok/s BF16, 3000-5000 tok/s FP8 |
| **Single-stream decode, batch=1** | 27 tok/s | DGX Spark v4 DFlash: 37 tok/s; per-B200 in batched workload: not separately reported |
| **Aggregate throughput, 8 streams** | Not yet measured | Google GKE per-B200: ~12K tok/s aggregate input+output |
| **Production stack** | NxDI + vLLM + chunked prefill + APC | vLLM + FlashInfer + FP8 KV + MTP spec + CUDA graphs |
| **MTP spec decoding** | Not yet wired | Wired, 2.5× decode, ~90% acceptance |
| **FP8 coverage** | MLP only | Full: QKV/O + KV cache + MLP |
| **Flash attention head_dim=256** | Non-flash path | FlashInfer native |

Single-stream prefill is ~5-10× behind NVIDIA. Decode is ~1.4× behind a comparable
NVIDIA setup. The decode gap is closeable via MTP + FP8 KV; the prefill gap is
partially closeable via FlashInfer-equivalent NKI work, but full parity requires
NKI maturity that's still years away.

**Important**: NVIDIA's vLLM Qwen3-Next blog explicitly states *"Further kernel
optimizations for GatedDeltaNet layers are on the roadmap."* Even on CUDA, the
GDN kernel is not fully optimized. The "NVIDIA has solved this" intuition is
overstated.

## Compile profile (baseline v3)

| Phase | Time |
|---|---|
| HLO generation (CTE + TKG) | ~14s |
| TKG NEFF compile | ~290s |
| CTE NEFF compile | ~310s |
| Weight layout optimization | ~18s |
| **Total build** | **~17 min** |
| Sharding (TP=4) | ~160s |
| Weight load | ~90s |
| Warmup | ~4s |

Artifact size on disk: **~82 GB** sharded NEFF + weights.

## File map

- **Modeling**: `contrib/models/Qwen3.6-27B/src/modeling_qwen35.py`
- **NKI kernels**: `contrib/models/Qwen3.6-27B/src/nki_kernels/nki_deltanet_chunked.py` (production)
- **Compile harness**: `/opt/dlami/nvme/qwen36_27b_compile_baseline.py` (on instance)
- **vLLM scaffold**: `contrib/models/Qwen3.6-27B/vllm/`
  - `start_baseline_v3.sh` — production launcher
  - `start_vllm_server.sh` — generic launcher
  - `qwen36_chat_proxy.py` — OpenAI-compatible proxy with thinking-mode-off default
  - `serve_qwen36.py` — vLLM CLI wrapper
  - `patch_nxdi_registry.py` — registers `qwen3_5` model type
- **Validation harnesses**: `inferentia-gdn/validation_scripts/`
  - `qwen36_27b_chat_quality_eval.py` — quality benchmarks
  - `qwen36_27b_vllm_chunked_eval.py` — chunked prefill perf
  - `qwen36_27b_vllm_concurrency_eval.py` — multi-stream throughput

## Decision: optimization arc is closed for kernel-internal prefill

Seven independent kernel-level attempts is sufficient evidence. The TE
pipeline on Trainium2 with the current chunked DeltaNet kernel is at a local
optimum. Further kernel rewrites have diminishing-returns probability and
should not be the next milestone.

The next milestone is **EAGLE speculative decoding** for decode tok/s gains.
After that, FP8 KV cache. After that, APC hit-rate tuning.

Tag the current state as the baseline for spec-decode work. Move on.

## Next baselines (planned, ordered by ROI)

| Tag | Target | Expected | Cite |
|---|---|---|---|
| `qwen36-27b-mtp-v1` | Decode 55-70 tok/s (2-2.5×) | After wiring native MTP heads to NxDI spec decode | NVIDIA achieves this on B200; MTP heads ship in Qwen3.6-27B weights |
| `qwen36-27b-fp8-kv-v1` | Decode +1.1-1.3× on top of MTP | After FP8 KV cache | Google GKE 1M tok/s uses this |
| `qwen36-27b-aggregate-throughput-v1` | 2-5× requests/sec | Multi-stream measurement at `max-num-seqs=16` | Aggregate is the fair benchmark vs NVIDIA's batched numbers |
| `qwen36-27b-production-v1` | All decode optimizations stacked | 60-80 tok/s decode, 418+ tok/s prefill, APC ≥80% hit rate | — |

## References

- vLLM Qwen3-Next blog: https://vllm.ai/blog/qwen3-next
- vLLM Qwen3.6-27B recipe: https://recipes.vllm.ai/Qwen/Qwen3.6-27B
- Google GKE 1M tok/s on B200: https://medium.com/google-cloud/1-million-tokens-per-second-qwen-3-5-27b-on-gke-with-b200-gpus-161da5c1b592
- NVIDIA Qwen3-Next blog: https://developer.nvidia.com/blog/new-open-source-qwen3-next-models-preview-hybrid-moe-architecture-delivering-improved-accuracy-and-accelerated-parallel-processing-across-nvidia-platform/
- Flash Linear Attention (FLA, sustcsonglin/flash-linear-attention): the Triton kernel library both vLLM-CUDA and lucebox-hub reference for GDN compute. Algorithm reference is portable; CUDA kernels are not.
