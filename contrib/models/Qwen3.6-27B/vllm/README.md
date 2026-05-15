# Qwen3.6-27B vLLM on Neuron

This folder contains the first-pass vLLM integration helpers for the
Qwen3.6-27B contrib model.

The current goal is **vLLM serving through the Neuron/NxDI plugin** for the
validated Qwen3.6 artifact, including long prompts through vLLM's native
chunked-prefill scheduler.

## Which vLLM Neuron Package?

Use the vLLM-on-Neuron environment that matches the installed Neuron SDK first.
For SDK 2.29, the AWS Neuron guide lists the NxDI/vLLM plugin stack as
`vLLM 0.16.0` with plugin version `0.5.0`. The
`vllm-project/vllm-neuron` repository is useful source/reference material, but
its README currently describes a beta plugin path tied to older `vLLM 0.11.0`
and SDK 2.26.1. Do not downgrade the working SDK 2.29 environment just to use
that repository.

On a DLAMI, prefer the preinstalled vLLM/Neuron environment when available. If
the instance does not have one, install the Neuron-compatible vLLM plugin/fork
using the current AWS guide, then run the contrib registry patch below.

## What Works First

- Register the contrib `qwen3_5` text model with the NxDI model registry inside
  the vLLM environment.
- Start vLLM with `VLLM_PLUGINS=neuron`.
- Load a small-context model or a precompiled artifact with
  `NEURON_COMPILED_ARTIFACTS`.
- Run a short OpenAI-compatible smoke prompt.

## Hybrid APC Production Boundary

Qwen3.6-27B is a hybrid model, so attention prefix caching alone is not a
complete production APC contract. The current stack has strong serving
primitives for attention-only models: block KV, block tables, slot mapping,
prefix caching, continuous batching, chunked prefill, and decode. For hybrid
attention plus GDN recurrence, the attention KV path is production-shaped, but
the GDN state path is still model-specific glue.

Current readiness:

| Layer | Current support | Production readiness |
| --- | ---: | ---: |
| Attention KV cache | Good | High on the existing NxDI/vLLM block-KV path |
| vLLM APC for attention blocks | Working baseline | Medium/high |
| GDN recurrent state cache | Implemented locally | Low/medium |
| GDN conv state cache | Implemented locally | Low/medium |
| Hybrid APC across attention + GDN | Not fully implemented | Low |
| Continuous batching with exact hybrid prefix reuse | Not supported by the local manager | Low |
| Speculation, FP8 cache, tiling, flash decode with hybrid state | Explicitly rejected by the local manager | Low |

The `HybridDeltaNetCacheManager` is therefore a contrib-local static/stateful
cache manager, not a production hybrid APC manager. It proves the model can
preserve recurrent and conv state, but it is batch-row based rather than
vLLM block-hash, refcount, eviction, and tenant-isolation based.

Production hybrid APC must define the usable prefix as the intersection of:

1. attention KV block hit;
2. GDN recurrent prefix-boundary checkpoint hit;
3. GDN conv prefix-boundary checkpoint hit.

For each GDN layer, the reusable checkpoint object needs:

```text
recurrent_state: [local_value_heads, key_dim, value_dim]
conv_state:      [conv_dim, conv_kernel_size - 1]
```

The recurrent state should stay FP32 for exact cold-vs-warm agreement until
BF16 equivalence is proven. Conv state can follow the model-compatible dtype,
but exactness still needs token-level validation. If the attention APC hit lands
inside a GDN checkpoint interval, restore the nearest earlier full GDN
checkpoint, replay the residual tokens, then run the suffix.

The launchers expose `--enable-hybrid-apc` and explicit hybrid cache dtype
knobs. In the current v0 implementation, `use_hybrid_apc_manager=True` creates
a bounded GDN checkpoint-slot bank and adds restore/commit tensors to the model
signature. The serving request-prep path must still fill those tensors from the
vLLM/NxDI cumulative-prefix hash lifecycle; otherwise the default zero masks run
as attention KV plus normal active-row GDN state with no GDN checkpoint reuse.
For v0, `gdn_checkpoint_interval` must equal the vLLM block size.

The production server launcher enables strict hybrid APC metadata by default.
That means request prep must provide vLLM/NxDI cumulative prefix hashes and real
attention block refs; local token-hash fallback is reserved for controlled
validation via `--allow-hybrid-apc-local-hash-fallback`. The live scheduler
integration should pass the full prompt before suffix slicing using
`hybrid_full_input_ids`/`full_input_ids`, attach `vllm_attention_hit_len`, pass
`cumulative_hashes_by_prefix_len`, and pass actual attention block refs at
commit time through `actual_attention_block_refs` or
`hybrid_actual_attention_block_refs`. Attention KV eviction should call the
model/store `on_attention_block_evicted` callback so GDN checkpoints do not
outlive the KV blocks they depend on.

## Chunked Prefill Note

The Neuron plugin disables vLLM chunked prefill by default and installs a custom
continuous-batching scheduler. For this Qwen3.6 artifact we need vLLM's native
chunked-prefill scheduler so prompts longer than the 512-token context graph are
fed to the precompiled model in 512-token chunks. The launcher sets
`DISABLE_NEURON_CUSTOM_SCHEDULER=1` when `--enable-vllm-chunked-prefill` is
passed. It also launches with `--generation-config vllm` so model
`generation_config.json` does not silently override deterministic sampling
defaults.

## Install The Contrib Registry Patch

Activate the vLLM/Neuron environment on the instance, then run:

```bash
cd /home/ubuntu/inferentia-gdn
contrib/models/Qwen3.6-27B/vllm/install_qwen36_vllm.sh
```

If your vLLM environment is not in a standard location:

```bash
contrib/models/Qwen3.6-27B/vllm/install_qwen36_vllm.sh \
  /opt/aws_neuronx_venv_pytorch_2_9_nxd_inference
```

The installer only patches the active environment. It does not modify core repo
files.

## Start vLLM

Small-context compile/load path:

```bash
contrib/models/Qwen3.6-27B/vllm/start_vllm_server.sh \
  --model-path /opt/dlami/nvme/models/Qwen3.6-27B \
  --max-model-len 512 \
  --port 8000
```

Precompiled artifact path:

```bash
contrib/models/Qwen3.6-27B/vllm/start_vllm_server.sh \
  --model-path /opt/dlami/nvme/models/Qwen3.6-27B \
  --compiled-artifacts /opt/dlami/nvme/qwen_artifacts/qwen36_27b_128k_fp8_mlp_only_vllm_statereset_run1 \
  --max-model-len 131072 \
  --seq-len 131072 \
  --cte-buckets 128,256,512 \
  --port 8000
```

Cold-prefill bucket waste is the first performance target. CTE buckets must stay
128-aligned because the fused DeltaNet CTE path operates in 128-token chunks.
Use one of the explicit profiles when compiling artifacts:

```bash
# Short-prompt latency
--cte-bucket-profile short     # [128,256,512,1024]

# General production
--cte-bucket-profile general   # [256,512,1024,2048]

# Long-context artifact
--cte-bucket-profile long      # [4096,8192,16384,32768]

# 262K load experiment
--cte-bucket-profile 262k      # [256]
```

`--cold-zero-conv-fast-path` is a manual ablation flag, not part of the
strict-final Trainium path. The guard is disabled while Torch is tracing so a
compiled graph cannot bake in the zero-prefix branch and then reuse it for a
continuation chunk or partial-prefix suffix. Leave it disabled for the first
strict-final run and for APC/partial-prefix serving unless separate zero-prefix
and stateful CTE graphs are introduced.

The launcher defaults to `TEXT_ONLY_CTE=1`, which is the intended profile for
this cold-prefill benchmark. A service that accepts real vision inputs needs a
separate multimodal artifact/config launched with `--no-text-only-cte`.

Long-prompt precompiled artifact path:

```bash
contrib/models/Qwen3.6-27B/vllm/start_vllm_server.sh \
  --model-path /opt/dlami/nvme/models/Qwen3.6-27B \
  --compiled-artifacts /opt/dlami/nvme/qwen_artifacts/qwen36_27b_128k_fp8_mlp_only_vllm_statereset_run1 \
  --max-model-len 131072 \
  --seq-len 131072 \
  --cte-buckets 256,512 \
  --block-size 256 \
  --enable-vllm-chunked-prefill \
  --port 8000
```

Native vLLM prefix-cache experiment:

```bash
contrib/models/Qwen3.6-27B/vllm/start_vllm_server.sh \
  --model-path /opt/dlami/nvme/models/Qwen3.6-27B \
  --compiled-artifacts /opt/dlami/nvme/qwen_artifacts/qwen36_27b_128k_fp8_mlp_only_vllm_statereset_run1 \
  --max-model-len 131072 \
  --seq-len 131072 \
  --cte-buckets 256,512 \
  --block-size 128 \
  --enable-vllm-chunked-prefill \
  --enable-prefix-caching \
  --gdn-checkpoint-interval 256 \
  --hybrid-gdn-recurrent-cache-dtype float32 \
  --hybrid-gdn-conv-cache-dtype bfloat16 \
  --mamba-cache-mode all \
  --mamba-ssm-cache-dtype float32 \
  --port 8000
```

Treat this as an experiment, not a production mode, until validation passes.
Standard vLLM APC reuses attention KV blocks; Qwen3.6 also needs DeltaNet
recurrent state and conv state as prefix-boundary checkpoints keyed by the
cumulative prefix hash. If native APC does not produce exact greedy matches and
a clear warm-hit speedup, the next step is a hybrid APC path that restores those
GDN checkpoints alongside attention KV.

For APC experiments, do not treat `256` as the only block size. It can be useful
for long-context amortization, but it is coarse for chat-style prefix reuse.
Run explicit sweeps at `64` and `128`; include `32` when hit granularity matters
enough to justify possible block-table/layout overhead. Keep the GDN checkpoint
interval separate from the attention block size.

Immediate Trainium experiments:

```text
262K TP=4, block_size=256, CTE buckets [256]
262K TP=4, block_size=128, CTE buckets [256]
128K TP=4, block_size=128, CTE buckets [256,512]
128K TP=4, block_size=256, CTE buckets [256,512]
```

Production chat proxy:

```bash
contrib/models/Qwen3.6-27B/vllm/start_vllm_server.sh \
  --model-path /opt/dlami/nvme/models/Qwen3.6-27B \
  --compiled-artifacts /opt/dlami/nvme/qwen_artifacts/qwen36_27b_128k_fp8_mlp_only_vllm_statereset_run1 \
  --max-model-len 131072 \
  --seq-len 131072 \
  --cte-bucket 512 \
  --block-size 256 \
  --enable-vllm-chunked-prefill \
  --port 8001
```

Then expose the guarded OpenAI-compatible endpoint on port 8000:

```bash
python contrib/models/Qwen3.6-27B/vllm/qwen36_chat_proxy.py \
  --model-path /opt/dlami/nvme/models/Qwen3.6-27B \
  --backend-url http://127.0.0.1:8001 \
  --port 8000
```

The proxy forces `chat_template_kwargs={"enable_thinking": false}` for
`/v1/chat/completions` by default. It rejects raw `/v1/completions` because raw
prompts bypass the Qwen chat template and can pollute the hybrid model state.
It also hoists `system` and `developer` messages to a single leading `system`
message because the Qwen chat template rejects system messages that appear later
in the conversation. Use `--allow-thinking` or `--allow-completions` only for
explicit debugging.

When `QWEN36_COLD_PREFILL_CONFIG` is present from the launcher, the proxy logs
`COLD_PREFILL_PROXY_REQUEST` and `COLD_PREFILL_PROXY_RESPONSE` rows with prompt
length, selected CTE buckets, padding, tile sizes, block size, feature flags,
and backend latency. Passing `--model-path` lets the proxy count prompt tokens
with the tokenizer instead of a whitespace fallback.

Offline long-prompt smoke:

```bash
python contrib/models/Qwen3.6-27B/vllm/run_offline_inference.py \
  --model-path /opt/dlami/nvme/models/Qwen3.6-27B \
  --compiled-artifacts /opt/dlami/nvme/qwen_artifacts/qwen36_27b_128k_fp8_mlp_only_vllm_statereset_run1 \
  --max-model-len 131072 \
  --seq-len 131072 \
  --cte-bucket 512 \
  --block-size 256 \
  --enable-vllm-chunked-prefill \
  --chat \
  --prompt "$(python - <<'PY'
print('Summarize this document in one paragraph. ' + 'Neuron inference ' * 700)
PY
)"
```

Offline token-exact prefix-cache validation:

```bash
python validation_scripts/qwen36_vllm_prefix_cache_offline.py \
  --repo-root /home/ubuntu/inferentia-gdn \
  --model-path /opt/dlami/nvme/models/Qwen3.6-27B \
  --compiled-artifacts /opt/dlami/nvme/qwen_artifacts/qwen36_27b_128k_fp8_mlp_only_vllm_statereset_run1 \
  --max-model-len 131072 \
  --seq-len 131072 \
  --cte-bucket 512 \
  --block-size 128 \
  --enable-vllm-chunked-prefill \
  --mamba-cache-mode all
```

Offline partial-prefix validation:

```bash
python validation_scripts/qwen36_vllm_prefix_cache_partial_offline.py \
  --repo-root /home/ubuntu/inferentia-gdn \
  --model-path /opt/dlami/nvme/models/Qwen3.6-27B \
  --compiled-artifacts /opt/dlami/nvme/qwen_artifacts/qwen36_27b_128k_fp8_mlp_only_vllm_statereset_run1 \
  --max-model-len 131072 \
  --seq-len 131072 \
  --cte-bucket 512 \
  --block-size 128 \
  --enable-vllm-chunked-prefill \
  --mamba-cache-mode all
```

Server-side prefix-cache validation through the guarded proxy:

```bash
python validation_scripts/qwen36_prefix_cache_validation.py \
  --base-url http://127.0.0.1:8000 \
  --model qwen3.6-27b-neuron-128k-fp8-mlp
```

The acceptance gate is strict: repeated greedy calls must produce identical
output, and warm-hit latency should be materially lower than cold-fill latency.
For hybrid Qwen3.6, prefix-cache validation is not complete until the GDN
recurrent/conv state behavior is proven, not just attention KV cache hits.

Hybrid APC exactness and HBM harness:

```bash
python validation_scripts/qwen36_hybrid_apc_validation.py exactness \
  --model-path /opt/dlami/nvme/models/Qwen3.6-27B \
  --compiled-artifacts /opt/dlami/nvme/qwen_artifacts/qwen36_hybrid_apc \
  --seq-len 2048 \
  --cte-buckets 256,512 \
  --block-size 256 \
  --gdn-checkpoint-interval 256 \
  --enable-vllm-chunked-prefill

python validation_scripts/qwen36_hybrid_apc_validation.py hbm \
  --context-lens 131072 262144 \
  --checkpoint-intervals 128 256 512
```

The exactness harness emits `COLD_PREFILL_METRICS` JSON lines for the cold
full-prefix and partial-prefix requests before the final report. Those lines use
the same metric schema as the offline benchmark runner and include HBM usage
when the Trainium runtime exposes it, so they can be archived with the
acceptance evidence.

Native APC validation run on Trn2 with the FP8 128K artifact:

- server exact-repeat, `~10.8K` prompt tokens: `26.68s` cold to `1.67s` warm,
  `16.0x` speedup, exact greedy text match;
- offline exact-repeat, token IDs exposed: `26.19s` cold to `2.38s` warm,
  `11.0x` speedup, exact greedy token-ID match;
- offline partial-prefix reuse, token IDs exposed: `25.52s` no-cache target to
  `1.70s` APC target after a different shared-prefix warmup request, `15.0x`
  speedup, exact greedy token-ID match.
- server hardening, exact repeat: `25.38s` cold to `1.55s` warm, `16.35x`
  speedup, exact text match;
- server hardening, cross-prefix reuse after unrelated prefix: `25.17s` cold to
  `1.36s` warm, exact text match;
- shared-prefix concurrency at 1/2/4 requests returned all requested markers
  exactly; the artifact still queues because it is compiled for `max_num_seqs=1`.

Validation run on Trn2 with the FP8 128K artifact:

- state-reset artifact: `/opt/dlami/nvme/qwen_artifacts/qwen36_27b_128k_fp8_mlp_only_vllm_statereset_run1`;
- OpenAI-compatible `/v1/chat/completions` behind the proxy passes focused
  quality checks without callers passing `chat_template_kwargs`;
- repeated short-after-long validation passes after 32K and 64K requests,
  confirming DeltaNet recurrent/conv state is reset for new requests;
- 32K and 64K needle retrieval prompts return all expected codes;
- measured prefill is `404-428 tok/s` from 512 through 64K prompt tokens;
- measured decode is `26.3-26.6 tok/s`;
- peak Neuron device memory is about `53.25 GB` decimal for the 64K eval.

Raw `/v1/completions` prompts are not chat-templated and can pollute the hybrid
state if sent directly to the backend. Keep the backend private and expose the
proxy on the public port for production calls.

## Offline Smoke

```bash
python contrib/models/Qwen3.6-27B/vllm/run_offline_inference.py \
  --model-path /opt/dlami/nvme/models/Qwen3.6-27B \
  --compiled-artifacts /opt/dlami/nvme/qwen_artifacts/qwen36_27b_128k_fp8_mlp_only_vllm_statereset_run1 \
  --max-model-len 131072 \
  --seq-len 131072 \
  --cte-buckets 128,256,512 \
  --chat \
  --prompt "What is 17 * 23? Answer with the number only."
```

## Cold-Prefill Benchmark

Use launcher dry-run mode to inspect generated config without starting vLLM:

```bash
USE_PYTORCH_CHUNK=1 contrib/models/Qwen3.6-27B/vllm/start_vllm_server.sh \
  --model-path /opt/dlami/nvme/models/Qwen3.6-27B \
  --cte-bucket-profile short \
  --dry-run
```

Run the cold-prefill matrix with deterministic one-token generation and write
JSON output for acceptance gates:

```bash
python validation_scripts/qwen36_cold_prefill_benchmark.py \
  --model-path /opt/dlami/nvme/models/Qwen3.6-27B \
  --compiled-artifacts /opt/dlami/nvme/qwen_artifacts/qwen36_27b_2k \
  --compiled-artifacts-by-len \
    8192=/opt/dlami/nvme/qwen_artifacts/qwen36_27b_8k \
    32768=/opt/dlami/nvme/qwen_artifacts/qwen36_27b_32k \
  --prompt-lengths 128 256 384 512 1024 2048 8192 32768 \
  --repetitions 5 \
  --output-json /tmp/qwen36_cold_prefill_matrix.json
```

By default the benchmark emits both `--max-tokens 1` rows for pure prefill
timing and `--max-tokens 32` rows for end-to-end timing. Override with
`--max-tokens-values` when narrowing a run. Use `--repetitions` to collect
enough samples for the acceptance report's p50/p95 latency summaries. When the
model tokenizer can be loaded from `--model-path`, prompt files are calibrated
to the requested target lengths and rows record `prompt_token_count`; otherwise
the runner's `actual_prompt_len` metric is used by acceptance. Runtime failures
are kept as rows and written to `--output-json` so artifact-load and DMA-spill
gates have evidence; pass `--fail-fast` only for interactive debugging. Use
`--compiled-artifacts-by-len LEN=PATH ...` when one matrix spans artifacts
compiled for different `seq_len` values; rows record the selected artifact path
so baseline/candidate comparisons can enforce the same prompt/model/artifact
inputs.

For long-context artifact checks, include the explicit 128K and 262K candidate
profiles:

```bash
python validation_scripts/qwen36_cold_prefill_benchmark.py \
  --model-path /opt/dlami/nvme/models/Qwen3.6-27B \
  --compiled-artifacts /opt/dlami/nvme/qwen_artifacts/qwen36_27b_2k \
  --compiled-artifacts-by-len \
    131072=/opt/dlami/nvme/qwen_artifacts/qwen36_27b_128k \
    262144=/opt/dlami/nvme/qwen_artifacts/qwen36_27b_262k \
  --prompt-lengths 128 256 384 512 2048 \
  --include-128k \
  --include-262k \
  --variants A_single512_old_chunked H_128k_candidate I_262k_recovery_block256 J_262k_recovery_block128 \
  --repetitions 3 \
  --output-json /tmp/qwen36_cold_prefill_long_context.json
```

Evaluate the matrix gates:

```bash
python validation_scripts/qwen36_cold_prefill_acceptance.py \
  /tmp/qwen36_cold_prefill_matrix.json \
  /tmp/qwen36_cold_prefill_long_context.json \
  --short-latency-speedup 1.5 \
  --baseline-cold-tok-per-s-target 420 \
  --baseline-cold-tok-per-s-tolerance 0.15 \
  --bucket-tok-regression-tolerance 0.20 \
  --prompt-token-tolerance 0.05 \
  --require-prompt-token-counts \
  --prefill-max-tokens 1 \
  --require-feature-deltas \
  --hbm-regression-tolerance 0.0
```

For a final sign-off run, add `--strict-final`. That preset requires prompt
rows for the fixed 128/256/384/512/1K/2K/8K/32K suite, prompt token counts,
the full A-G matrix on that fixed suite for both `max_tokens=1` and
`max_tokens=32`, row evidence for greedy sampling (`temperature=0`, `top_k=1`),
the complete cold-prefill instrumentation payload on every row,
benchmark schema/provenance fields on every row,
normalized launch-shape evidence (`tensor_parallel_size=4`,
`logical_nc_config=2`, `max_num_seqs=1`, `ctx_batch_size=1`),
positive `artifact_load_success` evidence on every compiled-artifact row,
positive `compiled_artifacts_path_exists` and
`compiled_artifacts_path_nonempty` evidence captured by the benchmark for every
compiled-artifact row,
explicit runtime `returncode=0` evidence on every row,
non-empty runtime `output_tail` evidence for DMA-spill auditing,
compiled-artifact consistency for the required 2K/8K/32K/128K/262K sequence
lengths,
generation latency/decode metrics and token exactness for
end-to-end rows including 128K/262K artifact candidates, at least three repeated
samples per required prefill row, per required 2K/8K tile-sweep case, and per
`max_tokens=32` generation row, including each required G tile-sweep case for
both prefill and generation, explicit p50/p95 prefill latency summaries for
each required row and G tile case, all required 2K/8K tile-sweep cases, A-G launch
profiles, repeated small dense-mask fallback rows at 256 and 512 tokens with
token exactness against the A-G matrix, feature-delta checks, HBM usage on every row,
GDN CTE kernel identity, GDN state diff fields on each candidate row, 128K and 262K artifact rows
including the 262K block-128 comparison, required long-artifact launch profiles
(`128/1024` q/kv tiles with block 128 and CTE buckets `256,512,1024,2048`
for 128K, block 256 then 128 with CTE profile `262k`/`[256]` for 262K),
matching baseline rows for long-context token exactness, a hybrid APC exactness
report proving full-prefix and partial-prefix token matches with cold-prefill
metrics and the strict cold-zero-disabled/chunked/text-only/compact validation
config, and an explicit
`--baseline-cold-tok-per-s-target` near the documented 420 tok/s cold baseline.
Strict-final checks that baseline target on the 2048-token baseline actual tok/s
row and uses bucket-normalized tok/s for each fixed short baseline prompt. For
8K+ rows, strict-final also requires concrete
`cte_attention_mask_path` / `dense_cte_mask_fallback` evidence showing the row
did not use the dense 4D SxS fallback path.
Use `--strict-min-samples` to raise that repeated-sample threshold.
`--strict-final` already requires HBM usage and GDN recurrent/conv state-diff
fields; HBM regression is checked on optimized candidate rows, while dense
fallback rows are treated as correctness controls. For exploratory non-strict runs, add `--require-hbm-usage` and
`--require-gdn-state-diff` once those fields are captured. The benchmark
harness captures a runtime line shaped as
`GDN_STATE_DIFF {"recurrent_max_abs_diff": ..., "conv_max_abs_diff": ...}` and
mirrors it into the row metrics consumed by the acceptance gate. It also accepts
`gdn_recurrent_state_max_abs_diff` / `gdn_conv_state_max_abs_diff` aliases,
normalizes them to canonical recurrent/conv keys, and preserves the raw payload
for audit provenance. If a debug hook writes state diffs to a sidecar instead
of stdout, pass `--gdn-state-diff-json /path/to/state_diff.json` to the
benchmark or strict-final orchestrator. The sidecar may be a keyed object using
`variant|prompt_len|max_tokens|repetition` keys, optionally with
`|q_tile|kv_tile|block_size` suffixes for tile sweeps, or a list of row records
containing those fields plus `gdn_state_diff`.
For final long-artifact acceptance, also pass `--require-128k` and
`--require-262k` against a matrix that includes the baseline rows plus
`H_128k_candidate`, `I_262k_recovery_block256`, and
`J_262k_recovery_block128`. If a long artifact run has no matching baseline
row, the acceptance report still enforces artifact load/run and DMA-spill
checks, and records long-context token exactness as skipped for that prompt
length.

To build the fixed-suite matrix, long-artifact matrix, and strict acceptance
report from one command, use the strict-final orchestrator. It also runs the
hybrid APC exactness validation and passes that JSON report into acceptance:

```bash
python validation_scripts/qwen36_cold_prefill_strict_final.py \
  --model-path /opt/dlami/nvme/models/Qwen3.6-27B \
  --artifacts-2k /opt/dlami/nvme/qwen_artifacts/qwen36_27b_2k \
  --artifacts-8k /opt/dlami/nvme/qwen_artifacts/qwen36_27b_8k \
  --artifacts-32k /opt/dlami/nvme/qwen_artifacts/qwen36_27b_32k \
  --artifacts-128k /opt/dlami/nvme/qwen_artifacts/qwen36_27b_128k \
  --artifacts-262k /opt/dlami/nvme/qwen_artifacts/qwen36_27b_262k \
  --output-dir /tmp/qwen36_cold_prefill_strict \
  --repetitions 3
```

Add `--preflight` first on the Trainium host to verify the local model,
non-empty compiled-artifact directories, output parent, Python executable, and
helper scripts before launching vLLM. It also checks that the checkout is on
`qwen36-cold-prefill-perf` by default; pass `--expected-branch` only if the
branch was intentionally renamed. The preflight step writes
`qwen36_cold_prefill_preflight.json` under `--output-dir` by default; keep that
with the run manifest, benchmark, acceptance, and per-phase `.log` artifacts.
The non-dry-run orchestrator writes `qwen36_cold_prefill_run_manifest.json`
with phase commands, return codes, log paths, expected output paths, and
non-empty evidence status for the preflight JSON and each output/log file. Add
`--dry-run` to inspect the generated benchmark and acceptance commands without
launching vLLM. After the strict-final run, use `--evidence-check` with the same
`--output-dir` to verify the preflight JSON passed and describes the same
bundle, the run manifest contains all phase commands and matching paths, all
phase return codes are 0, strict-final acceptance passed with no failed audit
items, and required benchmark/log files are present. It writes
`qwen36_cold_prefill_evidence_check.json` under `--output-dir`. Add
`--gdn-state-diff-json` here if GDN state-diff evidence is produced as a debug
sidecar rather than runtime stdout.
The same sequence is available as
`bash validation_scripts/qwen36_trainium_strict_final.sh` after exporting
`QWEN36_MODEL`, `QWEN36_ARTIFACT_2K`, `QWEN36_ARTIFACT_8K`,
`QWEN36_ARTIFACT_32K`, `QWEN36_ARTIFACT_128K`, `QWEN36_ARTIFACT_262K`, and
`QWEN36_OUT`. Set `QWEN36_OUTPUT_PREFIX` only if you need a non-default artifact
prefix, and `QWEN36_FAIL_FAST=1` if you want benchmark phases to stop after the
first runtime failure. The wrapper resolves the orchestrator from its own path,
so an absolute script path works even when the current directory is not the repo
root.
The orchestrator includes the dense fallback rows automatically and explicitly
runs both `--max-tokens-values 1 32`; for manual benchmark runs, add
`--include-dense-fallback` to collect the same 256/512-token fallback evidence.
For manual strict-final acceptance, first write the hybrid
APC report with `validation_scripts/qwen36_hybrid_apc_validation.py exactness
--output-json /tmp/qwen36_hybrid_apc_exactness.json`, then pass
`--hybrid-apc-report /tmp/qwen36_hybrid_apc_exactness.json` to
`qwen36_cold_prefill_acceptance.py --strict-final`.

For a focused GDN kernel comparison, run the old chunked, fused, and PyTorch
chunk toggles from the same prompt set:

```bash
python validation_scripts/qwen36_cold_prefill_benchmark.py \
  --model-path /opt/dlami/nvme/models/Qwen3.6-27B \
  --compiled-artifacts /opt/dlami/nvme/qwen_artifacts/qwen36_27b_2k \
  --prompt-lengths 128 512 2048 \
  --max-tokens-values 1 \
  --variants A_single512_old_chunked E_short_text_compact_fused K_short_text_compact_pytorch_chunk \
  --repetitions 5 \
  --output-json /tmp/qwen36_gdn_kernel_compare.json
```

The offline runner and minimal OpenAI-compatible server log
`GENERATION_METRICS` alongside `COLD_PREFILL_METRICS`; use that row for
first-token latency, decode latency, decode tok/s, and end-to-end generated
tok/s when validating benchmark and served requests. The benchmark harness
captures both metric rows when present and folds generation fields into the
acceptance report.
Benchmark rows also record prompt SHA-256, model path, compiled artifact path,
`max_model_len`, and `seq_len`; the acceptance script requires those controlled
inputs to match between the baseline and candidate rows by default. In
`--strict-final` mode, this controlled-input check also covers every
`max_tokens=32` generation row. Long-artifact rows also carry the selected CTE
bucket list, q/kv tile sizes, block size, text-only CTE flag, and compact-mask
flag into the acceptance audit. A-G rows carry their expected CTE bucket list,
text-only flag, compact-mask flag, and cold-zero-conv flag into the same audit.
Its JSON output includes an `audit_checklist` mapping the goal gates to concrete
evidence, including the optional baseline tok/s target, short-prompt speedup,
prompt-token accuracy, 2K regression, exactness, HBM, GDN kernel identity, GDN
state diff, and 128K/262K artifact checks.
It also includes `feature_delta_checks` for the A->G matrix so you can see the
per-step latency/token-throughput evidence for dynamic buckets, text-only CTE,
compact-mask guardrails, fused GDN CTE, cold-zero guardrails, and tile/block
sweeps. Pass
`--require-feature-deltas` to fail the report when any populated A->G step
regresses. When HBM usage is present, an increased target HBM footprint also
marks that feature step as regressed, and each feature-delta row includes the
HBM byte delta for audit. The tile/block sweep keeps every tile profile in
`tile_case_checks` and uses the best p50 profile per prompt for the gate.
Most A-G rows enable vLLM chunked prefill, so disabling the compact-mask flag is
not necessarily a dense-mask performance delta; `L_small_dense_mask_fallback` is
the explicit dense fallback check. The optional
`M_short_text_compact_fused_cold_zero_ablation` row is the isolated cold-zero
conv experiment after strict-final passes without that flag.

## Next Milestone

For cold-prefill latency, fix bucket waste before speculative decode or cache
quantization. The serving entrypoints now support multi-bucket CTE artifacts,
text-only CTE inputs, compact CTE masks, context-batch profiles, and attention
tile overrides.

For warm-prefix production APC, the required contract remains a unified
prefix-cache object whose attention KV, GDN recurrent state, and GDN conv state
are jointly addressable, evictable, restorable, and exact under continuous
batching.

Recommended order:

1. Dynamic CTE buckets: start with `[128,256,512]` for 2K short-prompt tests,
   `[256,512]` for 128K, and `[256]` for the 262K TP=4 load experiment.
2. Fused GDN CTE path validation: qwen chunked-prefill should use fused
   DeltaNet with restored initial state by default.
3. Text-only CTE and compact-mask validation: no full dummy vision reductions
   and no dense 4D causal masks in normal text serving.
4. Hybrid APC exactness: cold vs warm greedy token IDs, partial-prefix reuse,
   multi-hit chat history, continuous batching movement, and eviction pressure.
5. Attention block-size sweeps at `64` and `128`, with `32` included for
   granularity-sensitive chat workloads.
6. FP8 KV/cache only after the BF16/FP32 baseline is exact.
7. MTP/spec decode after recurrent-state rollback semantics are explicit.
