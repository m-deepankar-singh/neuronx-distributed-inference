# Qwen3.6 Fused DeltaNet Direct-Solve Follow-Up

This is a clean extraction on top of PR 164, `contrib/qwen36-27b-vllm-apc-pr` at `ac7df71`. It is meant to show the fused DeltaNet direct-solve follow-up without bringing in the full experimental branch stack.

## Branch Lineage

The actual development history was:

```text
PR 164 / vLLM APC baseline
  -> experimental
      -> qwen-fused-neumann-stable-decay
```

The `experimental` branch added substantial runtime and validation work after PR 164:

- Hybrid APC checkpoint cache, lifecycle, restore/commit masks, and strict metadata contracts.
- vLLM/NxDI scheduler bridge changes for cached chunked prefill, backed prefix reads, request-id propagation, and suffix continuation handling.
- Qwen chunked prefill fixes for CTE bucket alignment, prefix-cache slot mapping, GDN checkpoint commits, and chunk-boundary handling.
- FP8 128K artifact configuration guards, validation max-prompt alignment, and artifact audit checks.
- OpenAI/vLLM validation harnesses for exactness, context sweeps, TTFT/TPOT, decode benchmarking, memory capture, and API compatibility.
- Decode-path and sampling fixes, including on-device sampling/logits-path validation and chat-template thinking controls.

The final fused branch added the direct-solve fused DeltaNet fix on top of that experimental runtime stack.

## Major Changes From PR 164 To The Tested Branch

The full tested branch differs from PR 164 by roughly 105 source/result files. The important changes are:

- **Hybrid APC runtime:** checkpoint cache, restore/commit masks, backed prefix reads, checkpoint-slot lifecycle, and metadata validation.
- **vLLM scheduler bridge:** request-id propagation, cached chunked-prefill continuations, active suffix accounting, no-prefix fallback handling, and backed-prefix authorization.
- **Qwen model execution:** Hybrid APC chunked prefill, GDN checkpoint commit/restore, text-only CTE inputs, compact CTE masks, prefix/suffix boundary handling, and decode-path safety.
- **NxDI prefix-cache plumbing:** vectorized APC args, prefix-cache bucket selection, padded-row safety, cached decode rows, and async checkpoint lifecycle.
- **DeltaNet NKI kernels:** chunked/fused validation paths, DeltaNet backend compile controls, masked Neumann experiments, and the final fused direct triangular RHS solve.
- **FP8/artifact compile path:** Qwen FP8 compile config coverage, artifact config audits, 128K validation alignment, `pa_num_blocks` checks, and larger TKG bucket support.
- **Serving/API compatibility:** OpenAI-compatible proxy/server behavior, chat-template `enable_thinking=false`, stop-sequence handling, and startup/offline helpers.
- **Validation harnesses:** exactness validation, OpenAI chat APC validation, boundary APC probes, context sweeps, offline decode benchmark, BF16 sweep, artifact audit, and memory/perf capture.
- **Tests and results:** added Hybrid APC, scheduler, model-alias, compile-config, artifact-audit, sampling, async, prefix-cache, and DeltaNet tests plus recorded performance/memory artifacts.

## What Changed On Top Of PR 164

This clean branch extracts only the fused DeltaNet follow-up commits:

- Stabilized the Qwen fused DeltaNet CTE kernel.
- Added an isolated fused NKI validation script.
- Made the validator load the fused kernel directly.
- Replaced the fused kernel's Neumann power-doubling solve with a direct triangular RHS solve.
- Updated CPU DeltaNet decay regression coverage for realistic gate scales.
- Added compact validation artifacts for coherence, decode, prefill, and memory.

The artifact results below were produced from `qwen-fused-neumann-stable-decay`, so they validate the direct-solve fused kernel inside the full `experimental` lineage. They should not be read as proof that PR 164 plus only these extracted commits reproduces every Hybrid APC runtime fix from `experimental`.

## Why

The previous fused path could produce unstable or incoherent outputs with realistic Qwen gate values. The Neumann power-doubling solve is mathematically convenient, but it forms repeated full-matrix powers and is numerically fragile for this recurrence. The direct triangular RHS solve computes the causal recurrence without those large intermediate powers and matches the stable chunked-kernel approach.

## Validation

Local checks:

- `python3 -m py_compile contrib/models/Qwen3.6-27B/src/nki_kernels/nki_deltanet_fused.py contrib/models/Qwen3.6-27B/scripts/validate_deltanet_fused_nki.py contrib/models/Qwen3.6-27B/test/unit/test_deltanet_decay.py`
- `python3 -m pytest contrib/models/Qwen3.6-27B/test/unit/test_deltanet_decay.py -q`
- Result: `2 passed`

Trn2 artifact validation:

- Coherence pass: `true`
- Decode throughput: `21.63 tok/s`
- TPOT: `46.2 ms/token`
- Cold prefill: about `590 tok/s` from 4K through 16K
- Warm prefill: up to `36.3k tok/s` at 16K with APC reuse
- HBM peak sum: `60.1 GiB`

Known limitation:

- The compiled artifact used for this validation has `prefix_buckets` through `16384`; the 32K sweep failed with `Prefix len 16640 exceeds largest bucket 16384`. A long-context follow-up compile needs larger prefix buckets.
