# Qwen3.6 Fused DeltaNet Direct-Solve Follow-Up

This is a clean branch on top of PR 164, `contrib/qwen36-27b-vllm-apc-pr` at `ac7df71`. It is meant to extend the existing vLLM APC PR without bringing in the full experimental branch stack.

## What Changed On Top Of PR 164

- Stabilized the Qwen fused DeltaNet CTE kernel.
- Added an isolated fused NKI validation script.
- Made the validator load the fused kernel directly.
- Replaced the fused kernel's Neumann power-doubling solve with a direct triangular RHS solve.
- Updated CPU DeltaNet decay regression coverage for realistic gate scales.
- Added compact validation artifacts for coherence, decode, prefill, and memory.

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
