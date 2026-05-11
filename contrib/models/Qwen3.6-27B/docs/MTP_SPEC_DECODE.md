# Qwen3.6-27B Native MTP Speculative Decode

Status: CPU contract implemented; Trainium integration not wired yet.

## What Is Implemented

- `test/unit/test_mtp_cpu_reference.py` defines an executable CPU reference for
  the native Qwen3.6 MTP predictor tensor contract.
- The test validates the expected checkpoint keys:
  - `mtp.fc.weight`
  - `mtp.pre_fc_norm_embedding.weight`
  - `mtp.pre_fc_norm_hidden.weight`
  - `mtp.norm.weight`
  - `mtp.layers.0.*` attention, MLP, and norm weights
- The test also records the vLLM-style MTP weight remapping:
  - `mtp.fc.weight` -> `model.fc.weight`
  - `mtp.layers.0.self_attn.q_proj.weight` -> `model.layers.0.self_attn.q_proj.weight`
  - shared `embed_tokens.weight` and `lm_head.weight` stay shared.

## Current Gap

The Qwen3.6 contrib converter still skips `mtp.*` weights, and the hybrid cache
manager currently rejects speculative decoding. This means native MTP weights
exist in the checkpoint but are not yet loaded into the NxDI/Trainium execution
path.

## Next Implementation Steps

1. Add a Neuron-side MTP predictor module matching the CPU contract.
2. Extend conversion to preserve `mtp.*` weights for the MTP branch while keeping
   the target model unchanged.
3. Wire the predictor into NxDI speculative decoding with the hybrid cache state.
4. Validate with greedy baseline comparison, acceptance-rate measurement, and
   decode throughput.

The CPU tests are intentionally small and do not attempt full 27B parity. They
are a guardrail for the MTP tensor path before expensive Neuron compile attempts.
