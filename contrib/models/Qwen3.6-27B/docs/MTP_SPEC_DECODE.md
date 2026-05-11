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
- `Qwen35InferenceConfig(enable_mtp_weight_loading=True)` now preserves and
  converts native `mtp.*` weights into the same NxDI attention layout used by
  full-attention layers.
- `NeuronQwen35Model.compute_mtp_logits(...)` runs the native MTP predictor with
  shared target embeddings and `lm_head`, returning MTP logits, hidden states,
  and MTP cache tensors.
- `Qwen35InferenceConfig(enable_mtp_hidden_state_output=True)` appends target
  hidden states as the final model output for a future fused-spec scheduler.
- `NeuronQwen35MTPDraftForCausalLM` exposes Qwen's native MTP predictor as a
  separate NxDI draft model. Its converter keeps only shared `embed_tokens`,
  shared `lm_head`, and native `mtp.*` weights.

## Current Gap

The baseline target model still skips `mtp.*` weights by default, and the hybrid
cache manager currently rejects speculative decoding. This means native MTP
weights can now be loaded and called behind opt-in hooks, but they are not yet
connected to the NxDI generation scheduler or vLLM speculative request path.

## Next Implementation Steps

1. Wire `NeuronQwen35MTPDraftForCausalLM` into the fused-spec config and compile
   a small speculation artifact.
2. Add hybrid-cache state handling for accepted/rejected MTP drafts if the
   existing EAGLE-style scheduler cannot preserve DeltaNet state correctly.
3. Validate with greedy baseline comparison, acceptance-rate measurement, and
   decode throughput.

The CPU tests are intentionally small and do not attempt full 27B parity. They
are a guardrail for the MTP tensor path before expensive Neuron compile attempts.
