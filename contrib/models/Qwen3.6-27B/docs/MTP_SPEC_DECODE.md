# Qwen3.6-27B Native MTP Speculative Decode

Status: MTP draft/scaffold implemented; hardware compile not attempted yet.

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
- The EAGLE-style fused-spec path now has a Qwen-specific MTP guardrail for
  hybrid DeltaNet state: target verification can return per-step recurrent and
  conv states, and the fused-spec scheduler selects the state at the accepted
  prefix length before returning cache aliases.
- `test/integration/qwen36_27b_compile_mtp.py` builds a native-MTP config from
  the validated hybrid/chunked-prefill baseline settings.

## Current Gap

The baseline target model still skips `mtp.*` weights by default. Native MTP is
now connected to the NxDI EAGLE-style fused-spec scheduler behind opt-in flags,
but no Trainium artifact has been compiled or validated yet. vLLM request-level
plumbing for `qwen3_next_mtp` remains separate from the NxDI compile path.

## Next Implementation Steps

1. Compile a small native-MTP speculation artifact with
   `test/integration/qwen36_27b_compile_mtp.py`.
2. Validate compile/load, short greedy generation, 762-token exact-match gate,
   acceptance rate, and decode throughput.
3. Wire the validated artifact into the vLLM/Neuron serving path.

The CPU tests are intentionally small and do not attempt full 27B parity. They
are a guardrail for the MTP tensor path before expensive Neuron compile attempts.
