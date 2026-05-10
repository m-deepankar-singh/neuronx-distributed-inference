# Qwen3.6-27B vLLM Support Plan

Branch: `codex/qwen36-vllm-support`

## Research Summary

NVIDIA/vLLM support for Qwen3.5/Qwen3.6 is model-native, not a wrapper around a
separate server. vLLM has a Qwen GDN implementation that:

- uses a `GatedDeltaNetAttention` layer for Qwen3.5/Qwen3-Next style GDN;
- calls a custom `torch.ops.vllm.gdn_attention_core` op for the linear-attention
  core;
- supports MTP speculative decoding for Qwen3.6 recipes;
- supports prefix caching through vLLM cache manager machinery, but GDN/Mamba
  prefix caching is still described as experimental;
- requires GDN recurrent/SSM state to be cached in fp32 at block boundaries for
  reliable prefix reuse.

AWS Neuron vLLM support is different:

- NxDI integrates with vLLM through the vLLM-Neuron plugin.
- The `vllm-project/vllm-neuron` repo is the plugin source/reference, but its
  public README is a beta path for older SDK/vLLM combinations. For the current
  validated Neuron 2.29 environment, use the AWS vLLM-on-Neuron guide/version
  matrix first.
- Custom NxDI models are discovered through the NxDI model registry.
- `NEURON_COMPILED_ARTIFACTS` can load a precompiled artifact.
- AWS docs currently list chunked prefill as unsupported on Neuron vLLM.
- Our Qwen3.6 artifact needs chunked CTE because the context graph is compiled
  for 512-token chunks while cache length is 128K.

## Strategy

Do not try to replace the current NxDI server immediately. Add vLLM support in
milestones, keeping the standalone NxDI OpenAI server as the correctness and
long-context control.

## Phase 0: Local Branch And Scope

- Branch from the current FP8 baseline track.
- Keep current uncommitted OpenAI/prefix-cache WIP visible on this branch.
- Do not merge Python snapshot prefix caching into vLLM work; vLLM needs
  block-boundary hybrid state caching later.

## Phase 1: Minimal vLLM Load Path

Goal: prove vLLM can discover and load the Qwen3.6 contrib model at small
context.

Tasks:

- Add a vLLM install/registration helper under `contrib/models/Qwen3.6-27B/`.
- Detect whether the current environment already has vLLM and the Neuron plugin
  installed; do not install or downgrade packages from this helper.
- Mirror the DeepSeek contrib pattern: copy/register contrib model code into the
  vLLM/NxDI environment and ensure the model type key matches Hugging Face
  `config.json`.
- Add a README section with a `VLLM_PLUGINS=neuron vllm serve ...` command.
- Start with short context, no prefix caching, no chunked prefill, no
  speculation.

Gate:

- vLLM server starts.
- `/v1/models` responds.
- One short chat completion works.

## Phase 2: Precompiled Artifact Loading

Goal: vLLM loads our existing compiled artifact instead of recompiling.

Tasks:

- Launch with `NEURON_COMPILED_ARTIFACTS=/opt/dlami/nvme/qwen_artifacts/...`.
- Keep override config identical to artifact settings:
  - TP=4
  - seq_len/max_model_len=128K
  - context bucket=512
  - FP8 MLP settings
  - hybrid cache enabled
- Verify vLLM does not trigger a recompile.

Gate:

- Load succeeds from artifact.
- Short prompt output matches standalone NxDI server under greedy decoding.

## Phase 3: Long-Context Chunked Prefill Integration

Goal: make vLLM usable with the 128K chunked CTE artifact.

Problem:

- Neuron vLLM currently does not support chunked prefill.
- Our model cannot accept a 4K/16K/128K prompt as a single context call because
  the compiled context graph bucket is 512.

Tasks:

- Implement a model-runner-side chunked prefill shim for this model:
  - split prompt into 512-token CTE chunks;
  - call the compiled NxDI context model repeatedly;
  - preserve hybrid KV/GDN state across chunks;
  - hand off to token-generation model after the final chunk.
- Start single-request only. Continuous batching comes after correctness.

Gate:

- 4K prompt through vLLM works.
- 16K prompt through vLLM works.
- Greedy first 20 generated token IDs match standalone NxDI server.

## Phase 4: Production vLLM Features

Only after Phase 3:

- Enable continuous batching.
- Add block-boundary hybrid prefix caching:
  - attention KV block refs;
  - GDN recurrent state in fp32;
  - GDN conv state;
  - restore only at exact block boundaries.
- Evaluate vLLM EAGLE/MTP speculation separately.

## Explicit Non-Goals For First Pass

- No custom vLLM CUDA/Triton kernels.
- No Neuron vLLM prefix caching until chunked prefill works.
- No speculative decoding until vLLM baseline output matches NxDI.
- No attempt to make Python snapshot prefix cache part of vLLM.
