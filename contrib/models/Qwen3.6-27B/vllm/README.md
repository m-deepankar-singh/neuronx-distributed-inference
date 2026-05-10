# Qwen3.6-27B vLLM on Neuron

This folder contains the first-pass vLLM integration helpers for the
Qwen3.6-27B contrib model.

The current goal is **basic vLLM serving through the Neuron/NxDI plugin**.
Long-context production serving still needs a model-specific chunked-prefill
shim because the validated Qwen3.6 artifact uses a 512-token context-encoding
graph with a 128K cache.

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

## Known Limitation

AWS Neuron vLLM currently lists chunked prefill as unsupported. This matters for
this model: the 128K artifact is compiled with `seq_len=131072` but
`max_context_length=512`, so prompts longer than the context bucket require our
standalone chunked prefill driver until we port that driver into the vLLM model
runner.

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
  --compiled-artifacts /opt/dlami/nvme/qwen_artifacts/qwen36_27b_128k_fp8_mlp_only_run1 \
  --max-model-len 131072 \
  --seq-len 131072 \
  --cte-bucket 512 \
  --port 8000
```

For now, send prompts that fit within the compiled context bucket unless you are
testing the later vLLM chunked-prefill shim.

## Offline Smoke

```bash
python contrib/models/Qwen3.6-27B/vllm/run_offline_inference.py \
  --model-path /opt/dlami/nvme/models/Qwen3.6-27B \
  --compiled-artifacts /opt/dlami/nvme/qwen_artifacts/qwen36_27b_128k_fp8_mlp_only_run1 \
  --max-model-len 131072 \
  --seq-len 131072 \
  --cte-bucket 512 \
  --prompt "What is 17 * 23? Answer with the number only."
```

## Next Milestone

After basic vLLM load works, port the standalone NxDI server's chunked prefill
driver into the vLLM/Neuron model-runner path:

1. split long prompts into 512-token chunks;
2. run the compiled context model repeatedly;
3. preserve attention KV plus GDN recurrent/conv state across chunks;
4. hand off to token generation;
5. exact-match greedy tokens against the standalone NxDI server.
