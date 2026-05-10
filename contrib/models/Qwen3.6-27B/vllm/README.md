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

## Chunked Prefill Note

The Neuron plugin disables vLLM chunked prefill by default and installs a custom
continuous-batching scheduler. For this Qwen3.6 artifact we need vLLM's native
chunked-prefill scheduler so prompts longer than the 512-token context graph are
fed to the precompiled model in 512-token chunks. The launcher sets
`DISABLE_NEURON_CUSTOM_SCHEDULER=1` when `--enable-vllm-chunked-prefill` is
passed.

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

Long-prompt precompiled artifact path:

```bash
contrib/models/Qwen3.6-27B/vllm/start_vllm_server.sh \
  --model-path /opt/dlami/nvme/models/Qwen3.6-27B \
  --compiled-artifacts /opt/dlami/nvme/qwen_artifacts/qwen36_27b_128k_fp8_mlp_only_run1 \
  --max-model-len 131072 \
  --seq-len 131072 \
  --cte-bucket 512 \
  --block-size 256 \
  --enable-vllm-chunked-prefill \
  --port 8000
```

Offline long-prompt smoke:

```bash
python contrib/models/Qwen3.6-27B/vllm/run_offline_inference.py \
  --model-path /opt/dlami/nvme/models/Qwen3.6-27B \
  --compiled-artifacts /opt/dlami/nvme/qwen_artifacts/qwen36_27b_128k_fp8_mlp_only_run1 \
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

Validation run on Trn2 with the FP8 128K artifact:

- short prompt loaded and generated through vLLM;
- ~1K prompt with `--enable-vllm-chunked-prefill` completed in 3.705s;
- ~4K prompt that fails without chunking completed in 11.007s and generated
  valid token IDs.
- OpenAI-compatible `/v1/chat/completions` served a 1215-token prompt in
  3.87s with valid usage accounting.

## Offline Smoke

```bash
python contrib/models/Qwen3.6-27B/vllm/run_offline_inference.py \
  --model-path /opt/dlami/nvme/models/Qwen3.6-27B \
  --compiled-artifacts /opt/dlami/nvme/qwen_artifacts/qwen36_27b_128k_fp8_mlp_only_run1 \
  --max-model-len 131072 \
  --seq-len 131072 \
  --cte-bucket 512 \
  --chat \
  --prompt "What is 17 * 23? Answer with the number only."
```

## Next Milestone

Validate multi-turn prompts with `--enable-vllm-chunked-prefill`, then measure
TTFT/TPOT under concurrent requests.
