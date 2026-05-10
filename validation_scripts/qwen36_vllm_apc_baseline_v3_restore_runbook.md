# Qwen3.6 vLLM APC Baseline v3 Restore Runbook

Use this when a new Trainium instance comes up and the previous instance is
gone.

## 1. Checkout The Restore Branch

```bash
cd /home/ubuntu/inferentia-gdn
git fetch origin --tags
git checkout -B codex/qwen36-vllm-prefix-cache origin/codex/qwen36-vllm-prefix-cache
```

The immutable validation tag is still available as:

```bash
git checkout qwen36-27b-vllm-apc-baseline-v3
```

Use the branch for restore because it includes post-baseline convenience
scripts and the remote snapshot; the tag marks the validated runtime state.

## 2. Verify Required Paths

The fast path assumes these paths exist:

```text
/opt/dlami/nvme/models/Qwen3.6-27B
/opt/dlami/nvme/qwen_artifacts/qwen36_27b_128k_fp8_mlp_only_vllm_statereset_run1
/opt/aws_neuronx_venv_pytorch_inference_vllm_0_16
```

If the model or artifact is missing, restore it before serving. The artifact is
large and was not copied into git. The exact runtime config and validation
results are preserved, so it can be recreated if needed.

## 3. Start The Known-Good Server

```bash
cd /home/ubuntu/inferentia-gdn
contrib/models/Qwen3.6-27B/vllm/start_baseline_v3.sh \
  --restart \
  --install-registry
```

This starts:

- vLLM backend on `127.0.0.1:8001`;
- guarded OpenAI-compatible proxy on `0.0.0.0:8000`;
- APC with `--mamba-cache-mode align`;
- vLLM chunked prefill with `CTE=512`;
- the 128K FP8 MLP state-reset artifact.

It waits for `/v1/models` and runs the `17 * 23 -> 391` smoke test.

If your paths differ:

```bash
contrib/models/Qwen3.6-27B/vllm/start_baseline_v3.sh \
  --model-path /path/to/Qwen3.6-27B \
  --compiled-artifacts /path/to/qwen36_27b_128k_fp8_mlp_only_vllm_statereset_run1 \
  --venv /path/to/vllm-neuron-venv \
  --restart \
  --install-registry
```

## 4. External Smoke

From your laptop:

```bash
curl http://INSTANCE_IP:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "/opt/dlami/nvme/models/Qwen3.6-27B",
    "messages": [{"role": "user", "content": "What is 17 * 23? Answer with the number only."}],
    "max_tokens": 8,
    "temperature": 0,
    "top_k": 1
  }'
```

Expected answer: `391`.

## 5. Optional Baseline Revalidation

Run the fastest APC check:

```bash
python validation_scripts/qwen36_prefix_cache_validation.py \
  --base-url http://127.0.0.1:8000 \
  --model /opt/dlami/nvme/models/Qwen3.6-27B \
  --prefix-repeats 220 \
  --max-tokens 32 \
  --no-fail-on-speed
```

Run the full server hardening gate:

```bash
python validation_scripts/qwen36_vllm_apc_server_hardening_eval.py \
  --base-url http://127.0.0.1:8000 \
  --model /opt/dlami/nvme/models/Qwen3.6-27B \
  --prefix-repeats 220 \
  --max-tokens 32 \
  --concurrency-levels 1,2,4
```

## 6. Where The Previous Instance State Was Saved

Local snapshot:

```text
remote_snapshots/qwen36_v3_20260510T103627Z/
```

It includes process commands, `/v1/models`, backend/proxy logs, APC hardening
JSON, long-context quality JSONs, and the raw remote snapshot tarball.
