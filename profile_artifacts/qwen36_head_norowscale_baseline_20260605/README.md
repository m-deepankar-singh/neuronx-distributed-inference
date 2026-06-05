# Qwen3.6 256K Decode-Step Baseline Reproduction - 2026-06-05

## Purpose

This is the current slow-prefix, runtime-stable baseline used before comparing
against `codex/nki-deltanet-multihead-cte`.

It reproduces the May 28 split-QKV decode artifact class by keeping the
decode-step branch behavior and disabling the later row-scale split-QKV TKG
contract. It is suitable as a baseline for runtime stability and decode-smoke
coherence. It is not a strict semantic-quality pass.

## Live Runtime

As of 2026-06-05, this baseline is intentionally left running on:

* Runtime host: `ubuntu@16.26.184.190`
* Backend: `http://127.0.0.1:8001`
* Proxy: `http://127.0.0.1:8000`
* Backend log:
  `/home/ubuntu/validation_logs/fp8_256k_decode_nki/head_norowscale_baseline_backend_20260605T1228Z.log`
* Strict non-thinking proxy log:
  `/home/ubuntu/validation_logs/fp8_256k_decode_nki/head_norowscale_baseline_proxy_nothink_20260605T1239Z.log`

Health check:

```bash
ssh ubuntu@16.26.184.190 'curl -fsS http://127.0.0.1:8000/health && curl -fsS http://127.0.0.1:8001/health'
```

Current serving processes at validation time:

```text
Backend wrapper PID: 13203
EngineCore PID:      13304
Proxy PID:           15318
```

## Artifact

Compile host:

```text
ubuntu@16.26.135.243
```

Runtime host:

```text
ubuntu@16.26.184.190
```

Artifact path on both hosts:

```text
/mnt/trainium_artifacts/qwen_artifacts/qwen36_27b_256k_fp8_full_lmheadfp8_kvfp8_hybrid_apc_nki_decode_splitqkv_gdnrecbfloat16_sampletokens_b256_cte256_512_pfx16k_slots64_async_20260605T113222Z_head_norowscale
```

Compile log:

```text
/home/ubuntu/validation_logs/fp8_256k_decode_nki/qwen36_27b_256k_fp8_full_lmheadfp8_kvfp8_hybrid_apc_nki_decode_splitqkv_gdnrecbfloat16_sampletokens_b256_cte256_512_pfx16k_slots64_async_20260605T113222Z_head_norowscale_compile.log
```

Artifact size after rsync: about `33G`.

## Source Delta

Source checkout on compile host:

```text
/home/ubuntu/inferentia-gdn-decode-step-baseline-20260605
```

Base commit:

```text
834543e
```

Local patch:

* File:
  `/home/ubuntu/inferentia-gdn-decode-step-baseline-20260605/contrib/models/Qwen3.6-27B/src/modeling_qwen35.py`
* Purpose:
  default split-QKV token-generation NKI back to the May 28 known-good
  no-row-scale contract.
* Effective behavior:
  `QWEN36_SPLIT_QKV_TKG_ROW_SCALES=0` means split-QKV TKG compiles with
  `QuantizationType.NONE` and `qkv_w_scales=None`.

Why this matters:

* The May 28 usable artifact compiled split-QKV TKG with
  `QuantizationType.NONE`, `qkv_w_scales=None`.
* The later branch HEAD compiled it with `QuantizationType.ROW` and
  `qkv_w_scales`, then the recompiled artifact crashed inside
  `token_generation_model` with `NRT_EXEC_OOB`.
* This reproduction keeps HEAD fixes such as output-gate handling, but restores
  the old split-QKV decode kernel signature.

## Compile Command

```bash
ssh ubuntu@16.26.135.243
cd /home/ubuntu/inferentia-gdn-decode-step-baseline-20260605
TS=$(date -u +%Y%m%dT%H%M%SZ)_head_norowscale \
LOGDIR=/home/ubuntu/validation_logs/fp8_256k_decode_nki \
LOAD_AFTER_COMPILE=0 \
QWEN36_SPLIT_QKV_TKG_ROW_SCALES=0 \
bash tmp_compile_qwen256k_fp8_full_decode_splitqkv_sampletokens.sh
```

Compile passed with:

```text
Finished Compilation for all HLOs
CHECKPOINT_BANK_WEIGHTS_ADDED tp0_sharded_checkpoint.safetensors 48 48 torch.bfloat16
CHECKPOINT_BANK_WEIGHTS_ADDED tp1_sharded_checkpoint.safetensors 48 48 torch.bfloat16
CHECKPOINT_BANK_WEIGHTS_ADDED tp2_sharded_checkpoint.safetensors 48 48 torch.bfloat16
CHECKPOINT_BANK_WEIGHTS_ADDED tp3_sharded_checkpoint.safetensors 48 48 torch.bfloat16
COMPILE_DONE
```

The QKV TKG compile signature was verified in the compile log:

```text
quantization_type = QuantizationType.NONE
qkv_w_scales = None
```

## Transfer Command

```bash
ssh -A ubuntu@16.26.135.243 'rsync -aH --info=progress2 \
  /mnt/trainium_artifacts/qwen_artifacts/qwen36_27b_256k_fp8_full_lmheadfp8_kvfp8_hybrid_apc_nki_decode_splitqkv_gdnrecbfloat16_sampletokens_b256_cte256_512_pfx16k_slots64_async_20260605T113222Z_head_norowscale/ \
  ubuntu@16.26.184.190:/mnt/trainium_artifacts/qwen_artifacts/qwen36_27b_256k_fp8_full_lmheadfp8_kvfp8_hybrid_apc_nki_decode_splitqkv_gdnrecbfloat16_sampletokens_b256_cte256_512_pfx16k_slots64_async_20260605T113222Z_head_norowscale/'
```

## Backend Launch

```bash
ssh ubuntu@16.26.184.190
cd /home/ubuntu/inferentia-gdn-decode-step-baseline-20260605
export PATH=/opt/aws/neuron/bin:/opt/aws_neuronx_venv_pytorch_inference_vllm_0_16/bin:$PATH
export QWEN36_VLLM_LOGITS_DEBUG=1
export NEURON_RT_VISIBLE_CORES=0-3

nohup bash contrib/models/Qwen3.6-27B/vllm/start_vllm_server.sh \
  --model-path /home/ubuntu/models/Qwen3.6-27B \
  --compiled-artifacts /mnt/trainium_artifacts/qwen_artifacts/qwen36_27b_256k_fp8_full_lmheadfp8_kvfp8_hybrid_apc_nki_decode_splitqkv_gdnrecbfloat16_sampletokens_b256_cte256_512_pfx16k_slots64_async_20260605T113222Z_head_norowscale \
  --max-model-len 262144 \
  --seq-len 262144 \
  --cte-buckets "256 512" \
  --context-encoding-bucket-pairs "256:256 512:256 256:512 512:512 256:1024 512:1024 256:2048 512:2048 256:4096 512:4096 256:8192 512:8192 512:16384" \
  --token-generation-buckets "512 768 1024 1280 2048 2304 4096 4352 8192 8448 16384 16640 24576 24832 32768 33024 65536 65792 131072 131328 262144" \
  --tensor-parallel-size 4 \
  --logical-nc-config 2 \
  --max-num-seqs 1 \
  --ctx-batch-size 1 \
  --async-mode \
  --enable-prefix-caching \
  --enable-hybrid-apc \
  --enable-vllm-chunked-prefill \
  --block-size 256 \
  --gdn-checkpoint-interval 256 \
  --max-gdn-checkpoint-slots 64 \
  --gdn-recurrent-cache-dtype bfloat16 \
  --gdn-conv-cache-dtype bfloat16 \
  --hybrid-cache-mode all \
  --hybrid-apc-require-vllm-metadata \
  --hybrid-apc-enable-backed-prefix-reads \
  --num-gpu-blocks-override 1024 \
  --host 127.0.0.1 \
  --port 8001 \
  > /home/ubuntu/validation_logs/fp8_256k_decode_nki/head_norowscale_baseline_backend_20260605T1228Z.log 2>&1 &
```

## Proxy Launch

The live proxy is strict non-thinking mode:

```bash
ssh ubuntu@16.26.184.190
cd /home/ubuntu/inferentia-gdn-decode-step-baseline-20260605
export PATH=/opt/aws/neuron/bin:/opt/aws_neuronx_venv_pytorch_inference_vllm_0_16/bin:$PATH

nohup python3 contrib/models/Qwen3.6-27B/vllm/qwen36_chat_proxy.py \
  --host 0.0.0.0 \
  --port 8000 \
  --backend-url http://127.0.0.1:8001 \
  > /home/ubuntu/validation_logs/fp8_256k_decode_nki/head_norowscale_baseline_proxy_nothink_20260605T1239Z.log 2>&1 &
```

Do not use raw `/v1/completions` through this proxy. It intentionally rejects
raw completions for Qwen3.6 and requires `/v1/chat/completions`.

## Validation Result

Runtime-stability checks:

* Backend and proxy health passed.
* No `negative token_id`.
* No `out-of-vocab token_id`.
* No `fallback argmax`.
* No `finite=0` or logits NaN marker.
* No `NRT_EXEC` or `NRT_RESOURCE` during validation.

Cold chat validation passed HTTP 200 at:

```text
146, 160, 485, 505, 526, 1225, 2048, 2049, 2500
```

Natural-language smoke:

```text
Prompt: In one concise paragraph, explain why smaller batch sizes can reduce memory bandwidth pressure during inference.
Output: coherent explanatory paragraph, no mojibake, no repeated-exclamation failure.
```

Simple factual prompts:

```text
2 + 2 -> 4
capital of France -> Paris
```

Strict semantic caveat:

* Exact marker copy still fails. Example:
  `Return exactly this marker: BASELINE_OK_27B` produced template-like text
  beginning with `BASED ON THE INSTRUCTIONS PROVIDED...`.
* Exact `ALPHA-123` copy truncated to `ALPHA`.
* Multi-turn recall of `ZX-417` truncated to `ZX`.

Therefore this artifact should be described as:

```text
runtime-stable slow-prefix decode baseline; not strict semantic-quality-clean
```

## Speed Baseline

Usage-accounted streaming context bench:

```bash
/opt/aws_neuronx_venv_pytorch_inference_vllm_0_16/bin/python3 \
  validation_scripts/qwen36_chat_completion_context_bench.py \
  --base-url http://127.0.0.1:8000 \
  --model /home/ubuntu/models/Qwen3.6-27B \
  --model-path /home/ubuntu/models/Qwen3.6-27B \
  --lengths 512,16384 \
  --turns 1 \
  --repeats 1 \
  --concurrency 1 \
  --max-tokens 64 \
  --timeout 900 \
  --unique-per-request \
  --output-json /home/ubuntu/validation_logs/fp8_256k_decode_nki/head_norowscale_context_bench_512_16k_20260605T1240Z.json
```

Results:

```text
512 target:
  prompt tokens: 511
  TTFT: 0.9689s
  effective prompt tok/s: 527.2

16k target:
  prompt tokens: 16373
  TTFT: 46.4619s
  effective prompt tok/s: 352.4
```

This is the expected slow-prefix baseline class. It is not the >=3k tok/s
prefill target.

## Use As Comparison Anchor

When comparing against `codex/nki-deltanet-multihead-cte`, preserve these
invariants first:

* no invalid token fallback
* no Neuron runtime OOB/resource crash
* `/v1/chat/completions` through the proxy stays live
* cold boundaries through 2500 return HTTP 200 and non-mojibake text
* split-QKV TKG row-scale behavior is treated as a one-variable experiment,
  not assumed safe

