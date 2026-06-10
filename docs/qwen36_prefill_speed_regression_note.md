# Qwen3.6 FP8 Cold Prefill Speed Note

Date: 2026-05-28

## Reference Full-FP8 Result

Branch: `codex/full-fp8-qwen36`

Result file:
`profile_artifacts/qwen36_cte512_openai_20260521/qwen36_128k_fp8_cte512_openai_16k_perf_patchedscheduler.json`

Artifact:
`/mnt/trainium_artifacts/qwen_artifacts/qwen36_27b_128k_fp8_mlp_edgebf16_hybrid_apc_nki_chunked_b256_cte256_512_pfx16k_slots64_20260521T092332Z`

Runtime config highlights:

- `HYBRID_APC_PREFILL_CHUNK_TOKENS=512`
- `CTE_BUCKETS=256,512`
- `MAX_MODEL_LEN=65536`
- `SEQ_LEN=131072`
- patched scheduler run: `patchedscheduler16k_20260521T154444Z`

Cold prefill results:

| Target | Prompt tokens | TTFT s | Prefill tok/s |
|---:|---:|---:|---:|
| 1024 | 1007 | 1.881 | 535.33 |
| 4096 | 4085 | 6.942 | 588.45 |
| 8192 | 8189 | 13.489 | 607.11 |
| 16384 | 16381 | 27.539 | 594.82 |

Cold prefill average: `581.43 tok/s`.

## Regression Found On Current 256K Decode Artifact

Current artifact:
`/mnt/trainium_artifacts/qwen_artifacts/qwen36_27b_256k_fp8_full_lmheadfp8_kvfp8_hybrid_apc_nki_decode_stable_gdnrecbfloat16_sampletokens_b256_cte256_512_pfx16k_slots64_async_20260528T192955Z_trirowfix_rerun`

The first live restart served the correct artifact but used:

- `HYBRID_APC_PREFILL_CHUNK_TOKENS=0`
- startup default selected `--max-num-batched-tokens 256`

That caused 16K cold prefill to run as roughly 64 prefill scheduler/kernel chunks instead of 32.
The live 16K prefill measurement was about `279 tok/s`.

## Runtime Fix Applied

Restarted the live OpenAI-compatible server on TRN2 with:

- `--hybrid-apc-prefill-chunk-tokens 512`
- confirmed backend log: `Chunked prefill is enabled with max_num_batched_tokens=512`

Restart/eval root:
`/home/ubuntu/validation_logs/fp8_256k_decode_nki/live_trirowfix_chunk512_retry_20260528T205909Z`

Cold prefill benchmark:
`/home/ubuntu/validation_logs/fp8_256k_decode_nki/live_trirowfix_chunk512_retry_20260528T205909Z/prefill_bench/proxy_live_nov1.json`

Results with `max_tokens=1` through live proxy:

| Target | Prompt tokens | TTFT s | Prefill tok/s |
|---:|---:|---:|---:|
| 1024 | 1024 | 2.965 | 345.37 |
| 4096 | 4092 | 9.692 | 422.22 |
| 8192 | 8187 | 19.026 | 430.31 |
| 16384 | 16378 | 38.660 | 423.64 |

Average: `405.39 tok/s`.

## Interpretation

The 256-token chunk default was a real regression and fixing it recovered 16K cold prefill from about `279 tok/s` to about `424 tok/s`.

This does not recover the old full-FP8 branch result of about `595 tok/s` at 16K / `581 tok/s` average. The remaining gap is likely due to artifact/config path differences:

- old result used the 128K full-FP8 MLP edge-bf16 artifact;
- current result uses the newer 256K decode-focused artifact with FP8 lm_head, KV FP8, GDN bf16, sampletokens, and stable non-split decode settings;
- both use CTE 256/512 and prefix through 16K, but the broader runtime/config path is not identical.

When comparing cold prefill speed, use `max_tokens=1`, `usage.prompt_tokens / ttft_seconds`, and keep the prefill chunk size explicit.

## Branch Deep Dive

The current branch, `codex/nki-deltanet-decode-step`, is layered on top of
`codex/full-fp8-qwen36`. The old fast 16K numbers are present in both branches
under `profile_artifacts/qwen36_cte512_openai_20260521`, but they came from the
older 128K full-FP8 artifact and launcher config, not from the current 256K
decode-focused artifact.

The relevant old result used:

- artifact: `qwen36_27b_128k_fp8_mlp_edgebf16_hybrid_apc_nki_chunked_b256_cte256_512_pfx16k_slots64_20260521T092332Z`
- runtime max model length: `65536`
- compiled seq length: `131072`
- CTE buckets: `256,512`
- prefix buckets: `pfx16k`
- `HYBRID_APC_PREFILL_CHUNK_TOKENS=512`

The current trirow artifact uses:

- artifact: `qwen36_27b_256k_fp8_full_lmheadfp8_kvfp8_hybrid_apc_nki_decode_stable_gdnrecbfloat16_sampletokens_b256_cte256_512_pfx16k_slots64_async_20260528T192955Z_trirowfix_rerun`
- runtime max model length: `262144`
- compiled seq length: `262144`
- CTE buckets: `256,512`
- prefix buckets: `pfx16k`
- FP8 `lm_head`, FP8 KV cache, on-device sample-tokens path, and decode NKI path

There is also a later validated 256K full-FP8 branch path in
`codex/full-fp8-qwen36`:

- artifact: `qwen36_27b_256k_fp8_full_lmheadbf16_hybrid_apc_prod_pfx256k_segcte512stream_qpack4_boundfix_nki_fusedstable_directsolve_hostlogits_b256_cte3072_pfx256k_pa1025_slots64_tkg262144_async_20260527T052822Z`
- gated config: `pfx256k_segcte512stream_qpack4_boundfix_pa1025`
- max model length / seq length: `262144`
- CTE/prefix pair: `3072:262144`
- backend: segmented CTE, segment size `512`
- result: 261888-token cold prefill+decode `551.97s`, effective cold prompt throughput `474.46 tok/s`, warm refill+decode `10.76s`

So 256K max length is viable, but the validated 256K fast-prefill path is the
segmented CTE512 streaming + qpack4 + boundfix path. The current decode artifact
is not that lineage.

## Launcher Fix

The server launcher and offline runner previously defaulted Hybrid APC chunked
prefill to the smallest checkpoint-aligned compiled CTE bucket. For a
`256,512` artifact this picked `256` unless `--hybrid-apc-prefill-chunk-tokens
512` was passed explicitly. The code now defaults to the largest
checkpoint-aligned compiled CTE bucket, so the same artifact defaults to `512`.

The attempted launcher workaround that capped transient `max_context_length`
below the compiled prompt length was removed. vLLM-Neuron requires
`max_prompt_length` and `override_neuron_config.max_context_length` to match the
compiled artifact's max prompt length, so a 256K artifact cannot be served as a
runtime-only 128K/64K artifact.

## Trirow / Direct Solve Impact

The trirow update changes the DeltaNet fused CTE triangular solve from a full
128-row matmul per solve row to a one-row matmul plus a compiler-safe full-tile
one-hot update. It affects CTE/prefill, not token-generation decode TPOT.

Measured end-to-end impact was small:

- stable pre-trirow 16K eval: about `45.94s` TTFT, roughly `356 tok/s`
- trirow rerun 16K eval: about `45.71s` TTFT, roughly `358 tok/s`

The direct triangular solve change is not the cause of the large cold-prefill
speed loss. The large loss was first from the 256-token runtime chunk default,
and the remaining gap is from artifact/config lineage differences.

## Second Deep Dive: KV FP8 Cache Dequantization

The strongest remaining suspect is the current artifact's `--enable-kv-cache-quant`
path. The old fast pfx16k/pfx128k artifacts had `kv_cache_quant=false`. The
current 256K decode artifact has `kv_cache_quant=true` and `kv_quant_config` set
to direct-cast FP8.

The block KV cache manager currently dequantizes the whole block-layout cache
before selecting the active prefix blocks:

- `src/neuronx_distributed_inference/modules/kvcache/block_kv_cache_manager.py`
  fetches `k_cache`/`v_cache`, calls `_dequantize_cache(...)`, then calls
  `_get_block_cache_and_reshape_bhsd(...)`.
- Direct-cast dequantization is a tensor dtype conversion in
  `src/neuronx_distributed_inference/modules/kvcache/kv_cache_manager.py`.
- The current compile script enables this with `--enable-kv-cache-quant`.

For the current 256K runtime, this means each chunked CTE read can convert far
more KV cache than the active prefix needs. At 16K prompt length, the active
prefix is at most 64 blocks, but the compiled/runtime cache capacity is about
1025 blocks. That is roughly a 16x over-read/convert risk in the prefix-cache
read path before attention work starts.

This explains why the remaining regression survives the chunk-size fix:

- old fast pfx16k: `kv_cache_quant=false`, no full-cache dequant before block
  selection;
- old pfx128k: also `kv_cache_quant=false`, and still reached about `606 tok/s`
  at 16K even with a larger prefix-capable artifact;
- current 256K decode artifact: `kv_cache_quant=true`, so the compiled CTE graph
  contains extra KV cache conversion work.

Because this order is traced into the compiled CTE NEFF, a Python-only patch does
not fix the already-compiled artifact. The safe A/B is to compile a 256K
prefill-recovery artifact with KV cache quantization disabled first. The code fix
for a later KV-FP8 artifact is to select active block-cache entries first and
only dequantize the selected prefix/cache slice.

Code fix applied:

- `src/neuronx_distributed_inference/modules/kvcache/block_kv_cache_manager.py`
  now selects/reshapes active block-cache entries before calling
  `_dequantize_cache(...)`.
- `test/unit/modules/kvcache/test_block_kv_cache_manager.py` now verifies the
  prefix-cache quantized path dequantizes the selected `[B, H, active_prefix, D]`
  slice rather than the full PA cache.
- Remote Neuron-venv targeted test passed with
  `NEURON_PLATFORM_TARGET_OVERRIDE=trn2`: `3 passed, 17 deselected`.

Compile under test:

```text
host: ubuntu@16.26.225.228
pid: 125870
artifact:
  /mnt/trainium_artifacts/qwen_artifacts/qwen36_27b_256k_fp8_full_lmheadfp8_kvfp8_hybrid_apc_nki_decode_stable_gdnrecbfloat16_sampletokens_b256_cte256_512_pfx16k_slots64_async_20260529T073442Z_kvselectfix
log:
  /home/ubuntu/validation_logs/fp8_256k_decode_nki/qwen36_27b_256k_fp8_full_lmheadfp8_kvfp8_hybrid_apc_nki_decode_stable_gdnrecbfloat16_sampletokens_b256_cte256_512_pfx16k_slots64_async_20260529T073442Z_kvselectfix_compile.log
```

This compile keeps KV FP8 enabled, so it tests whether selecting before
dequantization recovers prefill throughput without giving up the decode-memory
benefit.

## KV-Select-Fix Result

The KV-select-before-dequant artifact compiled, transferred, and was loaded on
the TRN2 OpenAI-compatible server:

```text
/mnt/trainium_artifacts/qwen_artifacts/qwen36_27b_256k_fp8_full_lmheadfp8_kvfp8_hybrid_apc_nki_decode_stable_gdnrecbfloat16_sampletokens_b256_cte256_512_pfx16k_slots64_async_20260529T073442Z_kvselectfix
```

Live runtime config:

- max model/context length: `262144`
- CTE buckets: `256,512`
- Hybrid APC chunk size: `512`
- prefix buckets: `256,512,1024,2048,4096,8192,16384`
- KV cache quantization: `true`, direct-cast FP8
- stable non-split QKV TKG path, DeltaNet decode NKI enabled

True cold prefill after restart remained below the old full-FP8 branch result:

| Target | Prompt tokens | TTFT s | Prefill tok/s |
|---:|---:|---:|---:|
| 1024 | 1024 | 2.977 | 343.9 |
| 4096 | 4092 | 9.681 | 422.7 |
| 8192 | 8187 | 19.054 | 429.7 |
| 16384 | 16378 | 38.674 | 423.5 |

So selecting active blocks before dequantization did not recover old cold
prefill throughput. That weakens the "full-cache dequant before block select"
hypothesis for cold prefill. The remaining likely costs are:

- KV FP8 direct-cast writes/conversions during CTE cache update, not just reads;
- the 256K compiled max length / 1024 PA block capacity versus the old 128K /
  512-block artifact;
- the current quantization recipe (`fp8_full_lmheadfp8_kvfp8`) versus the old
  `fp8_mlp_edgebf16` lineage;
- the current decode-focused artifact using flat `attention_cte`, while the
  validated 256K full-FP8 path used segmented CTE512 streaming + qpack4 +
  boundfix.

Decode and coherence on this artifact are good:

- 8K prompt, 128 generated tokens: token-level TPOT about `23.9 ms/token`,
  about `41.8 tok/s`, computed from streaming `usage.completion_tokens`.
- Thinking smoke with `max_tokens=2048` emitted `<think>...</think>` plus a
  final answer and stopped normally.
- Semantic checks passed for arithmetic, marker copy, and short multi-turn
  recall.

Next A/B to isolate the remaining prefill regression: compile the same 256K
stable artifact without `--enable-kv-cache-quant`, keeping CTE buckets,
prefix buckets, chunk size, slots, decode NKI, and sampling path unchanged. If
cold prefill recovers toward 580 tok/s, KV FP8 write/direct-cast overhead is the
main cause. If it does not, compile the old quantization recipe
(`fp8_mlp_edgebf16`, no KV FP8) on the current branch, then test segmented CTE512
streaming/qpack4 as the 256K route.

## Error Notes

- First chunk-512 restart failed before model load because the script was launched without the Neuron vLLM virtualenv on `PATH`.
  - Error: `exec: python: not found`
  - Related sitecustomize error: `ModuleNotFoundError: No module named 'transformers'`
  - Mitigation: restarted with `/opt/aws_neuronx_venv_pytorch_inference_vllm_0_16/bin` first in `PATH`.

- First benchmark attempts used `--base-url .../v1`, but `qwen36_chat_completion_context_bench.py` already appends `/v1`.
  - Error: HTTP `404 {"detail":"Not Found"}`
  - Mitigation: reran with `--base-url http://127.0.0.1:8000`.

- Runtime-only attempt to serve the 256K artifact with old 128K/64K limits failed.
  - Attempt 1 error: `AssertionError: max_context_length cannot be more than max_length`
  - Attempt 2 error: `RuntimeError: Configuration mismatch: max_prompt_length in --additional-config (131072) does not match the Neuron model's compiled max prompt length (262144)`
  - Attempt 3 error: `ValueError: Conflicting max_prompt_length settings: override_neuron_config specifies max_context_length 131072 but the max_prompt_length is 262144`
  - Root cause: vLLM-Neuron validates runtime prompt/context limits against the compiled artifact. A 256K artifact must be served with 256K `max_prompt_length` / `max_context_length`.
  - Mitigation: restored the live server with valid 256K limits and `max_num_batched_tokens=512`.

- First local attempt to run the block KV cache unit test failed because the
  repo package path was missing.
  - Command: `python3 -m pytest test/unit/modules/kvcache/test_block_kv_cache_manager.py -k ...`
  - Error: `ModuleNotFoundError: No module named 'neuronx_distributed_inference'`
  - Mitigation: reran with `PYTHONPATH=src`.

- Second local attempt reached repo imports but failed because the local Mac
  environment does not have Neuron dependencies installed.
  - Command: `PYTHONPATH=src python3 -m pytest test/unit/modules/kvcache/test_block_kv_cache_manager.py -k ...`
  - Error: `ModuleNotFoundError: No module named 'neuronx_distributed'`
  - Mitigation: copied the changed file and test to the compile host and used
    `/opt/aws_neuronx_venv_pytorch_inference_vllm_0_16`.

- First remote unit-test attempt on the CPU compile host failed because Neuron
  target override was not set.
  - Command: remote `PYTHONPATH=src python -m pytest ...`
  - Error: `RuntimeError: Unsupported Platform - r7i.24xlarge`
  - Mitigation: reran with `NEURON_PLATFORM_TARGET_OVERRIDE=trn2`.
  - Verification: targeted remote test passed: `3 passed, 17 deselected`.

- Remote git status check in the compile-host workdir failed because that workdir
  is a copied source tree rather than a git checkout.
  - Command: remote `git status --short ...`
  - Error: `fatal: not a git repository (or any of the parent directories): .git`
  - Mitigation: verified files directly and used `scp` to place the patch.

- First compile launch attempt hit local sandbox/network denial.
  - Command: `ssh ... TS=... ./tmp_compile_qwen256k_fp8_full_decode_stable_sampletokens.sh`
  - Error: `ssh: connect to host 16.26.225.228 port 22: Operation not permitted`
  - Mitigation: reran with approved escalated SSH.
  - Verification: compile launched as PID `125870`; log path
    `/home/ubuntu/validation_logs/fp8_256k_decode_nki/qwen36_27b_256k_fp8_full_lmheadfp8_kvfp8_hybrid_apc_nki_decode_stable_gdnrecbfloat16_sampletokens_b256_cte256_512_pfx16k_slots64_async_20260529T073442Z_kvselectfix_compile.log`.
