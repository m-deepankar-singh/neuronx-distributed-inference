# Qwen3.6 Coherent Prefill-Speed Rebuild Notes

## 2026-06-05 coherent rebuild from decode-step baseline

Source branch: `codex/qwen36-prefill-speed-coherent`

Local clone: `/private/tmp/inferentia-gdn-prefill-speed-coherent`

Compile host: `ubuntu@16.26.135.243`

Runtime host target: `ubuntu@16.26.184.190`

Baseline source commit on compile host before rebuild patches: `834543e`

Ported rebuild commits:

- `157624a` remote / `4ee0abb` local: diagnostics and launch/compile plumbing
- `6be38e1` remote / `bac881e` local: Hybrid APC same-request active carry
- `dc54d22` remote / `0c0bf94` local: opt-in segmented DeltaNet prefill path
- `d00156d` remote / `f2c775c` local: FP8 compile policy flags

### Error log

1. Local unit-test invocation used an invalid dotted module path.
   - Command: `python3 -m unittest contrib.models.Qwen3.6-27B.test.unit.test_hybrid_apc_manager`
   - Error: `ModuleNotFoundError: No module named 'contrib.models.Qwen3'`
   - Context: local clean clone, after Hybrid APC active-carry edits.
   - Root cause: the `Qwen3.6-27B` directory name contains characters that cannot be addressed as a dotted Python module.
   - Mitigation: use unittest discovery with `PYTHONPATH=src`.
   - Verification: `PYTHONPATH=src python3 -m unittest discover -s contrib/models/Qwen3.6-27B/test/unit -p test_hybrid_apc_manager.py` passed 56 tests.

2. Local unit-test discovery initially missed the source path.
   - Command: `python3 -m unittest discover -s contrib/models/Qwen3.6-27B/test/unit -p test_hybrid_apc_manager.py`
   - Error: `ModuleNotFoundError: No module named 'neuronx_distributed_inference'`
   - Context: local clean clone, system Python.
   - Root cause: repo `src` was not on `PYTHONPATH`.
   - Mitigation: rerun with `PYTHONPATH=src`.
   - Verification: Hybrid APC and scheduler patch suites passed.

3. Remote tests under system Python did not have Torch.
   - Command: `python3 -m unittest discover ...` on `ubuntu@16.26.135.243`
   - Error: `ModuleNotFoundError: No module named 'torch'`
   - Context: compile host after applying the first three rebuild patches.
   - Root cause: system Python is not the Neuron/Torch environment.
   - Mitigation: use `/home/ubuntu/venvs/neuron_230_segmented_cte/bin/python`.
   - Verification: remote Hybrid APC suite passed 56 tests and scheduler suite passed 51 tests.

4. First remote compile launch could not execute the patch-created script.
   - Command: `./tmp_compile_qwen32k_segcte2048_gdnseg512.sh`
   - Error: `bash: line 1: ./tmp_compile_qwen32k_segcte2048_gdnseg512.sh: Permission denied`
   - Context: compile host source tree after `git am`; script file mode was not executable.
   - Root cause: patch-created scripts did not preserve executable mode.
   - Mitigation: invoke the driver as `bash tmp_compile_qwen32k_segcte2048_gdnseg512.sh`.
   - Verification: the next launch entered the Python compile entrypoint.

5. Second remote compile launch failed before compilation because the baseline compile harness lacked ported CLI flags.
   - Artifact base: `qwen36_32k_fp8_fp8all_lmheadfp8_gatesfp8_kvbf16_qkvnki_tiled_segmented_cte512_gdnseg512_cte2048_pfx32k_slots64_20260605T131438Z_coherent_rebuild_direct_scan0`
   - Log: `/home/ubuntu/validation_logs/fp8_256k_decode_nki/qwen36_32k_fp8_fp8all_lmheadfp8_gatesfp8_kvbf16_qkvnki_tiled_segmented_cte512_gdnseg512_cte2048_pfx32k_slots64_20260605T131438Z_coherent_rebuild_direct_scan0_compile.log`
   - Error: `qwen36_27b_compile_fp8.py: error: unrecognized arguments: --fp8-quantize-linear-attn-gates --disable-context-encoding-argmax-kernel`
   - Root cause: the compile driver already requested old-policy/debug flags, but the decode-step baseline compile entrypoint did not parse them.
   - Mitigation: added `--fp8-quantize-linear-attn-gates`, `--fp8-exclude-groups`, `--postprocess-only`, `--disable-argmax-kernel`, and `--disable-context-encoding-argmax-kernel` support in `qwen36_27b_compile_fp8.py`.
   - Verification: local and remote Python compile checks passed; local and remote Hybrid APC/scheduler unit suites passed.

6. CTE2048 compile with stock QKV CTE NKI failed during HLO generation.
   - Artifact base: `qwen36_32k_fp8_fp8all_lmheadfp8_gatesfp8_kvbf16_qkvnki_tiled_segmented_cte512_gdnseg512_cte2048_pfx32k_slots64_20260605T132224Z_coherent_rebuild_direct_scan0`
   - PID: `46754`
   - Log: `/home/ubuntu/validation_logs/fp8_256k_decode_nki/qwen36_32k_fp8_fp8all_lmheadfp8_gatesfp8_kvbf16_qkvnki_tiled_segmented_cte512_gdnseg512_cte2048_pfx32k_slots64_20260605T132224Z_coherent_rebuild_direct_scan0_compile.log`
   - Failing path: `src/neuronx_distributed_inference/modules/attention/gqa.py:_kernel_qkv_forward` calling `/home/ubuntu/nki-library-2.30/src/nkilib_src/nkilib/core/qkv/qkv_cte.py`
   - Error: `[NCC_INKI016] Kernel validation exception: [QKV CTE Kernel] weights.shape[1] must be <= 4096, but got 5120.`
   - Inputs/flags: `ENABLE_QKV_NKI_KERNELS=1`, `CTE_BUCKETS_RAW=2048`, `PREFIX_CTE_ATTENTION_BACKEND=segmented_cte`, `PREFIX_CTE_ATTENTION_SEGMENT_SIZE=512`, `QWEN36_DELTANET_FUSED_SEGMENT_TOKENS=512`, `QWEN36_DELTANET_MULTIHEAD_CTE=0`, `ENABLE_KV_CACHE_QUANT=0`, `QUANTIZE_LM_HEAD=1`, `FP8_QUANTIZE_LINEAR_ATTN_GATES=1`.
   - Refined root cause after NKI docs/source inspection: this was not a hidden-width limit. AWS/NKI QKV documents `fused_qkv_weights` as `[H, I]` and QKV CTE validates `I <= 4096`; Qwen3.6 TP=4 has `H=5120` and local `I=(24/4 + 2*(4/4))*256 = 2048`. The observed `weights.shape[1]=5120` means the FP8 fused-QKV parameter reached the kernel as `[I, H]`, so quantization had dropped the transposed `[H, I]` loader/layout contract.
   - Secondary root cause: the wrapper did not pass `QuantizationType.ROW` or `qkv_w_scale` for FP8 per-channel QKV weights, even though the NKI QKV API supports row-scale dequantization.
   - Mitigation: do not use stock `--enable-qkv-nki-kernels` for the first coherent-speed anchor. After the coherent standard-QKV anchor is green, fix QKV NKI as one isolated source slice by preserving the transposed layout through quantized module creation and passing row scales to the kernel.
   - Verification: standard-QKV fallback compile/runtime became the coherent anchor; QKV NKI source fix pending compile validation.

7. Standard-QKV fallback compile exposed a remaining Python `raise` inside the segmented CTE NKI kernel.
   - Artifact base: `qwen36_32k_fp8_fp8all_lmheadfp8_gatesfp8_kvbf16_standard_qkv_segmented_cte512_gdnseg512_cte2048_pfx32k_slots64_20260605T132507Z_coherent_rebuild_stdqkv_direct_scan0`
   - PID: `47806`
   - Log: `/home/ubuntu/validation_logs/fp8_256k_decode_nki/qwen36_32k_fp8_fp8all_lmheadfp8_gatesfp8_kvbf16_standard_qkv_segmented_cte512_gdnseg512_cte2048_pfx32k_slots64_20260605T132507Z_coherent_rebuild_stdqkv_direct_scan0_compile.log`
   - Failing path: `src/neuronx_distributed_inference/modules/attention/nki_kernels/qwen_segcte256/attention_segmented_cte_256.py:180`
   - Error: `NKI does not support 'raise' statements; use 'if/else' control flow within kernels, or 'assert' for fatal errors`
   - Inputs/flags: `ENABLE_QKV_NKI_KERNELS=0`, `CTE_BUCKETS_RAW=2048`, `PREFIX_CTE_ATTENTION_BACKEND=segmented_cte`, `PREFIX_CTE_ATTENTION_SEGMENT_SIZE=512`, `QWEN36_DELTANET_FUSED_SEGMENT_TOKENS=512`, `ENABLE_KV_CACHE_QUANT=0`, `QUANTIZE_LM_HEAD=1`, `FP8_QUANTIZE_LINEAR_ATTN_GATES=1`.
   - Root cause: Slice C replaced some NKI-hostile raises, but missed the `k_pre_transposed` guard in the segmented CTE load helper.
   - Mitigation: replace the `raise ValueError` with `kernel_assert(not k_pre_transposed, ...)`.
   - Verification: pending retry compile.

8. Completed standard-QKV CTE2048 compile produced BF16 recurrent checkpoint-bank slots.
   - Artifact base: `qwen36_32k_fp8_fp8all_lmheadfp8_gatesfp8_kvbf16_standard_qkv_segmented_cte512_gdnseg512_cte2048_pfx32k_slots64_20260605T132739Z_coherent_rebuild_stdqkv2_direct_scan0`
   - Log: `/home/ubuntu/validation_logs/fp8_256k_decode_nki/qwen36_32k_fp8_fp8all_lmheadfp8_gatesfp8_kvbf16_standard_qkv_segmented_cte512_gdnseg512_cte2048_pfx32k_slots64_20260605T132739Z_coherent_rebuild_stdqkv2_direct_scan0_compile.log`
   - Evidence: `COMPILE_DONE` was present and safetensor inspection showed each TP shard had 48 recurrent checkpoint tensors with dtype `torch.bfloat16` and 48 conv checkpoint tensors with dtype `torch.bfloat16`.
   - Inputs/flags: `--gdn-recurrent-cache-dtype float32`, `--gdn-conv-cache-dtype bfloat16`, `ENABLE_QKV_NKI_KERNELS=0`, `CTE_BUCKETS_RAW=2048`, `PREFIX_CTE_ATTENTION_BACKEND=segmented_cte`.
   - Root cause: `_ensure_hybrid_checkpoint_weights` used `NeuronConfig.torch_dtype` for both recurrent and conv checkpoint-bank tensors and skipped existing checkpoint-bank keys, so postprocess-only could not correct an already-added BF16 recurrent bank.
   - Mitigation: make `_ensure_hybrid_checkpoint_weights` derive recurrent and conv dtypes from `gdn_recurrent_cache_dtype` / `gdn_conv_cache_dtype`, and rewrite existing checkpoint-bank tensors when their dtype does not match.
   - Verification: postprocess-only repair changed recurrent banks to `torch.float32`, but runtime load then failed because the compiled graph expected BF16 checkpoint-bank tensors.

9. Postprocess-only FP32 recurrent-bank repair made the completed artifact unloadable.
   - Runtime host: `ubuntu@16.26.184.190`
   - Runtime log: `/home/ubuntu/validation_logs/fp8_256k_decode_nki/coherent_rebuild_stdqkv2_runtime_20260605T141619Z.log`
   - Artifact: `/mnt/trainium_artifacts/qwen_artifacts/qwen36_32k_fp8_fp8all_lmheadfp8_gatesfp8_kvbf16_standard_qkv_segmented_cte512_gdnseg512_cte2048_pfx32k_slots64_20260605T132739Z_coherent_rebuild_stdqkv2_direct_scan0`
   - Error: `Incorrect data type for checkpoint key hybrid_gdn_checkpoint_cache.recurrent_slots.0: received float, expected c10::BFloat16`
   - Root cause: the compiled `model.pt` was traced with BF16 checkpoint-bank parameter dtype. Rewriting safetensors to FP32 after compile cannot change the traced model's expected parameter dtype.
   - Mitigation: for this artifact, restore recurrent checkpoint banks to BF16 and launch with `--gdn-recurrent-cache-dtype bfloat16`. A true FP32 recurrent checkpoint-bank artifact requires a source fix before compile so the traced checkpoint-bank parameters are FP32.
   - Verification: pending BF16 restore and runtime launch.

10. First inline runtime probe used the wrong Python environment on the runtime host.
   - Runtime host: `ubuntu@16.26.184.190`
   - Command: `python3 - <<'PY' ... from transformers import AutoTokenizer ...`
   - Error: `ModuleNotFoundError: No module named 'transformers'`
   - Context: probing the live `coherent_rebuild_stdqkv2_direct_scan0` artifact through `http://127.0.0.1:8000/v1/chat/completions`.
   - Root cause: system Python on the runtime host is not the Neuron/vLLM environment.
   - Mitigation: use `/opt/aws_neuronx_venv_pytorch_inference_vllm_0_16/bin/python` for tokenizer-backed probes.
   - Verification: exact-token probes then ran successfully.

11. CTE2048 artifact is coherent through 2048 tokens but crashes on the first cached continuation at 2049.
   - Runtime host: `ubuntu@16.26.184.190`
   - Runtime log: `/home/ubuntu/validation_logs/fp8_256k_decode_nki/coherent_rebuild_stdqkv2_bf16rec_runtime_20260605T142519Z.log`
   - Artifact: `/mnt/trainium_artifacts/qwen_artifacts/qwen36_32k_fp8_fp8all_lmheadfp8_gatesfp8_kvbf16_standard_qkv_segmented_cte512_gdnseg512_cte2048_pfx32k_slots64_20260605T132739Z_coherent_rebuild_stdqkv2_direct_scan0`
   - Inputs/flags: standard QKV, CTE2048, segmented CTE512 prefix attention, GDN segment 512, KV BF16, recurrent checkpoint/cache BF16, Hybrid APC enabled, chunked prefill enabled.
   - Passing probes: exact prompt lengths `146`, `160`, `485`, `505`, `526`, `1225`, and `2048` all returned coherent text with matching `usage.prompt_tokens`.
   - Failing probe: exact prompt length `2049`; subsequent `2500`, `4092`, and `4096` requests failed because the engine was already dead.
   - Exact runtime error: `RuntimeError: shape '[1, 1]' is invalid for input of size 0` at `src/neuronx_distributed_inference/models/model_wrapper.py:1874`, called from `_process_async_inputs`.
   - Log evidence immediately before crash: `tag=token_generation_model index=0 name=input_ids shape=(1, 0)`, `position_ids shape=(1, 1) min=2048`, `slot_mapping shape=(1, 1) min=8448`, `computed_context_lens min=2048`, and `num_queries min=0`. The dumped scheduler output showed `scheduled_cached_reqs`, `num_computed_tokens=[2048]`, `num_scheduled_tokens=1`, `request_prefix_len=2049`, `active_suffix_len=1`, and `full_input_ids` present in Hybrid APC metadata.
   - Root cause hypothesis: vLLM V1 cached/chunked-prefill continuation schedules one active suffix token after the 2048-token CTE chunk, but the Neuron token-generation input path does not merge the request metadata for non-context executions. The model wrapper receives an impossible tuple: empty `input_ids` but non-empty active `position_ids` and `slot_mapping`.
   - Mitigation applied: add `_repair_cached_chunked_prefill_tkg_inputs` in `async_execution.py`; for non-context execution, attach Hybrid APC owner metadata, reconstruct the active suffix from `full_input_ids[computed_context_lens:computed_context_lens + active_suffix_len]`, set `num_queries` to the suffix length, and then build inert Hybrid APC args.
   - Verification: local `PYTHONPATH=src python3 -m py_compile src/neuronx_distributed_inference/modules/async_execution.py test/unit/modules/test_async_execution.py` passed; local `PYTHONPATH=src python3 -m unittest test.unit.modules.test_async_execution.TestCachedChunkedPrefillTkgRepair` passed. Runtime verification is pending source sync and vLLM restart.

12. After fixing 2049, cold 2500 fails because Hybrid APC is prepared twice for the same chunk continuation.
   - Runtime host: `ubuntu@16.26.184.190`
   - Runtime log: `/home/ubuntu/validation_logs/fp8_256k_decode_nki/coherent_rebuild_stdqkv2_cachedrepair_cold_runtime_20260605T144421Z.log`
   - Passing probe after fix #11: exact prompt length `2049` returned coherent text with `usage.prompt_tokens=2049`.
   - Failing probe: exact prompt length `2500`, unique cold marker to avoid cross-request prefix reuse.
   - Error: `ValueError: hybrid APC received an attention prefix hit without a matching GDN checkpoint; scheduler must intersect attention KV hits with GDN checkpoint hits or disable prefix reuse for this request`
   - Log evidence: first chunk prepared and committed `commit_prefix_len=2048 commit_slot=0`; second chunk prepared by async execution with `attention_hit_len=2048`, `request_prefix_len=2500`, `prepared_shape=(1, 452)`, `computed=tensor([[2048]])`, `num_queries=tensor([[452]])`, `restore_mask=tensor([0])`, `commit_mask=tensor([0])`; then `modeling_qwen35.py:_prepare_hybrid_apc_pad_inputs` prepared the same row again inside `pad_inputs` and raised.
   - Root cause: `pad_inputs` prepares when both Hybrid APC masks are zero. That is correct for raw cold inputs, but wrong for an already-prepared same-request continuation whose masks are intentionally inert because active carry is authoritative.
   - Mitigation applied: async execution now sets `_qwen36_hybrid_apc_skip_pad_prepare_once` on the selected model wrapper whenever it has already prepared Hybrid APC inputs; `Qwen35ModelWrapper._prepare_hybrid_apc_pad_inputs` consumes that one-shot flag and returns the prepared args unchanged.
   - Verification: local `PYTHONPATH=src python3 -m py_compile src/neuronx_distributed_inference/modules/async_execution.py contrib/models/Qwen3.6-27B/src/modeling_qwen35.py` passed; local `PYTHONPATH=src python3 -m unittest test.unit.modules.test_async_execution.TestCachedChunkedPrefillTkgRepair` passed. Runtime verification after source sync passed exact prompt lengths `2500`, `4092`, `4096`, `8192`, and `16384` with coherent output and matching `usage.prompt_tokens`.

13. The coherent CTE2048 standard-QKV anchor is too slow for the prefill-speed target.
   - Runtime host: `ubuntu@16.26.184.190`
   - Runtime log: `/home/ubuntu/validation_logs/fp8_256k_decode_nki/coherent_rebuild_stdqkv2_doubleprepfix_runtime_20260605T144922Z.log`
   - Artifact: `/mnt/trainium_artifacts/qwen_artifacts/qwen36_32k_fp8_fp8all_lmheadfp8_gatesfp8_kvbf16_standard_qkv_segmented_cte512_gdnseg512_cte2048_pfx32k_slots64_20260605T132739Z_coherent_rebuild_stdqkv2_direct_scan0`
   - Inputs/flags: standard QKV, CTE2048, segmented CTE512 prefix attention, GDN segment 512, KV BF16, recurrent checkpoint/cache BF16, Hybrid APC enabled.
   - Coherence evidence: exact `8192` prompt returned coherent text with `usage.prompt_tokens=8192`; exact `16384` prompt with `max_tokens=1` returned a normal first token and `usage.prompt_tokens=16384`.
   - Speed evidence: 16k `max_tokens=1` non-streaming wall time was `26.353s`, so conservative prompt throughput was `16384 / 26.353 = 621.7 prompt tok/s`.
   - Root cause hypothesis, refined: this anchor deliberately disabled QKV NKI because the FP8 fused-QKV path lost its `[H, I]` parameter layout and row-scale contract after quantization. Coherence is restored, but the speed path is still missing a validated QKV NKI rebuild.
   - Mitigation: keep this artifact as the coherent CTE2048 anchor; next speed work should port/adapt the working QKV tiled path for 5120-wide Qwen instead of reintroducing unrelated dense moat kernels.
   - Verification: final serve-log scan after 2500/4092/4096/8192/16384 showed no `negative token_id`, `out-of-vocab token_id`, `fallback argmax`, `finite=0`, `nan=`, `NRT_RESOURCE`, `EngineDeadError`, `RuntimeError`, `ValueError`, `Traceback`, or `InternalServerError`; backend and proxy health were OK.

14. QKV NKI FP8 source audit found a missing quantized layout/scale contract.
   - Local source paths: `src/neuronx_distributed_inference/modules/attention/gqa.py`, `src/neuronx_distributed_inference/modules/attention/utils.py`.
   - Remote docs/source evidence: AWS QKV API documents `fused_qkv_weights` as `[H, I]`; compile-host NKI Library 2.30 `qkv_cte_utils.py` validates `_H == H` and `I <= 4096`; `qkv_cte.py` supports `QuantizationType.ROW` with `qkv_w_scale` shape `[1, I]` or `[128, I]`.
   - Inputs/flags that expose it: `ENABLE_QKV_NKI_KERNELS=1`, `weight_dtype=fp8_full`, `QUANTIZE_LM_HEAD=1`, `FP8_QUANTIZE_LINEAR_ATTN_GATES=1`, Qwen3.6 TP=4.
   - Root cause: `GroupQueryAttention_QKV.__init__` transposed non-quantized `Wqkv.weight`, but did not install a `post_create_quantized_module_hook`, so FP8 quantization replaced the module with default `[I, H]` weight and `[I, 1]` scale metadata. `_kernel_qkv_forward` then derived `fused_qkv_size` from `Wqkv.weight.shape[1]`, mistaking hidden size `5120` for output width, and passed no row scale.
   - Mitigation applied locally: added `preprocess_quantized_qkv_nki_layer`, FP8-preserving transposed state-dict loaders, a `[128, I]` QKV row-scale loader, explicit QKV layout/scale guards, and `QuantizationType.ROW` / `qkv_w_scale` kernel arguments for quantized QKV.
   - Verification: local `py_compile` passed; remote compile-host `test/unit/modules/attention/test_gqa.py` passed 11 tests under the Neuron venv with `PATH=/opt/aws_neuronx_venv_pytorch_inference_vllm_0_16/bin:$PATH NEURON_PLATFORM_TARGET_OVERRIDE=trn2 NEURON_CC_FLAGS="--target trn2"`.

15. Local QKV unit-test run used a Python environment without Neuron/NxD.
   - Command: `PYTHONPATH=src python3 -m pytest -q test/unit/modules/attention/test_gqa.py`
   - Error: `ModuleNotFoundError: No module named 'neuronx_distributed'`
   - Context: local macOS/workspace Python after QKV NKI source edits.
   - Root cause: local Python lacks the Neuron/NxD package needed to import `neuronx_distributed_inference.modules.attention.gqa`.
   - Mitigation: use local `py_compile` for syntax and run the real unit test on `ubuntu@16.26.135.243` with `/home/ubuntu/venvs/neuron_230_segmented_cte/bin/python`.
   - Verification: local `PYTHONPATH=src python3 -m py_compile src/neuronx_distributed_inference/modules/attention/gqa.py test/unit/modules/attention/test_gqa.py` passed.

16. Remote QKV unit-test import exposed environment and circular-import issues before assertions ran.
   - Command 1: `PYTHONPATH=src /home/ubuntu/venvs/neuron_230_segmented_cte/bin/python -m pytest -q test/unit/modules/attention/test_gqa.py`
   - Error 1: `FileNotFoundError: [Errno 2] No such file or directory: 'libneuronpjrt-path'`
   - Root cause 1: `/opt/aws_neuronx_venv_pytorch_inference_vllm_0_16/bin` was not on `PATH`.
   - Command 2: same command with that venv bin on `PATH`.
   - Error 2: `RuntimeError: Unsupported Platform - r7i.24xlarge`; compile host is CPU-only and needs an explicit target override.
   - Mitigation 2: rerun with `NEURON_PLATFORM_TARGET_OVERRIDE=trn2 NEURON_CC_FLAGS="--target trn2"`.
   - Error 3 after the target override: `ImportError: cannot import name 'replicate_kv' from partially initialized module 'neuronx_distributed_inference.modules.attention.gqa'`; import path was `gqa.py -> lora_serving.__init__ -> lora_checkpoint.py -> gqa.replicate_kv`.
   - Root cause 3: `gqa.py` imported `is_lora_module` at module import time, which executes the LoRA package initializer before `replicate_kv` is defined.
   - Mitigation applied locally: replace the top-level LoRA helper import with a lazy `_is_lora_module()` wrapper and update QKV/O-proj call sites.
   - Verification: remote compile-host `PYTHONPATH=src /home/ubuntu/venvs/neuron_230_segmented_cte/bin/python -m py_compile src/neuronx_distributed_inference/modules/attention/gqa.py test/unit/modules/attention/test_gqa.py` passed with the corrected PATH/target override; remote `pytest -q test/unit/modules/attention/test_gqa.py` passed 11 tests with 11 warnings.

17. QKV NKI compile monitoring command had a local quoting error; the compile itself was unaffected.
   - Command: SSH monitor for `/home/ubuntu/validation_logs/fp8_256k_decode_nki/qwen36_32k_fp8_fp8all_lmheadfp8_gatesfp8_kvbf16_qkvnki_tiled_segmented_cte512_gdnseg512_cte2048_pfx32k_slots64_20260605T152927Z_direct_scan0_compile.log`.
   - Error: `bash: -c: line 1: unexpected EOF while looking for matching "\""`.
   - Context: compile host `ubuntu@16.26.135.243`, source `/home/ubuntu/inferentia-gdn-prefill-speed-coherent` at commit `f2c46e6`, PID `65418`, artifact `/mnt/trainium_artifacts/qwen_artifacts/qwen36_32k_fp8_fp8all_lmheadfp8_gatesfp8_kvbf16_qkvnki_tiled_segmented_cte512_gdnseg512_cte2048_pfx32k_slots64_20260605T152927Z_direct_scan0`.
   - Root cause: the monitor command mixed single quotes and a mismatched double quote around the remote `tail -80 "$log"` expression.
   - Mitigation: reran the monitor command with balanced quoting.
   - Verification: corrected monitor showed the compile still running, all HLOs completed successfully, no `QKV NKI expects`, `row-scale layout`, `NCC_INKI016`, traceback, or runtime exception in the log, and checkpoint-bank sharding had started.

18. First EC2-to-EC2 rsync of the completed QKV NKI artifact failed because compile host could not authenticate to runtime host.
   - Command: from local SSH into `ubuntu@16.26.135.243`, run `rsync -aH --partial --info=progress2 -e "ssh -o StrictHostKeyChecking=no" /mnt/trainium_artifacts/qwen_artifacts/qwen36_32k_fp8_fp8all_lmheadfp8_gatesfp8_kvbf16_qkvnki_tiled_segmented_cte512_gdnseg512_cte2048_pfx32k_slots64_20260605T152927Z_direct_scan0/ ubuntu@16.26.184.190:/mnt/trainium_artifacts/qwen_artifacts/qwen36_32k_fp8_fp8all_lmheadfp8_gatesfp8_kvbf16_qkvnki_tiled_segmented_cte512_gdnseg512_cte2048_pfx32k_slots64_20260605T152927Z_direct_scan0/`.
   - Error: `ubuntu@16.26.184.190: Permission denied (publickey).` followed by `rsync: connection unexpectedly closed (0 bytes received so far) [sender]` and `rsync error: unexplained error (code 255)`.
   - Context: compile host `ubuntu@16.26.135.243`; runtime host `ubuntu@16.26.184.190`; artifact completed with log `COMPILE_DONE`, all four checkpoint-bank insertions, `model.pt`, `neuron_config.json`, and nested `weights/tp{0..3}_sharded_checkpoint.safetensors`.
   - Root cause: the local workstation can SSH to both hosts, but the compile host did not yet have an SSH key authorized on the runtime host, so true EC2-to-EC2 rsync could not authenticate.
   - Mitigation: create or reuse a dedicated compile-host rsync key and append only its public key to runtime `~/.ssh/authorized_keys`; then rerun rsync from compile host to runtime host.
   - Verification: dedicated key `qwen-rsync-20260605` was authorized on `ubuntu@16.26.184.190`; retry transferred `39,067,142,602` bytes in about `0:02:45`; runtime artifact verified with `model.pt`, `neuron_config.json`, and four nested `weights/tp{0..3}_sharded_checkpoint.safetensors` files of `9,529,553,860` bytes each.

19. Runtime launch of the QKV NKI artifact failed before health because postprocess wrote FP32 checkpoint banks into a BF16-traced model.
   - Command: `MAX_MODEL_LEN=32768 SEQ_LEN=32768 CTE_BUCKETS=2048 CONTEXT_ENCODING_BUCKET_PAIRS="2048:256 2048:512 2048:1024 2048:2048 2048:4096 2048:8192 2048:16384 2048:32768" TOKEN_GENERATION_BUCKETS="512 16384 16640 32768" GDN_RECURRENT_CACHE_DTYPE=float32 GDN_CONV_CACHE_DTYPE=bfloat16 PORT=8001 bash tmp_launch_qwen36_segcte2048.sh /mnt/trainium_artifacts/qwen_artifacts/qwen36_32k_fp8_fp8all_lmheadfp8_gatesfp8_kvbf16_qkvnki_tiled_segmented_cte512_gdnseg512_cte2048_pfx32k_slots64_20260605T152927Z_direct_scan0 /home/ubuntu/validation_logs/fp8_256k_decode_nki/qkvnki_tiled_direct_scan0_runtime_20260605T160443Z.log`.
   - Error: `SERVER_EXITED_BEFORE_HEALTH`; root TorchScript error from `SPMDBucketModelScript.initialize`: `Incorrect data type for checkpoint key hybrid_gdn_checkpoint_cache.recurrent_slots.0: received float, expected c10::BFloat16` repeated for recurrent slots `0..47`; API server ended with `RuntimeError: Engine core initialization failed. See root cause above.`
   - Context: runtime host `ubuntu@16.26.184.190`, source commit `f2c46e6`, artifact compiled on `ubuntu@16.26.135.243`; compile log had `CHECKPOINT_BANK_WEIGHTS_ADDED tp0..tp3 48 48 torch.float32 torch.bfloat16` and `COMPILE_DONE`.
   - Root cause hypothesis: the compile/postprocess path honored the checkpoint-bank insertion dtype (`torch.float32`) but the traced model graph/state schema still expects recurrent checkpoint slots as `bfloat16`. This is the same class of mismatch previously warned about: do not postprocess-only a BF16-traced artifact to FP32 recurrent banks.
   - Mitigation: inspect artifact safetensors/model schema; if the trace truly expects BF16, convert recurrent checkpoint banks to BF16 and launch with `GDN_RECURRENT_CACHE_DTYPE=bfloat16` for this validation. Separately fix the compile path if the intended target is a truly FP32-traced recurrent cache.
   - Verification: created `/mnt/trainium_artifacts/qwen_artifacts/qwen36_32k_fp8_fp8all_lmheadfp8_gatesfp8_kvbf16_qkvnki_tiled_segmented_cte512_gdnseg512_cte2048_pfx32k_slots64_20260605T152927Z_direct_scan0_recbf16fix`; converted 48 recurrent checkpoint tensors per TP shard to `torch.bfloat16`; confirmed recurrent and conv checkpoint slots are both `torch.bfloat16`; relaunch with `GDN_RECURRENT_CACHE_DTYPE=bfloat16 GDN_CONV_CACHE_DTYPE=bfloat16` reached `HEALTH_OK attempt=61`.

20. QKV NKI BF16-corrected artifact had one empty raw completion in the unique 4k sweep.
   - Runtime host/log: `ubuntu@16.26.184.190`, `/home/ubuntu/validation_logs/fp8_256k_decode_nki/qkvnki_tiled_direct_scan0_recbf16_runtime_20260605T161129Z.log`.
   - Artifact: `/mnt/trainium_artifacts/qwen_artifacts/qwen36_32k_fp8_fp8all_lmheadfp8_gatesfp8_kvbf16_qkvnki_tiled_segmented_cte512_gdnseg512_cte2048_pfx32k_slots64_20260605T152927Z_direct_scan0_recbf16fix`.
   - Probe command context: self-contained OpenAI `/v1/completions` and `/v1/chat/completions` probes on `127.0.0.1:8001`, tokenizer-verified exact prompt lengths, `temperature=0`, no `logprobs`.
   - Failure: unique raw completion at exact prompt length `4103` returned HTTP 200 with `usage_prompt=4103`, `usage_completion=1`, no text, and the probe marked `bad=["empty"]`.
   - Neighboring evidence: primary boundaries `146,160,485,505,526,1225,2048,2049,2500,4092,4096` all passed; repeated exact `2500` passed 3/3; unique raw sweep `4088..4102` and `4104` passed; chat multi-turn `160,1225,2500` passed.
   - Root cause hypothesis: this may be a legitimate EOS/local prompt artifact rather than cache corruption because adjacent lengths and multi-turn probes are coherent, but it must be rerun with fresh markers and serve-log inspection before calling the artifact coherent.
   - Mitigation: rerun targeted exact `4103` variants with fresh markers and scan the serve log for invalid token fallback, NaN logits, or EOS-only sampling.
   - Verification: five fresh exact-`4103` variants all returned non-empty coherent text with `usage_prompt=4103`, `usage_completion=24`, and no badness markers; final serve-log scans showed no `negative token_id`, `out-of-vocab token_id`, `fallback argmax`, `finite=0`, `nan=`, `NRT_RESOURCE`, engine crash, traceback, runtime error, or internal server error.

21. QKV NKI CTE2048 artifact is coherent but does not recover the cold-prefill speed target.
   - Runtime host/log: `ubuntu@16.26.184.190`, `/home/ubuntu/validation_logs/fp8_256k_decode_nki/qkvnki_tiled_direct_scan0_recbf16_runtime_20260605T161129Z.log`.
   - Artifact under test: `/mnt/trainium_artifacts/qwen_artifacts/qwen36_32k_fp8_fp8all_lmheadfp8_gatesfp8_kvbf16_qkvnki_tiled_segmented_cte512_gdnseg512_cte2048_pfx32k_slots64_20260605T152927Z_direct_scan0_recbf16fix`.
   - Launch flags: `MAX_MODEL_LEN=32768`, `SEQ_LEN=32768`, `CTE_BUCKETS=2048`, context pairs `2048:{256,512,1024,2048,4096,8192,16384,32768}`, token buckets `{512,16384,16640,32768}`, `GDN_RECURRENT_CACHE_DTYPE=bfloat16`, `GDN_CONV_CACHE_DTYPE=bfloat16`, `PORT=8001`, Hybrid APC enabled, segmented CTE prefix attention, GDN segment 512.
   - Runtime config evidence: live server command includes `fused_qkv=true`, `qkv_kernel_enabled=true`, `qkv_nki_kernel_enabled=true`, `kv_cache_quant=false`, `prefix_cte_attention_backend="segmented_cte"`, `prefix_cte_attention_segment_size=512`, `max_prompt_length=32768`.
   - Coherence matrix:
     - Raw exact boundaries `146,160,485,505,526,1225,2048,2049,2500,4092,4096`: pass with matching `usage.prompt_tokens`.
     - Repeated exact `2500`: pass 3/3.
     - Unique sweep `4088..4104`: pass except one non-reproducing empty at `4103`; five fresh exact-`4103` variants passed.
     - Chat multi-turn exact prompt tokens `160,1225,2500`: pass.
   - 16k cold-prefill benchmark: streaming `/v1/completions`, `max_tokens=1`, `stream_options.include_usage=true`, exact `usage.prompt_tokens=16384`; runs were `26.1831s` / `625.7 tok/s`, `25.8873s` / `632.9 tok/s`, and `25.8859s` / `632.9 tok/s`.
   - Interpretation: the QKV NKI layout/scale fix is compile-safe and coherence-safe, and the runtime says the QKV NKI path is enabled, but it is not the missing prefill-speed lever in this segmented CTE2048 configuration. The next speed work should profile before adding another kernel; likely bottlenecks are segmented CTE attention/GDN segmentation scheduling or remaining dense projections, not QKV layout.
   - Verification: server remains healthy on `127.0.0.1:8001`; final serve-log scan is clean.

22. Full-runtime 16k Neuron profiling run was not usable for TTFT, but it exposed async timeouts and confirmed that naive whole-server profiling is too disruptive.
   - Runtime host: `ubuntu@16.26.184.190`
   - Script: `/home/ubuntu/inferentia-gdn-prefill-speed-coherent/tmp_profile_qwen36_qkvnki_16k_prefill.sh`
   - Profile root: `/home/ubuntu/validation_logs/fp8_256k_decode_nki/qkvnki_prefill_profile_20260605T163533Z`
   - Artifact under test: `/mnt/trainium_artifacts/qwen_artifacts/qwen36_32k_fp8_fp8all_lmheadfp8_gatesfp8_kvbf16_qkvnki_tiled_segmented_cte512_gdnseg512_cte2048_pfx32k_slots64_20260605T152927Z_direct_scan0_recbf16fix`
   - Launch context: server was relaunched with `NEURON_RT_INSPECT_ENABLE=1`, `NEURON_RT_INSPECT_DEVICE_PROFILE=1`, and `NEURON_RT_INSPECT_OUTPUT_DIR` pointing at the profile root, then a single exact 16k `max_tokens=1` completion request was sent.
   - Error/evidence: `prefill_request.json` recorded `actual_prompt_tokens=16384`, `chunks=1`, `http=200`, `total_seconds=603.2149450778961`, `ttft_seconds=null`, `usage=null`, and `text=""`. The profiled server log emitted repeated Neuron runtime warnings: `Timeout polling for async exec completion on nc ... (waited 2 minutes)` and later 4 minutes.
   - Root cause hypothesis: enabling runtime device-profile collection on the full vLLM server makes the multi-NEFF async execution path stall enough that the request result cannot be used as a prefill benchmark. The useful profile data must be captured from specific context NEFFs, not by timing a live full-server request under profiling.
   - Mitigation applied: stopped the profiled server path, manually relaunched the normal non-profiled server on port `8001`, and kept the generated inspect directory only for NEFF identification.
   - Verification: the normal server was relaunched and reached `HEALTH_OK`; only a `layout_opt` NTFF was complete enough to summarize from the full-server run, with `total_time=0.048718152211`, `hardware_flops=1750128852992`, and `transpose_flops=1385322430464`, which is not the context bottleneck.

23. Profiling stop command returned exit 255 because the remote kill pattern likely matched the SSH command shell.
   - Command context: local SSH into `ubuntu@16.26.184.190` to stop the profiled vLLM server and any matching `tmp_profile_qwen36_qkvnki_16k_prefill` process after the 603s stalled request.
   - Error: SSH returned exit code `255` during the remote stop sequence.
   - Root cause hypothesis: the remote `pkill -9 -f "[t]mp_profile_qwen36_qkvnki_16k_prefill"` pattern still matched or disrupted the remote command/session being executed under SSH, closing the connection before a clean status could be returned.
   - Mitigation applied: followed with explicit process checks and a normal launcher restart instead of relying on that stop command result.
   - Verification: runtime host process state was checked afterward and the non-profiled server was relaunched to `HEALTH_OK`.

24. Direct context-NEFF profiler initially failed on the NEFF checksum lookup.
   - Runtime host: `ubuntu@16.26.184.190`
   - Script: `/home/ubuntu/inferentia-gdn-prefill-speed-coherent/tmp_profile_qwen36_context_neffs_from_inspect.sh`
   - Command context: map compile-workdir context NEFFs to runtime inspect NEFFs by SHA and run `neuron-explorer capture/view` for selected context graphs.
   - Error: `xargs: sha256sum: terminated by signal 13`, and the script exited with code `125` under `pipefail`.
   - Root cause: `find_neff_by_sha` used a `find | xargs sha256sum | awk` pipeline; when `awk` stopped after the first match, upstream `sha256sum` received SIGPIPE, which became fatal under `pipefail`.
   - Mitigation applied: rewrote the lookup to loop over `find` results and compute `sha256sum` one file at a time, returning immediately on a matching digest without a SIGPIPE-prone pipeline.
   - Verification: rerunning the script progressed past NEFF mapping and captured the first direct context profile.

25. Direct profiling shows the cold context graph is already the 16k speed bottleneck before prefix reads.
   - Runtime profile root: `/home/ubuntu/validation_logs/fp8_256k_decode_nki/qkvnki_prefill_profile_20260605T163533Z/direct_context_captures_20260605T165525Z`
   - Compile workdir: `/mnt/trainium_artifacts/qwen_artifacts/_nxd_work_32k_fp8_fp8all_lmheadfp8_gatesfp8_kvbf16_qkvnki_tiled_segmented_cte512_gdnseg512_cte2048_20260605T152927Z_direct_scan0`
   - NEFF profiled: `context_encoding_model/_tp0_bk0/graph.neff` mapped by SHA to runtime inspect NEFF `neff_642651114238503_vnc_0.neff`; this is the `context_bk0_pfx0` graph.
   - Metrics from `neuron-explorer view --output-format summary-json`: `total_time=3.132167956813`, `total_exec_time=3.13216400481`, `total_active_time=1.258159555751`, `tensor_engine_active_time=0.565739726504`, `tensor_engine_active_time_percent=0.1806224105171051`, `vector_engine_active_time=0.492130449926`, `vector_engine_active_time_percent=0.15712134748570308`, `dma_active_time=0.406673390022`, `dma_active_time_percent=0.1298376701470992`, `hbm_read_bytes=31017606071`, `hbm_write_bytes=11101448403`, `spill_reload_bytes=116516096`, `hardware_flops=19279660899840`, `transpose_flops=339695382912`, `mfu_estimated_percent=0.06277731969607976`, `mm_arithmetic_intensity=449.67689216811834`, `peak_flops_bandwidth_ratio=109.83687150837989`.
   - Interpretation: one cold 2048-token context graph takes about `3.13s` on device; a 16k cold prompt uses eight such chunks, so `8 * 3.13s = 25.1s`, matching the measured 16k wall time of about 25.9-26.2s. The bottleneck is therefore inside the compiled context graph, not OpenAI/vLLM Python scheduling or prefix-cache bookkeeping.
   - Root cause hypothesis: because `pfx0` is already slow, segmented prefix attention is not the first-order cold-prefill bottleneck. The leading remaining one-variable suspect is GDN internal segmentation (`QWEN36_DELTANET_FUSED_SEGMENT_TOKENS=512`) or the base GDN solve path inside the context graph.
   - Mitigation in flight: launched a one-variable no-GDN-segmentation compile with `QWEN36_DELTANET_FUSED_SEGMENT_TOKENS=0`, keeping QKV NKI, CTE2048, segmented CTE512 prefix attention, KV BF16, lm_head FP8, gates FP8, direct solve scan0, and multihead DeltaNet CTE off.
   - Verification pending: compare the new no-GDN-seg artifact against the same coherence matrix and a 16k usage-accounted cold-prefill benchmark. If coherent and faster, the speed loss is from the GDN segmentation loop; if coherent and still slow, profile the no-GDN context NEFF and compare per-engine metrics against this profile.

26. The no-GDN-segmentation CTE2048 compile completed cleanly with BF16 checkpoint banks.
   - Compile host: `ubuntu@16.26.135.243`
   - Source: `/home/ubuntu/inferentia-gdn-prefill-speed-coherent`
   - Command shape: `ENABLE_QKV_NKI_KERNELS=1`, `CTE_BUCKETS_RAW=2048`, `PREFIX_CTE_ATTENTION_BACKEND=segmented_cte`, `PREFIX_CTE_ATTENTION_SEGMENT_SIZE=512`, `QWEN36_DELTANET_FUSED_SEGMENT_TOKENS=0`, `QWEN36_DELTANET_MULTIHEAD_CTE=0`, `ENABLE_KV_CACHE_QUANT=0`, `QUANTIZE_LM_HEAD=1`, `FP8_QUANTIZE_LINEAR_ATTN_GATES=1`, `QWEN36_DELTANET_SOLVE_MODE=direct`, `QWEN36_DELTANET_SOLVE_SCAN_STEPS=0`, `GDN_RECURRENT_CACHE_DTYPE=bfloat16`, `GDN_CONV_CACHE_DTYPE=bfloat16`.
   - PID/log: PID `73303`, `/home/ubuntu/validation_logs/fp8_256k_decode_nki/qwen36_32k_fp8_fp8all_lmheadfp8_gatesfp8_kvbf16_qkvnki_tiled_segmented_cte512_gdnseg0_cte2048_pfx32k_slots64_20260605T171105Z_nogdnseg_direct_scan0_compile.log`.
   - Env log: `/home/ubuntu/validation_logs/fp8_256k_decode_nki/qwen36_32k_fp8_fp8all_lmheadfp8_gatesfp8_kvbf16_qkvnki_tiled_segmented_cte512_gdnseg0_cte2048_pfx32k_slots64_20260605T171105Z_nogdnseg_direct_scan0_env.txt`; confirmed `QWEN36_DELTANET_FUSED_SEGMENT_TOKENS=0`.
   - Artifact: `/mnt/trainium_artifacts/qwen_artifacts/qwen36_32k_fp8_fp8all_lmheadfp8_gatesfp8_kvbf16_qkvnki_tiled_segmented_cte512_gdnseg0_cte2048_pfx32k_slots64_20260605T171105Z_nogdnseg_direct_scan0`.
   - Compile evidence: HLO generation completed, priority HLO compiled, all HLOs completed with `INFO:Neuron:Finished Compilation for all HLOs in 418.1721477508545 seconds`, sharding completed, then `COMPILE_DONE`.
   - Checkpoint-bank evidence: log lines `CHECKPOINT_BANK_WEIGHTS_ADDED tp{0..3}_sharded_checkpoint.safetensors 48 48 torch.bfloat16 torch.bfloat16`.
   - Artifact verification: `model.pt` size `707806474`, `neuron_config.json` size `106546`, and four safetensor shards of `8321591748` bytes each. Runtime and compile-host safetensor checks both found 48 recurrent and 48 conv checkpoint-bank tensors per TP shard, all `torch.bfloat16`.
   - Resource note: compile host disk dropped to about `9.2G` free during checkpoint-bank insertion and ended around `16G` free. If another compile is needed on this host, clean old artifacts first instead of relying on this margin.
   - Transfer: EC2-to-EC2 rsync from compile host to runtime host `ubuntu@16.26.184.190` with `~/.ssh/qwen_rsync_ed25519`; transferred `33,994,280,012` bytes in about `0:02:08`.

27. Runtime validation had two missing-file/path errors before the maintained probes were run.
   - Runtime host: `ubuntu@16.26.184.190`
   - Error 1 command: `/home/ubuntu/venvs/neuron_230_segmented_cte/bin/python` used for artifact dtype verification.
   - Error 1 text: `bash: line 1: /home/ubuntu/venvs/neuron_230_segmented_cte/bin/python: No such file or directory`
   - Root cause 1: runtime host uses `/opt/aws_neuronx_venv_pytorch_inference_vllm_0_16`, not the compile-host helper venv.
   - Mitigation 1: reran safetensor dtype verification with `/opt/aws_neuronx_venv_pytorch_inference_vllm_0_16/bin/python`; verification passed.
   - Error 2 command: `python /tmp/tmp_bisect_probe3.py`
   - Error 2 text: `bash: line 1: python: command not found`
   - Error 3 command: `python3 /tmp/tmp_bisect_probe3.py`
   - Error 3 text: `python3: can't open file '/tmp/tmp_bisect_probe3.py': [Errno 2] No such file or directory`
   - Root cause 2/3: this runtime instance does not have the old ad hoc `/tmp/tmp_bisect_probe3.py`, and `python` is not on PATH.
   - Mitigation 2/3: used maintained repo validators instead: `validation_scripts/qwen36_openai_boundary_apc_probe.py` for exact token-id raw completions and `validation_scripts/qwen36_chat_completion_context_bench.py` for multi-turn chat streaming usage.

28. The no-GDN-segmentation artifact is coherent but still slow at 16k cold prefill.
   - Runtime host/log: `ubuntu@16.26.184.190`, `/home/ubuntu/validation_logs/fp8_256k_decode_nki/qkvnki_tiled_direct_scan0_nogdnseg_runtime_20260605T174340Z.log`.
   - Launch command shape: `MAX_MODEL_LEN=32768`, `SEQ_LEN=32768`, `CTE_BUCKETS=2048`, context pairs `2048:{256,512,1024,2048,4096,8192,16384,32768}`, token buckets `{512,16384,16640,32768}`, `GDN_RECURRENT_CACHE_DTYPE=bfloat16`, `GDN_CONV_CACHE_DTYPE=bfloat16`, `QWEN36_DELTANET_FUSED_SEGMENT_TOKENS=0`, `QWEN36_DELTANET_MULTIHEAD_CTE=0`, `QWEN36_DELTANET_SOLVE_MODE=direct`, `QWEN36_DELTANET_SOLVE_SCAN_STEPS=0`, port `8001`.
   - Live server config evidence: command line contains `fused_qkv=true`, `qkv_kernel_enabled=true`, `qkv_nki_kernel_enabled=true`, `kv_cache_quant=false`, `prefix_cte_attention_backend="segmented_cte"`, `prefix_cte_attention_segment_size=512`, `max_prompt_length=32768`, and BF16 recurrent/conv GDN cache dtypes.
   - Health: launch reached `HEALTH_OK attempt=35`; final health check remained `HEALTH_OK` with server PID `31549`.
   - Primary raw exact boundary probe: lengths `146,160,485,505,526,1225,2048,2049,2500,4092,4096` all returned HTTP 200, valid OpenAI bodies, matching `usage.prompt_tokens`, and non-empty coherent text. JSONL: `/home/ubuntu/validation_logs/fp8_256k_decode_nki/nogdnseg_boundary_probe_20260605T174559Z.jsonl`.
   - Unique 4k sweep: lengths `4088..4104` all passed, including `4103`, with non-empty coherent text and matching `usage.prompt_tokens`. JSONL: `/home/ubuntu/validation_logs/fp8_256k_decode_nki/nogdnseg_4k_sweep_20260605T174706Z.jsonl`.
   - Repeated 2500 probe: 3/3 passed; repeats 1 and 2 showed `2048` token prefix-cache hits and coherent text. JSONL: `/home/ubuntu/validation_logs/fp8_256k_decode_nki/nogdnseg_repeat2500_20260605T174935Z.jsonl`.
   - Multi-turn chat probe: target prompt sizes `160,1225,2500` passed with non-empty streamed content (`ack 4` or `lambda`) and usage-derived completion tokens. JSON: `/home/ubuntu/validation_logs/fp8_256k_decode_nki/nogdnseg_chat_multiturn_20260605T175024Z.json`.
   - Long raw probe: lengths `8192` and `16384` passed with coherent non-empty text; 16k case had `usage.prompt_tokens=16384`. JSONL: `/home/ubuntu/validation_logs/fp8_256k_decode_nki/nogdnseg_long_probe_20260605T175110Z.jsonl`.
   - Final serve-log scan after validation: no `negative token_id`, `out-of-vocab token_id`, `fallback argmax`, `finite=0`, `nan=`, `NaN`, `NRT_RESOURCE`, engine death, traceback, internal server error, or dtype mismatch markers.
   - 16k usage-accounted streaming cold-prefill benchmark (`/v1/completions`, exact 16384 token-id prompt, `max_tokens=1`, `stream_options.include_usage=true`): run 0 `25.6959s TTFT`, `637.6 tok/s`; run 1 `25.6728s TTFT`, `638.2 tok/s`; run 2 `25.6755s TTFT`, `638.1 tok/s`.
   - Conclusion: disabling GDN internal segmentation did not recover prefill speed. It preserves coherence but is effectively the same speed class as the prior coherent QKV-NKI segmented-CTE artifact. The bottleneck is not the `QWEN36_DELTANET_FUSED_SEGMENT_TOKENS=512` loop.

29. Compile-host cleanup was required before the next compile.
   - Compile host: `ubuntu@16.26.135.243`
   - Initial disk state before cleanup: `/dev/root 484G`, `397G used`, `87G avail` after cleanup; before cleanup the same filesystem had only about `16G` free, too little for another full artifact and checkpoint-bank insertion.
   - Cleanup command context: delete compile-host-only copies that were already transferred/validated or superseded:
     - `/mnt/trainium_artifacts/qwen_artifacts/qwen36_32k_fp8_fp8all_lmheadfp8_gatesfp8_kvbf16_qkvnki_tiled_segmented_cte512_gdnseg0_cte2048_pfx32k_slots64_20260605T171105Z_nogdnseg_direct_scan0`
     - `/mnt/trainium_artifacts/qwen_artifacts/qwen36_32k_fp8_fp8all_lmheadfp8_gatesfp8_kvbf16_qkvnki_tiled_segmented_cte512_gdnseg512_cte2048_pfx32k_slots64_20260605T152927Z_direct_scan0`
     - `/mnt/trainium_artifacts/qwen_artifacts/_nxd_work_32k_fp8_fp8all_lmheadfp8_gatesfp8_kvbf16_qkvnki_tiled_segmented_cte512_gdnseg0_cte2048_20260605T171105Z_nogdnseg_direct_scan0`
     - `/mnt/trainium_artifacts/qwen_artifacts/_nxd_work_32k_fp8_fp8all_lmheadfp8_gatesfp8_kvbf16_qkvnki_tiled_segmented_cte512_gdnseg512_cte2048_20260605T152927Z_direct_scan0`
   - Root cause/hypothesis: compile artifacts are about 32-37G each; keeping multiple validated attempts on the compile host leaves too little room for the next compile. Runtime host still has the validated no-GDN artifact live; the compile-host copy was not needed for serving.
   - Mitigation result: disk free increased to `87G`, enough for the next one-variable compile.

30. Automation setup for the next compile failed at the Codex app tool layer.
   - Tool calls attempted: `codex_app.automation_update` with `mode=create` and `mode=suggested_create`, `kind=heartbeat`, 10-minute RRULE, and both full and minimal monitor prompts.
   - Error text returned by tool: `automation_update received invalid arguments.`
   - Context: this was before launching the attention-CTE compile, because compiles should have monitors/automations.
   - Root cause hypothesis: the app automation tool rejected heartbeat create payloads in this resumed-goal context; minimal and explicit-thread variants both failed, so the issue was not prompt length alone.
   - Mitigation: proceeded with manual monitoring using deterministic PID/log/artifact paths, and noted that no stale automation exists for this compile.
   - Remaining risk: if this thread is not actively monitored, no heartbeat will wake it automatically. Manual monitor command is listed below.

31. Attention-CTE one-variable compile completed and validated; it is coherent but not faster.
   - Compile host: `ubuntu@16.26.135.243`
   - Source: `/home/ubuntu/inferentia-gdn-prefill-speed-coherent`
   - PID: `81458`
   - PID file: `/home/ubuntu/validation_logs/fp8_256k_decode_nki/qwen36_32k_fp8_fp8all_lmheadfp8_gatesfp8_kvbf16_qkvnki_tiled_attention_cte512_gdnseg0_cte2048_pfx32k_slots64_20260605T180405Z_attncte_direct_scan0_compile.pid`
   - Compile log: `/home/ubuntu/validation_logs/fp8_256k_decode_nki/qwen36_32k_fp8_fp8all_lmheadfp8_gatesfp8_kvbf16_qkvnki_tiled_attention_cte512_gdnseg0_cte2048_pfx32k_slots64_20260605T180405Z_attncte_direct_scan0_compile.log`
   - Env log: `/home/ubuntu/validation_logs/fp8_256k_decode_nki/qwen36_32k_fp8_fp8all_lmheadfp8_gatesfp8_kvbf16_qkvnki_tiled_attention_cte512_gdnseg0_cte2048_pfx32k_slots64_20260605T180405Z_attncte_direct_scan0_env.txt`
   - Artifact: `/mnt/trainium_artifacts/qwen_artifacts/qwen36_32k_fp8_fp8all_lmheadfp8_gatesfp8_kvbf16_qkvnki_tiled_attention_cte512_gdnseg0_cte2048_pfx32k_slots64_20260605T180405Z_attncte_direct_scan0`
   - Workdir: `/mnt/trainium_artifacts/qwen_artifacts/_nxd_work_32k_fp8_fp8all_lmheadfp8_gatesfp8_kvbf16_qkvnki_tiled_attention_cte512_gdnseg0_cte2048_20260605T180405Z_attncte_direct_scan0`
   - Command shape: `TS=20260605T180405Z_attncte`, `ENABLE_QKV_NKI_KERNELS=1`, `CTE_BUCKETS_RAW=2048`, `PREFIX_CTE_ATTENTION_BACKEND=attention_cte`, `PREFIX_CTE_ATTENTION_SEGMENT_SIZE=512`, `QWEN36_DELTANET_FUSED_SEGMENT_TOKENS=0`, `QWEN36_DELTANET_MULTIHEAD_CTE=0`, `ENABLE_KV_CACHE_QUANT=0`, `QUANTIZE_LM_HEAD=1`, `FP8_QUANTIZE_LINEAR_ATTN_GATES=1`, `QWEN36_DELTANET_SOLVE_MODE=direct`, `QWEN36_DELTANET_SOLVE_SCAN_STEPS=0`, `GDN_RECURRENT_CACHE_DTYPE=bfloat16`, `GDN_CONV_CACHE_DTYPE=bfloat16`, `bash tmp_compile_qwen32k_segcte2048_gdnseg512.sh`.
   - Env verification: env log contains `PREFIX_CTE_ATTENTION_BACKEND=attention_cte`, `QWEN36_DELTANET_FUSED_SEGMENT_TOKENS=0`, `ENABLE_QKV_NKI_KERNELS=1`, `ENABLE_KV_CACHE_QUANT=0`, and BF16 GDN dtypes. `CONTEXT_TRACE_SHAPE` also records `"prefix_cte_attention_backend": "attention_cte"`.
   - Compile result: `Finished Compilation for all HLOs in 418.6668481826782 seconds`, `CHECKPOINT_BANK_WEIGHTS_ADDED` for `tp0..tp3` with `48 48 torch.bfloat16 torch.bfloat16`, and `COMPILE_DONE`.
   - Artifact files verified on compile and runtime hosts: `model.pt` `748425994` bytes, `neuron_config.json` `106546` bytes, and `weights/tp0..tp3_sharded_checkpoint.safetensors` `8321591748` bytes each.
   - Runtime host: `ubuntu@16.26.184.190`
   - Runtime artifact: `/mnt/trainium_artifacts/qwen_artifacts/qwen36_32k_fp8_fp8all_lmheadfp8_gatesfp8_kvbf16_qkvnki_tiled_attention_cte512_gdnseg0_cte2048_pfx32k_slots64_20260605T180405Z_attncte_direct_scan0`
   - Runtime log: `/home/ubuntu/validation_logs/fp8_256k_decode_nki/qkvnki_tiled_attention_cte_direct_scan0_runtime_20260605T183200Z.log`
   - Launch command shape: `MAX_MODEL_LEN=32768`, `SEQ_LEN=32768`, `CTE_BUCKETS=2048`, context pairs `2048:{256,512,1024,2048,4096,8192,16384,32768}`, token buckets `{512,16384,16640,32768}`, `GDN_RECURRENT_CACHE_DTYPE=bfloat16`, `GDN_CONV_CACHE_DTYPE=bfloat16`, `QWEN36_DELTANET_FUSED_SEGMENT_TOKENS=0`, `QWEN36_DELTANET_MULTIHEAD_CTE=0`, `QWEN36_DELTANET_SOLVE_MODE=direct`, `QWEN36_DELTANET_SOLVE_SCAN_STEPS=0`, `PREFIX_CTE_ATTENTION_BACKEND=attention_cte`, port `8001`.
   - Health: server reached `/health` on attempt 6 with serve PID `33585` and EngineCore PID `33737`.
   - Config evidence: runtime command line contains `prefix_cte_attention_backend="attention_cte"`, `qkv_nki_kernel_enabled=true`, `kv_cache_quant=false`, `context_encoding_buckets=[2048]`, bucket pairs through `32768`, BF16 recurrent/conv GDN cache dtypes, and `use_hybrid_apc_manager=true`.
   - Primary raw exact boundary probe: lengths `146,160,485,505,526,1225,2048,2049,2500,4092,4096` all returned HTTP 200, valid OpenAI bodies, matching `usage.prompt_tokens`, and coherent non-empty text. JSONL: `/home/ubuntu/validation_logs/fp8_256k_decode_nki/attncte_boundary_probe_20260605T183800Z.jsonl`.
   - Unique 4k sweep: lengths `4088..4104` all passed, including `4103`, with valid coherent text. JSONL: `/home/ubuntu/validation_logs/fp8_256k_decode_nki/attncte_4k_sweep_20260605T184000Z.jsonl`.
   - Repeated 2500 probe: 3/3 passed; repeats 1 and 2 showed `2048` token prefix-cache hits and coherent text. JSONL: `/home/ubuntu/validation_logs/fp8_256k_decode_nki/attncte_repeat2500_20260605T184300Z.jsonl`.
   - Multi-turn chat probe: target prompt sizes `160,1225,2500` passed with streamed content (`ack 4` / `lambda`) and usage-derived completion tokens. JSON: `/home/ubuntu/validation_logs/fp8_256k_decode_nki/attncte_chat_multiturn_20260605T184500Z.json`.
   - Long raw probe: lengths `8192` and `16384` passed with coherent text. JSONL: `/home/ubuntu/validation_logs/fp8_256k_decode_nki/attncte_long_probe_20260605T184700Z.jsonl`.
   - Final serve-log scan after validation: no `negative token_id`, `out-of-vocab token_id`, `fallback argmax`, `finite=0`, `nan=`, `NaN`, `NRT_RESOURCE`, engine death, traceback, internal server error, or dtype mismatch markers.
   - 16k usage-accounted streaming cold-prefill benchmark (`max_tokens=1`, `stream_options.include_usage=true`): run 0 `16373 / 26.123764895997738 = 626.747 tok/s`; run 1 `16373 / 26.10490949099767 = 627.200 tok/s`; run 2 `16373 / 26.1018272729998 = 627.274 tok/s`; mean `627.074 tok/s`. JSON: `/home/ubuntu/validation_logs/fp8_256k_decode_nki/attncte_16k_cold_prefill_20260605T184900Z.json`.
   - Errors encountered and mitigated:
     - Local monitor poll wrapped approved `ssh` behind local `sleep` and failed with `ssh: connect to host 16.26.135.243 port 22: Operation not permitted`, exit `255`. Root cause: sandbox/prefix issue in local command shape, not compile-host failure. Mitigation: reran as direct `ssh` and remote-side `sleep`.
     - Safetensor dtype check under system `python3` failed with `ModuleNotFoundError: No module named 'safetensors'`, exit `1`. Root cause: system Python missing the package. Mitigation: reran with `/home/ubuntu/venvs/neuron_230_segmented_cte/bin/python`; dtype check passed.
     - The launch command's SSH wrapper stayed attached after `nohup`; runtime process was healthy with PPID 1 and log/PID files were present. Root cause hypothesis: remote shell/job-control attachment, not model failure. Mitigation: verified process list/log/health separately.
     - The 16k benchmark exec wrapper did not return even though no benchmark process remained and the JSON result file was complete. Root cause hypothesis: stale local SSH exec session. Mitigation: parsed completed JSON output directly.
   - Conclusion: switching from `segmented_cte` to `attention_cte` does not recover cold-prefill speed. This artifact is coherent, but its 16k cold prefill is the same slow class as the coherent `segmented_cte` artifacts (`~627 tok/s` vs `~638 tok/s`). The speed bottleneck is not `QWEN36_DELTANET_FUSED_SEGMENT_TOKENS=512` and not the `segmented_cte` prefix-attention backend by itself.

32. Direct NEFF profiling of the coherent attention-CTE artifact explains the slow 16k TTFT.
   - Runtime host: `ubuntu@16.26.184.190`
   - Profile root: `/home/ubuntu/validation_logs/fp8_256k_decode_nki/attncte_direct_neff_profile_20260605T1900Z`
   - Loose context NEFF source copied from compile host workdir: `/mnt/trainium_artifacts/qwen_artifacts/_nxd_work_32k_fp8_fp8all_lmheadfp8_gatesfp8_kvbf16_qkvnki_tiled_attention_cte512_gdnseg0_cte2048_20260605T180405Z_attncte_direct_scan0/context_encoding_model`
   - Copied runtime NEFF directory: `/home/ubuntu/validation_logs/fp8_256k_decode_nki/attncte_context_neffs_20260605T1900Z/context_encoding_model`
   - Error encountered: first selected-NEFF `rsync` failed with `rsync: [Receiver] mkdir ".../attncte_context_neffs_20260605T1900Z/context_encoding_model" failed: No such file or directory (2)`, exit code `11`. Root cause: destination parent directory did not exist. Mitigation: created the destination directory on runtime host and retried the selected-file transfer successfully.
   - Error encountered: first direct profiling command dropped the SSH session with exit `255` after writing only profile-root/tool lines. Root cause hypothesis: the remote cleanup step disrupted the SSH/session path, similar to the earlier `pkill` issue. Evidence: no capture files existed afterward and `/health` was `000` because the server had been stopped. Mitigation: resumed with a safer profiling command that skipped cleanup entirely, because the server was already down.
   - Profile method: `neuron-explorer capture` directly on loose `graph.neff` files with 4 collectives workers, `--num-exec=2`, `--profile-nth-exec=2`, `--ignore-exec-errors`, and `--io-from=neff`, followed by `neuron-explorer view --output-format summary-json --ignore-nc-buf-usage`.
   - `context_bk0_pfx0` NEFF: `/home/ubuntu/validation_logs/fp8_256k_decode_nki/attncte_context_neffs_20260605T1900Z/context_encoding_model/_tp0_bk0/graph.neff`
   - `context_bk0_pfx0` metrics: `total_time=3.105348121219`, `total_exec_time=3.10534743024`, `total_active_time=1.240761278717`, `tensor_engine_active_time=0.570718131305`, `tensor_engine_active_time_percent=0.18378555608798072`, `vector_engine_active_time=0.498965804321`, `vector_engine_active_time_percent=0.1606795067231083`, `dma_active_time=0.373811388766`, `dma_active_time_percent=0.12037664512127577`, `hbm_read_bytes=29320828336`, `hbm_write_bytes=9405615955`, `spill_reload_bytes=31581440`, `hardware_flops=19515351916032`, `transpose_flops=513317590272`, `mfu_estimated_percent=0.06321398487510277`, `mm_arithmetic_intensity=490.6733544389992`.
   - `context_bk7_pfx16384` NEFF: `/home/ubuntu/validation_logs/fp8_256k_decode_nki/attncte_context_neffs_20260605T1900Z/context_encoding_model/_tp0_bk7/graph.neff`
   - `context_bk7_pfx16384` metrics: `total_time=3.353049140852`, `total_exec_time=3.353044819683`, `total_active_time=1.399721471025`, `tensor_engine_active_time=0.621091953868`, `tensor_engine_active_time_percent=0.1852319867015675`, `vector_engine_active_time=0.584730660997`, `vector_engine_active_time_percent=0.17438773976584837`, `dma_active_time=0.519647728016`, `dma_active_time_percent=0.15497766545823993`, `hbm_read_bytes=51877232190`, `hbm_write_bytes=25550816079`, `spill_reload_bytes=31581440`, `hardware_flops=21529004604928`, `transpose_flops=1002688222464`, `mfu_estimated_percent=0.06479861762820996`, `mm_arithmetic_intensity=265.1018182862057`.
   - Relaunch verification: server was relaunched from the same attention-CTE artifact, `/health` returned OK on attempt 19, serve PID `35884`, EngineCore PID `36114`, with `prefix_cte_attention_backend="attention_cte"`, QKV NKI enabled, KV quant off, BF16 recurrent/conv dtypes.
   - Profile-query attempt: `neuron-explorer view --ingest-only` for `context_bk0_pfx0` completed successfully, but querying parquet with the runtime venv failed because pandas had no parquet backend: `ImportError: Unable to find a usable engine; tried using: 'pyarrow', 'fastparquet'`, exit `1`. The runtime venv also lacks `duckdb`, `pyarrow`, `fastparquet`, and `polars`.
   - Profile-query API attempt: `neuron-explorer view --disable-ui --data-path ... --display-name attncte_bk0_pfx0` did not open `127.0.0.1:3002` after more than 6 minutes, while consuming about `475%` CPU and `32.5%` memory. It was stopped to avoid burning the runtime host. Model server health after stopping the API attempt was `health_http=200`.
   - Interpretation: the coherent context graph is intrinsically slow at roughly `3.1s` per 2048-token cold chunk. A 16k cold prompt requires 8 chunks, so `8 * 3.105s = 24.84s` before overhead, matching the measured `26.1s` TTFT and `~627 tok/s`. This clears Python scheduling, Hybrid APC bookkeeping, GDN segmentation, and segmented-vs-attention CTE as first-order explanations for the 3k gap. The next step must compare the compiled context graph against the old fast CTE2048 artifact or profile the old fast artifact directly.

33. Implemented Qwen-safe QKV CTE fused QK-norm + RoPE wiring for the next speed slice.
   - Source branch/workdir: `/private/tmp/inferentia-gdn-prefill-speed-coherent`, branch `codex/qwen36-prefill-speed-coherent`.
   - NKI API evidence: compile host source `/home/ubuntu/nki-library-2.30/src/nkilib_src/nkilib/core/qkv/qkv.py` exposes `qk_norm_pre_rope: Optional[QKNormConfig]`, `qk_norm_pre_rope_q_gamma`, and `qk_norm_pre_rope_k_gamma`; `/home/ubuntu/nki-library-2.30/src/nkilib_src/nkilib/core/utils/common_types.py` defines `QKNormConfig` with RMSNorm defaults and `[1, d_head]` Q/K gamma weights. AWS NKI docs for the QKV kernel also document `qk_norm_pre_rope` and `qk_norm_post_rope`.
   - Root cause hypothesis addressed: the existing `qkv_cte_nki_kernel_fuse_rope` flag could fuse RoPE in the QKV kernel while Qwen's `q_layernorm`/`k_layernorm` stayed in the Python `move_heads_front` path. For Qwen3/Qwen3.6 those norms are semantically pre-RoPE, so blindly enabling the existing flag risks changing the math to RoPE-before-QK-norm and explains why the old `qknormrope`-named fast artifact was incoherent.
   - Code fix: `src/neuronx_distributed_inference/modules/attention/attention_base.py` now detects pre-RoPE Q/K layernorms when QKV CTE fused RoPE is enabled, passes those modules and `qk_norm_eps` into the QKV projection, and skips the Python `move_heads_front` layernorm only for that fused path. It fails loudly if those modules are `torch.nn.LayerNorm`, because the NKI QKV path only supports RMSNorm-style QK norm here.
   - Code fix: `src/neuronx_distributed_inference/modules/attention/gqa.py` now builds `QKNormConfig(q_norm=RMS_NORM, k_norm=RMS_NORM, eps=...)` and passes `qk_norm_pre_rope_q_gamma` / `qk_norm_pre_rope_k_gamma` into `qkv_kernel` when fused RoPE and pre-RoPE QK norm are both active.
   - Compile harness fix: `contrib/models/Qwen3.6-27B/test/integration/qwen36_27b_compile_fp8.py` adds `--enable-qkv-cte-nki-kernel-fuse-rope`, validates that it requires `--enable-qkv-nki-kernels`, and records it in the emitted config summary. `tmp_compile_qwen32k_segcte2048_gdnseg512.sh` adds `ENABLE_QKV_CTE_NKI_KERNEL_FUSE_ROPE=1` and tags artifacts as `qkvnki_qknormrope`.
   - Errors encountered and mitigated:
     - Remote NKI source lookup on runtime host `ubuntu@16.26.184.190` failed for `/home/ubuntu/nki-library-2.30/src/nkilib_src/nkilib/core/qkv/qkv.py` with `grep: ... No such file or directory`, exit `2`. Root cause: runtime host does not carry the NKI source checkout; compile host does. Mitigation: inspected `/home/ubuntu/nki-library-2.30/...` on compile host `ubuntu@16.26.135.243`.
     - Local search command `rg -n ... src tests -g '*.py'` failed with `rg: tests: No such file or directory (os error 2)`, exit `2`. Root cause: this repo uses `test/`, not `tests/`. Mitigation: reran with `rg --files | rg '(^|/)test|tests|validation'`.
     - Local zsh command containing unquoted `tmp_compile_qwen36*` failed with `zsh:1: no matches found: tmp_compile_qwen36*`, exit `1`. Root cause: unmatched zsh glob. Mitigation: used concrete script names.
     - Local search command including nonexistent `scripts` failed with `rg: scripts: No such file or directory (os error 2)`, exit `2`. Root cause: no `scripts/` directory in this checkout. Mitigation: searched concrete existing paths.
     - First `apply_patch` for `qwen36_27b_compile_fp8.py` failed because the expected context did not match the local file. Root cause: broad patch context around the QKV config block. Mitigation: split into smaller exact-context patches.
     - Local pytest failed before collection: `test_gqa.py` hit `ModuleNotFoundError: No module named 'neuronx_distributed_inference'`; `test_attention_base.py` hit `ModuleNotFoundError: No module named 'torch_xla'`. Root cause: local Mac environment lacks the repo `PYTHONPATH` and Neuron/Torch-XLA dependencies. Mitigation: synced changed files to compile host and ran tests in `/opt/aws_neuronx_venv_pytorch_inference_vllm_0_16`.
     - First remote pytest attempt failed during import with `RuntimeError: Unsupported Platform - r7i.24xlarge ... supply a compiler target argument`, and the attention-base targeted run reported no collectors because collection failed. Root cause: compile host is CPU (`r7i`) and needs the Neuron target override for imports. Mitigation: reran with `NEURON_PLATFORM_TARGET_OVERRIDE=trn2` and `NEURON_CC_FLAGS="--target trn2 --lnc 2"`.
     - First GQA remote test run with target override failed one new test: `TypeError: GroupQueryAttention_QKV._qk_rmsnorm_gamma() takes 2 positional arguments but 3 were given`. Root cause: the unit-test helper bound a `staticmethod` as an instance method on a mock. Mitigation: changed the test helper to attach `_qk_rmsnorm_gamma` without `__get__`.
   - Verification passed:
     - Local syntax: `python3 -m py_compile` passed for `gqa.py`, `attention_base.py`, and `qwen36_27b_compile_fp8.py`; `bash -n tmp_compile_qwen32k_segcte2048_gdnseg512.sh` passed.
     - Remote targeted tests under compile venv with `NEURON_PLATFORM_TARGET_OVERRIDE=trn2`: `python -m pytest test/unit/modules/attention/test_gqa.py -q` passed `13 passed`; targeted attention-base tests `test_prep_qkv_tensors_qkv_cte_fuse_rope_nki_kernel` and `test_prep_qkv_tensors_fused_rope_passes_qwen_pre_rope_qk_norm` passed `7 passed`.

34. Full-head QKV CTE fused RoPE is not directly usable for Qwen3.6 partial RoPE; added qk-norm-only fallback.
   - Failed compile: `qwen36_32k_fp8_fp8all_lmheadfp8_gatesfp8_kvbf16_qkvnki_qknormrope_attention_cte512_gdnseg0_cte2048_pfx32k_slots64_20260605T194451Z_qknormrope_direct_scan0`
   - PID: `92686`
   - Compile log: `/home/ubuntu/validation_logs/fp8_256k_decode_nki/qwen36_32k_fp8_fp8all_lmheadfp8_gatesfp8_kvbf16_qkvnki_qknormrope_attention_cte512_gdnseg0_cte2048_pfx32k_slots64_20260605T194451Z_qknormrope_direct_scan0_compile.log`
   - Env evidence: `ENABLE_QKV_CTE_NKI_KERNEL_FUSE_ROPE=1`, `ENABLE_QKV_NKI_KERNELS=1`, `PREFIX_CTE_ATTENTION_BACKEND=attention_cte`, `QWEN36_DELTANET_FUSED_SEGMENT_TOKENS=0`, `GDN_RECURRENT_CACHE_DTYPE=bfloat16`.
   - Failure stage: first context HLO trace, inside `qkv_kernel[self.logical_nc_config]` before neuron-cc compilation.
   - Exact error: `AssertionError: error: failed to compile NKI kernel: ... [NCC_INKI016] Kernel validation exception: [QKV CTE Kernel] cos_cache and sin_cache must have the shape of (B, S, d_head) where S = 2048, but got cos_cache.shape = (1, 2048, 64), sin_cache.shape = (1, 2048, 64).`
   - Root cause: stock NKI QKV CTE fused RoPE expects full-head RoPE caches `[B, S, d_head]` and rotates the head by splitting `d_head` into two halves. Qwen3.6 uses partial RoPE: `rope_dim=64` with pass-through dimensions, as shown in `contrib/models/Qwen3.6-27B/src/modeling_qwen35.py` where only `Q[..., :self.rope_dim]` and `K[..., :self.rope_dim]` are rotated. Expanding the cache to full width would still be mathematically wrong because the NKI full-head pairing is different from Qwen's partial-RoPE pairing.
   - Automation: `monitor-qwen-qknormrope-compile` was deleted after this compile failure was fully identified and reported.
   - Fix/mitigation: added a separate `qkv_cte_nki_kernel_fuse_qk_norm` config/CLI/script flag. The wrapper can now fuse Qwen's pre-RoPE Q/K RMSNorm into the QKV CTE kernel while keeping Qwen partial RoPE in the existing post-QKV path. If full-head fused RoPE is requested for a model whose cache width equals `head_dim`, it still fuses RoPE; if cache width is partial, the wrapper deliberately passes `cos_cache=None` / `sin_cache=None` to the QKV kernel and applies RoPE afterward.
   - Files changed: `src/neuronx_distributed_inference/models/config.py`, `src/neuronx_distributed_inference/modules/attention/attention_base.py`, `src/neuronx_distributed_inference/modules/attention/gqa.py`, `contrib/models/Qwen3.6-27B/test/integration/qwen36_27b_compile_fp8.py`, `tmp_compile_qwen32k_segcte2048_gdnseg512.sh`, and attention unit tests.
   - Verification passed:
     - Local syntax: `python3 -m py_compile` passed for `config.py`, `gqa.py`, `attention_base.py`, and `qwen36_27b_compile_fp8.py`; `bash -n tmp_compile_qwen32k_segcte2048_gdnseg512.sh` passed.
     - Remote targeted tests under compile venv with `NEURON_PLATFORM_TARGET_OVERRIDE=trn2`: `python -m pytest test/unit/modules/attention/test_gqa.py -q` passed `14 passed`; focused attention-base tests including `test_prep_qkv_tensors_fuses_qk_norm_without_fusing_rope` and `test_prep_qkv_tensors_does_not_fuse_partial_rope_cache` passed `9 passed`.

35. QKV CTE qk-norm-only compile completed and validated; coherent but still slow.
   - Source branch/workdir: `/private/tmp/inferentia-gdn-prefill-speed-coherent`, branch `codex/qwen36-prefill-speed-coherent`.
   - Local commits: `244f6d2 Add Qwen-safe QKV fused qk norm rope`, `3356b88 Add QKV CTE qk norm only fusion`.
   - Compile host/source: `ubuntu@16.26.135.243`, `/home/ubuntu/inferentia-gdn-prefill-speed-coherent`, remote commit `8c308e4`.
   - Compile PID/log/env:
     - PID `95304`
     - `/home/ubuntu/validation_logs/fp8_256k_decode_nki/qwen36_32k_fp8_fp8all_lmheadfp8_gatesfp8_kvbf16_qkvnki_qknorm_attention_cte512_gdnseg0_cte2048_pfx32k_slots64_20260605T195500Z_qknorm_direct_scan0_compile.log`
     - `/home/ubuntu/validation_logs/fp8_256k_decode_nki/qwen36_32k_fp8_fp8all_lmheadfp8_gatesfp8_kvbf16_qkvnki_qknorm_attention_cte512_gdnseg0_cte2048_pfx32k_slots64_20260605T195500Z_qknorm_direct_scan0_env.txt`
   - Artifact: `/mnt/trainium_artifacts/qwen_artifacts/qwen36_32k_fp8_fp8all_lmheadfp8_gatesfp8_kvbf16_qkvnki_qknorm_attention_cte512_gdnseg0_cte2048_pfx32k_slots64_20260605T195500Z_qknorm_direct_scan0`.
   - Compile shape: `ENABLE_QKV_NKI_KERNELS=1`, `ENABLE_QKV_CTE_NKI_KERNEL_FUSE_QK_NORM=1`, `ENABLE_QKV_CTE_NKI_KERNEL_FUSE_ROPE=0`, `PREFIX_CTE_ATTENTION_BACKEND=attention_cte`, `QWEN36_DELTANET_FUSED_SEGMENT_TOKENS=0`, `QWEN36_DELTANET_MULTIHEAD_CTE=0`, `QWEN36_DELTANET_SOLVE_MODE=direct`, `QWEN36_DELTANET_SOLVE_SCAN_STEPS=0`, `GDN_RECURRENT_CACHE_DTYPE=bfloat16`, `GDN_CONV_CACHE_DTYPE=bfloat16`, `CTE_BUCKETS_RAW=2048`, `MAX_CONTEXT_LENGTH=32768`, `ENABLE_KV_CACHE_QUANT=0`, `QUANTIZE_LM_HEAD=1`, `FP8_QUANTIZE_LINEAR_ATTN_GATES=1`.
   - Compile evidence: all 13 HLOs compiled, `INFO:Neuron:Finished Compilation for all HLOs in 418.9846622943878 seconds`, `CHECKPOINT_BANK_WEIGHTS_ADDED tp0..tp3 48 48 torch.bfloat16 torch.bfloat16`, and `COMPILE_DONE`.
   - Config evidence: compile trace contained `"enable_qkv_cte_nki_kernel_fuse_qk_norm": true`, `"enable_qkv_cte_nki_kernel_fuse_rope": false`, and `"prefix_cte_attention_backend": "attention_cte"`. Env log confirmed `ENABLE_QKV_CTE_NKI_KERNEL_FUSE_QK_NORM=1`, `ENABLE_QKV_CTE_NKI_KERNEL_FUSE_ROPE=0`, `PREFIX_CTE_ATTENTION_BACKEND=attention_cte`, and `GDN_RECURRENT_CACHE_DTYPE=bfloat16`.
   - Artifact verification: runtime artifact has `model.pt` `714M`, `neuron_config.json` `105K`, and four `weights/tp{0..3}_sharded_checkpoint.safetensors` shards of `7.8G` each.
   - Transfer: first direct rsync failed with `ubuntu@16.26.184.190: Permission denied (publickey)` and `rsync error ... code 255`; direct SSH was fixed by authorizing compile-host public key `/home/ubuntu/.ssh/qwen_rsync_ed25519.pub` on the runtime host. A second rsync still failed because local SSH stripped the quoted `-e 'ssh -i ...'` identity string, causing rsync to use the default key and hit the same publickey error. A `bash -lc` test also failed with SSH usage text because the command string was split. Mitigation: copied `/tmp/qwen_rsync_qknorm_artifact.sh` to the compile host and ran it there; final EC2-to-EC2 rsync transferred `34,034,976,633` bytes in about `0:02:04`.
   - Launch: first wrapper launch failed before vLLM because `/home/ubuntu/inferentia-gdn-prefill-speed-coherent/tmp_launch_qwen36_segcte2048.sh` was not executable: `Permission denied`. Mitigation: changed wrapper to invoke `bash tmp_launch_qwen36_segcte2048.sh`. Relaunch reached `HEALTH_OK attempt=47`.
   - Launch flags: wrapper `/home/ubuntu/launch_qwen36_qknorm_32k_runtime.sh` used `MAX_MODEL_LEN=32768`, `SEQ_LEN=32768`, `CTE_BUCKETS=2048`, context pairs `2048:{256,512,1024,2048,4096,8192,16384,32768}`, token buckets `{512,16384,16640,32768}`, `GDN_RECURRENT_CACHE_DTYPE=bfloat16`, `GDN_CONV_CACHE_DTYPE=bfloat16`, `NUM_GPU_BLOCKS_OVERRIDE=128`, `PORT=8001`, `QWEN36_DELTANET_MULTIHEAD_CTE=0`, `QWEN36_DELTANET_FUSED_SEGMENT_TOKENS=0`, `QWEN36_DELTANET_SOLVE_MODE=direct`, `QWEN36_DELTANET_SOLVE_SCAN_STEPS=0`.
   - Runtime config evidence: live log shows pre-compiled artifact loaded from the qknorm artifact path, hybrid KV-cache spec for `16/64` attention layers, KV quant off, QKV NKI enabled, BF16 recurrent/conv dtypes, and `prefix_cte_attention_backend="attention_cte"`.
   - Validation setup errors:
     - `/tmp/tmp_bisect_probe3.py` was missing on both runtime and compile hosts. Mitigation: used maintained validators from `validation_scripts/` and copied small helper scripts `/tmp/qwen36_scan_runtime_log.py` and `/tmp/qwen36_16k_cold_prefill_bench.py` to runtime.
     - A remote structured JSON check failed because SSH stripped the `python3 -c` quoting, producing `bash: -c: line 1: syntax error near unexpected token '('`. Mitigation: verified flags from compile trace and env log instead.
     - A multi-word `grep` log scan failed with `grep: token_id: No such file or directory`, `grep: argmax: No such file or directory`, and similar path errors because patterns were split by the remote shell. Mitigation: replaced it with exact-string Python log scanning.
     - First multi-turn chat probe failed with HTTP `404` because the script defaulted to model id `Qwen3.6-27B`; server model id was `/home/ubuntu/models/Qwen3.6-27B`. Mitigation: reran with `--model /home/ubuntu/models/Qwen3.6-27B`.
   - Coherence matrix:
     - Primary raw exact boundaries `146,160,485,505,526,1225,2048,2049,2500,4092,4096`: pass; all HTTP 200, valid OpenAI bodies, matching `usage.prompt_tokens`, non-empty coherent text. JSONL: `/home/ubuntu/validation_logs/fp8_256k_decode_nki/qknorm_boundary_probe_20260605T2035Z.jsonl`.
     - Repeated exact `2500`: pass 3/3; each row had `2048` prompt-token prefix hit and coherent text. JSONL: `/home/ubuntu/validation_logs/fp8_256k_decode_nki/qknorm_repeat2500_20260605T2037Z.jsonl`.
     - Unique 4k sweep `4088..4104`: pass all 17 rows, including `4103`; no empty text or mojibake. JSONL: `/home/ubuntu/validation_logs/fp8_256k_decode_nki/qknorm_4k_sweep_20260605T2038Z.jsonl`.
     - Long raw exact `8192` and `16384`: pass with coherent non-empty text and matching usage. JSONL: `/home/ubuntu/validation_logs/fp8_256k_decode_nki/qknorm_long_probe_20260605T2041Z.jsonl`.
     - Multi-turn chat `160,1225,2500`: pass after rerun with correct model id; streamed usage present; outputs included `ack 8`, `.`, and `lambda`. JSON: `/home/ubuntu/validation_logs/fp8_256k_decode_nki/qknorm_chat_multiturn_20260605T2043Z.json`.
   - Runtime log scans: exact-string scanner found zero matches for `negative token_id`, `out-of-vocab token_id`, `fallback argmax`, `finite=0`, `nan=`, `NaN`, `NRT_RESOURCE`, `Traceback`, `RuntimeError`, or `Internal Server Error` after coherence probes and again after the speed benchmark.
   - 16k cold-prefill speed: usage-accounted streaming `/v1/completions`, exact `usage.prompt_tokens=16384`, `max_tokens=1`, unique cold prompts:
     - run 0: TTFT `26.088903736999782s`, `628.006 tok/s`
     - run 1: TTFT `26.08751873199799s`, `628.040 tok/s`
     - run 2: TTFT `26.083300692997s`, `628.141 tok/s`
     - mean `628.063 tok/s`
     - JSON: `/home/ubuntu/validation_logs/fp8_256k_decode_nki/qknorm_16k_cold_prefill_20260605T2045Z.json`.
   - Conclusion: qk-norm-only QKV CTE fusion is compile-safe and coherence-safe, but it does not improve cold prefill over the coherent attention-CTE anchor. It should not be pursued further as the primary speed lever.

36. Output-projection NKI compile toggle was added, but the one-variable output-proj speed slice failed at HLO trace and must not be enabled for Qwen3.6 yet.
   - Source branch/workdir: `/private/tmp/inferentia-gdn-prefill-speed-coherent`, branch `codex/qwen36-prefill-speed-coherent`.
   - Local commit: `3465373 Add output projection NKI compile toggle`.
   - Compile-driver change: `tmp_compile_qwen32k_segcte2048_gdnseg512.sh` now exposes `ENABLE_OUT_PROJ_NKI_KERNEL=0|1`, tags artifacts as `outprojstd` or `outprojnki`, writes the flag to the env log, and passes `--enable-out-proj-nki-kernel` only when explicitly enabled. Local `bash -n tmp_compile_qwen32k_segcte2048_gdnseg512.sh` passed.
   - Compile host/source: `ubuntu@16.26.135.243`, `/home/ubuntu/inferentia-gdn-prefill-speed-coherent`, remote HEAD `8c308e4` plus synced compile-driver change from local commit `3465373`.
   - Compile PID/log/env:
     - PID `108387`
     - `/home/ubuntu/validation_logs/fp8_256k_decode_nki/qwen36_32k_fp8_fp8all_lmheadfp8_gatesfp8_kvbf16_qkvnki_qknorm_outprojnki_attention_cte512_gdnseg0_cte2048_pfx32k_slots64_20260605T205300Z_outproj_direct_scan0_compile.log`
     - `/home/ubuntu/validation_logs/fp8_256k_decode_nki/qwen36_32k_fp8_fp8all_lmheadfp8_gatesfp8_kvbf16_qkvnki_qknorm_outprojnki_attention_cte512_gdnseg0_cte2048_pfx32k_slots64_20260605T205300Z_outproj_direct_scan0_env.txt`
   - Artifact target: `/mnt/trainium_artifacts/qwen_artifacts/qwen36_32k_fp8_fp8all_lmheadfp8_gatesfp8_kvbf16_qkvnki_qknorm_outprojnki_attention_cte512_gdnseg0_cte2048_pfx32k_slots64_20260605T205300Z_outproj_direct_scan0`.
   - Compile shape: one-variable delta from the coherent qk-norm anchor: `ENABLE_OUT_PROJ_NKI_KERNEL=1`, with `ENABLE_QKV_NKI_KERNELS=1`, `ENABLE_QKV_CTE_NKI_KERNEL_FUSE_QK_NORM=1`, `ENABLE_QKV_CTE_NKI_KERNEL_FUSE_ROPE=0`, `PREFIX_CTE_ATTENTION_BACKEND=attention_cte`, `QWEN36_DELTANET_FUSED_SEGMENT_TOKENS=0`, `QWEN36_DELTANET_MULTIHEAD_CTE=0`, `QWEN36_DELTANET_SOLVE_MODE=direct`, `QWEN36_DELTANET_SOLVE_SCAN_STEPS=0`, `GDN_RECURRENT_CACHE_DTYPE=bfloat16`, `GDN_CONV_CACHE_DTYPE=bfloat16`, `CTE_BUCKETS_RAW=2048`, `MAX_CONTEXT_LENGTH=32768`, `ENABLE_KV_CACHE_QUANT=0`, `QUANTIZE_LM_HEAD=1`, and `FP8_QUANTIZE_LINEAR_ATTN_GATES=1`.
   - Env evidence: compile trace and env log both show `enable_out_proj_nki_kernel=true`, `enable_qkv_cte_nki_kernel_fuse_qk_norm=true`, `enable_qkv_cte_nki_kernel_fuse_rope=false`, `prefix_cte_attention_backend="attention_cte"`, KV quant off, and BF16 recurrent/conv caches.
   - Failure stage: first context HLO trace, before neuron-cc/NKI compilation, while tracing `context_encoding_model` for shape `[1,2048]`.
   - Exact error: `RuntimeError: Shapes are not compatible for broadcasting: bf16[1,2048,5120] vs. bf16[1,2048,1536]. Expected dimension 2 of shape bf16[1,2048,5120] (5120) to match dimension 2 of shape bf16[1,2048,1536] (1536). Either that or that any of them is either 1 or unbounded. Try reshaping one of the tensors to match the other.`
   - Root cause hypothesis, strongly supported by source inspection: the generic `GroupQueryAttention_O._kernel_o_proj()` path derives output width `H` from the transposed/partitioned `RowParallelLinear` weight and returns a TP-local attention width (`1536`, Qwen local heads * head dim) instead of the model hidden width (`5120`). The failure then surfaces in `modeling_qwen35.py` at the residual add in the standard attention path. This is a Qwen weight-layout/kernel-support gap, not an output coherence bug.
   - Fix/mitigation applied: do not enable `ENABLE_OUT_PROJ_NKI_KERNEL` in the coherent-speed rebuild. Leave the flag available for a future Qwen-specific output-proj kernel/layout fix, but keep default `0` and exclude it from the current speed path.
   - Errors encountered and mitigated:
     - Compile-host disk was too low before launch: `/mnt/trainium_artifacts` showed only `18G` free (`/dev/root 484G 466G 18G 97%`). Mitigation, with explicit approval: deleted obsolete compile-host artifact/workdir copies (`head_norowscale`, old attention-CTE/qk-norm workdirs, standard-QKV coherent copy, and old 256k kkt_hier copy), freeing space to `159G` available.
     - Remote `git pull --ff-only` failed with `There is no tracking information for the current branch`, exit `1`. Mitigation: tried explicit branch pull.
     - Explicit remote pull failed with `fatal: couldn't find remote ref codex/qwen36-prefill-speed-coherent`, exit `1`, because the EC2 clone's `origin` points to `/home/ubuntu/inferentia-gdn-decode-step-baseline-20260605`, not the local pushed branch. Mitigation: synced the single committed compile-driver file to the compile host with `scp` and verified the remote diff.
     - Local search command including nonexistent `scripts` failed with `rg: scripts: No such file or directory (os error 2)`, exit `2`. Root cause: no `scripts/` directory in this checkout. Mitigation: searched concrete existing paths.
     - A remote grep evidence command using `|` inside the pattern was split by the shell/tooling, causing commands such as `COMPILE_DONE: command not found`, `Traceback: command not found`, and `grep: Compilation: No such file or directory`. A second grep with a multi-word pattern still treated `Compilation` as a file. Mitigation: used direct `tail`/source inspection for the exact error and avoid multi-word/piped grep patterns in remote one-liners.
   - Verification result: compile process exited; `ps -p 108387` returned no running process, no `COMPILE_DONE`, no `model.pt`, and no `Finished Compilation for all HLOs`. Automation `monitor-qwen-outproj-compile` should be deleted after this failure is fully reported.

### Current next step

We now have five coherent CTE2048 artifacts in the same slow prefill class:

- Standard QKV + segmented attention + GDN seg512: coherent, about `622 tok/s`.
- QKV NKI + segmented attention + GDN seg512: coherent, about `626-633 tok/s`.
- QKV NKI + segmented attention + GDN seg0: coherent, about `638 tok/s`.
- QKV NKI + attention_cte + GDN seg0: coherent, about `627 tok/s`.
- QKV NKI + qk-norm-only QKV CTE fusion + attention_cte + GDN seg0: coherent, about `628 tok/s`.

The next compile should not touch QKV/QK-norm/RoPE/output-proj again. Output-proj NKI is now ruled out for Qwen3.6 until the generic kernel path is fixed for Qwen's RowParallel attention output layout.

The next useful step is profiling/diffing, not another blind speed kernel:

1. Profile/direct-run the current attention-CTE context NEFF with `NEURON_RT_INSPECT`/`neuron-profile`, and compare top kernels with the prior slow segmented-CTE profile and the fast full-FP8/CTE2048 artifact profile if available.
2. Diff the compile-time config and generated HLO/NEFF set between the old fast CTE2048 artifact and the coherent artifacts, focusing on context graph shape, number of 2048 calls per 16k request, modular flow flags, qkv kernel selection, attention implementation, and hidden fallback to standard/dense kernels.
3. Only after profiling identifies the heavy kernel, add the next speed slice. Candidate slices should be one-variable and coherence-gated. Do not enable packed qkvgate, quantized MLP NKI, multihead DeltaNet CTE, FP8 KV, full-head fused RoPE, or output-proj NKI by default.

37. Output-projection NKI weight-layout fix compiled, but runtime coherence failed; output-proj NKI is now ruled out as a safe speed slice.
   - Source branch/workdir: `/private/tmp/inferentia-gdn-prefill-speed-coherent`, branch `codex/qwen36-prefill-speed-coherent`.
   - Fix commit: `a0e93a5 Fix output projection NKI weight layout`.
   - What changed: `GroupQueryAttention_O` now preserves/normalizes the NKI output-projection weight contract as `[local_heads * head_dim, hidden_size]` and recovers transposed weights before calling `output_projection_cte`. Unit coverage added in `test/unit/modules/attention/test_gqa.py`.
   - Local/remote test verification:
     - Local `py_compile` passed.
     - Local pytest failed because this workstation lacks Neuron deps: first `ModuleNotFoundError: No module named 'neuronx_distributed_inference'`, then with `PYTHONPATH=src`, `ModuleNotFoundError: No module named 'neuronx_distributed'`. This is an environment gap, not a code failure.
     - Remote focused test initially failed because direct venv Python did not have `libneuronpjrt-path` on `PATH`: `FileNotFoundError: [Errno 2] No such file or directory: 'libneuronpjrt-path'`. Mitigation: sourced `/opt/aws_neuronx_venv_pytorch_inference_vllm_0_16/bin/activate`.
     - Remote focused output-proj tests passed: `2 passed, 14 deselected`.
     - Remote full GQA tests passed: `16 passed, 11 warnings`.
   - Compile host/source: `ubuntu@16.26.135.243`, `/home/ubuntu/inferentia-gdn-prefill-speed-coherent`, remote HEAD `8c308e4` plus synced local source changes from commits `3465373` and `a0e93a5`.
   - Compile PID/log/env:
     - PID `111592`
     - `/home/ubuntu/validation_logs/fp8_256k_decode_nki/qwen36_32k_fp8_fp8all_lmheadfp8_gatesfp8_kvbf16_qkvnki_qknorm_outprojnki_attention_cte512_gdnseg0_cte2048_pfx32k_slots64_20260605T210828Z_outprojfix_direct_scan0_compile.log`
     - `/home/ubuntu/validation_logs/fp8_256k_decode_nki/qwen36_32k_fp8_fp8all_lmheadfp8_gatesfp8_kvbf16_qkvnki_qknorm_outprojnki_attention_cte512_gdnseg0_cte2048_pfx32k_slots64_20260605T210828Z_outprojfix_direct_scan0_env.txt`
   - Artifact:
     - Compile host and runtime host: `/mnt/trainium_artifacts/qwen_artifacts/qwen36_32k_fp8_fp8all_lmheadfp8_gatesfp8_kvbf16_qkvnki_qknorm_outprojnki_attention_cte512_gdnseg0_cte2048_pfx32k_slots64_20260605T210828Z_outprojfix_direct_scan0`
     - `model.pt` size `748992330`, `neuron_config.json` size `106590`.
   - Compile shape: one-variable delta from the coherent qk-norm anchor: `ENABLE_OUT_PROJ_NKI_KERNEL=1`, `ENABLE_QKV_NKI_KERNELS=1`, `ENABLE_QKV_CTE_NKI_KERNEL_FUSE_QK_NORM=1`, `ENABLE_QKV_CTE_NKI_KERNEL_FUSE_ROPE=0`, `PREFIX_CTE_ATTENTION_BACKEND=attention_cte`, `QWEN36_DELTANET_FUSED_SEGMENT_TOKENS=0`, `QWEN36_DELTANET_MULTIHEAD_CTE=0`, `QWEN36_DELTANET_SOLVE_MODE=direct`, `QWEN36_DELTANET_SOLVE_SCAN_STEPS=0`, `GDN_RECURRENT_CACHE_DTYPE=bfloat16`, `GDN_CONV_CACHE_DTYPE=bfloat16`, `CTE_BUCKETS_RAW=2048`, `MAX_CONTEXT_LENGTH=32768`, `ENABLE_KV_CACHE_QUANT=0`, `QUANTIZE_LM_HEAD=1`, and `FP8_QUANTIZE_LINEAR_ATTN_GATES=1`.
   - Compile result: success. Log shows `Finished Compilation for all HLOs in 421.13827085494995 seconds`, `CHECKPOINT_BANK_WEIGHTS_ADDED` for `tp0`..`tp3` with `48 48 torch.bfloat16 torch.bfloat16`, and `COMPILE_DONE`.
   - Transfer: EC2-to-EC2 `rsync` from compile host to runtime host `ubuntu@16.26.184.190` completed, `34,035,465,912` bytes transferred.
   - Runtime launch:
     - First launch command failed before touching the server because it passed the artifact through `env ART=... "$ART"`; shell expansion occurred before `env` set `ART`, so the launch script printed `usage: tmp_launch_qwen36_segcte2048.sh ARTIFACT [LOG]`. Mitigation: reran with literal artifact/log arguments.
     - Successful output-proj runtime log: `/home/ubuntu/validation_logs/fp8_256k_decode_nki/qwen36_outprojfix_runtime_20260605T2140Z.log`.
     - Runtime loaded the artifact successfully and reported hybrid KV allocation for `16/64` attention layers.
   - Boundary validation:
     - First validation invocation failed before request because system Python lacked Transformers: `ModuleNotFoundError: No module named 'transformers'`. Mitigation: reran after sourcing the Neuron/vLLM venv.
     - Second validation invocation hit HTTP 404 for every row because `--base-url http://127.0.0.1:8001/v1` caused the script to request `/v1/v1/chat/completions`. Mitigation: reran with `--base-url http://127.0.0.1:8001`.
     - Correct boundary JSONL: `/home/ubuntu/validation_logs/fp8_256k_decode_nki/qwen36_outprojfix_boundary_20260605T2150Z.jsonl`.
     - HTTP/OpenAI body validity passed for all 22 rows, but content was incoherent from short context:
       - `146` and `160`: repeated `.swap`.
       - `485`: mixed Thai/Chinese/English fragments such as `.swap`, `一条龙`, `把好`, `Specifier`, `finn`, `Pant`, `Bota`.
       - `505`/`526`: mojibake-like multilingual fragments and repeated `把好`.
       - `1225+`: degenerate repeated `把好`, `_quit_quit`, `finn`, and similar fragments.
       - Repeats with prefix-cache hits stayed incoherent, so this is not just a cold-cache artifact.
   - Runtime log scan: no `negative token_id`, `out-of-vocab token_id`, `fallback argmax`, `finite=0`, `nan=`, or `NRT_RESOURCE` matches. This is a clean wrong-output/logit corruption, not a sampler fallback or invalid-token path.
   - Conclusion: the weight-layout fix fixed compile-time shape assembly, but the generic output-projection NKI kernel is still numerically/semantically wrong for Qwen3.6 in this path. Keep `ENABLE_OUT_PROJ_NKI_KERNEL=0`; do not revisit output-proj NKI as a speed lever until there is a separate CPU/device equivalence test for the exact Qwen attention output layout and FP8 scale handling.
   - Restore state: runtime host `ubuntu@16.26.184.190` is restored to the coherent qk-norm anchor on port `8001`.
     - Artifact: `/mnt/trainium_artifacts/qwen_artifacts/qwen36_32k_fp8_fp8all_lmheadfp8_gatesfp8_kvbf16_qkvnki_qknorm_attention_cte512_gdnseg0_cte2048_pfx32k_slots64_20260605T195500Z_qknorm_direct_scan0`.
     - Restore log: `/home/ubuntu/validation_logs/fp8_256k_decode_nki/qwen36_qknorm_restore_runtime_20260605T2158Z.log`.
     - Sanity JSONL: `/home/ubuntu/validation_logs/fp8_256k_decode_nki/qwen36_qknorm_restore_sanity_20260605T2202Z.jsonl`.
     - Sanity passed at `146` and `485`, with coherent text and valid OpenAI bodies.

For this completed attention-CTE artifact, launch with BF16 recurrent banks:

```bash
GDN_RECURRENT_CACHE_DTYPE=bfloat16 \
bash tmp_launch_qwen36_segcte2048.sh ...
```

Compile the same coherent-speed anchor with standard QKV after the segmented CTE `kernel_assert` fix:

```bash
ENABLE_QKV_NKI_KERNELS=0 \
CTE_BUCKETS_RAW=2048 \
PREFIX_CTE_ATTENTION_BACKEND=segmented_cte \
PREFIX_CTE_ATTENTION_SEGMENT_SIZE=512 \
QWEN36_DELTANET_FUSED_SEGMENT_TOKENS=512 \
QWEN36_DELTANET_MULTIHEAD_CTE=0 \
ENABLE_KV_CACHE_QUANT=0 \
QUANTIZE_LM_HEAD=1 \
FP8_QUANTIZE_LINEAR_ATTN_GATES=1 \
bash tmp_compile_qwen32k_segcte2048_gdnseg512.sh
```

38. Sample-token-only compile slice is in flight under automation; no manual polling needed unless the user asks.
   - Source branch/workdir: `/private/tmp/inferentia-gdn-prefill-speed-coherent`, branch `codex/qwen36-prefill-speed-coherent`.
   - Commit: `0901798 Add Qwen tokens-only sampling compile option`.
   - What changed: `tmp_compile_qwen32k_segcte2048_gdnseg512.sh` now supports `OUTPUT_LOGITS_WITH_ON_DEVICE_SAMPLING=0|1` and tags artifacts as `sampletokonly` or `sampletoklogits`; aliases are covered for tokens-only on-device sampling; stale compile-config test defaults were updated for the current QKV CTE fusion flags, argmax flags, FP8 gate flag, and FP32 recurrent checkpoint-bank policy.
   - Local checks: `bash -n tmp_compile_qwen32k_segcte2048_gdnseg512.sh` passed; Python syntax checks for the updated unit tests passed.
   - Remote checks on compile host `ubuntu@16.26.135.243`:
     - Correct shell check passed: `bash -n tmp_compile_qwen32k_segcte2048_gdnseg512.sh`.
     - Focused alias tests passed: `3 passed in 1.11s`.
     - Full updated two-file unit suite passed: `75 passed, 3 subtests passed in 1.35s`.
   - Errors encountered and mitigated:
     - Incorrect command: `PYTHONPATH=src /home/ubuntu/venvs/neuron_230_segmented_cte/bin/python -m py_compile tmp_compile_qwen32k_segcte2048_gdnseg512.sh ...`.
       Error: `SyntaxError: invalid decimal literal` at the Bash parameter expansion in `MODEL=${MODEL:-/home/ubuntu/models/Qwen3.6-27B}`. Root cause: tried to run Python compilation on a shell script. Mitigation: reran with `bash -n`; shell syntax passed.
     - Broad compile-config pytest initially failed with stale fixture/defaults, first at `AttributeError: 'Namespace' object has no attribute 'enable_qkv_cte_nki_kernel_fuse_rope'`, then `disable_argmax_kernel`, then `fp8_quantize_linear_attn_gates`; the checkpoint-bank test also expected `torch.bfloat16` recurrent slots while current policy writes recurrent `torch.float32` and conv `torch.bfloat16`. Mitigation: updated the unit fixture defaults and expected recurrent dtype; reran remote suite successfully.
   - Active compile automation: `monitor-qwen-sampletokonly-compile`, every 10 minutes.
   - Compile host/source: `ubuntu@16.26.135.243`, `/home/ubuntu/inferentia-gdn-prefill-speed-coherent`, local commit `0901798` synced.
   - Compile PID/log/env:
     - PID `123175`
     - `/home/ubuntu/validation_logs/fp8_256k_decode_nki/qwen36_32k_fp8_fp8all_lmheadfp8_gatesfp8_kvbf16_qkvnki_qknorm_outprojstd_sampletokonly_attention_cte512_gdnseg0_cte2048_pfx32k_slots64_20260605T221059Z_sampletokonly_direct_scan0_compile.log`
     - `/home/ubuntu/validation_logs/fp8_256k_decode_nki/qwen36_32k_fp8_fp8all_lmheadfp8_gatesfp8_kvbf16_qkvnki_qknorm_outprojstd_sampletokonly_attention_cte512_gdnseg0_cte2048_pfx32k_slots64_20260605T221059Z_sampletokonly_direct_scan0_env.txt`
   - Artifact target: `/mnt/trainium_artifacts/qwen_artifacts/qwen36_32k_fp8_fp8all_lmheadfp8_gatesfp8_kvbf16_qkvnki_qknorm_outprojstd_sampletokonly_attention_cte512_gdnseg0_cte2048_pfx32k_slots64_20260605T221059Z_sampletokonly_direct_scan0`.
   - Compile shape: one-variable delta from coherent qk-norm anchor: `OUTPUT_LOGITS_WITH_ON_DEVICE_SAMPLING=0`, `ENABLE_QKV_NKI_KERNELS=1`, `ENABLE_QKV_CTE_NKI_KERNEL_FUSE_QK_NORM=1`, `ENABLE_QKV_CTE_NKI_KERNEL_FUSE_ROPE=0`, `ENABLE_OUT_PROJ_NKI_KERNEL=0`, `PREFIX_CTE_ATTENTION_BACKEND=attention_cte`, `QWEN36_DELTANET_FUSED_SEGMENT_TOKENS=0`, `QWEN36_DELTANET_MULTIHEAD_CTE=0`, `QWEN36_DELTANET_SOLVE_MODE=direct`, `QWEN36_DELTANET_SOLVE_SCAN_STEPS=0`, `GDN_RECURRENT_CACHE_DTYPE=bfloat16`, `GDN_CONV_CACHE_DTYPE=bfloat16`, `CTE_BUCKETS_RAW=2048`, `MAX_CONTEXT_LENGTH=32768`, `ENABLE_KV_CACHE_QUANT=0`, `QUANTIZE_LM_HEAD=1`, and `FP8_QUANTIZE_LINEAR_ATTN_GATES=1`.
   - Early evidence before stopping manual polling: env log shows `SAMPLING=on_device_greedy_sampletokonly` and `OUTPUT_LOGITS_WITH_ON_DEVICE_SAMPLING=0`; process command has no `--output-logits-with-on-device-sampling`; compile trace shows `disable_context_encoding_argmax_kernel=true`, `enable_qkv_cte_nki_kernel_fuse_qk_norm=true`, `enable_out_proj_nki_kernel=false`, and `enable_kv_cache_quant=false`.
   - HLO evidence: HLO generation completed in `302.8270480632782 seconds`, with each context HLO around `30-32s`. This is effectively identical to the coherent qk-norm anchor's HLO generation time (`300.4844801425934s`), so dropping returned debug logits does not recover the old short HLO generation path. The compile is still worth validating for runtime TTFT, but the current best hypothesis shifts toward the old `hostlogits`/`lmheadbf16` graph shape or another old compile-policy difference.
   - User instruction after launch: do not manually monitor continuously; rely on automation unless asked for a check or a heartbeat reports a meaningful result.

39. Next compile decision tree after the sample-token-only automation verdict.
   - Do not launch another compile until `monitor-qwen-sampletokonly-compile` reports a final compile/runtime verdict.
   - If sample-token-only is incoherent, revert to the coherent qk-norm anchor and do not pursue sampling-output changes further.
   - If sample-token-only is coherent but still slow, the next one-variable compile should flip only sampling mode to host-side logits:

```bash
TS=<timestamp>_hostlogits \
DISABLE_ON_DEVICE_SAMPLING=1 \
OUTPUT_LOGITS_WITH_ON_DEVICE_SAMPLING=0 \
ENABLE_QKV_NKI_KERNELS=1 \
ENABLE_QKV_CTE_NKI_KERNEL_FUSE_QK_NORM=1 \
ENABLE_QKV_CTE_NKI_KERNEL_FUSE_ROPE=0 \
ENABLE_OUT_PROJ_NKI_KERNEL=0 \
PREFIX_CTE_ATTENTION_BACKEND=attention_cte \
QWEN36_DELTANET_FUSED_SEGMENT_TOKENS=0 \
QWEN36_DELTANET_MULTIHEAD_CTE=0 \
QWEN36_DELTANET_SOLVE_MODE=direct \
QWEN36_DELTANET_SOLVE_SCAN_STEPS=0 \
GDN_RECURRENT_CACHE_DTYPE=bfloat16 \
GDN_CONV_CACHE_DTYPE=bfloat16 \
CTE_BUCKETS_RAW=2048 \
SEQ_LEN=32768 \
MAX_CONTEXT_LENGTH=32768 \
ENABLE_KV_CACHE_QUANT=0 \
QUANTIZE_LM_HEAD=1 \
FP8_QUANTIZE_LINEAR_ATTN_GATES=1 \
bash tmp_compile_qwen32k_segcte2048_gdnseg512.sh
```

   - Rationale: this isolates hostlogits from lm_head dtype. The old fast family used `hostlogits` and `lmheadbf16`, but changing both at once would confound the result.
   - If hostlogits-only is coherent but still slow, the next one-variable compile should keep hostlogits and flip only `QUANTIZE_LM_HEAD=0` to reproduce the old `lmheadbf16` policy.
   - Before either compile, create a new automation with exact PID/log/env/artifact paths. Do not reuse `monitor-qwen-sampletokonly-compile`.

40. Compile automation and profiling contract for the next speed slice.
   - Every new compile gets its own heartbeat automation before launch. The prompt must include source commit, compile host, PID/log/env/artifact/workdir paths, expected env flags, success markers, rsync target, runtime launch command shape, coherence matrix, log-scan strings, and the usage-accounted 16k cold-prefill benchmark. Do not reuse an older automation id for a different artifact.
   - After a compile succeeds, validation order is fixed: artifact file/config check -> EC2-to-EC2 rsync -> launch -> `/health` -> boundary coherence -> 4k sweep -> multi-turn -> long probes -> runtime log scan -> 16k cold-prefill benchmark. Speed numbers are not meaningful until the artifact is coherent and log-clean.
   - Runtime log scan must include exact strings: `negative token_id`, `out-of-vocab token_id`, `fallback argmax`, `finite=0`, `nan=`, `NaN`, `NRT_RESOURCE`, `Traceback`, `RuntimeError`, and `Internal Server Error`.
   - Do not profile by enabling `NEURON_RT_INSPECT_DEVICE_PROFILE=1` on the full live vLLM request path. That already stalled a 16k request for about 603 seconds with no usage payload and repeated Neuron async-exec polling timeouts.
   - If a coherent artifact is still slow, profile direct context NEFFs instead: map loose `context_encoding_model/_tp*_bk*/graph.neff` files to runtime inspect NEFFs, run `neuron-explorer capture` with `--io-from=neff`, `--num-exec=2`, `--profile-nth-exec=2`, `--ignore-exec-errors`, and summarize with `neuron-explorer view --output-format summary-json --ignore-nc-buf-usage`.
   - Keep the speed slice one-variable: sample-token-only -> hostlogits -> lm_head BF16. Do not mix in output-proj NKI, packed qkvgate, fused RoPE, GDN segmentation, segmented attention, multihead DeltaNet CTE, or FP8 KV while the current sampling/lm_head hypothesis is unresolved.

41. Compile-driver dry-run guard for sampling-mode slices.
   - Change: `tmp_compile_qwen32k_segcte2048_gdnseg512.sh` now supports `COMPILE_DRY_RUN=1`, which writes the env log and prints `BASE`, artifact, workdir, log, env log, and PID paths without launching the compile subprocess. This makes future automation prompts and one-variable compile flags testable before a long compile.
   - Change: the env log now labels `DISABLE_ON_DEVICE_SAMPLING=1` as `SAMPLING=host_logits` instead of the misleading `on_device_greedy_hostlogits`. On-device sample-token-only remains `SAMPLING=on_device_greedy_sampletokonly`.
   - Added local unit coverage in `contrib/models/Qwen3.6-27B/test/unit/test_qwen36_compile_driver.py` for both `sampletokonly` and `hostlogits` dry-run modes; each asserts the artifact tag, env-log sampling label, flag values, and that no PID file is created.
   - Error encountered: running `python3 -m unittest contrib.models.Qwen3.6-27B.test.unit.test_qwen36_compile_driver` failed because Python treated `Qwen3.6-27B` as a dotted module path and attempted to import `contrib.models.Qwen3`, raising `ModuleNotFoundError: No module named 'contrib.models.Qwen3'`. The same happened when passing the path through `python3 -m unittest`.
   - Mitigation: run these tests directly by file path (`python3 contrib/models/Qwen3.6-27B/test/unit/test_qwen36_compile_driver.py`, etc.), which avoids package parsing of the model directory name.
   - Verification passed locally: `bash -n tmp_compile_qwen32k_segcte2048_gdnseg512.sh`; direct execution of `test_qwen36_compile_driver.py` (`2 tests`), `test_qwen36_compile_fp8_config.py` (`31 tests`), and `test_qwen36_model_aliases.py` (`44 tests`).

42. Automation prompt generator for future compiles.
   - Added `validation_scripts/qwen36_compile_monitor_prompt.py`, which reads a compile env log and renders a self-contained Codex automation prompt containing compile host, runtime host, source commit, PID/log/env/artifact/workdir paths, expected flags, success markers, rsync/launch instructions, coherence matrix, log-scan strings, speed gate, and duplicate-compile guard.
   - Intended use after a `COMPILE_DRY_RUN=1` preflight: pass the rendered prompt to `codex_app.automation_update` as a fresh heartbeat automation before launching the compile. Do not include schedule details in the prompt; the automation tool owns cadence.
   - The generated prompt now explicitly tells heartbeat monitors to return a quiet/DONT_NOTIFY status while a compile is still running with no new failure signal, and to notify only for terminal compile failure, completed compile readiness, runtime validation failure, or coherent speed results. This preserves the "automation for compiles, no manual watch loop" workflow.
   - The generated prompt now also tells monitors to run `validation_scripts/qwen36_speed_slice_decision.py` after `runtime_validation_summary.json` is written, and to create a fresh automation before any next compile rather than launching it directly from the current monitor.
   - Added unit coverage in `test/unit/scripts/test_qwen36_compile_monitor_prompt.py` for env-log parsing, prompt content, and missing required path validation.
   - Error encountered during smoke test: rendering a prompt from the dry-run env log failed with `ValueError: env log is missing required key 'ENVLOG'` because the compile driver printed `ENVLOG` to stdout but did not write `ENVLOG` into the env log itself. Mitigation: the driver now writes `ENVLOG=${ENVLOG}` to env logs, and the prompt generator falls back to the `--env-log` argument for older env logs that lack the key.
   - Verification passed locally after mitigation: `python3 -m py_compile validation_scripts/qwen36_compile_monitor_prompt.py test/unit/scripts/test_qwen36_compile_monitor_prompt.py`; `python3 -m pytest test/unit/scripts/test_qwen36_compile_monitor_prompt.py`; `COMPILE_DRY_RUN=1 ... bash tmp_compile_qwen32k_segcte2048_gdnseg512.sh`; and prompt rendering from the generated env log.

43. Reusable runtime log scan gate.
   - Added `validation_scripts/qwen36_runtime_log_scan.py` to replace ad hoc remote `grep`/inline Python log scans. This matters because a prior multi-word `grep` scan split patterns like `negative token_id` into shell path arguments and failed with `grep: token_id: No such file or directory`.
   - Default markers: `negative token_id`, `out-of-vocab token_id`, `fallback argmax`, `finite=0`, `nan=`, `NaN`, `NRT_RESOURCE`, `Traceback`, `RuntimeError`, `Internal Server Error`, `InternalServerError`, and `EngineDeadError`.
   - Usage: `python3 validation_scripts/qwen36_runtime_log_scan.py --json <serve.log>` exits `0` only when all scanned files are clean; exits `1` and reports line numbers/text when any marker is present.
   - The compile automation prompt generator now explicitly instructs future monitors to use this scanner before reporting speed.

44. Maintained raw completion cold-prefill benchmark.
   - Added `validation_scripts/qwen36_raw_completion_prefill_bench.py` to replace the exact raw benchmark that was previously embedded only in `tmp_profile_qwen36_qkvnki_16k_prefill.sh`.
   - It sends exact token-id prompts to `/v1/completions`, streams with `stream_options: {"include_usage": true}`, records TTFT, and computes `prefill_tok_s` from `usage.prompt_tokens / TTFT`.
   - By default, missing usage is a failure. `--allow-usage-fallback` exists but labels token source as `actual_prompt_tokens`; do not use fallback numbers as the primary speed claim unless usage is genuinely unavailable and that limitation is stated.
   - The compile automation prompt generator now tells future monitors to run `validation_scripts/qwen36_raw_completion_prefill_bench.py --lengths 16384 --repeats 3 --max-tokens 1` only after coherence and log scan pass.

45. One-command runtime validation matrix.
   - Added `validation_scripts/qwen36_runtime_validation_matrix.py`, which orchestrates existing maintained gates in order: primary raw boundaries, repeated 2500 prompt, unique 4k sweep, long raw prompts, chat multi-turn probe, runtime log scan, then raw cold-prefill speed.
   - The driver requires `--serve-log` unless `--skip-log-scan` is explicit. It skips the speed step automatically if any coherence or log-scan step fails, so a broken artifact cannot accidentally produce a speed-only "pass".
   - Example after launch: `python3 validation_scripts/qwen36_runtime_validation_matrix.py --base-url http://127.0.0.1:8001 --model-path /home/ubuntu/models/Qwen3.6-27B --serve-log <serve.log> --output-dir <validation-output-dir>`.
   - The compile automation prompt generator now names this matrix driver as the preferred validation path after `/health`.
   - Error encountered during local tests: importing the matrix script with `importlib.util.module_from_spec(...); spec.loader.exec_module(...)` without registering it in `sys.modules` made the `@dataclass` decorator fail with `AttributeError: 'NoneType' object has no attribute '__dict__'`. Mitigation: the unit test loader now inserts the module into `sys.modules` before executing it.
   - Verification passed locally: `python3 -m py_compile validation_scripts/qwen36_runtime_validation_matrix.py test/unit/scripts/test_qwen36_runtime_validation_matrix.py validation_scripts/qwen36_compile_monitor_prompt.py test/unit/scripts/test_qwen36_compile_monitor_prompt.py`; `python3 -m pytest test/unit/scripts/test_qwen36_runtime_validation_matrix.py test/unit/scripts/test_qwen36_compile_monitor_prompt.py`; and `python3 validation_scripts/qwen36_runtime_validation_matrix.py --help`.

46. Chat validation model-id auto-detection.
   - Prior runtime validation had a real failure mode where the chat multi-turn probe used a stale hard-coded model id and had to be rerun with the served model id. To avoid repeating that, `validation_scripts/qwen36_chat_completion_context_bench.py` now supports `--model auto`, matching the raw boundary and raw speed probes.
   - `validation_scripts/qwen36_runtime_validation_matrix.py` now defaults `--chat-model auto`, so the matrix driver detects `/v1/models` for chat requests by default.

47. Compile readiness status checker.
   - Added `validation_scripts/qwen36_compile_status.py`, which reads `LOG`, `ARTIFACT`, and `PIDFILE` from a compile env log and emits a structured JSON verdict: `ready`, `running`, `failed`, or `incomplete`.
   - Ready requires `Finished Compilation for all HLOs`, `COMPILE_DONE`, no configured failure markers, `CHECKPOINT_BANK_WEIGHTS_ADDED` for tp0..tp3, checkpoint-bank dtypes matching `GDN_RECURRENT_CACHE_DTYPE` and `GDN_CONV_CACHE_DTYPE` from the env log, and artifact files `model.pt` plus `neuron_config.json`.
   - The compile automation prompt generator now instructs future monitors to run `validation_scripts/qwen36_compile_status.py --env-log <env-log>` before rsync/runtime validation.
   - Verification passed locally: `python3 -m py_compile validation_scripts/qwen36_compile_status.py test/unit/scripts/test_qwen36_compile_status.py validation_scripts/qwen36_compile_monitor_prompt.py test/unit/scripts/test_qwen36_compile_monitor_prompt.py`; `python3 -m pytest test/unit/scripts/test_qwen36_compile_status.py test/unit/scripts/test_qwen36_compile_monitor_prompt.py`; `python3 validation_scripts/qwen36_compile_status.py --help`; and a synthetic env/log/artifact CLI smoke that returned `state=ready`.
   - Follow-up: dtype validation was added because prior artifacts could compile with checkpoint-bank dtype mismatches that only surfaced at launch. `validation_scripts/qwen36_compile_status.py` now normalizes `torch.bfloat16`/`bfloat16` and `torch.float32`/`float32`, and reports `checkpoint_dtype_mismatches` as readiness failures.
   - Verification for dtype mismatch: a synthetic env log requesting `GDN_RECURRENT_CACHE_DTYPE=float32` against a BF16 checkpoint-bank compile log returned `ready=false`, `state=incomplete`, and recurrent `checkpoint_dtype_mismatches` for `tp0..tp3`.

48. Removed stale `/tmp/tmp_bisect_probe3.py` automation dependency.
   - The old ad hoc `/tmp/tmp_bisect_probe3.py` probe was previously missing on both runtime and compile hosts. The compile monitor prompt now explicitly points to the maintained `validation_scripts/qwen36_runtime_validation_matrix.py` driver instead of telling future monitors to use the missing tmp helper.
   - The maintained matrix still covers the intended gates: exact boundaries, repeated 2500 prompt, unique 4k sweep, long raw prompts, chat multi-turn, runtime log scan, and raw cold-prefill speed after coherence.

49. Direct context-NEFF profile comparison helper.
   - Added `validation_scripts/qwen36_profile_summary_compare.py` to consume `neuron-explorer view --output-format summary-json` outputs without pandas/duckdb/pyarrow. It estimates 16k cold-prefill tok/s from context-graph `total_time`, computes the per-context time required to hit the target tok/s, reports active/idle gap and dominant active engine, and can attach `qwen36_raw_completion_prefill_bench.py` speed JSON.
   - Intended use after a coherent but slow artifact: profile selected loose `context_encoding_model/_tp*_bk*/graph.neff` files directly, save the `summary-json`, then run `python3 validation_scripts/qwen36_profile_summary_compare.py --summary current=<summary.json> --speed-json <raw_prefill_speed.json>`. For the 16k/CTE2048 case, the helper uses 8 chunks and makes the target graph time explicit instead of relying on hand-written arithmetic.
   - Error encountered during local test: `python3 -m pytest test/unit/scripts/test_qwen36_profile_summary_compare.py` failed one assertion in `test_build_report_compares_candidate_to_baseline`; expected `682.6666666666667`, actual `682.6666666666666`. Root cause: exact equality on binary floating-point arithmetic. Mitigation: changed the assertion to compare with an absolute tolerance.

50. Maintained direct context-NEFF profiling runner.
   - Added `validation_scripts/qwen36_context_neff_profile.py` to replace the durable parts of `tmp_profile_qwen36_context_neffs_from_inspect.sh`. It discovers loose `context_encoding_model/_tp*_bk*/graph.neff` files, labels buckets as `context_bkN_pfxM`, writes a dry-run capture plan by default, and runs `neuron-explorer capture/view` only when `--run` is explicit.
   - The runner intentionally does not stop or relaunch vLLM. Direct NEFF profiling should be done on an idle Trainium host or after the monitor/validator has stopped the server; the script avoids `pkill`/server cleanup because prior cleanup commands dropped SSH sessions and obscured the profile result.
   - Intended use on the runtime host after copying loose context NEFFs: `python3 validation_scripts/qwen36_context_neff_profile.py --context-neff-root <context_encoding_model> --output-dir <profile-dir> --buckets 0,7 --run`; then compare summary JSON files with `qwen36_profile_summary_compare.py`.

51. 3k cold-prefill speed gate is now explicit.
   - `validation_scripts/qwen36_raw_completion_prefill_bench.py` now accepts `--min-prefill-tok-s`; when set above zero it records `speed_gate` in the output JSON and returns nonzero if mean measured prefill tok/s is below the threshold.
   - `validation_scripts/qwen36_runtime_validation_matrix.py` defaults `--min-prefill-tok-s` to `3000.0` and passes it to the raw speed benchmark, so a coherent/log-clean artifact at ~600 tok/s is a validation failure rather than a pass with a disappointing number.
   - The compile automation prompt generator now tells monitors to run the runtime matrix and raw benchmark with `--min-prefill-tok-s 3000`.

52. Validation-tool manifest for remote/source drift.
   - Added `validation_scripts/qwen36_validation_tool_manifest.py` to build/verify SHA256 manifests for the maintained Qwen validation/profiling scripts. This addresses the current EC2 workflow where the compile-host clone often receives source changes via manual `scp` instead of a reliable git remote branch.
   - Default manifest coverage includes the compile driver, compile readiness, monitor prompt generation, runtime coherence matrix, log scan, raw prefill speed gate, boundary/chat probes, direct context-NEFF profiling, profile summary comparison, and the manifest tool itself. `--include-tests` adds the matching unit tests, including the compile-driver dry-run guard tests.
   - Intended workflow before remote validation: build a local manifest, sync/copy the listed files if needed, then run `python3 validation_scripts/qwen36_validation_tool_manifest.py --repo <remote-source> --manifest <manifest.json> --mode verify` on the host that will run validators. `--mode file-list` emits the repo-relative files for a copy command.
   - Verification passed locally: manifest build for 12 files, self-verify with `mismatch_count=0`, file-list output, and `python3 -m pytest test/unit/scripts` (`67 passed`).

53. Compile-driver speed-slice guard for the next sampling/lm_head experiments.
   - `tmp_compile_qwen32k_segcte2048_gdnseg512.sh` now accepts optional `SPEED_SLICE=sampletokonly|hostlogits|hostlogits_lmheadbf16`. When set, it fails before compile if the env drifts from the coherent qk-norm attention-CTE anchor: QKV NKI + QK norm on, fused RoPE off, output-proj NKI off, KV quant off, attention_cte, GDN segmentation off, multihead CTE off, direct scan0 solve, BF16 GDN caches, and CTE2048.
   - `SPEED_SLICE=hostlogits` requires `DISABLE_ON_DEVICE_SAMPLING=1`, `OUTPUT_LOGITS_WITH_ON_DEVICE_SAMPLING=0`, and `QUANTIZE_LM_HEAD=1`, so it cannot accidentally combine the hostlogits experiment with the later lm_head BF16 experiment. `SPEED_SLICE=hostlogits_lmheadbf16` is the explicit second flip.
   - Error encountered during local tests: `python3 contrib/models/Qwen3.6-27B/test/unit/test_qwen36_compile_driver.py` initially failed `test_speed_slice_rejects_non_anchor_attention_backend` with `TypeError: ... got multiple values for keyword argument 'PREFIX_CTE_ATTENTION_BACKEND'`. Root cause: the test passed the same kwarg through the anchor override dict and explicitly. Mitigation: copy the anchor dict and override that key before calling the dry-run helper.
   - Verification passed locally after mitigation: `bash -n tmp_compile_qwen32k_segcte2048_gdnseg512.sh`; `python3 -m py_compile contrib/models/Qwen3.6-27B/test/unit/test_qwen36_compile_driver.py`; and direct compile-driver unittest (`6 tests`).

54. Automation payload generation for future compiles.
   - `validation_scripts/qwen36_compile_monitor_prompt.py` now supports `--automation-json`, which emits the `codex_app.automation_update` heartbeat-create payload from the same compile env log used to render the monitor prompt. The payload sets `destination=thread`, a generated unique monitor name, active status, the monitor prompt, and a configurable minute interval.
   - Intended workflow for the next speed slice: run `COMPILE_DRY_RUN=1 ... bash tmp_compile_qwen32k_segcte2048_gdnseg512.sh`, then run `python3 validation_scripts/qwen36_compile_monitor_prompt.py --env-log <env-log> --automation-json --automation-name <fresh-monitor-name>`. Use that payload to create the heartbeat automation before launching the real compile. This avoids hand-copying PID/log/artifact paths and keeps the "automation for compiles, no manual watch loop" rule intact.
   - Unit coverage added for generated payload names, interval validation, and CLI JSON output.
   - Verification passed locally: `python3 -m py_compile validation_scripts/qwen36_compile_monitor_prompt.py test/unit/scripts/test_qwen36_compile_monitor_prompt.py`; `python3 -m pytest test/unit/scripts/test_qwen36_compile_monitor_prompt.py` (`7 passed`); `python3 -m pytest test/unit/scripts` (`67 passed`); prompt-generator `--help`; manifest build for 12 files; and manifest verify with `mismatch_count=0`.

55. Runtime verdict to next speed-slice decision helper.
   - Added `validation_scripts/qwen36_speed_slice_decision.py`, which consumes a compile env log, `runtime_validation_summary.json`, and the raw prefill speed JSON to emit one explicit next action. This keeps the rebuild flow coherence-first and one-variable: incoherent/log-dirty artifacts stop speed work; missing/malformed speed data asks to rerun speed validation; coherent artifacts that pass the 3k gate are kept; coherent-but-slow `sampletokonly` advances only to `hostlogits`; coherent-but-slow `hostlogits` advances only to `hostlogits_lmheadbf16`; coherent-but-slow `hostlogits_lmheadbf16` profiles instead of inventing another compile.
   - The output JSON includes current artifact/base, current `SPEED_SLICE`, coherence/log-scan verdict, speed gate, mean prefill tok/s, `decision`, optional `next_speed_slice`, required flags for that next slice, and a `next_preflight` block.
   - `next_preflight` emits a three-step command sequence for `launch_next_speed_slice`: run the compile driver with `COMPILE_DRY_RUN=1`, create a fresh heartbeat automation from the dry-run `ENVLOG`, then run the real compile command only after automation exists. It includes both shell-ready commands and structured `dry_run_env`/`launch_env` maps. The generated env maps spell out the coherent speed-slice anchor flags (`attention_cte`, GDN segment 0, direct scan0, BF16 GDN caches, KV BF16, qk-norm QKV, no fused RoPE, no output-proj NKI, CTE2048) so future slices do not silently fall back to the compile driver's segmented defaults.
   - Issue found and fixed: the first `next_preflight` version did not pin `TS`, while `tmp_compile_qwen32k_segcte2048_gdnseg512.sh` defaults `TS=$(date -u +%Y%m%dT%H%M%SZ)` on every invocation and uses it in `BASE`, `ENVLOG`, `PIDFILE`, artifact, and workdir paths. Root cause: the generated dry-run command and later launch command could point at different timestamped artifacts, leaving the automation monitoring the wrong log/artifact. Mitigation: `next_preflight` now generates one `TS` and includes it in both commands; the compile driver also writes/prints `TS` so monitors can verify the artifact identity.
   - Follow-up fix: generated heartbeat monitor names now include the pinned `TS`, both in `qwen36_compile_monitor_prompt.py` defaults and in the `next_preflight` automation payload template. The truncation helper preserves the timestamp suffix even when the artifact `BASE` is long, preventing repeated `hostlogits` or `hostlogits_lmheadbf16` attempts from all using the same generic monitor name.
   - Runtime validation now supports `--compile-env-log <env-log>`, which makes `validation_scripts/qwen36_runtime_validation_matrix.py` write `<output-dir>/speed_slice_decision.json` and annotate `runtime_validation_summary.json` with the decision. Future compile monitor prompts pass this flag and then read the generated decision JSON; they still explicitly forbid launching the next compile from the current monitor.
   - The validation-tool manifest now includes this helper and its tests, so remote/source drift checks cover the decision logic used after compile automation reports a validation result.
   - Error encountered during prompt test update: `python3 -m pytest test/unit/scripts/test_qwen36_runtime_validation_matrix.py test/unit/scripts/test_qwen36_compile_monitor_prompt.py test/unit/scripts/test_qwen36_speed_slice_decision.py` initially failed `test_render_prompt_contains_compile_validation_and_runtime_gates` because the test still expected literal `qwen36_speed_slice_decision.py` in the prompt after the matrix started owning decision generation. Root cause: stale assertion. Mitigation: assert `--compile-env-log` and `speed_slice_decision.json` instead. Verification after fix: focused pytest (`24 passed`).
   - Verification passed locally after structured `next_preflight` and matrix decision-output updates: `python3 -m py_compile validation_scripts/qwen36_runtime_validation_matrix.py test/unit/scripts/test_qwen36_runtime_validation_matrix.py validation_scripts/qwen36_compile_monitor_prompt.py test/unit/scripts/test_qwen36_compile_monitor_prompt.py validation_scripts/qwen36_speed_slice_decision.py test/unit/scripts/test_qwen36_speed_slice_decision.py`; focused pytest (`24 passed`); `python3 -m pytest test/unit/scripts` (`72 passed`); `python3 validation_scripts/qwen36_runtime_validation_matrix.py --help`; manifest build for 12 files; and manifest verify with `mismatch_count=0`.

56. Direct-NEFF profile preflight for the terminal slow-coherent case.
   - Added `profile_preflight` to `validation_scripts/qwen36_speed_slice_decision.py` for the case where `hostlogits_lmheadbf16` is coherent/log-clean but still below the 3k 16k cold-prefill gate. This makes the next action explicit: do not invent another speed compile; profile the loose context NEFFs directly.
   - The emitted preflight includes `context_neff_root=<WORKDIR>/context_encoding_model`, a timestamped profile output directory under `LOGDIR`, `qwen36_context_neff_profile.py` plan/run commands with `--buckets 0,7 --enable-dge --run`, and a `qwen36_profile_summary_compare.py` command template tied to the raw speed JSON and CTE bucket size.
   - Updated the compile monitor prompt so heartbeat monitors follow the emitted `profile_preflight` on `profile_slow_coherent`, stop vLLM or use idle Trainium cores first, and avoid `NEURON_RT_INSPECT_DEVICE_PROFILE=1` on live vLLM requests. This matches the Neuron profiling skill workflow and the previous observed failure where full-server profiling stalled a 16k request for about 603 seconds.
   - Error encountered during focused local tests: `python3 -m pytest test/unit/scripts/test_qwen36_speed_slice_decision.py test/unit/scripts/test_qwen36_compile_monitor_prompt.py` failed `test_slow_final_planned_slice_profiles_instead_of_branching` with `AssertionError: assert "current='<SUMMARY_JSON_FROM_context_neff_profile_results>'" in "python3 ... --summary 'current=<SUMMARY_JSON_FROM_context_neff_profile_results>' ..."`. Root cause: the test expected quotes around only the placeholder, while `shlex.quote` correctly quotes the whole `current=<...>` shell argument. Mitigation: changed the assertion to match the actual shell-safe argument.
   - Follow-up: `validation_scripts/qwen36_runtime_validation_matrix.py` now copies `next_preflight` and `profile_preflight` into the compact `runtime_validation_summary.json` annotation, not only the standalone `speed_slice_decision.json`. This prevents a monitor or later handoff reader from seeing only `profile_slow_coherent` without the safe direct-NEFF profiling command path.

57. Compile-status polling mode for heartbeat automations.
   - Added `--zero-when-running` to `validation_scripts/qwen36_compile_status.py`. Strict mode still exits nonzero unless the artifact is ready, but heartbeat polling can now return exit 0 for `state=running` while preserving the JSON state. This prevents an automation wrapper from treating an active compile as a terminal failure just because the readiness helper is not ready yet.
   - Updated the compile monitor prompt to use `qwen36_compile_status.py --env-log <envlog> --zero-when-running` for quiet polling, and to require strict `ready=true` from `qwen36_compile_status.py --env-log <envlog>` before rsync/runtime validation.
   - Follow-up bug found during code review: `check_status` classified any live PID as `state=running` before considering `failure_lines`. With `--zero-when-running`, a compile log containing `Traceback` or `RuntimeError` could therefore return exit 0 if the wrapper PID was still alive. Mitigation: failure markers now take precedence over live PID, and a regression test verifies that live PID + `Traceback` still returns `state=failed` and exit 1 even with `--zero-when-running`.
