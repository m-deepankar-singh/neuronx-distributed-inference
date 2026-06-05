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

31. Attention-CTE one-variable compile is in flight to test the remaining cold-prefill speed suspect.
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
   - Initial status: PID alive; context HLO generation started; no immediate traceback, NCC error, or disk error.

### Current next step

Monitor the one-variable no-GDN-segmentation compile:

```bash
ssh ubuntu@16.26.135.243 \
  'pidfile=/home/ubuntu/validation_logs/fp8_256k_decode_nki/qwen36_32k_fp8_fp8all_lmheadfp8_gatesfp8_kvbf16_qkvnki_tiled_segmented_cte512_gdnseg0_cte2048_pfx32k_slots64_20260605T171105Z_nogdnseg_direct_scan0_compile.pid; \
   log=/home/ubuntu/validation_logs/fp8_256k_decode_nki/qwen36_32k_fp8_fp8all_lmheadfp8_gatesfp8_kvbf16_qkvnki_tiled_segmented_cte512_gdnseg0_cte2048_pfx32k_slots64_20260605T171105Z_nogdnseg_direct_scan0_compile.log; \
   pid=$(cat "$pidfile" 2>/dev/null || true); \
   ps -p "$pid" -o pid,etime,stat,cmd || true; \
   grep -nE "Finished generating HLO|Finished Compilation|CHECKPOINT_BANK_WEIGHTS_ADDED|COMPILE_DONE|Traceback|RuntimeError|Exception|NCC_|No space left|Killed" "$log" | tail -120'
```

Artifact under compile:

```bash
/mnt/trainium_artifacts/qwen_artifacts/qwen36_32k_fp8_fp8all_lmheadfp8_gatesfp8_kvbf16_qkvnki_tiled_segmented_cte512_gdnseg0_cte2048_pfx32k_slots64_20260605T171105Z_nogdnseg_direct_scan0
```

Compile command shape:

```bash
TS=20260605T171105Z_nogdnseg \
ENABLE_QKV_NKI_KERNELS=1 \
CTE_BUCKETS_RAW=2048 \
PREFIX_CTE_ATTENTION_BACKEND=segmented_cte \
PREFIX_CTE_ATTENTION_SEGMENT_SIZE=512 \
QWEN36_DELTANET_FUSED_SEGMENT_TOKENS=0 \
QWEN36_DELTANET_MULTIHEAD_CTE=0 \
ENABLE_KV_CACHE_QUANT=0 \
QUANTIZE_LM_HEAD=1 \
FP8_QUANTIZE_LINEAR_ATTN_GATES=1 \
QWEN36_DELTANET_SOLVE_MODE=direct \
QWEN36_DELTANET_SOLVE_SCAN_STEPS=0 \
GDN_RECURRENT_CACHE_DTYPE=bfloat16 \
GDN_CONV_CACHE_DTYPE=bfloat16 \
bash tmp_compile_qwen32k_segcte2048_gdnseg512.sh
```

Result from the no-GDN-seg experiment: coherent, but still only about `638 tok/s` at exact 16k cold prefill. Do not repeat this compile as a speed fix.

The next isolated speed experiment should move off `segmented_cte` for cold prefill and compile the same coherent policy with `PREFIX_CTE_ATTENTION_BACKEND=attention_cte`, keeping all else fixed (`QKV NKI`, `CTE2048`, `KV BF16`, `GDN segment tokens 0`, `direct scan0`, `multihead DeltaNet CTE off`). Reason: raw `pfx0` context is slow even without prefix reads and remains slow when GDN segmentation is disabled, so the remaining compile-baked suspect in the cold context graph is segmented CTE attention/model integration overhead.

Manual monitor command for the current attention-CTE compile:

```bash
ssh ubuntu@16.26.135.243 \
  'pidfile=/home/ubuntu/validation_logs/fp8_256k_decode_nki/qwen36_32k_fp8_fp8all_lmheadfp8_gatesfp8_kvbf16_qkvnki_tiled_attention_cte512_gdnseg0_cte2048_pfx32k_slots64_20260605T180405Z_attncte_direct_scan0_compile.pid; \
   log=/home/ubuntu/validation_logs/fp8_256k_decode_nki/qwen36_32k_fp8_fp8all_lmheadfp8_gatesfp8_kvbf16_qkvnki_tiled_attention_cte512_gdnseg0_cte2048_pfx32k_slots64_20260605T180405Z_attncte_direct_scan0_compile.log; \
   pid=$(cat "$pidfile" 2>/dev/null || true); \
   ps -p "$pid" -o pid,etime,stat,cmd || true; \
   grep -nE "Finished generating HLO|Compilation Successfully Completed|Finished Compilation|CHECKPOINT_BANK_WEIGHTS_ADDED|COMPILE_DONE|Traceback|RuntimeError|Exception|NCC_|No space left|Killed" "$log" | tail -160; \
   df -h /mnt/trainium_artifacts | tail -1'
```

Do not postprocess-only a BF16-traced artifact to FP32 recurrent banks. For this completed artifact, launch with BF16 recurrent banks:

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
