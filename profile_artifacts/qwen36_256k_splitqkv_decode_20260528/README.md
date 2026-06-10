# Qwen3.6 256K FP8 Split-QKV Decode Results - 2026-05-28

## Usable Artifact

- Artifact: `/mnt/trainium_artifacts/qwen_artifacts/qwen36_27b_256k_fp8_full_lmheadfp8_kvfp8_hybrid_apc_nki_decode_splitqkv_gdnrecbfloat16_sampletokens_b256_cte256_512_pfx16k_slots64_async_20260528T074851Z`
- Eval root: `trn2/splitqkv_decode_eval_20260528T074851Z`
- Compile logs: `compile_logs/*074851Z*`
- Result source: OpenAI/vLLM streaming usage, not content chunk count.

## Decode Speed

| Target prompt tokens | Completion tokens | TTFT seconds | Token TPOT seconds | Decode tok/s |
| --- | ---: | ---: | ---: | ---: |
| 512 | 256 | 2.111228919995483 | 0.028016996199998995 | 35.69262003897605 |
| 16384 | 256 | 48.01256207100232 | 0.03260043315294395 | 30.67443905755884 |

The coherence smoke for the usable artifact completed, but output quality still needs a separate quality pass.

## OpenAI Server Smoke

- Live-server log snapshot: `trn2/openai_splitqkv_good_20260528T103834Z`
- Backend was started on `127.0.0.1:8001`.
- Proxy was started on `0.0.0.0:8000` with `--allow-thinking`.
- Streaming was verified with `stream=true`; the proxy returns standard SSE `data:` chunks and `[DONE]`.
- Thinking mode is accepted by the proxy, but this artifact did not reliably generate a native `</think>` delimiter during smoke prompts.

## Failed MLP-Kernel Follow-Up

- Artifact attempted: `/mnt/trainium_artifacts/qwen_artifacts/qwen36_27b_256k_fp8_full_lmheadfp8_kvfp8_hybrid_apc_nki_decode_splitqkv_mlpker_gdnrecbfloat16_sampletokens_b256_cte256_512_pfx16k_slots64_async_20260528T093650Z`
- Compile status: compile completed; see `compile_logs/*093650Z*`.
- Runtime status: not usable; see `trn2/splitqkv_decode_eval_20260528T093650Z/server/backend.log`.
- Error evidence: Neuron runtime reported output tensors too small, for example `output67` had tensor size `17408` with minimum size `2228224`, followed by `NRT_INVALID in nrt_execute()` and `nrt_execute_repeat ... graph.neff with status 2`.
- Best current hypothesis: enabling the MLP TKG kernel changed the compiled graph output contract, while runtime buffer allocation still matched the older smaller output shape contract.
- Mitigation used: do not use the MLP-kernel artifact for serving; serve the split-QKV-only artifact above.
