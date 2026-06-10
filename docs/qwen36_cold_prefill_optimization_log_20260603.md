# Qwen3.6 Cold Prefill Optimization Log

Date: 2026-06-03

This note records the cold-prefill optimization attempts for Qwen3.6-27B FP8
Hybrid APC on TRN2, what each change was trying to prove, and the measured
outcome. Use this as the current working ledger before starting another compile.

## Current Best

Current best measured artifact:

```text
/mnt/trainium_artifacts/qwen_artifacts/qwen36_27b_32k_fp8_cte2048_gpfx2_b256_pfx16k_mlpcte_qkvnki_outprojnki_qknormrope_qkvgate_20260602T225035Z_scan2
```

Runtime:

```text
host: 16.50.153.185
instance: trn2.3xlarge
visible cores: 0-3
benchmark:
  /home/ubuntu/validation_logs/fp8_256k_decode_nki/qkvgate_steady_16k_20260602T232347Z.json
```

Result:

| Metric | Value |
|---|---:|
| Prompt length | 16,384 tokens |
| CTE bucket | 2,048 |
| Prefix mode | grouped prefix 2 |
| Measured runs | 3 |
| Avg cold prefill | 3,620.46 tok/s |
| Min / max | 3,617.56 / 3,623.61 tok/s |

Correctness note: this artifact still falls back to dummy generated token `0`
for the one-token output path. It is a prefill-performance artifact, not a
correctness artifact.

Output coherence diagnostic on 2026-06-03:

- Runtime command: `validation_scripts/qwen36_steady_cold_prefill_bench.py`
  on `16.50.153.185` with the qkvgate artifact above, 2,048 prompt tokens,
  one measured run, one generated token, and
  `QWEN36_SAMPLE_LOGITS_COMPARE_JSONL` enabled.
- Log root:
  `/home/ubuntu/validation_logs/fp8_256k_decode_nki/coherence_diag_qkvgate_20260603T064417Z`.
- Evidence: CTE sampling returned invalid token id `2147483647`
  (`0x7fffffff`) for the completed prefill row. The Python repair path then
  used returned logits shaped `[1, 1, 62080]`; this is one TP shard of the
  248,320 padded vocabulary, so the fallback argmax became local token `0`.
  The generated text was `"!"` and `real_tokens_passed=false`.
- Root cause: Qwen's custom model forward returned vocab-parallel
  `output_logits` without gathering across TP, unlike the base NxDI model.
  Therefore the invalid sampled-token fallback could not compute a global
  argmax and silently converted the handoff failure into dummy token `0`.
- Fix in local code: gather Qwen `output_logits` with `_gather_along_dim(...,
  partition_dim=2)` when `output_logits=True` and on-device sampling is
  enabled, and make the vLLM repair path reject partial-vocab logits instead
  of using a local-shard argmax.
- Verification: local `py_compile` passed for the changed model and vLLM patch
  files; focused unit suites passed (`test_qwen36_model_aliases.py`: 57 tests,
  `test_vllm_scheduler_patch.py`: 55 tests). The fix was copied to the compile
  and runtime hosts and the rebuilt gathered-logits artifact loaded:

  ```text
  /mnt/trainium_artifacts/qwen_artifacts/qwen36_27b_32k_fp8_cte2048_gpfx2_b256_pfx16k_mlpcte_qkvnki_outprojnki_qknormrope_qkvgate_20260603T065659Z_scan2
  ```

  That artifact returned full-vocab logits (`[1, 1, 248320]`), so the
  vocab-shard fallback bug is fixed. It is still not a correctness artifact.
  The 2,048-token validation prompt produced all-NaN returned logits and dummy
  token `0`:

  ```text
  /home/ubuntu/validation_logs/fp8_256k_decode_nki/coherence_diag_qkvgate_logitsdebug_20260603T074012Z
  raw_output[1] shape=(1, 1, 248320) dtype=torch.float32 finite=0/248320 nan=248320
  logits shape=(1, 1, 248320) dtype=torch.float32 finite=0/248320 nan=248320
  ```

  A no-compile CPU/HF greedy reference gate was then run from saved HF goldens.
  For short prompts the same artifact produced finite logits, but the logits
  were wrong: 0/10 first generated tokens matched HF. Example first case:
  HF expected token `271`; Neuron produced token `96714` (`"围"`), and debug
  showed full-vocab finite logits with argmax `96714`.

  ```text
  /home/ubuntu/validation_logs/fp8_256k_decode_nki/cpu_reference_gate_20260603/qkvgate_gather_vs_hf_10prompts_1tok.json
  overall_matches=0
  overall_positions=10
  overall_match_rate=0.0
  ```

  Interpretation: the sampler and returned-logits gather are no longer the
  primary explanation for short prompts. The compiled graph is producing the
  wrong distribution, while longer validation prompts can still hit the
  previously documented all-NaN CTE/logits path.

## Optimization Timeline

| Step | Change | Result | Keep? | Notes |
|---|---|---:|---|---|
| Runtime chunk fix | Set Hybrid APC prefill chunk to compiled CTE 512 instead of accidental 256 | ~424 tok/s at 16k | Yes | Recovered a real regression from ~279 tok/s, but not enough. |
| KV select before dequant | Select active block KV cache before FP8 dequantization | ~423 tok/s at 16k | Code ok, perf neutral | Disproved full-cache dequant-before-select as the main bottleneck. |
| Reduced 32k perf probe, CTE512 | Smaller 32k artifact, output logits, CTE512 attention CTE | ~1.4k tok/s initially | Probe only | Big speedup versus 400 tok/s lineage, but output coherence still broken. |
| CTE512 attention CTE baseline | 32k reduced artifact with attention CTE and pfx16k | ~2.6k tok/s | Baseline | Became the first useful cold-prefill baseline for kernel work. |
| Block128 solve | DeltaNet solve block size 128, exact path | ~2.1k tok/s | No | Worse than 2.6k baseline. More parallel solve work did not dominate end-to-end. |
| FLA-style / scan experiments | Parallel scan variants for DeltaNet solve, including scan4 | Flat or worse than best baseline | No | CPU/kernel-level promise did not translate end-to-end; GDN was not the only bottleneck. |
| AutoCP CTE | FlashQLA-inspired AutoCP prepass for GDN/CTE | No useful gain | No | AutoCP pattern was not the right isolated win in this traced Neuron graph. |
| Segmented CTE variants | Segment prefix attention into smaller chunks | Not current best | Maybe later | Useful for long context stability, but current 16k reduced probe did better with attention CTE + grouping. |
| GQA-native / skip 1024 prefix | CTE1024 native GQA-style path | Did not beat best | No | Did not move past the 2.6k/3k range. |
| MLP-CTE NKI | Enable quantized MLP CTE NKI path | ~3,069.97 tok/s | Yes | First dense-kernel win after GDN attempts. |
| QKV-NKI + MLP-CTE | Enable QKV NKI with MLP CTE | ~3,222.27 tok/s | Yes | Modest but real gain. |
| Output-proj NKI | Add output projection NKI on top of QKV+MLP | ~3,274.7 tok/s | Yes | Small gain; proved dense constant cost matters. |
| Grouped prefix attention, CTE2048 | Process grouped query chunks / reduce prefix overhead | ~3,318.1 tok/s | Yes | Became best before Q/K norm+RoPE. |
| Q/K RMSNorm + partial RoPE NKI | Fuse Qwen Q/K head RMSNorm plus partial RoPE in NKI | 3,472.04 tok/s | Yes | +4.6% over ~3,318 tok/s. Real but modest. |
| Q/K norm+RoPE context profile | Profile qknormrope context NEFFs by prefix bucket | Prefix increment is only ~20% at 16k | Yes | Prefix attention is not enough by itself; dense floor is ~80% of 16k CTE time. |
| Output-gate projection NKI | Route Qwen attention output gate through NKILib QKV projection kernel | 3,055.79 tok/s | No | Regressed from the 3,472.04 tok/s best. The standalone gate custom call compiled and loaded, but added cost instead of removing dense floor. |
| Packed QKV+gate | Pack full-attention projection as `[Q \| gate \| K \| V]` and split gate from QKV NKI output | 3,620.46 tok/s | Yes | +4.3% over qknormrope. CTE-sized BF16 hardware probe passed against CPU reference; the first compile failed at sharding, then the GQA preshard hook was fixed for `[Q, gate, K, V]` and the retry became current best. |
| Packed QKV+gate context profile | Profile qkvgate context NEFFs by prefix bucket | Prefix increment is ~16.9% at 16k | Yes | Dense floor grew more dominant: 2048:0 was 0.5021s and 2048:16384 was 0.6041s. Compile log still exposed separate `sigmoid`/`mul` before output projection, motivating the next candidate. |
| Gated output-projection fusion | Fuse `attn_output * sigmoid(gate)` into the ROW FP8 output projection NKI kernel | 3,054.58 tok/s | No | Hardware probes passed and full compile/transfer succeeded, but measured 16k cold prefill regressed by ~15.6% from qkvgate. Keep the opt-in code for reference, but do not promote the artifact. |

## Key Lessons

1. The early 400 tok/s path was mostly artifact/runtime configuration, not a
   single bad kernel. Fixing chunk size and moving to the reduced 32k probe
   changed the scale of the problem.

2. GDN/DeltaNet solve optimizations alone did not move end-to-end enough. The
   block128, scan, and AutoCP attempts either regressed or stayed flat because
   dense QKV/MLP/out-projection and prefix attention still consumed large fixed
   cost.

3. Dense kernel work did help. MLP-CTE, QKV-NKI, output-projection NKI, grouped
   prefix, Q/K norm+RoPE, and packed QKV+gate produced the real path from ~2.6k
   to ~3.62k tok/s.

4. CPU/reference improvements are not sufficient evidence. Multiple changes
   looked faster in isolated tests but were flat once compiled into the full
   NxDI/vLLM graph.

5. Not every fusion wins. Folding the attention output gate into output
   projection removed a graph-level sigmoid/multiply boundary, but the custom
   ROW FP8 kernel cost more than the boundary it replaced in the full graph.

6. H100-style gains mostly come from FlashAttention/FlashQLA-style amortization
   and fused dense paths. On TRN2, the closest productive direction so far has
   been reducing full-graph dense/prefix overhead rather than only rewriting the
   triangular solve.

## Errors And Mitigations

| Area | Failure | Root cause | Mitigation |
|---|---|---|---|
| NKI probe | `failed to resolve name 'nki.isa.rsqrt'` | Installed SDK exposes `nisa.activation(..., op=nl.rsqrt)` instead | Switched Q/K norm kernel to `nisa.activation`; 128 and 512 token hardware probes passed. |
| Compile packaging | `safetensors_rust.SafetensorError: No space left on device` | Post-compile checkpoint-bank rewrite creates large `.tmp` shard files | Deleted old artifacts, repaired `tp3` shard in place, verified all shards have 48 recurrent + 48 conv banks. |
| Runtime load | `NRT_INVALID in nrt_init()` | Set `NEURON_RT_VISIBLE_CORES=0-7` on `trn2.3xlarge`, but only cores `0-3` exist | Reran benchmark with `NEURON_RT_VISIBLE_CORES=0-3`; artifact loaded and benchmark completed. |
| Transfer | Local `ssh-agent` socket blocked; older rsync lacked `--info=progress2`; nested SSH consumed heredoc stdin | Local sandbox/socket path and command compatibility issues | Used escalated temporary ssh-agent, rsync `--progress --stats`, and `ssh -n` for nested SSH. |
| Output-gate NKI first compile | `RuntimeError: Shapes are not compatible for broadcasting: f32[1,1,5120] vs. f32[1536,1,128]` in `output_gate_proj` during HLO tracing | The module weight/scale had been preprocessed for NKI layout, but `_should_use_qwen_output_gate_nki()` rejected `q_len == 1`, so token-generation tracing called the standard quantized linear with NKI-layout weights | Removed the `q_len > 1` / SP fallback from the gate-NKI predicate and routed split-QKV TKG through gate NKI when enabled. Local unit/import/py_compile checks passed; TRN2 BF16 q_len=1 gate probe passed before restarting full compile. |
| Packed QKV+gate first probe | `tmp_validate_qwen36_qkvgate_qkv_kernel.py` initially failed with `q_max_abs=5.3016`, `gate_max_abs=4.9095`, `k_max_abs=4.5740`, `v_max_abs=4.5399` on `16.50.153.185` | The probe used unit-variance hidden inputs and compared against a strict FP32 CPU matmul; this was much harsher than the existing QKV/gate validator and made BF16 accumulation error look like layout failure. A deterministic channel-order probe showed the output order was contiguous. | Scaled probe inputs by `1/sqrt(hidden)` to match the accepted gate validator and checked packed output directly against CPU reference. CTE-sized probe then passed with Q/gate/K/V max abs <= 0.1657. |
| Packed QKV+gate first compile | PID `272189` on `16.51.94.87`, log `/home/ubuntu/validation_logs/fp8_256k_decode_nki/qwen36_27b_32k_fp8_cte2048_gpfx2_b256_pfx16k_mlpcte_qkvnki_outprojnki_qknormrope_qkvgate_20260602T223037Z_scan2_compile.log`, artifact `...qknormrope_qkvgate_20260602T223037Z_scan2`. After HLO compile, checkpoint sharding raised `RuntimeError: split_with_sizes expects split_sizes to sum exactly to 14336 (input tensor's size at dimension 0), but got split_sizes=[6144, 1024, 1024]`. | Packed `Wqkv` was correctly laid out as `[Q, gate, K, V]`, but `GroupQueryAttention_QKV.preshard_hook()` still split/repacked fused QKV as `[Q, K, V]` and set old Q-head metadata for the sharder. | Patched the GQA preshard hook to detect `qwen_qkv_gate_packed`, split `[Q, gate, K, V]`, pad/shard gate like Q for GQA strategies, re-cat `[Q, gate, K, V]`, and set doubled Q-head metadata. Local `py_compile`, compile-script `bash -n`, and Qwen config/model-alias/vLLM unit tests passed. Direct `test_gqa.py` collection remains blocked by the repo's existing `lora_serving`/`gqa` circular import in this environment, so the next full compile is the verification for the sharding stage. |
| Local NKILib contract check | `python3 tmp_validate_outproj_nki_contract.py` failed locally with `ModuleNotFoundError: No module named 'nkilib'`. | Local macOS Python environment does not include the Neuron/NKILib packages. | Ran the contract check in the remote Neuron venv on `16.50.153.185`; `outproj_nki_contract: PASS`. |
| Unit-test invocation | `python3 -m unittest contrib/models/Qwen3.6-27B/test/unit/test_qwen36_compile_fp8_config.py` failed with `ModuleNotFoundError: No module named 'contrib.models.Qwen3'`. | `unittest` interpreted the filesystem path containing `Qwen3.6-27B` as a dotted module path. | Reran with discovery: `python3 -m unittest discover -s contrib/models/Qwen3.6-27B/test/unit -p test_qwen36_compile_fp8_config.py`; 45 tests passed. |
| GQA edit cleanup | A patch left a duplicate `_kernel_gated_o_proj` tail after `forward()` in `src/neuronx_distributed_inference/modules/attention/gqa.py`; `sed -n '1190,1425p'` showed a stray indented `return self._kernel_o_proj(...)` and second `forward_gated`/`forward`. | Patch context around the fallback branch was too broad. | Removed the duplicate block; local `py_compile` and focused unit tests passed. |
| Runtime SSH sandbox | Initial remote `py_compile` and contract-check SSH commands to `16.50.153.185` exited 255 with `ssh: connect to host 16.50.153.185 port 22: Operation not permitted`. | Local sandbox blocked outbound SSH. | Reran the same SSH commands with approved escalation; remote checks passed. |
| Runtime file copy | Command `scp contrib/models/Qwen3.6-27B/src/modeling_qwen35.py contrib/models/Qwen3.6-27B/vllm/qwen36_hybrid_apc_scheduler_patch.py ubuntu@16.50.153.185:/home/ubuntu/inferentia-gdn-multihead-cte-20260531T1350Z/` exited 0 but copied both coherence-fix files into the runtime repo root instead of their subdirectories. Context: copying local output-logits gather and vLLM fallback guard changes to runtime host `16.50.153.185`, repo `/home/ubuntu/inferentia-gdn-multihead-cte-20260531T1350Z`. | The `scp` destination was the repo directory and did not preserve repo-relative paths; the failure mode was wrong placement with no nonzero exit. | Re-copied each file to its exact remote path: `.../contrib/models/Qwen3.6-27B/src/modeling_qwen35.py` and `.../contrib/models/Qwen3.6-27B/vllm/qwen36_hybrid_apc_scheduler_patch.py`. Verification: remote `python3 -m py_compile contrib/models/Qwen3.6-27B/src/modeling_qwen35.py contrib/models/Qwen3.6-27B/vllm/qwen36_hybrid_apc_scheduler_patch.py` passed on `16.50.153.185`; misplaced root copies were left as harmless extra files. |
| Gated output-proj hardware probe env | `tmp_validate_qwen36_gated_outproj_nki.py` first failed with `RuntimeError: float8_e4m3fn is not supported in nki. Set UNSAFE_FP8FNCAST...`. | The standalone probe did not set the FP8 unsafe-cast environment used by the model wrapper. | Added `XLA_HANDLE_SPECIAL_SCALAR=1` and `UNSAFE_FP8FNCAST=1`; moved to the next compiler-flag failure. |
| Gated output-proj hardware probe compiler flags | The same probe then failed in neuronx-cc with `[NCC_EVRF051] Data type F8E4M3FN is not supported on TRN1/TRN2... use --experimental-unsafe-fp8e4m3fn-as-fp8e4m3`. | Standalone XLA compile bypassed the full model wrapper's internal HLO tensorizer flags. | Added default `NEURON_CC_FLAGS` with `--internal-hlo2tensorizer-options='--experimental-unsafe-fp8e4m3fn-as-fp8e4m3 --verify-hlo=true'`; base and folded-head probes passed. |
| Compile monitor quoting | A monitor command using `ssh compile "PID=293509; LOG=...; ps -p ${PID} ...; tail -60 ${LOG}"` printed `ps` error `process ID list syntax error`. | Local shell expanded `${PID}` and `${LOG}` inside the double-quoted SSH command before it reached the compile host. | Reran monitors with literal PID/path values. |
| Gated output-proj transfer | Direct compile-host-to-runtime `rsync --append-verify` failed at about 84% with receiver `write failed .../weights/tp3_sharded_checkpoint.safetensors: No space left on device (28)` and sender `Broken pipe (32)`. Runtime `/dev/root` was `484G 484G 0 100%`; partial artifact was 29G. | Runtime artifact volume was full. | With approval, removed one older non-best artifact, freeing 32G. Resumed rsync; exit 0, runtime artifact verified at 34G with 6 files. |
| Runtime discovery | Remote script discovery using `rg --files | rg ...` failed with exit 127 and `bash: line 1: rg: command not found`. | Runtime host shell lacks ripgrep. | Used `find`/`grep` for discovery. |
| Stale discovery session | Attempting `write_stdin` with Ctrl-C for session `38740` returned `stdin is closed for this session`. | The non-TTY exec session did not keep stdin open. | Verified no matching remote grep process remained; continued with benchmark work and checked process state separately. |
| Local process listing sandbox | `ps -axo pid,ppid,stat,etime,command | grep ...` failed locally with exit 1 and `zsh:1: operation not permitted: ps`. | Sandbox restricted the broader `ps` invocation. | Reran a narrower approved process listing with escalation. |
| Benchmark shell quoting | First gated-output-proj benchmark launch failed before Python with exit 2: `syntax error near unexpected token '2'`; evidence showed `--output-json` and log redirect paths were empty. | `${OUT}` and `${LOG}` expanded locally inside a double-quoted SSH command. | Relaunched with single-quoted remote shell text and literal paths. |
| Benchmark interpreter | Corrected benchmark wrapper reached Python but exited 1 with `ModuleNotFoundError: No module named 'transformers'`. | `/usr/bin/python3` on runtime lacks inference/vLLM dependencies. | Used `/opt/aws_neuronx_venv_pytorch_inference_vllm_0_16/bin/python`; moved to PATH failure. |
| Benchmark venv PATH | Running the venv Python directly exited 1 during EngineCore init: `FileNotFoundError: [Errno 2] No such file or directory: 'libneuronpjrt-path'`. | Calling the venv Python did not put the venv helper binaries on `PATH`. | Activated the venv before running the benchmark; `which libneuronpjrt-path` resolved to the venv path, and the benchmark completed. |
| Gathered-logits coherence validation | `validation_scripts/qwen36_hybrid_apc_validation.py --require-real-tokens` against artifact `qwen36_27b_32k_fp8_cte2048_gpfx2_b256_pfx16k_mlpcte_qkvnki_outprojnki_qknormrope_qkvgate_20260603T065659Z_scan2` exited status 2. Log root: `/home/ubuntu/validation_logs/fp8_256k_decode_nki/coherence_diag_qkvgate_gather_20260603T073549Z`. Evidence: `result.json` had `correctness_passed=false`, generated token `[0]`, text `"!"`, `real_tokens_passed=false`; `sample_logits_compare.jsonl` had sampled token `[2147483647]`, logits shape `[1,1,248320]`, argmax `[0]`. | The output-logits gather fixed the prior partial-vocab shape, but the returned logits for this validation prompt were still numerically unusable. | Reran with raw/logit debug and no real-token requirement at `/home/ubuntu/validation_logs/fp8_256k_decode_nki/coherence_diag_qkvgate_logitsdebug_20260603T074012Z`; it confirmed `raw_output[1]`/`logits` were all NaN (`finite=0/248320 nan=248320`). Remaining blocker: compiled graph/logits path, likely the documented CTE numerical path. |
| Runtime artifact capacity | Before transferring the gathered-logits artifact, runtime `/mnt/trainium_artifacts/qwen_artifacts` had about 27G free for a 34G artifact. | The runtime artifact volume retained older probe artifacts. | Removed non-best artifact `qwen36_27b_32k_fp8_cte2048_gpfx2_b256_pfx16k_mlpcte_qkvnki_outprojnki_qknormrope_qkvgate_gatedoutproj_20260603T000018Z_scan2` with approval; free space increased to about 61G. The best qkvgate artifact was retained. |
| Direct EC2-to-EC2 rsync | Direct rsync launched on compile host `16.51.94.87` to runtime `16.50.153.185` failed with exit 255: `Permission denied (publickey)` and `rsync error: unexplained error (code 255)`. | The compile host did not have credentials to SSH to the runtime host. | Verified SSH agent forwarding from local to compile host (`AGENT_FORWARD_OK`), then resumed with `ssh -A ubuntu@16.51.94.87 'rsync -a --partial --append-verify --progress --stats ... ubuntu@16.50.153.185:...'`. Transfer exited 0 with 8 files and total size `36,276,070,642`. |
| Interrupted local transfer | A local `scp -3` fallback was still running after the user requested EC2-to-EC2 rsync. Attempting `write_stdin` Ctrl-C against session `62762` failed with `stdin is closed for this session`. Local process listing showed scp PID `64975` with ssh children `64976` and `65007`; terminating them made the scp session exit code 1. | The transfer command was launched as a non-TTY exec session, so stdin was unavailable for Ctrl-C. | Used approved `ps -axo pid,command` to identify the local scp/ssh process tree and sent `kill -TERM` to those PIDs. Replaced the transfer with agent-forwarded EC2-to-EC2 rsync, which completed and verified the runtime artifact. |
| Accidental long compile | The `nki_chunked` compile command was locally aborted, but the remote process had already started on compile host `16.51.94.87`: PID `311375`, artifact `qwen36_27b_32k_fp8_cte2048_gpfx2_b256_pfx16k_mlpcte_qkvnki_outprojnki_qknormrope_qkvgate_nki_chunked_20260603T074316Z_scan2`, log `/home/ubuntu/validation_logs/fp8_256k_decode_nki/qwen36_27b_32k_fp8_cte2048_gpfx2_b256_pfx16k_mlpcte_qkvnki_outprojnki_qknormrope_qkvgate_nki_chunked_20260603T074316Z_scan2_compile.log`. At inspection it had been running about 2m56s and was generating HLO. | The SSH command reached the remote host before the local tool call was aborted. This conflicted with the follow-up decision to run CPU/reference gates before another long compile. | Sent `kill -TERM 311375` on the compile host and verified `STOPPED`. The partial compile log only shows HLO-generation warnings and no `COMPILE_DONE`; no artifact from this run should be treated as valid. |
| CPU-reference summary command | First attempt to summarize the remote CPU-reference JSON failed locally before SSH ran: `zsh:1: unmatched '`, exit code 1. Command context: nested `ssh ... python3 -c ...` one-liner for `/home/ubuntu/validation_logs/fp8_256k_decode_nki/cpu_reference_gate_20260603/qkvgate_gather_vs_hf_10prompts_1tok.json`. | Bad nested shell quoting in the local command string. | Reran with safer `jq` quoting. Verification output: `overall 0/10 match_rate 0.0`, with per-prompt HF vs Neuron first-token ids printed. No remote state changed during the failed quoting attempt. |
| CPU/HF first-token gate | No-compile saved-HF-golden gate completed with status 0 but failed coherence: `/home/ubuntu/validation_logs/fp8_256k_decode_nki/cpu_reference_gate_20260603/qkvgate_gather_vs_hf_10prompts_1tok.json` reported `overall_matches=0`, `overall_positions=10`, `overall_match_rate=0.0`. Per-prompt examples: idx 0 HF `[271]` vs Neuron `[96714]` (`"围"`), idx 2 HF `[11751]` vs Neuron `[141534]` (`"斗士"`), idx 8 HF `[381]` vs Neuron `[7]` (`"("`). | The saved HF goldens were generated by the repo's CPU/HF reference flow and previously matched a comparable FP8 artifact at 97.5%; this artifact now produces a systematically different first-token distribution. The one-prompt debug run showed finite full-vocab logits with argmax equal to the wrong Neuron token, so this is not the partial-vocab fallback. | Kept the harness patch that propagates `context_encoding_bucket_pairs` and truncates expected HF tokens to `--max-tokens`. Current mitigation is to use this saved-HF gate before further compiles; remaining fix must target the compiled graph/kernel configuration, not only vLLM sampling. |

## Current-Best Profile Snapshot

Profile root:

```text
/home/ubuntu/validation_logs/fp8_256k_decode_nki/qkvgate_context_profile_fast_nodge_20260602T232734Z
```

Context bucket timing:

| Bucket | Total exec time | Cycles | Model FLOPs |
|---|---:|---:|---:|
| 2048:0 | 0.5021s | 602.6M | 42.397T |
| 2048:256 | 0.5042s | 605.1M | 42.449T |
| 2048:512 | 0.5055s | 606.6M | 42.500T |
| 2048:1024 | 0.5083s | 609.9M | 42.604T |
| 2048:2048 | 0.5156s | 618.7M | 42.810T |
| 2048:4096 | 0.5278s | 633.3M | 43.222T |
| 2048:8192 | 0.5526s | 663.1M | 44.047T |
| 2048:16384 | 0.6041s | 724.9M | 45.696T |

At the 16k prefix bucket, the prefix-dependent increment is about 16.9% of
total CTE time. That caps perfect prefix-attention optimization to roughly
1.20x on this shape. The qkvgate profile and the gated-output-projection
regression both point to the same constraint: most remaining time is dense
floor or graph/runtime overhead, and a fusion must remove more than it adds.

## Recommended Next Move

Do not start another full compile without a stronger profile-backed hypothesis.
The packed QKV+gate artifact is the retained best; gated output-projection
fusion is measured non-beneficial on the 16k cold-prefill target.

Expected next useful targets:

1. If dense floor can be decomposed further: rank per-layer/per-op cost inside
   the 2048:0 bucket before another fusion attempt.
2. If prefix attention is revisited: prioritize a design that reuses KV tiles
   across active chunks without increasing dense boundary cost.
3. If GDN is still visible but not dominant: revisit lower precision GDN math or
   better state prepass only with profile evidence.

Current realistic near-term target is 4k-5k tok/s. Reaching H100-like 17k tok/s
will need a larger architectural gap closure: FlashAttention-style attention
amortization, lower precision GDN/state math, tighter scheduler buckets, and
probably correctness cleanup so we can use the same serving path for perf and
real outputs.
