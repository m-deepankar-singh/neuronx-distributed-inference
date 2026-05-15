# Qwen3.6 Cold-Prefill Goal Audit

Source objective: `docs/goal.md`

Status: not complete. The local patch stack is implemented on
`qwen36-cold-prefill-perf`, but final Trainium evidence is still required.

## Local Deliverables

| Requirement | Evidence | Status |
| --- | --- | --- |
| Separate branch on top of `experimental` | `git branch --show-current` -> `qwen36-cold-prefill-perf`; `git merge-base --is-ancestor experimental HEAD` exited 0 | Done |
| Patch 01 instrumentation | `run_offline_inference.py` emits `COLD_PREFILL_METRICS` and `GENERATION_METRICS`; launcher exports `QWEN36_COLD_PREFILL_CONFIG`; hybrid APC validation emits `COLD_PREFILL_METRICS` | Done locally |
| Patch 02 dynamic CTE buckets | `short`, `general`, `long`, and `262k` profiles in runner and launcher; 128-alignment tests in `test_vllm_serving_config.py` | Done locally |
| Patch 03 text-only CTE hardening | `validate_text_only_cte_vision_inputs`; model assertion; `test_cold_prefill_guards.py` | Done locally |
| Patch 04 compact CTE mask hardening | `prepare_cte_attention_mask`; long dense-mask rejection; small dense fallback test | Done locally |
| Patch 05 fused GDN CTE default | `USE_NKI_FUSED=1` default, `GDN_CTE_KERNEL=fused_initial_state` logging, old chunked and PyTorch toggle variants | Done locally |
| Patch 06 safe cold-zero conv fast path | `safe_cold_zero_conv_fast_path`; tracing disables the fast path; zero-prefix, hybrid-restore, and long chunk-continuation guard tests | Done locally |
| Patch 07 attention tile/block sweep | Benchmark `G_tile_block_sweep`; strict 2K/8K tile-sweep prefill and generation coverage | Done locally |
| Patch 08 benchmark and gates | `qwen36_cold_prefill_benchmark.py` and `qwen36_cold_prefill_acceptance.py`; strict-final audit checklist | Done locally |
| Per-artifact long-context collection | `--compiled-artifacts-by-len LEN=PATH`; rows record selected artifact path | Done locally |
| Launcher config verification | Dry-run tests inspect `COLD_PREFILL_CONFIG`, 2K short profile, 128K candidate, 262K recovery profile, and GDN kernel toggles | Done locally |
| Positive strict-final verifier path | Synthetic complete matrix in `test_cold_prefill_acceptance.py` proves every strict-final audit item can pass when evidence is complete | Done locally |
| Strict-final run orchestration | `qwen36_cold_prefill_strict_final.py` builds fixed-suite, long-artifact, hybrid APC exactness, and strict acceptance commands from explicit artifact paths; benchmark phases explicitly pass `--max-tokens-values 1 32` | Done locally |
| Strict-final Trainium preflight | `qwen36_cold_prefill_strict_final.py --preflight` checks the expected git branch, local model path, non-empty 2K/8K/32K/128K/262K artifact directories, output parent, Python executable, and helper scripts before launching vLLM; it writes `qwen36_cold_prefill_preflight.json` under `--output-dir` with expected output/log paths | Done locally |
| Strict-final phase log capture | Non-dry-run orchestration tees `matrix`, `long_context`, `hybrid_apc`, and `acceptance` stdout/stderr into per-phase `.log` files under `--output-dir` | Done locally |
| Strict-final run manifest | Non-dry-run orchestration writes `qwen36_cold_prefill_run_manifest.json` with phase commands, return codes, elapsed seconds, log paths, expected output paths, per-output/log non-empty evidence status, and final pass status | Done locally |
| Post-run evidence bundle check | `qwen36_cold_prefill_strict_final.py --evidence-check` verifies the preflight JSON passed and describes this output bundle, run manifest passed and contains all phase commands, expected output/log paths match, phase return codes are 0, acceptance JSON passed in strict-final mode, and the acceptance audit checklist has no failed items after the Trainium run; it writes `qwen36_cold_prefill_evidence_check.json` | Done locally |
| Trainium instance runbook | `docs/trainium_strict_final_runbook.md` lists branch checks, path exports, wrapper usage, preflight JSON, dry-run, strict-final collection, evidence-bundle verification, expected JSON/log artifacts, and failure triage | Done locally |
| Trainium wrapper script | `validation_scripts/qwen36_trainium_strict_final.sh` runs preflight, dry-run, strict-final collection, and evidence-check from the exported instance paths; unit coverage verifies it resolves the repo from its own script path and invokes all four phases from another working directory | Done locally |
| GDN state-diff sidecar ingestion | `qwen36_cold_prefill_benchmark.py --gdn-state-diff-json` merges debug sidecar recurrent/conv diffs into benchmark rows when stdout lacks `GDN_STATE_DIFF` | Done locally |
| Small dense-mask fallback exactness | `L_small_dense_mask_fallback` benchmark rows disable chunked prefill and compact masks for 256/512-token prompts; strict-final acceptance compares A-G token IDs to those fallback rows and requires repeated dense-fallback samples | Done locally |
| Hybrid APC partial-prefix exactness gate | `qwen36_hybrid_apc_validation.py exactness --output-json` writes a self-describing report; strict-final orchestration runs it and acceptance requires full-prefix/partial-prefix exactness, cold-prefill metrics, and the strict cold-zero-disabled/chunked/text-only/compact validation config | Done locally |
| Greedy sampling evidence | Benchmark rows record `temperature=0` and `top_k=1`; strict-final acceptance fails rows missing the required greedy sampling config | Done locally |
| Instrumentation completeness gate | Strict-final acceptance requires every row to include the Patch 01 cold-prefill metrics: prompt length, CTE bucket/padding, ctx/tile/block, feature flags, fused flag, latency, tok/s, and HBM usage | Done locally |
| Benchmark row provenance gate | Benchmark rows include schema version, script/runner paths, command, prompt file, CTE args, flags, env toggles, repetition, runtime duration, and chunked-prefill state; strict-final acceptance rejects rows missing that provenance | Done locally |
| Strict launch-shape gate | Benchmark/offline/hybrid validation rows report `tensor_parallel_size=4`, `logical_nc_config=2`, `max_num_seqs=1`, and `ctx_batch_size=1`; strict-final acceptance rejects mismatches | Done locally |
| Artifact load-success evidence gate | Strict-final acceptance requires every compiled-artifact row to report `artifact_load_success=True`, not merely omit a failure | Done locally |
| Artifact path-existence evidence gate | Benchmark rows record `compiled_artifacts_resolved`, `compiled_artifacts_path_exists`, and `compiled_artifacts_path_nonempty`; strict-final acceptance requires positive path-existence and non-empty evidence on every compiled-artifact row | Done locally |
| Runtime returncode evidence gate | Strict-final acceptance requires every row to report `returncode=0`, not merely omit a return code | Done locally |
| Runtime log-tail DMA-spill gate | Strict-final acceptance requires non-empty `output_tail` evidence on every row and rejects tails containing DMA spill failures | Done locally |
| Compiled-artifact consistency gate | Strict-final acceptance requires rows for 2K, 8K, 32K, 128K, and 262K sequence lengths, with every row carrying `compiled_artifacts` and all rows for a given `seq_len` using one artifact path | Done locally |
| Short baseline reproduction gate | Strict-final acceptance checks the documented ~420 tok/s baseline on the 2048-token baseline actual tok/s row and uses bucket-normalized tok/s for fixed short baseline prompts | Done locally |
| P50/P95 latency summary gate | Strict-final acceptance requires p50/p95 prefill latency summaries for each required prefill row and each required G tile-sweep case | Done locally |
| Long-context dense-mask evidence gate | Strict-final acceptance requires 8K+ rows to report `cte_attention_mask_path` / `dense_cte_mask_fallback` evidence and rejects `dense_4d_fallback` | Done locally |

## Verification Run Locally

| Command | Result |
| --- | --- |
| Focused unit suite for serving config, cold-prefill guards, benchmark, acceptance, strict-final orchestration, proxy/server metrics, and hybrid validation metrics | Passed, 155 tests |
| `python3 -m py_compile` on edited Python files | Passed |
| `bash -n contrib/models/Qwen3.6-27B/vllm/start_vllm_server.sh` and `bash -n validation_scripts/qwen36_trainium_strict_final.sh` | Passed |
| `git diff --check` | Passed |
| `python3 -m pytest contrib/models/Qwen3.6-27B/test/unit` | Passed locally: 182 passed, 4 skipped because `neuronx_distributed_inference` is not installed |
| `python3 validation_scripts/qwen36_cold_prefill_strict_final.py ... --preflight` | Passed locally with non-empty placeholder `/tmp` paths; verifies the preflight checker reports Python, model/artifact directories, non-empty status, helper scripts, output parent, and writes preflight JSON |
| `python3 validation_scripts/qwen36_cold_prefill_strict_final.py ... --dry-run` | Passed; generated matrix command includes `--include-dense-fallback`, fixed prompt suite, long-artifact command, hybrid APC exactness command, and strict acceptance command |
| `python3 validation_scripts/qwen36_cold_prefill_strict_final.py --output-dir /tmp/qwen36_evidence_check_cli --evidence-check` | Passed against a synthetic complete evidence bundle; verified preflight pass status/path consistency, run manifest commands/path consistency, output JSONs, phase logs, phase return codes, strict-final acceptance pass status, and no failed acceptance audit items |

## Remaining Completion Gates

These gates require a Trainium environment, compiled artifacts, and real
runtime output. Do not mark the goal complete until all are captured and
accepted by `qwen36_cold_prefill_acceptance.py --strict-final`.

| Gate | Required Evidence |
| --- | --- |
| Baseline reproduction | `A_single512_old_chunked` 2048-token actual tok/s near documented `420 tok/s`, plus fixed short baseline prompts checked with bucket-normalized tok/s |
| Instrumentation completeness | Every strict-final row includes the required cold-prefill metrics payload |
| Benchmark provenance | Every strict-final row includes benchmark schema/provenance fields from `qwen36_cold_prefill_benchmark.py` |
| Launch shape | Every strict-final row records the objective launch shape: TP=4, LNC=2, max-num-seqs=1, ctx-batch-size=1 |
| Artifact load success | Every compiled-artifact row reports `artifact_load_success=True` |
| Artifact path evidence | Every compiled-artifact row reports `compiled_artifacts_path_exists=True` and `compiled_artifacts_path_nonempty=True` from the benchmark collection host |
| Runtime success | Every strict-final row reports `returncode=0` |
| DMA-spill auditability | Every strict-final row includes non-empty runtime `output_tail` evidence and no row reports a DMA spill failure |
| Artifact consistency | Rows exist for required 2K/8K/32K/128K/262K sequence lengths and each sequence length uses one compiled artifact path across variants |
| Greedy sampling | Every strict-final row records `temperature=0` and `top_k=1` |
| Short prompt speedup | `>=1.5x` p50 cold-prefill latency improvement for prompts under 512 tokens |
| Latency summaries | p50/p95 prefill latency summaries are present for every required strict-final prefill target and G tile-sweep case |
| 2K no-regression | Candidate p50 latency not worse than baseline beyond configured tolerance |
| Token exactness | `max_tokens=1` and `max_tokens=32` token IDs match baseline for required prompt suite and long artifact rows; 256/512-token A-G rows also match the dense-mask fallback rows |
| HBM safety | HBM usage present on every strict-final row and no optimized-candidate HBM regression/DMA spill |
| GDN correctness | Recurrent and conv state-diff fields present and within tolerance for candidate rows |
| Hybrid APC exactness | `qwen36_hybrid_apc_validation.py exactness` report with full-prefix and partial-prefix token exactness, `cold_full`/`cold_partial` cold-prefill metrics, and strict cold-zero-disabled/chunked/text-only/compact validation config |
| Long-context load/run | 128K artifact row, 262K block-256 row, and 262K block-128 comparison row load and run |
| Tile sweep | Required 2K/8K tile cases for prefill and `max_tokens=32` generation rows have repeated samples |
| Long-context mask path | 8K+ rows report concrete attention-mask path evidence and do not use dense 4D SxS fallback |
| Feature deltas | A->G deltas for buckets, text-only CTE, compact-mask guardrail, fused GDN, cold-zero-disabled guardrail, and tile/block sweep do not regress in latency, tok/s, token exactness, or HBM when HBM is present; cold-zero conv itself is an optional post-strict-final ablation row |
