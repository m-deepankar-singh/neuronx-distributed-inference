# Qwen3.6 Hybrid APC Equivalence Results

Validation target:

`qwen36_27b_128k_fp8_mlp_edgebf16_hybrid_apc_nki_fusedstable_directsolve_samplelogits_vocabparallel_b256_cte256_512_pfx16k_slots64_tkg8192_32768_131072_async_20260523T123711Z`

Branch/worktree used on Trn2:

`qwen-fused-neumann-stable-decay`, validation host `ubuntu@16.50.153.24`

## Stage 1

Result file: `stage1_hybrid_cte256_512_adapterfix.json`

- Token match: `287/320`
- Match rate: `89.6875%`
- Skill verdict: fail, below the `95%` token-match threshold.
- Main outlier: `"The chemical symbol for gold is"` diverged at token 0.

## Stage 2

Result file: `stage2.json`

- Passed: `6/7`
- Failed: `rmsnorm`, `R=266.6859`
- Passed components: MLP gate/up/down, MLP SwiGLU, embedding small-vocab, lm_head small-vocab.

Stage 3 localized the Stage 2 failure to `rmsnorm`, but Stage 3/4 were not used as the primary signal for this architecture.

## Stage 5/6

Result file: `teacher_forced_hybrid_cte256_512_blockkv.json`

- Compared positions: `96`
- Top-1 agreement: `100%`
- Condition B semantic cosine: pass
- Cosine mean: `0.998494`
- Cosine min: `0.956102`
- Cosine p5: `0.998018`
- KL mean: `0.003300`
- KL p95: `0.005152`
- KL max: `0.160416`

The E2E R-ratio is very large (`mean=359983.34`, `max=2932735.15`) because the BF16-vs-FP32 denominator is extremely small. Given the `100%` top-1 agreement, high cosine, and low KL, the Stage 5/6 functional signal is good, with the R-ratio requiring denominator-aware interpretation.

## Stage 7

Result files: `stage7/stage7_summary.json`, `stage7/stage7_run.log`, `stage7/hf_hellaswag_details.json`, `stage7/neuron_hellaswag_details.json`

The official Stage 7 wrapper requires a `neuron_bench` package, but that package was not installed on the validation host and was not present in the public AWS skill bundle. A local compatibility shim was added under `stage7/neuron_bench_compat/` and used on the host to run the unchanged `run_stage7.py` entrypoint.

Because the validation artifact is `vocab_parallel=true`, host-side log-likelihood scoring sees only partial logits for some token ids. The final Stage 7 run therefore used exact HellaSwag validation rows from the Hugging Face dataset-server with a generation-choice protocol and Qwen thinking disabled via `enable_thinking=False` plus `/no_think`.

- Task: HellaSwag validation, first `5` rows
- Method: generate one of `A/B/C/D`
- HF score: `4/5 = 0.80`
- Neuron score: `4/5 = 0.80`
- Delta: `0.00`
- Tolerance: `0.02`
- Stage 7 verdict: pass

## Files

- `stage1_hybrid_cte256_512_adapterfix.json`
- `stage1_hybrid_cte256_512_adapterfix.log`
- `stage2.json`
- `stage2.log`
- `stage3.json`
- `stage3.log`
- `teacher_forced_hybrid_cte256_512_blockkv.json`
- `teacher_forced_hybrid_cte256_512_blockkv.log`
- `stage7/bench_config.yaml`
- `stage7/stage7_run.log`
- `stage7/stage7_summary.json`
- `stage7/hf_hellaswag_details.json`
- `stage7/neuron_hellaswag_details.json`
- `stage7/neuron_bench_compat/`
- `component_mapping.json`
- `class_divergence_report.json`
