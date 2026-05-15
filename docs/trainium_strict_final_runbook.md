# Trainium Strict-Final Runbook

Use this when you get access to the Trainium instance. The goal is to collect
real runtime evidence for the Qwen3.6 cold-prefill patch stack and pass strict
final acceptance.

## 1. Prepare The Checkout

From the repo root:

```bash
git status --short
git branch --show-current
```

Expected branch:

```text
qwen36-cold-prefill-perf
```

If the instance checkout is stale, bring this branch up to date before running
anything. Do not run strict-final from `experimental` directly; this work is a
separate patch stack on top of the hybrid APC work.

The strict-final `--preflight` command below also checks this branch by default.
Override it only if the branch was intentionally renamed:

```bash
--expected-branch qwen36-cold-prefill-perf
```

## 2. Set Paths

Replace these paths with the actual model and compiled artifact directories on
the instance:

```bash
export QWEN36_MODEL=/opt/dlami/nvme/models/Qwen3.6-27B
export QWEN36_ARTIFACT_2K=/opt/dlami/nvme/qwen_artifacts/qwen36_27b_2k
export QWEN36_ARTIFACT_8K=/opt/dlami/nvme/qwen_artifacts/qwen36_27b_8k
export QWEN36_ARTIFACT_32K=/opt/dlami/nvme/qwen_artifacts/qwen36_27b_32k
export QWEN36_ARTIFACT_128K=/opt/dlami/nvme/qwen_artifacts/qwen36_27b_128k
export QWEN36_ARTIFACT_262K=/opt/dlami/nvme/qwen_artifacts/qwen36_27b_262k
export QWEN36_OUT=/tmp/qwen36_cold_prefill_strict
```

The artifact directories must exist and be non-empty. Strict-final acceptance
will reject rows missing `compiled_artifacts_path_exists=True` and
`compiled_artifacts_path_nonempty=True`.

## 3. Run Preflight

Run this first. It should not launch vLLM.

```bash
python3 validation_scripts/qwen36_cold_prefill_strict_final.py \
  --model-path "$QWEN36_MODEL" \
  --artifacts-2k "$QWEN36_ARTIFACT_2K" \
  --artifacts-8k "$QWEN36_ARTIFACT_8K" \
  --artifacts-32k "$QWEN36_ARTIFACT_32K" \
  --artifacts-128k "$QWEN36_ARTIFACT_128K" \
  --artifacts-262k "$QWEN36_ARTIFACT_262K" \
  --output-dir "$QWEN36_OUT" \
  --preflight
```

Every line should start with `OK`. Fix any `MISSING` line before continuing.
This also writes:

```text
$QWEN36_OUT/qwen36_cold_prefill_preflight.json
```

Keep that JSON file with the benchmark artifacts. It records the resolved model
and artifact paths, non-empty checks, helper script checks, runtime shape, and
expected strict-final output and log paths. It also records the expected git
branch and run manifest path.

## 4. Optional One-Command Wrapper

After setting the paths above, the wrapper runs preflight, dry-run,
strict-final collection, and evidence-check in order:

```bash
bash validation_scripts/qwen36_trainium_strict_final.sh
```

The wrapper resolves the repo path from its own location, so it can also be run
with an absolute path from another working directory.

Optional environment overrides:

```bash
export PYTHON=python3
export QWEN36_REPETITIONS=3
export QWEN36_STRICT_MIN_SAMPLES=3
export QWEN36_EXPECTED_BRANCH=qwen36-cold-prefill-perf
export QWEN36_OUTPUT_PREFIX=qwen36_cold_prefill
export QWEN36_FAIL_FAST=1
export QWEN36_GDN_STATE_DIFF_JSON=/path/to/gdn_state_diff.json
export QWEN36_NUM_GPU_BLOCKS_OVERRIDE=32768
```

Use the manual steps below if you want to inspect or rerun individual phases.

## 5. Inspect The Generated Commands

Run the dry-run next:

```bash
python3 validation_scripts/qwen36_cold_prefill_strict_final.py \
  --model-path "$QWEN36_MODEL" \
  --artifacts-2k "$QWEN36_ARTIFACT_2K" \
  --artifacts-8k "$QWEN36_ARTIFACT_8K" \
  --artifacts-32k "$QWEN36_ARTIFACT_32K" \
  --artifacts-128k "$QWEN36_ARTIFACT_128K" \
  --artifacts-262k "$QWEN36_ARTIFACT_262K" \
  --output-dir "$QWEN36_OUT" \
  --dry-run
```

Confirm the generated commands include:

- `--include-dense-fallback`
- `--max-tokens-values 1 32`
- `--tensor-parallel-size 4`
- `--logical-nc-config 2`
- `--max-num-seqs 1`
- `--ctx-batch-size 1`
- 8K and 32K artifact overrides in the matrix command
- 128K and 262K artifact overrides in the long-context command
- hybrid APC exactness command without `--cold-zero-conv-fast-path`
- acceptance command with `--strict-final`

## 6. Run Strict-Final Collection

Run without `--preflight` or `--dry-run`:

```bash
python3 validation_scripts/qwen36_cold_prefill_strict_final.py \
  --model-path "$QWEN36_MODEL" \
  --artifacts-2k "$QWEN36_ARTIFACT_2K" \
  --artifacts-8k "$QWEN36_ARTIFACT_8K" \
  --artifacts-32k "$QWEN36_ARTIFACT_32K" \
  --artifacts-128k "$QWEN36_ARTIFACT_128K" \
  --artifacts-262k "$QWEN36_ARTIFACT_262K" \
  --output-dir "$QWEN36_OUT" \
  --repetitions 3
```

If GDN recurrent/conv state diffs are produced by a debug sidecar instead of
runtime stdout, add:

```bash
--gdn-state-diff-json /path/to/gdn_state_diff.json
```

## 7. Expected Output Artifacts

The run should write:

```text
$QWEN36_OUT/qwen36_cold_prefill_preflight.json
$QWEN36_OUT/qwen36_cold_prefill_run_manifest.json
$QWEN36_OUT/qwen36_cold_prefill_matrix.json
$QWEN36_OUT/qwen36_cold_prefill_long_context.json
$QWEN36_OUT/qwen36_cold_prefill_hybrid_apc_exactness.json
$QWEN36_OUT/qwen36_cold_prefill_strict_acceptance.json
$QWEN36_OUT/qwen36_cold_prefill_matrix.log
$QWEN36_OUT/qwen36_cold_prefill_long_context.log
$QWEN36_OUT/qwen36_cold_prefill_hybrid_apc.log
$QWEN36_OUT/qwen36_cold_prefill_acceptance.log
```

The final acceptance JSON must report `"passed": true`.
The run manifest should contain all four phase return codes and `"passed": true`.
It also includes `evidence_status`, which should show non-empty JSON outputs and
phase logs.

## 8. Verify The Evidence Bundle

After strict-final finishes, run the evidence check. This does not launch vLLM;
it inspects the preflight JSON, run manifest, benchmark outputs, acceptance
JSON, and phase logs already written under `QWEN36_OUT`.

```bash
python3 validation_scripts/qwen36_cold_prefill_strict_final.py \
  --output-dir "$QWEN36_OUT" \
  --evidence-check
```

Every line should be `OK`. This writes:

```text
$QWEN36_OUT/qwen36_cold_prefill_evidence_check.json
```

Keep this file with the rest of the evidence bundle. It is a quick audit that
preflight passed, the preflight and manifest both describe this output
directory, the run manifest includes all four phase commands, all four phases
returned 0, acceptance was run in strict-final mode, the acceptance audit
checklist has no failed items, and every required JSON/log artifact is present
and non-empty.

## 9. Failure Triage

If acceptance fails, inspect `failures` and `audit_checklist` in:

```text
$QWEN36_OUT/qwen36_cold_prefill_strict_acceptance.json
```

Common blockers:

- Baseline `A_single512_old_chunked` does not reproduce the documented ~420 tok/s
  2048-token actual tok/s or the short-prompt bucket-normalized tok/s gate.
- Missing HBM usage in `COLD_PREFILL_METRICS`.
- Missing GDN recurrent/conv state diff fields.
- 128K or 262K artifact row did not load/run.
- 262K block-256 or block-128 row reports DMA spill-ring allocation failure.
- Token IDs differ from baseline for `max_tokens=1` or `max_tokens=32`.
- 8K+ rows report dense 4D SxS mask fallback.
- Artifact rows are missing path existence/non-empty evidence.

Keep the JSON artifacts, phase logs, run manifest, preflight report, and
evidence-check report. They are the evidence needed to complete the goal audit.
