# Trainium Instance Backup - 2026-05-08

Source instance: `ubuntu@16.50.246.134`
Remote repo: `/home/ubuntu/inferentia-gdn`
Local backup root: `remote_instance_backup_20260508/`

## Remote Git State

Current remote branch at backup time: `codex/qwen36-27b-64k-internal`

Remote branch tips observed:

- `chunked-prefill-deltanet-head-batched` -> `a9d2169` Record Qwen3.6 DeltaNet dispatch ablation
- `codex/qwen-hybrid-all-models-backup-20260505-174543` -> `499c563` Backup mixed Qwen hybrid working state
- `codex/qwen35-2b-jim-fixes` -> `330f589` Remove hybrid cache from Qwen3.5 2B follow-up
- `codex/qwen35-2b-jim-fixes-hybrid-backup` -> `ff9a8e1` Add Qwen3.5 2B Jim follow-up fixes
- `codex/qwen35-4b-9b-contrib` -> `c06a09f` Add Qwen3.5 4B and 9B contrib models
- `codex/qwen35-4b-9b-contrib-main` -> `c1795b0` Keep Qwen3.5 contrib on stable dummy KV path
- `codex/qwen35-hybrid-cache-manager-backup` -> `8e1288a` Scope Qwen3.5 hybrid cache to 4B and 9B
- `codex/qwen36-27b-64k-internal` -> `e9118a2` Add Qwen3.6 27B profile decision report
- `codex/qwen36-27b-attn-cache-ablation` -> `f56b7c5` Record Qwen3.6 attention cache ablation results
- `codex/qwen36-27b-jim-fixes` -> `9909515` Stabilize Qwen3.6 27B fused DeltaNet kernel
- `codex/qwen36-27b-jim-fixes-hybrid-backup` -> `27f5b36` Add Qwen3.6 27B Jim follow-up fixes
- `main` -> `d2c3047` A0 env + A1 ref_gdn + invariants (7/7 PASS)

Remote working tree was clean at backup time.

## Backed Up Locally

### Git Patch Stacks

`patches/` contains `git format-patch main..<branch>` exports for the remote branches above.

### Validation Scripts

`validation_scripts/` contains the instance-only runner/server scripts:

- `qwen36_27b_openai_server.py`
- `qwen36_27b_manual_chunk_runner.py`
- `qwen36_27b_long_context_eval.py`
- `qwen_compile_sample.py`
- `profile_chunked_prefill.py`
- `run_monitored_mgs.sh`
- `run_sampled_hbm_mgs.sh`
- `download_qwen36_27b.log`

### Reports And Logs

`reports/` contains the Qwen3.6-27B Markdown reports.

`logs/` contains the key final inference, OpenAI server, long-context, profile, and stress-failure logs.

## Not Backed Up

Large binary/model artifacts were not copied locally:

- `/opt/dlami/nvme/models/Qwen3.6-27B`
- `/opt/dlami/nvme/qwen_artifacts/*`
- `/home/ubuntu/qwen_artifacts/*`

Those are tens to hundreds of GB. The code/scripts/reports needed to rebuild or rerun are backed up, but the compiled 64K artifact itself is still instance-local.
