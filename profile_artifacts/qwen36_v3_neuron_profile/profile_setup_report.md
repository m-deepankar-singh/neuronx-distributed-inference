# Qwen3.6-27B v3 Neuron Profile Setup

Date: 2026-05-14
Instance: trn2.48xlarge, 64 NeuronCores visible
Branch: contrib/qwen36-27b-vllm-apc-pr @ 6d6ae62
Profile venv: /opt/dlami/nvme/venvs/qwen36-v3-profile
Env helper: /opt/dlami/nvme/qwen36_profile_env.sh

## Inputs

- HF model: /opt/dlami/nvme/models/Qwen3.6-27B (52G)
- MLP-only FP8 checkpoint: /opt/dlami/nvme/models/Qwen3.6-27B-MLP-FP8 (36G)
- Compiled artifact: /opt/dlami/nvme/qwen_artifacts/qwen36_27b_v3_profile (35G)
- Profile root: /opt/dlami/nvme/profiles/qwen36_v3

## Compile

- seq_len: 131072
- CTE bucket: 512
- TP degree: 4
- logical_nc_config: 2
- compile status: PASS
- load-after-compile: PASS
- important env fix: NEURON_PLATFORM_TARGET_OVERRIDE=trn2
- important path fix: use installed neuronx_distributed_inference core package; only the Qwen3.6 contrib model is imported from the checkout.

## Preserved NEFFs

- /opt/dlami/nvme/profiles/qwen36_v3/neffs/context_encoding_model_graph.neff
- /opt/dlami/nvme/profiles/qwen36_v3/neffs/token_generation_model_graph.neff
- Compiler logs are in the same neffs directory.

## Captures

Token generation profile:
- Directory: /opt/dlami/nvme/profiles/qwen36_v3/token_generation_model
- Capture: 4 ranks, 5 executions
- Rank-0 summary: /opt/dlami/nvme/profiles/qwen36_v3/token_generation_model/summary_rank0.txt
- Rank-0 total_time: 0.029915778256 s
- Rank-0 HBM read bytes: 10,785,258,698
- Rank-0 HBM write bytes: 61,167,932
- Rank-0 DMA active time percent: 89.30%

Context encoding profile:
- Directory: /opt/dlami/nvme/profiles/qwen36_v3/context_encoding_model
- Capture: 4 ranks, 1 execution
- Rank-0 summary: /opt/dlami/nvme/profiles/qwen36_v3/context_encoding_model/summary_rank0.txt
- Rank-0 total_time: 1.448317723667 s
- Rank-0 HBM read bytes: 67,550,784,514
- Rank-0 HBM write bytes: 34,337,804,460
- Rank-0 DMA active time percent: 34.91%
- Rank-0 tensor engine active time percent: 18.23%
- Rank-0 vector engine active time percent: 27.65%
- Rank-0 scalar engine active time percent: 19.15%

## Notes

- `neuron-profile view` multi-rank directory parsing crashed in this tool build. Single-rank export using `-s profile_rank_0.ntff` works.
- InfluxDB is not installed, so UI ingestion is not configured. Use summary export scripts or install InfluxDB if UI browsing is required.
- Raw CTE profiles are large: about 5.1G per rank for one execution.
