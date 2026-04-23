# inferentia-gdn

Qwen 3.5 hybrid (Gated DeltaNet) on AWS Inferentia via custom NKI kernel + NxD Inference + vLLM.

## Layout
- `ref/`    PyTorch reference GDN forward, validated against FLA
- `nki/`    NKI kernel implementation (Phase A)
- `tests/`  Correctness tests (CPU-sim + device)
- `bench/`  Throughput / latency benchmarks
- `notes/`  Design notes, bug log, progress
- `scripts/` Sync + remote-run helpers

## Dev workflow

This repo is the **source of truth**. The remote trn1/inf2 instance is a runner.

1. Edit locally.
2. `scripts/sync.sh` -- rsync to `ubuntu@<ip>:~/inferentia-gdn/`
3. `scripts/run_remote.sh tests/foo.py` -- activates venv, runs script on device.

Put the current spot-instance IP in `.remote-ip` (gitignored).

## Remote venv (on the trn1)

```
source /opt/aws_neuronx_venv_pytorch_2_9_nxd_inference/bin/activate
```

## Plan

See `~/.claude/plans/atomic-mixing-quail.md` for the full phase breakdown.
