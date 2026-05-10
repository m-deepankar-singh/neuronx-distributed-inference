# Qwen3.6-27B FP8 MLP Kernel Experiment

Branch: `codex/qwen36-fp8-gemm-ablation`

Baseline: `qwen36-27b-vllm-apc-baseline-v3`

## Change

Added a compile-time flag to the FP8 harness:

- `--enable-quantized-mlp-kernel`
- maps to `NeuronConfig(quantized_mlp_kernel_enabled=True)`

Everything else stayed aligned with baseline v3:

- Qwen3.6-27B
- 128K max context
- CTE bucket 512
- TP=4
- MLP-only FP8 checkpoint
- hybrid cache and vLLM APC path unchanged

## Artifact

Compiled and loaded successfully on Trn2:

`/opt/dlami/nvme/qwen_artifacts/qwen36_27b_128k_fp8_mlp_kernel_run1`

Artifact config confirms:

`"quantized_mlp_kernel_enabled": true`

Artifact size: `35G`

## Validation

Compile/load:

- HLO generation passed
- NEFF compilation passed
- `LOAD_AFTER_COMPILE_OK`
- vLLM smoke prompt returned `391`

Focused chat quality:

- math: passed
- Olympics factual: passed
- JSON instruction: passed
- Metal Gear Solid summary: passed
- safe SSH key guidance: passed

## Performance

Cold no-prefix-cache chat measurements:

| Target Prompt | Measured Prompt | TTFT | Prefill tok/s | Decode tok/s |
| --- | ---: | ---: | ---: | ---: |
| 512 | 488 | 1.22s | 400.2 | 26.7 |
| 2048 | 1969 | 4.77s | 412.7 | 26.5 |
| 16000 | 15422 | 36.92s | 417.7 | 26.4 |

Baseline v3 is about 413-420 tok/s prefill and 26-27 tok/s decode, so the
quantized MLP kernel flag does not produce a material speedup for this model.

## Conclusion

This is correct but not a useful speedup. The MLP-only FP8 checkpoint already
captures the practical benefit available in this path, or the remaining
runtime bottleneck is outside the MLP GEMM kernels.

The instance was restored to baseline v3 after the experiment.
