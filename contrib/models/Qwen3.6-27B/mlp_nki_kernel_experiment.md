# Qwen3.6-27B MLP NKI Kernel Experiment

Branch: `codex/qwen36-mlp-nki-kernel`

Baseline: `qwen36-27b-vllm-apc-baseline-v3`

## Change

Ported the stock NxDI/Llama-style MLP NKI path into the custom Qwen3.6 dense
MLP:

- `Qwen35MLP` now reads `mlp_kernel_enabled` and
  `quantized_mlp_kernel_enabled`.
- FP8 quantized MLP weights attach `preprocess_quantized_linear_layer` hooks.
- CTE uses `rmsnorm_quant_kernel` plus `nkilib.core.mlp.mlp`.
- TKG uses the smaller MLP kernel path directly to avoid the LNC=2 outer
  dimension failure for sequence length 1.
- Compile harness now exposes:
  - `--enable-mlp-kernel`
  - `--enable-quantized-mlp-kernel`

The initial CTE-only implementation failed on TKG HLO with:

`Outer dimension of size 1 is not big enough to distribute work among 2 programs`

The final implementation adds the TKG split and compiles.

## Artifact

Compiled and loaded successfully on Trn2:

`/opt/dlami/nvme/qwen_artifacts/qwen36_27b_128k_fp8_mlp_nki_kernel_run3`

Artifact config confirms:

- `"mlp_kernel_enabled": true`
- `"quantized_mlp_kernel_enabled": true`

Artifact size: `36G`

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
| 512 | 488 | 1.23s | 398.3 | 27.4 |
| 2048 | 1969 | 4.79s | 411.0 | 27.2 |
| 16000 | 15422 | 37.07s | 416.0 | 27.1 |

Baseline v3 is about 413-420 tok/s prefill and 26-27 tok/s decode. This is
correct, but it is not a material speedup.

## Conclusion

The true MLP NKI/FP8 execution path works for the custom Qwen3.6 model, but MLP
GEMM is not the dominant runtime bottleneck at batch 1, CTE=512. Cold prefill
remains dominated elsewhere, likely GDN and/or attention/orchestration.

The instance was restored to baseline v3 after the experiment.
