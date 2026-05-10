# Qwen3.6-27B Head-Grouped GDN Experiment

Branch: `codex/qwen36-gdn-headgroup-megakernel`

Baseline: `qwen36-27b-baseline-v3`

## Change

Added an opt-in grouped-head fused DeltaNet CTE path:

- `use_qwen_headgroup_gdn_prefill`
- `qwen_gdn_head_group_size`
- new NKI entry point: `deltanet_fused_chunked_fwd_headgroup`

The implementation is conservative. It moves the local-head loop inside the
NKI custom call but preserves the validated per-head chunk math.

## Validation

Artifact compiled and loaded:

`/opt/dlami/nvme/qwen_artifacts/qwen36_27b_128k_fp8_mlp_headgroup4_gdn_run1`

Compile/load result:

- HLO generation passed
- NEFF compilation passed
- `LOAD_AFTER_COMPILE_OK`
- vLLM smoke prompt returned `391`
- focused chat quality checks passed 5/5

## Performance

Cold no-prefix-cache chat measurements:

| Target Prompt | Measured Prompt | TTFT | Prefill tok/s | Decode tok/s |
| --- | ---: | ---: | ---: | ---: |
| 512 | 487 | 1.21s | 402.3 | 26.8 |
| 2048 | 1969 | 4.73s | 416.1 | 26.5 |
| 16000 | 15422 | 36.62s | 421.2 | 26.8 |

Baseline v3 is about 413-420 tok/s prefill and 26-27 tok/s decode, so this
conservative grouped-head wrapper does not produce a material speedup.

## Conclusion

This path is correct but not useful as an optimization. Neuron appears to
compile the original per-head custom calls into a static graph, so simply
moving the loop from Python/Torch into NKI does not remove a meaningful runtime
bottleneck.

The next GDN optimization must change the actual kernel tiling or fuse more
work, not only wrap the per-head calls:

- true parallel head tiling inside the kernel, or
- fuse surrounding GDN layer work such as projection/conv/gate prep, or
- target another bottleneck such as FP8 GEMM or speculative decoding.

The instance was restored to baseline v3 after the experiment.
