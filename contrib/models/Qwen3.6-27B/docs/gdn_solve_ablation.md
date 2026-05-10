# Qwen3.6-27B DeltaNet Solve-Isolation Ablation

Baseline branch: `codex/qwen36-gdn-core-rewrite`

This branch is measurement-only. It intentionally replaces the stable
triangular solve:

```text
N = inv(I - A)
```

with:

```text
N = I
```

inside `nki_deltanet_chunked.py`.

The artifact is expected to produce invalid model-quality outputs. Only TTFT and
prefill throughput are meaningful.

Decision rule:

- If solve-noop approaches the recurrent-core no-op speed (~1238 tok/s at 16K),
  the triangular solve is the recurrent-core wall.
- If solve-noop is only modestly faster than baseline (~418 tok/s at 16K), the
  post-solve state/output interaction path dominates.
- If it lands between those points, both matter and the larger delta determines
  the next rewrite target.
