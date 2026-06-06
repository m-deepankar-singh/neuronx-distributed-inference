#!/usr/bin/env python3
"""Compare Qwen3.6 context-NEFF profile summaries against prefill targets."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


_METRIC_KEYS = [
    "total_time",
    "total_exec_time",
    "total_active_time",
    "tensor_engine_active_time",
    "tensor_engine_active_time_percent",
    "vector_engine_active_time",
    "vector_engine_active_time_percent",
    "dma_active_time",
    "dma_active_time_percent",
    "hbm_read_bytes",
    "hbm_write_bytes",
    "spill_reload_bytes",
    "hardware_flops",
    "transpose_flops",
    "mfu_estimated_percent",
    "mm_arithmetic_intensity",
    "peak_flops_bandwidth_ratio",
]


def _load_json(path: Path) -> Any:
    with path.open() as handle:
        return json.load(handle)


def _as_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(parsed) or math.isinf(parsed):
        return None
    return parsed


def _as_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _has_summary_metrics(value: dict[str, Any]) -> bool:
    return any(key in value for key in _METRIC_KEYS)


def _walk_summary_dicts(value: Any) -> list[dict[str, Any]]:
    found: list[dict[str, Any]] = []
    if isinstance(value, dict):
        if _has_summary_metrics(value):
            found.append(value)
        for key in ("Summary", "summary", "summaries", "data", "rows", "results"):
            if key in value:
                found.extend(_walk_summary_dicts(value[key]))
    elif isinstance(value, list):
        for item in value:
            found.extend(_walk_summary_dicts(item))
    return found


def extract_summary_metrics(payload: Any) -> dict[str, float]:
    summaries = _walk_summary_dicts(payload)
    if not summaries:
        raise ValueError("no neuron-explorer summary metrics found")
    metrics: dict[str, float] = {}
    for key in _METRIC_KEYS:
        value = _as_float(summaries[0].get(key))
        if value is not None:
            metrics[key] = value
    if "total_time" not in metrics:
        raise ValueError("summary metrics are missing required total_time")
    return metrics


def _parse_summary_arg(raw: str) -> tuple[str, Path]:
    if "=" in raw:
        label, path = raw.split("=", 1)
        label = label.strip()
    else:
        path = raw
        label = Path(path).stem
    if not label:
        raise ValueError(f"empty profile label in {raw!r}")
    return label, Path(path).expanduser()


def summarize_profile(
    *,
    label: str,
    path: Path,
    prompt_tokens: int,
    context_tokens: int,
    target_prefill_tok_s: float,
) -> dict[str, Any]:
    metrics = extract_summary_metrics(_load_json(path))
    chunks = math.ceil(prompt_tokens / context_tokens)
    total_time = metrics["total_time"]
    estimated_prefill_seconds = chunks * total_time
    estimated_prefill_tok_s = (
        prompt_tokens / estimated_prefill_seconds
        if estimated_prefill_seconds > 0
        else None
    )
    target_prompt_seconds = prompt_tokens / target_prefill_tok_s
    target_context_time = target_prompt_seconds / chunks
    total_active = metrics.get("total_active_time")
    active_gap = total_time - total_active if total_active is not None else None
    engine_times = {
        "tensor": metrics.get("tensor_engine_active_time"),
        "vector": metrics.get("vector_engine_active_time"),
        "dma": metrics.get("dma_active_time"),
    }
    present_engine_times = {
        key: value for key, value in engine_times.items() if value is not None
    }
    dominant_engine = (
        max(present_engine_times, key=present_engine_times.get)
        if present_engine_times
        else None
    )
    estimated_speedup_needed = (
        target_prefill_tok_s / estimated_prefill_tok_s
        if estimated_prefill_tok_s and estimated_prefill_tok_s > 0
        else None
    )
    return {
        "label": label,
        "path": str(path),
        "metrics": metrics,
        "context_tokens": context_tokens,
        "prompt_tokens": prompt_tokens,
        "chunks_for_prompt": chunks,
        "total_time_seconds": total_time,
        "estimated_prompt_seconds": estimated_prefill_seconds,
        "estimated_prefill_tok_s": estimated_prefill_tok_s,
        "target_prefill_tok_s": target_prefill_tok_s,
        "target_context_time_seconds": target_context_time,
        "estimated_speedup_needed": estimated_speedup_needed,
        "meets_target_by_profile": (
            estimated_prefill_tok_s is not None
            and estimated_prefill_tok_s >= target_prefill_tok_s
        ),
        "total_active_time_seconds": total_active,
        "active_gap_seconds": active_gap,
        "active_fraction": (
            total_active / total_time
            if total_active is not None and total_time > 0
            else None
        ),
        "dominant_active_engine": dominant_engine,
        "hbm_total_bytes": metrics.get("hbm_read_bytes", 0.0)
        + metrics.get("hbm_write_bytes", 0.0),
    }


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _speed_summary_contract_errors(
    payload: dict[str, Any],
    *,
    expected_prompt_tokens: int,
    min_runs: int,
) -> list[str]:
    errors: list[str] = []
    lengths = payload.get("lengths")
    if lengths != [expected_prompt_tokens]:
        errors.append("lengths")
    repeats = _as_int(payload.get("repeats"))
    if repeats is None or repeats < min_runs:
        errors.append("repeats")
    if payload.get("allow_usage_fallback") is not False:
        errors.append("allow_usage_fallback")
    if payload.get("prefill_tokens_all_match_actual") is not True:
        errors.append("prefill_tokens_all_match_actual")
    rows = payload.get("results")
    if not isinstance(rows, list):
        errors.append("results_missing")
        return errors
    if repeats is not None and len(rows) != repeats:
        errors.append("results_count")
    observed_repeats: set[int] = set()
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            errors.append(f"result_{index}_not_object")
            continue
        repeat = _as_int(row.get("repeat"))
        if repeat is not None:
            observed_repeats.add(repeat)
        for field in ("target_prompt_tokens", "actual_prompt_tokens", "prefill_tokens"):
            if _as_int(row.get(field)) != expected_prompt_tokens:
                errors.append(f"result_{index}_{field}")
        usage = row.get("usage")
        usage_tokens = (
            _as_int(usage.get("prompt_tokens")) if isinstance(usage, dict) else None
        )
        if usage_tokens != expected_prompt_tokens:
            errors.append(f"result_{index}_usage_prompt_tokens")
        if row.get("prefill_token_source") != "usage":
            errors.append(f"result_{index}_prefill_token_source")
        if row.get("prefill_tokens_match_actual") is not True:
            errors.append(f"result_{index}_prefill_tokens_match_actual")
        status = _as_int(row.get("status"))
        if status is None or status >= 400:
            errors.append(f"result_{index}_status")
        ttft = _as_float(row.get("ttft_seconds"))
        if ttft is None or ttft <= 0:
            errors.append(f"result_{index}_ttft_seconds")
        speed = _as_float(row.get("prefill_tok_s"))
        if speed is None or speed <= 0:
            errors.append(f"result_{index}_prefill_tok_s")
    if repeats is not None and observed_repeats != set(range(repeats)):
        errors.append("results_repeats")
    return errors


def extract_speed_summary(
    payload: Any,
    *,
    expected_prompt_tokens: int | None = None,
    min_runs: int = 1,
) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise ValueError("speed JSON must be an object")
    if expected_prompt_tokens is not None:
        errors = _speed_summary_contract_errors(
            payload,
            expected_prompt_tokens=expected_prompt_tokens,
            min_runs=min_runs,
        )
        if errors:
            raise ValueError(
                "speed JSON does not match the usage-accounted prompt contract: "
                + ",".join(errors)
            )
    rows = payload.get("results")
    speeds: list[float] = []
    ttfts: list[float] = []
    prompt_tokens: list[float] = []
    if isinstance(rows, list):
        for row in rows:
            if not isinstance(row, dict):
                continue
            speed = _as_float(row.get("prefill_tok_s"))
            ttft = _as_float(row.get("ttft_seconds"))
            tokens = _as_float(row.get("prefill_tokens"))
            if speed is not None:
                speeds.append(speed)
            if ttft is not None:
                ttfts.append(ttft)
            if tokens is not None:
                prompt_tokens.append(tokens)
    top_speed = _as_float(payload.get("prefill_tok_s_mean"))
    return {
        "path_prefill_tok_s_mean": top_speed,
        "row_prefill_tok_s_mean": _mean(speeds),
        "ttft_seconds_mean": _mean(ttfts),
        "prefill_tokens_mean": _mean(prompt_tokens),
        "run_count": len(rows) if isinstance(rows, list) else 0,
        "passed": payload.get("passed"),
    }


def compare_profiles(profiles: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if len(profiles) < 2:
        return []
    baseline = profiles[0]
    baseline_time = baseline["total_time_seconds"]
    comparisons: list[dict[str, Any]] = []
    for candidate in profiles[1:]:
        candidate_time = candidate["total_time_seconds"]
        comparisons.append(
            {
                "baseline": baseline["label"],
                "candidate": candidate["label"],
                "context_time_ratio_candidate_over_baseline": (
                    candidate_time / baseline_time if baseline_time > 0 else None
                ),
                "context_time_speedup_baseline_over_candidate": (
                    baseline_time / candidate_time if candidate_time > 0 else None
                ),
                "estimated_prefill_tok_s_delta": (
                    candidate["estimated_prefill_tok_s"]
                    - baseline["estimated_prefill_tok_s"]
                    if candidate["estimated_prefill_tok_s"] is not None
                    and baseline["estimated_prefill_tok_s"] is not None
                    else None
                ),
            }
        )
    return comparisons


def build_report(
    *,
    summaries: list[str],
    prompt_tokens: int,
    context_tokens: int,
    target_prefill_tok_s: float,
    speed_json: Path | None = None,
) -> dict[str, Any]:
    if prompt_tokens <= 0:
        raise ValueError("--prompt-tokens must be positive")
    if context_tokens <= 0:
        raise ValueError("--context-tokens must be positive")
    if target_prefill_tok_s <= 0:
        raise ValueError("--target-prefill-tok-s must be positive")
    profiles = [
        summarize_profile(
            label=label,
            path=path,
            prompt_tokens=prompt_tokens,
            context_tokens=context_tokens,
            target_prefill_tok_s=target_prefill_tok_s,
        )
        for label, path in (_parse_summary_arg(raw) for raw in summaries)
    ]
    speed_summary = (
        extract_speed_summary(
            _load_json(speed_json),
            expected_prompt_tokens=prompt_tokens,
            min_runs=3,
        )
        if speed_json is not None
        else None
    )
    return {
        "prompt_tokens": prompt_tokens,
        "context_tokens": context_tokens,
        "chunks_for_prompt": math.ceil(prompt_tokens / context_tokens),
        "target_prefill_tok_s": target_prefill_tok_s,
        "target_prompt_seconds": prompt_tokens / target_prefill_tok_s,
        "target_context_time_seconds": (
            prompt_tokens / target_prefill_tok_s
        )
        / math.ceil(prompt_tokens / context_tokens),
        "profiles": profiles,
        "comparisons": compare_profiles(profiles),
        "speed_summary": speed_summary,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--summary",
        action="append",
        required=True,
        help="Profile summary JSON, optionally as label=/path/summary.json",
    )
    parser.add_argument("--prompt-tokens", type=int, default=16384)
    parser.add_argument("--context-tokens", type=int, default=2048)
    parser.add_argument("--target-prefill-tok-s", type=float, default=3000.0)
    parser.add_argument("--speed-json", type=Path, default=None)
    parser.add_argument("--output-json", type=Path, default=None)
    args = parser.parse_args()

    report = build_report(
        summaries=args.summary,
        prompt_tokens=args.prompt_tokens,
        context_tokens=args.context_tokens,
        target_prefill_tok_s=args.target_prefill_tok_s,
        speed_json=args.speed_json.expanduser() if args.speed_json else None,
    )
    encoded = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(encoded)
    print(encoded, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
