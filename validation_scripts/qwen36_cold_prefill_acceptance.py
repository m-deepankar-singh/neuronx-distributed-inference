#!/usr/bin/env python3
"""Acceptance gates for Qwen3.6 cold-prefill benchmark output."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any


SHORT_PROMPT_LIMIT = 512
MANDATORY_PROMPT_LENGTHS = (128, 256, 384, 512, 1024, 2048, 8192, 32768)
FEATURE_MATRIX_VARIANTS = (
    "A_single512_old_chunked",
    "B_short_buckets_old_chunked",
    "C_short_text_only_old_chunked",
    "D_short_text_compact_old_chunked",
    "E_short_text_compact_fused",
    "F_short_text_compact_fused_cold_zero",
    "G_tile_block_sweep",
)
EXPECTED_GDN_CTE_KERNEL_BY_VARIANT = {
    "A_single512_old_chunked": "nki_chunked",
    "B_short_buckets_old_chunked": "nki_chunked",
    "C_short_text_only_old_chunked": "nki_chunked",
    "D_short_text_compact_old_chunked": "nki_chunked",
    "E_short_text_compact_fused": "fused_initial_state",
    "F_short_text_compact_fused_cold_zero": "fused_initial_state",
    "G_tile_block_sweep": "fused_initial_state",
    "H_128k_candidate": "fused_initial_state",
    "I_262k_recovery_block256": "fused_initial_state",
    "J_262k_recovery_block128": "fused_initial_state",
    "K_short_text_compact_pytorch_chunk": "pytorch_chunk",
}
EXPECTED_FEATURE_LAUNCH_PROFILE_BY_VARIANT = {
    "A_single512_old_chunked": {
        "cte_buckets": (512,),
        "text_only_cte_enabled": False,
        "compact_mask_enabled": False,
        "cold_zero_conv_fast_path_enabled": False,
    },
    "B_short_buckets_old_chunked": {
        "cte_buckets": (128, 256, 512, 1024),
        "text_only_cte_enabled": False,
        "compact_mask_enabled": False,
        "cold_zero_conv_fast_path_enabled": False,
    },
    "C_short_text_only_old_chunked": {
        "cte_buckets": (128, 256, 512, 1024),
        "text_only_cte_enabled": True,
        "compact_mask_enabled": False,
        "cold_zero_conv_fast_path_enabled": False,
    },
    "D_short_text_compact_old_chunked": {
        "cte_buckets": (128, 256, 512, 1024),
        "text_only_cte_enabled": True,
        "compact_mask_enabled": True,
        "cold_zero_conv_fast_path_enabled": False,
    },
    "E_short_text_compact_fused": {
        "cte_buckets": (128, 256, 512, 1024),
        "text_only_cte_enabled": True,
        "compact_mask_enabled": True,
        "cold_zero_conv_fast_path_enabled": False,
    },
    "F_short_text_compact_fused_cold_zero": {
        "cte_buckets": (128, 256, 512, 1024),
        "text_only_cte_enabled": True,
        "compact_mask_enabled": True,
        "cold_zero_conv_fast_path_enabled": True,
    },
    "G_tile_block_sweep": {
        "cte_buckets": (128, 256, 512, 1024),
        "text_only_cte_enabled": True,
        "compact_mask_enabled": True,
        "cold_zero_conv_fast_path_enabled": True,
    },
}
LONG_ARTIFACT_GENERATION_TARGETS = (
    ("H_128k_candidate", 131072),
    ("I_262k_recovery_block256", 262144),
    ("J_262k_recovery_block128", 262144),
)
LONG_ARTIFACT_REQUIRED_TILE_CASES = {
    ("H_128k_candidate", 131072): (128, 1024, 128),
    ("I_262k_recovery_block256", 262144): (128, 1024, 256),
    ("J_262k_recovery_block128", 262144): (128, 1024, 128),
}
LONG_ARTIFACT_REQUIRED_CTE_BUCKETS = {
    ("H_128k_candidate", 131072): (256, 512, 1024, 2048),
    ("I_262k_recovery_block256", 262144): (256,),
    ("J_262k_recovery_block128", 262144): (256,),
}
REQUIRED_TILE_SWEEP_PROMPTS = (2048, 8192)
REQUIRED_TILE_SWEEP_CASES = (
    (128, 512, 128),
    (128, 1024, 128),
    (128, 2048, 128),
    (256, 1024, 128),
    (128, 1024, 256),
)
DENSE_FALLBACK_VARIANT = "L_small_dense_mask_fallback"
DENSE_FALLBACK_PROMPT_LENGTHS = (256, 512)
STRICT_FINAL_REQUIRED_SEQ_LENS = (2048, 8192, 32768, 131072, 262144)
DEFAULT_BASELINE = "A_single512_old_chunked"
DEFAULT_CANDIDATE = "F_short_text_compact_fused_cold_zero"
DEFAULT_GDN_DIFF_TOLERANCE = 1.0e-2
DEFAULT_STRICT_MIN_SAMPLES = 3
GOAL_BASELINE_COLD_TOK_PER_S_TARGET = 420.0
EXPECTED_STRICT_LAUNCH_SHAPE = {
    "tensor_parallel_size": 4,
    "logical_nc_config": 2,
    "max_num_seqs": 1,
    "ctx_batch_size": 1,
}
TILE_CASE_FIELDS = ("kernel_q_tile_size", "kernel_kv_tile_size", "block_size")
REQUIRED_GENERATION_METRICS = (
    "first_token_latency_ms",
    "decode_tok_per_s",
    "end_to_end_generated_tok_per_s",
)
REQUIRED_COLD_PREFILL_METRICS = (
    "actual_prompt_len",
    "selected_cte_bucket",
    "padding_tokens",
    "padding_ratio",
    "ctx_batch_size",
    "block_size",
    "kernel_q_tile_size",
    "kernel_kv_tile_size",
    "text_only_cte_enabled",
    "compact_mask_enabled",
    "cold_zero_conv_fast_path_enabled",
    "use_nki_fused",
    "prefill_latency_ms",
    "actual_tok_per_s",
    "bucket_tok_per_s",
    "hbm_usage",
)
BENCHMARK_SCHEMA_VERSION = 1
REQUIRED_BENCHMARK_ROW_FIELDS = (
    "benchmark_schema_version",
    "benchmark_script",
    "runner",
    "prompt_file",
    "command",
    "cte_args",
    "flags",
    "env",
    "repetition",
    "elapsed_seconds",
    "enable_vllm_chunked_prefill",
    "output_tail",
)


def _load_rows(path: Path) -> list[dict[str, Any]]:
    text = path.read_text().strip()
    if not text:
        return []
    if text.startswith("["):
        data = json.loads(text)
        if not isinstance(data, list):
            raise ValueError(f"{path} must contain a JSON array or JSONL rows")
        return data
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def _load_all_rows(paths: list[Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in paths:
        rows.extend(_load_rows(path))
    return rows


def _load_hybrid_apc_report(path: Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    return json.loads(path.expanduser().read_text(encoding="utf-8"))


def _metric(row: dict[str, Any], name: str) -> float | None:
    metrics = row.get("metrics")
    if not isinstance(metrics, dict):
        return None
    value = metrics.get(name)
    if value is None:
        return None
    return float(value)


def _first_float(row: dict[str, Any], names: tuple[str, ...]) -> float | None:
    containers = [row]
    if isinstance(row.get("metrics"), dict):
        containers.append(row["metrics"])
    if isinstance(row.get("gdn_state_diff"), dict):
        containers.append(row["gdn_state_diff"])
    metrics = row.get("metrics")
    if isinstance(metrics, dict) and isinstance(metrics.get("gdn_state_diff"), dict):
        containers.append(metrics["gdn_state_diff"])

    for container in containers:
        for name in names:
            value = container.get(name)
            if value is not None:
                return float(value)
    return None


def _hbm_used_bytes(row: dict[str, Any]) -> float | None:
    metrics = row.get("metrics")
    if not isinstance(metrics, dict):
        return None
    for key in ("hbm_used_bytes", "hbm_bytes_used", "memory_used_bytes"):
        value = metrics.get(key)
        if value is not None:
            return float(value)

    hbm_usage = metrics.get("hbm_usage")
    if not isinstance(hbm_usage, dict):
        return None
    for key in ("bytes_used", "used_bytes", "hbm_used_bytes"):
        value = hbm_usage.get(key)
        if value is not None:
            return float(value)
    if "kb_total" in hbm_usage and "kb_free" in hbm_usage:
        return float(hbm_usage["kb_total"] - hbm_usage["kb_free"]) * 1024.0
    if "bytes_total" in hbm_usage and "bytes_free" in hbm_usage:
        return float(hbm_usage["bytes_total"] - hbm_usage["bytes_free"])
    return None


def _int_or_none(value) -> int | None:
    if value is None:
        return None
    return int(value)


def _tile_case(row: dict[str, Any]) -> dict[str, int | None]:
    metrics = row.get("metrics") if isinstance(row.get("metrics"), dict) else {}
    return {
        field: _int_or_none(row.get(field, metrics.get(field)))
        for field in TILE_CASE_FIELDS
    }


def _tile_case_key(row: dict[str, Any]) -> tuple[int | None, ...]:
    tile_case = _tile_case(row)
    return tuple(tile_case[field] for field in TILE_CASE_FIELDS)


def _gdn_cte_kernel(row: dict[str, Any]) -> str | None:
    metrics = row.get("metrics") if isinstance(row.get("metrics"), dict) else {}
    value = row.get("gdn_cte_kernel", metrics.get("gdn_cte_kernel"))
    return str(value) if value is not None else None


def _row_bool_or_none(row: dict[str, Any], name: str) -> bool | None:
    metrics = row.get("metrics") if isinstance(row.get("metrics"), dict) else {}
    value = row.get(name, metrics.get(name))
    return bool(value) if value is not None else None


def _row_int_or_none(row: dict[str, Any], name: str) -> int | None:
    metrics = row.get("metrics") if isinstance(row.get("metrics"), dict) else {}
    value = row.get(name, metrics.get(name))
    return int(value) if value is not None else None


def _int_list_or_none(value) -> list[int] | None:
    if not isinstance(value, (list, tuple)):
        return None
    try:
        return [int(item) for item in value]
    except (TypeError, ValueError):
        return None


def _p50(values: list[float]) -> float | None:
    if not values:
        return None
    return float(statistics.median(values))


def _p95(values: list[float]) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = max(0, math.ceil(0.95 * len(ordered)) - 1)
    return float(ordered[index])


def _latency_values(rows: list[dict[str, Any]]) -> list[float]:
    return [
        value
        for row in rows
        if (value := _metric(row, "prefill_latency_ms")) is not None
    ]


def _metric_values(rows: list[dict[str, Any]], name: str) -> list[float]:
    return [value for row in rows if (value := _metric(row, name)) is not None]


def _rows_by_variant_and_prompt(rows: list[dict[str, Any]]):
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row.get("variant"), int(row.get("target_prompt_tokens", -1)))].append(row)
    return grouped


def _comparison_signature(row: dict[str, Any]) -> dict[str, Any]:
    metrics = row.get("metrics") if isinstance(row.get("metrics"), dict) else {}
    return {
        "prompt_sha256": row.get("prompt_sha256"),
        "model_path": row.get("model_path"),
        "compiled_artifacts": row.get("compiled_artifacts"),
        "max_model_len": row.get("max_model_len", metrics.get("max_model_len")),
        "seq_len": row.get("seq_len", metrics.get("seq_len")),
    }


def _row_seq_len(row: dict[str, Any]) -> int | None:
    metrics = row.get("metrics") if isinstance(row.get("metrics"), dict) else {}
    value = row.get("seq_len", metrics.get("seq_len"))
    if value is None:
        return None
    return int(value)


def _require(condition: bool, failures: list[str], message: str):
    if not condition:
        failures.append(message)


def _p50_hbm_for_rows(rows: list[dict[str, Any]]):
    return _p50(
        [value for row in rows if (value := _hbm_used_bytes(row)) is not None]
    )


def _audit_item(name: str, status: str, evidence=None, reason: str | None = None):
    item = {"name": name, "status": status}
    if evidence is not None:
        item["evidence"] = evidence
    if reason:
        item["reason"] = reason
    return item


def _dense_fallback_check_passed(check: dict[str, Any]) -> bool:
    if "exact_with_dense_fallback" in check:
        return check["exact_with_dense_fallback"] is True
    return (
        check.get("present") is True
        and check.get("dense_cte_mask_fallback") is True
    )


def _hybrid_apc_metric_fields_present(metrics: dict[str, Any] | None) -> bool:
    if not isinstance(metrics, dict):
        return False
    required_fields = (
        "actual_prompt_len",
        "prefill_latency_ms",
        "actual_tok_per_s",
        "bucket_tok_per_s",
    )
    return all(metrics.get(field) is not None for field in required_fields)


def _hybrid_apc_config_matches_strict_final(report: dict[str, Any]) -> bool:
    config = report.get("validation_config")
    if not isinstance(config, dict):
        return False
    expected = {
        "enable_vllm_chunked_prefill": True,
        "text_only_cte": True,
        "compact_cte_attention_mask": True,
        "cold_zero_conv_fast_path": True,
        "hybrid_apc_require_vllm_metadata": True,
        "max_num_seqs": 1,
        "block_size": 128,
        "gdn_checkpoint_interval": 128,
        "kernel_q_tile_size": 128,
        "kernel_kv_tile_size": 1024,
    }
    return all(config.get(key) == value for key, value in expected.items())


def _feature_delta_prompt_check(
    *,
    source_variant: str,
    target_variant: str,
    prompt_len: int,
    source_rows: list[dict[str, Any]],
    target_rows: list[dict[str, Any]],
    baseline_tokens,
    target_tile_case: dict[str, int | None] | None = None,
) -> dict[str, Any]:
    source_latency = _p50(_metric_values(source_rows, "prefill_latency_ms"))
    target_latency = _p50(_metric_values(target_rows, "prefill_latency_ms"))
    source_tps = _p50(_metric_values(source_rows, "actual_tok_per_s"))
    target_tps = _p50(_metric_values(target_rows, "actual_tok_per_s"))
    source_hbm = _p50_hbm_for_rows(source_rows)
    target_hbm = _p50_hbm_for_rows(target_rows)
    target_tokens = next(
        (
            row.get("token_ids")
            for row in target_rows
            if row.get("token_ids") is not None
        ),
        None,
    )
    token_exact = (
        None
        if baseline_tokens is None or target_tokens is None
        else target_tokens == baseline_tokens
    )
    check = {
        "prompt_len": prompt_len,
        "source_variant": source_variant,
        "target_variant": target_variant,
        "source_p50_ms": source_latency,
        "target_p50_ms": target_latency,
        "latency_ratio": (
            target_latency / source_latency
            if source_latency and target_latency is not None
            else None
        ),
        "source_actual_tok_per_s": source_tps,
        "target_actual_tok_per_s": target_tps,
        "source_hbm_used_bytes": source_hbm,
        "target_hbm_used_bytes": target_hbm,
        "hbm_delta_bytes": (
            target_hbm - source_hbm
            if source_hbm is not None and target_hbm is not None
            else None
        ),
        "token_exact_with_baseline": token_exact,
    }
    if target_tile_case is not None:
        check["target_tile_case"] = target_tile_case
        check["is_best_tile_case"] = False
    return check


def _feature_delta_check_passed(check: dict[str, Any]) -> bool:
    latency_and_tps_passed = (
        check["target_p50_ms"] is not None
        and check["source_p50_ms"] is not None
        and check["target_p50_ms"] <= check["source_p50_ms"]
        and check["target_actual_tok_per_s"] is not None
        and check["source_actual_tok_per_s"] is not None
        and check["target_actual_tok_per_s"]
        >= check["source_actual_tok_per_s"]
        and check["token_exact_with_baseline"] is not False
    )
    if not latency_and_tps_passed:
        return False
    if (
        check["source_hbm_used_bytes"] is not None
        and check["target_hbm_used_bytes"] is not None
    ):
        return check["target_hbm_used_bytes"] <= check["source_hbm_used_bytes"]
    return True


def evaluate(
    rows: list[dict[str, Any]],
    *,
    baseline_variant: str,
    candidate_variant: str,
    short_latency_speedup: float,
    two_k_regression_tolerance: float,
    bucket_tok_regression_tolerance: float = 0.20,
    baseline_cold_tok_per_s_target: float | None = None,
    baseline_cold_tok_per_s_tolerance: float = 0.15,
    prompt_token_tolerance: float = 0.05,
    require_prompt_token_counts: bool = False,
    prefill_max_tokens: int = 1,
    recurrent_state_tolerance: float = DEFAULT_GDN_DIFF_TOLERANCE,
    conv_state_tolerance: float = DEFAULT_GDN_DIFF_TOLERANCE,
    require_gdn_state_diff: bool = False,
    hbm_regression_tolerance: float | None = None,
    require_hbm_usage: bool = False,
    require_controlled_inputs: bool = True,
    require_feature_deltas: bool = False,
    require_128k: bool = False,
    require_262k: bool = False,
    strict_final: bool = False,
    strict_min_samples: int = DEFAULT_STRICT_MIN_SAMPLES,
    hybrid_apc_report: dict[str, Any] | None = None,
) -> dict[str, Any]:
    failures: list[str] = []
    warnings: list[str] = []
    if strict_final:
        require_prompt_token_counts = True
        require_gdn_state_diff = True
        require_hbm_usage = True
        require_feature_deltas = True
        require_128k = True
        require_262k = True
        if hbm_regression_tolerance is None:
            hbm_regression_tolerance = 0.0
        if baseline_cold_tok_per_s_target is None:
            failures.append(
                "strict final acceptance requires --baseline-cold-tok-per-s-target"
            )
        else:
            lower_goal_target = GOAL_BASELINE_COLD_TOK_PER_S_TARGET * (
                1.0 - baseline_cold_tok_per_s_tolerance
            )
            upper_goal_target = GOAL_BASELINE_COLD_TOK_PER_S_TARGET * (
                1.0 + baseline_cold_tok_per_s_tolerance
            )
            _require(
                lower_goal_target
                <= baseline_cold_tok_per_s_target
                <= upper_goal_target,
                failures,
                (
                    "strict final acceptance baseline target must reproduce "
                    f"the documented ~{GOAL_BASELINE_COLD_TOK_PER_S_TARGET:.0f} tok/s baseline"
                ),
            )

    gate_rows = [
        row
        for row in rows
        if row.get("max_tokens") is None or int(row.get("max_tokens")) == prefill_max_tokens
    ]
    all_grouped = _rows_by_variant_and_prompt(rows)
    grouped = _rows_by_variant_and_prompt(gate_rows)
    controlled_input_rows = rows if strict_final else gate_rows
    controlled_grouped = _rows_by_variant_and_prompt(controlled_input_rows)

    hybrid_apc_exactness_checks = []
    if strict_final:
        if hybrid_apc_report is None:
            failures.append(
                "strict final acceptance requires a hybrid APC exactness report"
            )
            hybrid_apc_exactness_checks.append(
                {
                    "present": False,
                    "full_prefix_exact": None,
                    "partial_prefix_exact": None,
                    "cold_full_metrics_present": False,
                    "cold_partial_metrics_present": False,
                    "strict_config_match": False,
                }
            )
        else:
            cold_metrics = hybrid_apc_report.get("cold_prefill_metrics")
            cold_full_metrics = (
                cold_metrics.get("cold_full")
                if isinstance(cold_metrics, dict)
                else None
            )
            cold_partial_metrics = (
                cold_metrics.get("cold_partial")
                if isinstance(cold_metrics, dict)
                else None
            )
            full_exact = hybrid_apc_report.get("full_prefix_exact") is True
            partial_exact = hybrid_apc_report.get("partial_prefix_exact") is True
            cold_full_metrics_present = _hybrid_apc_metric_fields_present(
                cold_full_metrics
            )
            cold_partial_metrics_present = _hybrid_apc_metric_fields_present(
                cold_partial_metrics
            )
            strict_config_match = _hybrid_apc_config_matches_strict_final(
                hybrid_apc_report
            )
            hybrid_apc_exactness_checks.append(
                {
                    "present": True,
                    "full_prefix_exact": full_exact,
                    "partial_prefix_exact": partial_exact,
                    "cold_full_metrics_present": cold_full_metrics_present,
                    "cold_partial_metrics_present": cold_partial_metrics_present,
                    "strict_config_match": strict_config_match,
                }
            )
            if not full_exact:
                failures.append("hybrid APC full-prefix token exactness failed")
            if not partial_exact:
                failures.append("hybrid APC partial-prefix token exactness failed")
            if not cold_full_metrics_present:
                failures.append(
                    "hybrid APC exactness report missing cold_full COLD_PREFILL_METRICS"
                )
            if not cold_partial_metrics_present:
                failures.append(
                    "hybrid APC exactness report missing cold_partial COLD_PREFILL_METRICS"
                )
            if not strict_config_match:
                failures.append(
                    "hybrid APC exactness report did not use the strict-final cold-zero/chunked validation config"
                )

    sampling_config_checks = []
    if strict_final:
        for row in rows:
            metrics = row.get("metrics") if isinstance(row.get("metrics"), dict) else {}
            temperature = row.get("temperature", metrics.get("temperature"))
            top_k = row.get("top_k", metrics.get("top_k"))
            matches_sampling = temperature == 0 and top_k == 1
            sampling_config_checks.append(
                {
                    "variant": row.get("variant"),
                    "prompt_len": row.get("target_prompt_tokens"),
                    "max_tokens": row.get("max_tokens"),
                    "temperature": temperature,
                    "top_k": top_k,
                    "matches_greedy_sampling": matches_sampling,
                }
            )
            if not matches_sampling:
                failures.append(
                    (
                        f"{row.get('variant')} prompt={row.get('target_prompt_tokens')} "
                        f"max_tokens={row.get('max_tokens')} did not use "
                        "temperature=0 and top_k=1"
                    )
                )

    for row in rows:
        _require(
            row.get("returncode", 0) == 0,
            failures,
            f"{row.get('variant')} prompt={row.get('target_prompt_tokens')} failed with returncode={row.get('returncode')}",
        )
        _require(
            isinstance(row.get("metrics"), dict),
            failures,
            f"{row.get('variant')} prompt={row.get('target_prompt_tokens')} missing COLD_PREFILL_METRICS",
        )
        if row.get("compiled_artifacts") and row.get("artifact_load_success") is False:
            failures.append(
                f"{row.get('variant')} prompt={row.get('target_prompt_tokens')} artifact load/run did not succeed"
            )
        output_text = "\n".join(str(line) for line in row.get("output_tail") or [])
        if "DMA" in output_text and "spill" in output_text.lower():
            failures.append(
                f"{row.get('variant')} prompt={row.get('target_prompt_tokens')} hit DMA spill failure"
            )
    _require(
        bool(gate_rows),
        failures,
        f"no rows found for acceptance prefill max_tokens={prefill_max_tokens}",
    )

    artifact_load_success_checks = []
    artifact_path_checks = []
    runtime_returncode_checks = []
    runtime_output_tail_checks = []
    if strict_final:
        for row in rows:
            returncode = row.get("returncode")
            returncode_ok = returncode == 0
            runtime_returncode_checks.append(
                {
                    "variant": row.get("variant"),
                    "prompt_len": row.get("target_prompt_tokens"),
                    "max_tokens": row.get("max_tokens"),
                    "returncode": returncode,
                    "present": returncode_ok,
                }
            )
            if not returncode_ok:
                failures.append(
                    (
                        f"{row.get('variant')} prompt={row.get('target_prompt_tokens')} "
                        f"max_tokens={row.get('max_tokens')} missing successful "
                        "runtime returncode evidence"
                    )
                )
            output_tail = row.get("output_tail")
            output_tail_present = isinstance(output_tail, list) and bool(output_tail)
            output_text = "\n".join(str(line) for line in output_tail or [])
            dma_spill_detected = "DMA" in output_text and "spill" in output_text.lower()
            runtime_output_tail_checks.append(
                {
                    "variant": row.get("variant"),
                    "prompt_len": row.get("target_prompt_tokens"),
                    "max_tokens": row.get("max_tokens"),
                    "output_tail_present": output_tail_present,
                    "dma_spill_detected": dma_spill_detected,
                    "present": output_tail_present and not dma_spill_detected,
                }
            )
            if not output_tail_present:
                failures.append(
                    (
                        f"{row.get('variant')} prompt={row.get('target_prompt_tokens')} "
                        f"max_tokens={row.get('max_tokens')} missing runtime "
                        "output_tail evidence for DMA-spill audit"
                    )
                )
            if not row.get("compiled_artifacts"):
                continue
            path_exists = row.get("compiled_artifacts_path_exists")
            path_nonempty = row.get("compiled_artifacts_path_nonempty")
            path_present = path_exists is True and path_nonempty is True
            artifact_path_checks.append(
                {
                    "variant": row.get("variant"),
                    "prompt_len": row.get("target_prompt_tokens"),
                    "max_tokens": row.get("max_tokens"),
                    "compiled_artifacts": row.get("compiled_artifacts"),
                    "compiled_artifacts_resolved": row.get(
                        "compiled_artifacts_resolved"
                    ),
                    "compiled_artifacts_path_exists": path_exists,
                    "compiled_artifacts_path_nonempty": path_nonempty,
                    "present": path_present,
                }
            )
            if not path_present:
                failures.append(
                    (
                        f"{row.get('variant')} prompt={row.get('target_prompt_tokens')} "
                        f"max_tokens={row.get('max_tokens')} missing compiled "
                        "artifact path existence/non-empty evidence"
                    )
                )
            artifact_load_success = row.get("artifact_load_success")
            present = artifact_load_success is True
            artifact_load_success_checks.append(
                {
                    "variant": row.get("variant"),
                    "prompt_len": row.get("target_prompt_tokens"),
                    "max_tokens": row.get("max_tokens"),
                    "compiled_artifacts": row.get("compiled_artifacts"),
                    "artifact_load_success": artifact_load_success,
                    "present": present,
                }
            )
            if not present:
                failures.append(
                    (
                        f"{row.get('variant')} prompt={row.get('target_prompt_tokens')} "
                        f"max_tokens={row.get('max_tokens')} missing successful "
                        "artifact load/run evidence"
                    )
                )

    cold_prefill_metric_checks = []
    if strict_final:
        for row in rows:
            metrics = row.get("metrics") if isinstance(row.get("metrics"), dict) else {}
            missing_metrics = [
                name
                for name in REQUIRED_COLD_PREFILL_METRICS
                if metrics.get(name) is None
            ]
            cold_prefill_metric_checks.append(
                {
                    "variant": row.get("variant"),
                    "prompt_len": row.get("target_prompt_tokens"),
                    "max_tokens": row.get("max_tokens"),
                    "missing_metrics": missing_metrics,
                    "present": not missing_metrics,
                }
            )
            if missing_metrics:
                failures.append(
                    (
                        f"{row.get('variant')} prompt={row.get('target_prompt_tokens')} "
                        f"max_tokens={row.get('max_tokens')} missing cold-prefill "
                        f"instrumentation metrics: {missing_metrics}"
                    )
                )

    benchmark_row_provenance_checks = []
    if strict_final:
        for row in rows:
            missing_fields = [
                field
                for field in REQUIRED_BENCHMARK_ROW_FIELDS
                if row.get(field) is None
            ]
            schema_matches = row.get("benchmark_schema_version") == BENCHMARK_SCHEMA_VERSION
            command_present = isinstance(row.get("command"), list) and bool(row["command"])
            cte_args_present = isinstance(row.get("cte_args"), list) and bool(row["cte_args"])
            flags_present = isinstance(row.get("flags"), list)
            env_present = isinstance(row.get("env"), dict)
            prompt_file_present = bool(row.get("prompt_file"))
            elapsed_present = row.get("elapsed_seconds") is not None
            matches_provenance = (
                not missing_fields
                and schema_matches
                and command_present
                and cte_args_present
                and flags_present
                and env_present
                and prompt_file_present
                and elapsed_present
            )
            benchmark_row_provenance_checks.append(
                {
                    "variant": row.get("variant"),
                    "prompt_len": row.get("target_prompt_tokens"),
                    "max_tokens": row.get("max_tokens"),
                    "missing_fields": missing_fields,
                    "schema_matches": schema_matches,
                    "command_present": command_present,
                    "cte_args_present": cte_args_present,
                    "flags_present": flags_present,
                    "env_present": env_present,
                    "prompt_file_present": prompt_file_present,
                    "elapsed_present": elapsed_present,
                    "matches_provenance": matches_provenance,
                }
            )
            if not matches_provenance:
                failures.append(
                    (
                        f"{row.get('variant')} prompt={row.get('target_prompt_tokens')} "
                        f"max_tokens={row.get('max_tokens')} missing benchmark "
                        "row provenance"
                    )
                )

    strict_launch_shape_checks = []
    if strict_final:
        for row in rows:
            actual_shape = {
                name: _row_int_or_none(row, name)
                for name in EXPECTED_STRICT_LAUNCH_SHAPE
            }
            missing_fields = [
                name for name, value in actual_shape.items() if value is None
            ]
            mismatched_fields = [
                name
                for name, value in actual_shape.items()
                if value is not None and value != EXPECTED_STRICT_LAUNCH_SHAPE[name]
            ]
            matches_launch_shape = not missing_fields and not mismatched_fields
            strict_launch_shape_checks.append(
                {
                    "variant": row.get("variant"),
                    "prompt_len": row.get("target_prompt_tokens"),
                    "max_tokens": row.get("max_tokens"),
                    "expected_shape": EXPECTED_STRICT_LAUNCH_SHAPE,
                    "actual_shape": actual_shape,
                    "missing_fields": missing_fields,
                    "mismatched_fields": mismatched_fields,
                    "matches_launch_shape": matches_launch_shape,
                }
            )
            if not matches_launch_shape:
                failures.append(
                    (
                        f"{row.get('variant')} prompt={row.get('target_prompt_tokens')} "
                        f"max_tokens={row.get('max_tokens')} did not use strict-final "
                        f"launch shape {EXPECTED_STRICT_LAUNCH_SHAPE}; "
                        f"missing={missing_fields} mismatched={mismatched_fields}"
                    )
                )

    artifact_consistency_checks = []
    if strict_final:
        for seq_len in STRICT_FINAL_REQUIRED_SEQ_LENS:
            seq_rows = [row for row in rows if _row_seq_len(row) == seq_len]
            artifacts = [
                row.get("compiled_artifacts")
                for row in seq_rows
                if row.get("compiled_artifacts")
            ]
            artifact_set = sorted(set(artifacts))
            present = bool(seq_rows)
            all_rows_have_artifacts = present and len(artifacts) == len(seq_rows)
            single_artifact = len(artifact_set) == 1
            artifact_consistency_checks.append(
                {
                    "seq_len": seq_len,
                    "row_count": len(seq_rows),
                    "artifacts": artifact_set,
                    "present": present,
                    "all_rows_have_artifacts": all_rows_have_artifacts,
                    "single_artifact_for_seq_len": single_artifact,
                }
            )
            if not present:
                failures.append(
                    f"strict final acceptance requires seq_len={seq_len} rows"
                )
            elif not all_rows_have_artifacts:
                failures.append(
                    (
                        f"strict final acceptance requires compiled_artifacts on "
                        f"every seq_len={seq_len} row"
                    )
                )
            elif not single_artifact:
                failures.append(
                    (
                        f"strict final acceptance requires one compiled artifact "
                        f"for seq_len={seq_len}; saw {artifact_set}"
                    )
                )

    prompt_suite_checks = []
    if strict_final:
        for prompt_len in MANDATORY_PROMPT_LENGTHS:
            for variant in (baseline_variant, candidate_variant):
                present = bool(grouped.get((variant, prompt_len), []))
                prompt_suite_checks.append(
                    {
                        "variant": variant,
                        "prompt_len": prompt_len,
                        "present": present,
                    }
                )
                _require(
                    present,
                    failures,
                    f"strict final acceptance requires {variant} prompt={prompt_len} row",
                )

    feature_matrix_row_checks = []
    if strict_final:
        for prompt_len in MANDATORY_PROMPT_LENGTHS:
            for variant in FEATURE_MATRIX_VARIANTS:
                present = bool(grouped.get((variant, prompt_len), []))
                feature_matrix_row_checks.append(
                    {
                        "variant": variant,
                        "prompt_len": prompt_len,
                        "present": present,
                    }
                )
                _require(
                    present,
                    failures,
                    f"strict final acceptance requires {variant} prompt={prompt_len} matrix row",
                )

    generation_row_checks = []
    generation_tile_case_checks = []
    generation_metric_checks = []
    generation_exactness_checks = []
    if strict_final:
        for prompt_len in MANDATORY_PROMPT_LENGTHS:
            baseline_generation_rows = [
                row
                for row in all_grouped.get((baseline_variant, prompt_len), [])
                if row.get("max_tokens") is not None
                and int(row.get("max_tokens")) == 32
            ]
            baseline_generation_tokens = next(
                (
                    row.get("token_ids")
                    for row in baseline_generation_rows
                    if row.get("token_ids") is not None
                ),
                None,
            )
            if baseline_generation_rows and baseline_generation_tokens is None:
                failures.append(
                    (
                        f"{baseline_variant} prompt={prompt_len} max_tokens=32 "
                        "missing baseline token IDs"
                    )
                )
            for variant in FEATURE_MATRIX_VARIANTS:
                generation_rows = [
                    row
                    for row in all_grouped.get((variant, prompt_len), [])
                    if row.get("max_tokens") is not None
                    and int(row.get("max_tokens")) == 32
                ]
                present = bool(generation_rows)
                generation_row_checks.append(
                    {
                        "variant": variant,
                        "prompt_len": prompt_len,
                        "max_tokens": 32,
                        "present": present,
                    }
                )
                _require(
                    present,
                    failures,
                    f"strict final acceptance requires {variant} prompt={prompt_len} max_tokens=32 row",
                )
                for row in generation_rows:
                    metrics = row.get("metrics") if isinstance(row.get("metrics"), dict) else {}
                    missing_metrics = [
                        name for name in REQUIRED_GENERATION_METRICS if metrics.get(name) is None
                    ]
                    generation_metric_checks.append(
                        {
                            "variant": variant,
                            "prompt_len": prompt_len,
                            "max_tokens": row.get("max_tokens"),
                            "missing_metrics": missing_metrics,
                            "present": not missing_metrics,
                        }
                    )
                    if missing_metrics:
                        failures.append(
                            (
                                f"{variant} prompt={prompt_len} max_tokens={row.get('max_tokens')} "
                                f"missing generation metrics: {missing_metrics}"
                            )
                        )
                if variant == baseline_variant:
                    continue
                for row in generation_rows:
                    token_ids = row.get("token_ids")
                    exact = (
                        None
                        if baseline_generation_tokens is None or token_ids is None
                        else token_ids == baseline_generation_tokens
                    )
                    generation_exactness_checks.append(
                        {
                            "variant": variant,
                            "prompt_len": prompt_len,
                            "max_tokens": row.get("max_tokens"),
                            "exact": exact,
                        }
                    )
                    if token_ids is None:
                        failures.append(
                            f"{variant} prompt={prompt_len} max_tokens=32 missing token IDs"
                        )
                    elif baseline_generation_tokens is not None:
                        _require(
                            exact is True,
                            failures,
                            (
                                f"{variant} prompt={prompt_len} max_tokens=32 "
                                "token IDs differ from baseline"
                            ),
                        )

        for variant, prompt_len in LONG_ARTIFACT_GENERATION_TARGETS:
            baseline_generation_rows = [
                row
                for row in all_grouped.get((baseline_variant, prompt_len), [])
                if row.get("max_tokens") is not None
                and int(row.get("max_tokens")) == 32
            ]
            baseline_generation_tokens = next(
                (
                    row.get("token_ids")
                    for row in baseline_generation_rows
                    if row.get("token_ids") is not None
                ),
                None,
            )
            if not baseline_generation_rows:
                failures.append(
                    (
                        "strict final acceptance requires "
                        f"{baseline_variant} prompt={prompt_len} max_tokens=32 baseline row"
                    )
                )
            elif baseline_generation_tokens is None:
                failures.append(
                    (
                        f"{baseline_variant} prompt={prompt_len} max_tokens=32 "
                        "missing baseline token IDs"
                    )
                )

            generation_rows = [
                row
                for row in all_grouped.get((variant, prompt_len), [])
                if row.get("max_tokens") is not None
                and int(row.get("max_tokens")) == 32
            ]
            present = bool(generation_rows)
            generation_row_checks.append(
                {
                    "variant": variant,
                    "prompt_len": prompt_len,
                    "max_tokens": 32,
                    "present": present,
                    "long_artifact": True,
                }
            )
            _require(
                present,
                failures,
                (
                    "strict final acceptance requires "
                    f"{variant} prompt={prompt_len} max_tokens=32 long-artifact row"
                ),
            )
            for row in generation_rows:
                metrics = row.get("metrics") if isinstance(row.get("metrics"), dict) else {}
                missing_metrics = [
                    name for name in REQUIRED_GENERATION_METRICS if metrics.get(name) is None
                ]
                generation_metric_checks.append(
                    {
                        "variant": variant,
                        "prompt_len": prompt_len,
                        "max_tokens": row.get("max_tokens"),
                        "missing_metrics": missing_metrics,
                        "present": not missing_metrics,
                        "long_artifact": True,
                    }
                )
                if missing_metrics:
                    failures.append(
                        (
                            f"{variant} prompt={prompt_len} max_tokens={row.get('max_tokens')} "
                            f"missing generation metrics: {missing_metrics}"
                        )
                    )

                token_ids = row.get("token_ids")
                exact = (
                    None
                    if baseline_generation_tokens is None or token_ids is None
                    else token_ids == baseline_generation_tokens
                )
                generation_exactness_checks.append(
                    {
                        "variant": variant,
                        "prompt_len": prompt_len,
                        "max_tokens": row.get("max_tokens"),
                        "exact": exact,
                        "long_artifact": True,
                    }
                )
                if token_ids is None:
                    failures.append(
                        f"{variant} prompt={prompt_len} max_tokens=32 missing token IDs"
                    )
                elif baseline_generation_tokens is not None:
                    _require(
                        exact is True,
                        failures,
                        (
                            f"{variant} prompt={prompt_len} max_tokens=32 "
                            "token IDs differ from baseline"
                        ),
                    )

        for prompt_len in REQUIRED_TILE_SWEEP_PROMPTS:
            generation_rows = [
                row
                for row in all_grouped.get(("G_tile_block_sweep", prompt_len), [])
                if row.get("max_tokens") is not None
                and int(row.get("max_tokens")) == 32
            ]
            for tile_case_key in REQUIRED_TILE_SWEEP_CASES:
                tile_case = {
                    field: tile_case_key[index]
                    for index, field in enumerate(TILE_CASE_FIELDS)
                }
                matching_rows = [
                    row for row in generation_rows if _tile_case_key(row) == tile_case_key
                ]
                present = bool(matching_rows)
                generation_tile_case_checks.append(
                    {
                        "variant": "G_tile_block_sweep",
                        "prompt_len": prompt_len,
                        "max_tokens": 32,
                        "tile_case": tile_case,
                        "present": present,
                    }
                )
                _require(
                    present,
                    failures,
                    (
                        "strict final acceptance requires G_tile_block_sweep "
                        f"prompt={prompt_len} max_tokens=32 tile_case={tile_case}"
                    ),
                )

    strict_sample_count_checks = []
    required_prefill_latency_targets = set()
    if strict_final:
        long_prompt_lengths = {prompt_len for _variant, prompt_len in LONG_ARTIFACT_GENERATION_TARGETS}
        prefill_targets = {
            (variant, prompt_len)
            for prompt_len in MANDATORY_PROMPT_LENGTHS
            for variant in FEATURE_MATRIX_VARIANTS
        }
        prefill_targets.update(LONG_ARTIFACT_GENERATION_TARGETS)
        prefill_targets.update(
            (baseline_variant, prompt_len) for prompt_len in long_prompt_lengths
        )
        prefill_targets.update(
            (DENSE_FALLBACK_VARIANT, prompt_len)
            for prompt_len in DENSE_FALLBACK_PROMPT_LENGTHS
        )
        required_prefill_latency_targets = set(prefill_targets)
        generation_targets = set(prefill_targets)

        for variant, prompt_len in sorted(prefill_targets):
            sample_count = len(
                [
                    row
                    for row in grouped.get((variant, prompt_len), [])
                    if _metric(row, "prefill_latency_ms") is not None
                ]
            )
            present = sample_count >= strict_min_samples
            strict_sample_count_checks.append(
                {
                    "kind": "prefill",
                    "variant": variant,
                    "prompt_len": prompt_len,
                    "sample_count": sample_count,
                    "required_samples": strict_min_samples,
                    "present": present,
                }
            )
            if not present:
                failures.append(
                    (
                        "strict final acceptance requires at least "
                        f"{strict_min_samples} prefill samples for {variant} "
                        f"prompt={prompt_len}; saw {sample_count}"
                    )
                )

        for prompt_len in REQUIRED_TILE_SWEEP_PROMPTS:
            tile_rows = grouped.get(("G_tile_block_sweep", prompt_len), [])
            for tile_case_key in REQUIRED_TILE_SWEEP_CASES:
                tile_case = {
                    field: tile_case_key[index]
                    for index, field in enumerate(TILE_CASE_FIELDS)
                }
                sample_count = len(
                    [
                        row
                        for row in tile_rows
                        if _tile_case_key(row) == tile_case_key
                        and _metric(row, "prefill_latency_ms") is not None
                    ]
                )
                present = sample_count >= strict_min_samples
                strict_sample_count_checks.append(
                    {
                        "kind": "tile_prefill",
                        "variant": "G_tile_block_sweep",
                        "prompt_len": prompt_len,
                        "tile_case": tile_case,
                        "sample_count": sample_count,
                        "required_samples": strict_min_samples,
                        "present": present,
                    }
                )
                if not present:
                    failures.append(
                        (
                            "strict final acceptance requires at least "
                            f"{strict_min_samples} prefill samples for "
                            f"G_tile_block_sweep prompt={prompt_len} "
                            f"tile_case={tile_case}; saw {sample_count}"
                        )
                    )

        for variant, prompt_len in sorted(generation_targets):
            sample_count = 0
            for row in all_grouped.get((variant, prompt_len), []):
                if row.get("max_tokens") is None or int(row.get("max_tokens")) != 32:
                    continue
                metrics = row.get("metrics") if isinstance(row.get("metrics"), dict) else {}
                if all(metrics.get(name) is not None for name in REQUIRED_GENERATION_METRICS):
                    sample_count += 1
            present = sample_count >= strict_min_samples
            strict_sample_count_checks.append(
                {
                    "kind": "generation",
                    "variant": variant,
                    "prompt_len": prompt_len,
                    "max_tokens": 32,
                    "sample_count": sample_count,
                    "required_samples": strict_min_samples,
                    "present": present,
                }
            )
            if not present:
                failures.append(
                    (
                        "strict final acceptance requires at least "
                        f"{strict_min_samples} max_tokens=32 generation samples "
                        f"for {variant} prompt={prompt_len}; saw {sample_count}"
                    )
                )

        for prompt_len in REQUIRED_TILE_SWEEP_PROMPTS:
            tile_rows = all_grouped.get(("G_tile_block_sweep", prompt_len), [])
            for tile_case_key in REQUIRED_TILE_SWEEP_CASES:
                tile_case = {
                    field: tile_case_key[index]
                    for index, field in enumerate(TILE_CASE_FIELDS)
                }
                sample_count = 0
                for row in tile_rows:
                    if _tile_case_key(row) != tile_case_key:
                        continue
                    if row.get("max_tokens") is None or int(row.get("max_tokens")) != 32:
                        continue
                    metrics = row.get("metrics") if isinstance(row.get("metrics"), dict) else {}
                    if all(metrics.get(name) is not None for name in REQUIRED_GENERATION_METRICS):
                        sample_count += 1
                present = sample_count >= strict_min_samples
                strict_sample_count_checks.append(
                    {
                        "kind": "tile_generation",
                        "variant": "G_tile_block_sweep",
                        "prompt_len": prompt_len,
                        "max_tokens": 32,
                        "tile_case": tile_case,
                        "sample_count": sample_count,
                        "required_samples": strict_min_samples,
                        "present": present,
                    }
                )
                if not present:
                    failures.append(
                        (
                            "strict final acceptance requires at least "
                            f"{strict_min_samples} max_tokens=32 generation samples "
                            f"for G_tile_block_sweep prompt={prompt_len} "
                            f"tile_case={tile_case}; saw {sample_count}"
                        )
                    )

    tile_sweep_case_checks = []
    if strict_final:
        for prompt_len in REQUIRED_TILE_SWEEP_PROMPTS:
            actual_cases = {
                _tile_case_key(row)
                for row in grouped.get(("G_tile_block_sweep", prompt_len), [])
            }
            for tile_case_key in REQUIRED_TILE_SWEEP_CASES:
                present = tile_case_key in actual_cases
                tile_case = {
                    field: tile_case_key[index]
                    for index, field in enumerate(TILE_CASE_FIELDS)
                }
                tile_sweep_case_checks.append(
                    {
                        "prompt_len": prompt_len,
                        "tile_case": tile_case,
                        "present": present,
                    }
                )
                _require(
                    present,
                    failures,
                    (
                        "strict final acceptance requires G_tile_block_sweep "
                        f"prompt={prompt_len} tile_case={tile_case}"
                    ),
                )

    dense_fallback_exactness_checks = []
    if strict_final:
        for prompt_len in DENSE_FALLBACK_PROMPT_LENGTHS:
            for max_tokens in (1, 32):
                dense_rows = [
                    row
                    for row in all_grouped.get((DENSE_FALLBACK_VARIANT, prompt_len), [])
                    if row.get("max_tokens") is not None
                    and int(row.get("max_tokens")) == max_tokens
                ]
                dense_tokens = next(
                    (
                        row.get("token_ids")
                        for row in dense_rows
                        if row.get("token_ids") is not None
                    ),
                    None,
                )
                dense_mask_evidence = [
                    row.get("metrics", {}).get("dense_cte_mask_fallback")
                    for row in dense_rows
                    if isinstance(row.get("metrics"), dict)
                ]
                dense_present = bool(dense_rows) and dense_tokens is not None
                dense_path_present = bool(dense_mask_evidence) and all(
                    value is True for value in dense_mask_evidence
                )
                dense_fallback_exactness_checks.append(
                    {
                        "variant": DENSE_FALLBACK_VARIANT,
                        "prompt_len": prompt_len,
                        "max_tokens": max_tokens,
                        "present": dense_present,
                        "dense_cte_mask_fallback": dense_path_present,
                    }
                )
                if not dense_rows:
                    failures.append(
                        (
                            "strict final acceptance requires "
                            f"{DENSE_FALLBACK_VARIANT} prompt={prompt_len} "
                            f"max_tokens={max_tokens} row"
                        )
                    )
                    continue
                if dense_tokens is None:
                    failures.append(
                        (
                            f"{DENSE_FALLBACK_VARIANT} prompt={prompt_len} "
                            f"max_tokens={max_tokens} missing token IDs"
                        )
                    )
                    continue
                if not dense_path_present:
                    failures.append(
                        (
                            f"{DENSE_FALLBACK_VARIANT} prompt={prompt_len} "
                            f"max_tokens={max_tokens} did not report dense CTE "
                            "mask fallback"
                        )
                    )

                for variant in FEATURE_MATRIX_VARIANTS:
                    variant_rows = [
                        row
                        for row in all_grouped.get((variant, prompt_len), [])
                        if row.get("max_tokens") is not None
                        and int(row.get("max_tokens")) == max_tokens
                    ]
                    variant_tokens = next(
                        (
                            row.get("token_ids")
                            for row in variant_rows
                            if row.get("token_ids") is not None
                        ),
                        None,
                    )
                    exact = (
                        None
                        if dense_tokens is None or variant_tokens is None
                        else variant_tokens == dense_tokens
                    )
                    dense_fallback_exactness_checks.append(
                        {
                            "variant": variant,
                            "prompt_len": prompt_len,
                            "max_tokens": max_tokens,
                            "exact_with_dense_fallback": exact,
                        }
                    )
                    if variant_tokens is None:
                        failures.append(
                            (
                                f"{variant} prompt={prompt_len} "
                                f"max_tokens={max_tokens} missing token IDs for "
                                "dense fallback exactness"
                            )
                        )
                    elif dense_tokens is not None:
                        _require(
                            exact is True,
                            failures,
                            (
                                f"{variant} prompt={prompt_len} "
                                f"max_tokens={max_tokens} token IDs differ from "
                                "dense fallback"
                            ),
                        )

    controlled_input_checks = []
    for prompt_len in sorted(
        {prompt for _variant, prompt in controlled_grouped if prompt > 0}
    ):
        baseline_rows = controlled_grouped.get((baseline_variant, prompt_len), [])
        if not baseline_rows:
            continue
        baseline_signature = _comparison_signature(baseline_rows[0])
        missing_fields = [
            name for name, value in baseline_signature.items() if value is None
        ]
        if missing_fields and require_controlled_inputs:
            failures.append(
                f"{baseline_variant} prompt={prompt_len} missing controlled input fields: {missing_fields}"
            )
        for variant in sorted(
            {variant for variant, prompt in controlled_grouped if prompt == prompt_len}
        ):
            for row in controlled_grouped[(variant, prompt_len)]:
                signature = _comparison_signature(row)
                check = {
                    "variant": variant,
                    "prompt_len": prompt_len,
                    "max_tokens": row.get("max_tokens"),
                    "matches_baseline": signature == baseline_signature,
                    "baseline_signature": baseline_signature,
                    "signature": signature,
                }
                controlled_input_checks.append(check)
                if require_controlled_inputs and signature != baseline_signature:
                    failures.append(
                        (
                            f"{variant} prompt={prompt_len} max_tokens={row.get('max_tokens')} "
                            "did not use the same prompt/model/artifact/length as baseline"
                        )
                    )

    prompt_token_checks = []
    for row in gate_rows:
        target = row.get("target_prompt_tokens")
        if target is None:
            continue
        target = int(target)
        actual = row.get("prompt_token_count")
        metrics = row.get("metrics")
        if actual is None and isinstance(metrics, dict):
            actual = metrics.get("actual_prompt_len")
        if actual is None:
            if require_prompt_token_counts:
                failures.append(
                    f"{row.get('variant')} prompt={target} missing actual prompt token count"
                )
            continue
        actual = int(actual)
        error_ratio = abs(actual - target) / max(target, 1)
        check = {
            "variant": row.get("variant"),
            "target_prompt_tokens": target,
            "actual_prompt_tokens": actual,
            "error_ratio": error_ratio,
            "within_tolerance": error_ratio <= prompt_token_tolerance,
        }
        prompt_token_checks.append(check)
        if error_ratio > prompt_token_tolerance:
            failures.append(
                f"{row.get('variant')} target prompt={target} actual prompt={actual} exceeds token tolerance"
            )

    short_results = []
    latency_summaries = []
    for (variant, prompt_len), variant_rows in sorted(grouped.items()):
        if variant is None or prompt_len <= 0:
            continue
        latencies = _latency_values(variant_rows)
        if not latencies:
            continue
        latency_summaries.append(
            {
                "variant": variant,
                "prompt_len": prompt_len,
                "sample_count": len(latencies),
                "p50_ms": _p50(latencies),
                "p95_ms": _p95(latencies),
            }
        )

    prefill_latency_summary_checks = []
    tile_prefill_latency_summary_checks = []
    latency_summary_by_target = {
        (summary["variant"], summary["prompt_len"]): summary
        for summary in latency_summaries
    }
    if strict_final:
        for variant, prompt_len in sorted(required_prefill_latency_targets):
            summary = latency_summary_by_target.get((variant, prompt_len))
            sample_count = summary["sample_count"] if summary is not None else 0
            p50_ms = summary["p50_ms"] if summary is not None else None
            p95_ms = summary["p95_ms"] if summary is not None else None
            present = (
                sample_count >= strict_min_samples
                and p50_ms is not None
                and p95_ms is not None
            )
            prefill_latency_summary_checks.append(
                {
                    "variant": variant,
                    "prompt_len": prompt_len,
                    "sample_count": sample_count,
                    "required_samples": strict_min_samples,
                    "p50_ms": p50_ms,
                    "p95_ms": p95_ms,
                    "present": present,
                }
            )
            if not present:
                failures.append(
                    (
                        "strict final acceptance requires p50/p95 prefill "
                        f"latency summary for {variant} prompt={prompt_len}; "
                        f"saw {sample_count} samples"
                    )
                )

        for prompt_len in REQUIRED_TILE_SWEEP_PROMPTS:
            tile_rows = grouped.get(("G_tile_block_sweep", prompt_len), [])
            for tile_case_key in REQUIRED_TILE_SWEEP_CASES:
                tile_case = {
                    field: tile_case_key[index]
                    for index, field in enumerate(TILE_CASE_FIELDS)
                }
                latencies = [
                    value
                    for row in tile_rows
                    if _tile_case_key(row) == tile_case_key
                    and (value := _metric(row, "prefill_latency_ms")) is not None
                ]
                sample_count = len(latencies)
                p50_ms = _p50(latencies)
                p95_ms = _p95(latencies)
                present = (
                    sample_count >= strict_min_samples
                    and p50_ms is not None
                    and p95_ms is not None
                )
                tile_prefill_latency_summary_checks.append(
                    {
                        "variant": "G_tile_block_sweep",
                        "prompt_len": prompt_len,
                        "tile_case": tile_case,
                        "sample_count": sample_count,
                        "required_samples": strict_min_samples,
                        "p50_ms": p50_ms,
                        "p95_ms": p95_ms,
                        "present": present,
                    }
                )
                if not present:
                    failures.append(
                        (
                            "strict final acceptance requires p50/p95 prefill "
                            "latency summary for G_tile_block_sweep "
                            f"prompt={prompt_len} tile_case={tile_case}; "
                            f"saw {sample_count} samples"
                        )
                    )

    generation_summaries = []
    for (variant, prompt_len), variant_rows in sorted(all_grouped.items()):
        if variant is None or prompt_len <= 0:
            continue
        first_token_latencies = [
            value
            for row in variant_rows
            if (value := _metric(row, "first_token_latency_ms")) is not None
        ]
        decode_rates = [
            value
            for row in variant_rows
            if (value := _metric(row, "decode_tok_per_s")) is not None
        ]
        end_to_end_rates = [
            value
            for row in variant_rows
            if (value := _metric(row, "end_to_end_generated_tok_per_s")) is not None
        ]
        if first_token_latencies or decode_rates or end_to_end_rates:
            generation_summaries.append(
                {
                    "variant": variant,
                    "prompt_len": prompt_len,
                    "first_token_sample_count": len(first_token_latencies),
                    "first_token_p50_ms": _p50(first_token_latencies),
                    "first_token_p95_ms": _p95(first_token_latencies),
                    "decode_tok_per_s_sample_count": len(decode_rates),
                    "decode_tok_per_s_p50": _p50(decode_rates),
                    "decode_tok_per_s_p95": _p95(decode_rates),
                    "end_to_end_generated_tok_per_s_sample_count": len(end_to_end_rates),
                    "end_to_end_generated_tok_per_s_p50": _p50(end_to_end_rates),
                    "end_to_end_generated_tok_per_s_p95": _p95(end_to_end_rates),
                }
            )

    for prompt_len in sorted(
        {
            prompt_len
            for variant, prompt_len in grouped
            if prompt_len > 0 and prompt_len <= SHORT_PROMPT_LIMIT
        }
    ):
        baseline_latencies = _latency_values(
            grouped.get((baseline_variant, prompt_len), [])
        )
        candidate_latencies = _latency_values(
            grouped.get((candidate_variant, prompt_len), [])
        )
        baseline_p50 = _p50(baseline_latencies)
        baseline_p95 = _p95(baseline_latencies)
        candidate_p50 = _p50(candidate_latencies)
        candidate_p95 = _p95(candidate_latencies)
        if baseline_p50 is None or candidate_p50 is None:
            failures.append(
                f"missing baseline/candidate latency for short prompt={prompt_len}"
            )
            continue
        speedup = baseline_p50 / candidate_p50 if candidate_p50 else 0.0
        short_results.append(
            {
                "prompt_len": prompt_len,
                "baseline_p50_ms": baseline_p50,
                "baseline_p95_ms": baseline_p95,
                "candidate_p50_ms": candidate_p50,
                "candidate_p95_ms": candidate_p95,
                "speedup": speedup,
            }
        )
        _require(
            speedup >= short_latency_speedup,
            failures,
            (
                f"short prompt={prompt_len} speedup {speedup:.3f}x is below "
                f"{short_latency_speedup:.3f}x"
            ),
        )

        baseline_tps = _p50(
            [
                value
                for row in grouped.get((baseline_variant, prompt_len), [])
                if (value := _metric(row, "actual_tok_per_s")) is not None
            ]
        )
        candidate_tps = _p50(
            [
                value
                for row in grouped.get((candidate_variant, prompt_len), [])
                if (value := _metric(row, "actual_tok_per_s")) is not None
            ]
        )
        _require(
            baseline_tps is not None and candidate_tps is not None and candidate_tps > baseline_tps,
            failures,
            f"short prompt={prompt_len} actual tok/s did not improve",
        )
        short_results[-1]["baseline_actual_tok_per_s"] = baseline_tps
        short_results[-1]["candidate_actual_tok_per_s"] = candidate_tps
        baseline_bucket_tps = _p50(
            [
                value
                for row in grouped.get((baseline_variant, prompt_len), [])
                if (value := _metric(row, "bucket_tok_per_s")) is not None
            ]
        )
        candidate_bucket_tps = _p50(
            [
                value
                for row in grouped.get((candidate_variant, prompt_len), [])
                if (value := _metric(row, "bucket_tok_per_s")) is not None
            ]
        )
        short_results[-1]["baseline_bucket_tok_per_s"] = baseline_bucket_tps
        short_results[-1]["candidate_bucket_tok_per_s"] = candidate_bucket_tps
        if baseline_bucket_tps is None or candidate_bucket_tps is None:
            failures.append(f"short prompt={prompt_len} bucket tok/s missing")
        else:
            min_bucket_tps = baseline_bucket_tps * (1.0 - bucket_tok_regression_tolerance)
            _require(
                candidate_bucket_tps >= min_bucket_tps,
                failures,
                (
                    f"short prompt={prompt_len} bucket tok/s regressed beyond "
                    f"{bucket_tok_regression_tolerance:.3f} tolerance"
                ),
            )
    _require(
        bool(short_results),
        failures,
        f"no short-prompt rows found for <= {SHORT_PROMPT_LIMIT} tokens",
    )

    baseline_actual_tps_values = [
        value
        for (variant, _prompt), variant_rows in grouped.items()
        if variant == baseline_variant
        for value in _metric_values(variant_rows, "actual_tok_per_s")
    ]
    baseline_cold_tok_per_s = _p50(baseline_actual_tps_values)
    baseline_cold_gate = None
    if baseline_cold_tok_per_s_target is not None:
        lower = baseline_cold_tok_per_s_target * (
            1.0 - baseline_cold_tok_per_s_tolerance
        )
        upper = baseline_cold_tok_per_s_target * (
            1.0 + baseline_cold_tok_per_s_tolerance
        )
        baseline_cold_gate = {
            "target_tok_per_s": baseline_cold_tok_per_s_target,
            "tolerance": baseline_cold_tok_per_s_tolerance,
            "lower_tok_per_s": lower,
            "upper_tok_per_s": upper,
            "baseline_p50_actual_tok_per_s": baseline_cold_tok_per_s,
        }
        _require(
            baseline_cold_tok_per_s is not None
            and lower <= baseline_cold_tok_per_s <= upper,
            failures,
            "baseline cold tok/s did not reproduce the expected target range",
        )

    baseline_cold_prompt_checks = []
    if strict_final and baseline_cold_tok_per_s_target is not None:
        lower = baseline_cold_tok_per_s_target * (
            1.0 - baseline_cold_tok_per_s_tolerance
        )
        upper = baseline_cold_tok_per_s_target * (
            1.0 + baseline_cold_tok_per_s_tolerance
        )
        for prompt_len in MANDATORY_PROMPT_LENGTHS:
            if prompt_len > SHORT_PROMPT_LIMIT:
                continue
            values = _metric_values(
                grouped.get((baseline_variant, prompt_len), []),
                "actual_tok_per_s",
            )
            prompt_p50 = _p50(values)
            within_range = prompt_p50 is not None and lower <= prompt_p50 <= upper
            baseline_cold_prompt_checks.append(
                {
                    "variant": baseline_variant,
                    "prompt_len": prompt_len,
                    "sample_count": len(values),
                    "p50_actual_tok_per_s": prompt_p50,
                    "lower_tok_per_s": lower,
                    "upper_tok_per_s": upper,
                    "within_target_range": within_range,
                }
            )
            if not within_range:
                failures.append(
                    (
                        f"{baseline_variant} prompt={prompt_len} baseline cold "
                        "tok/s did not reproduce the expected target range"
                    )
                )

    baseline_2k_p50 = _p50(_latency_values(grouped.get((baseline_variant, 2048), [])))
    candidate_2k_p50 = None
    if baseline_2k_p50 is None:
        warnings.append("2K no-regression gate skipped; baseline 2048-token row missing")
        if strict_final:
            failures.append("strict final acceptance requires a 2048-token baseline row")
    else:
        candidate_2k_p50 = _p50(
            _latency_values(grouped.get((candidate_variant, 2048), []))
        )
        if candidate_2k_p50 is None:
            failures.append("missing 2K candidate latency")
        else:
            _require(
                candidate_2k_p50
                <= baseline_2k_p50 * (1.0 + two_k_regression_tolerance),
                failures,
                "2K candidate prefill latency regressed beyond tolerance",
            )

    exactness_checks = []
    for prompt_len in sorted({prompt for _variant, prompt in grouped if prompt > 0}):
        baseline_rows = grouped.get((baseline_variant, prompt_len), [])
        baseline_tokens = next(
            (row.get("token_ids") for row in baseline_rows if row.get("token_ids") is not None),
            None,
        )
        if baseline_tokens is None:
            if strict_final and not baseline_rows:
                failures.append(
                    (
                        "strict final token exactness requires a baseline row "
                        f"for prompt={prompt_len}"
                    )
                )
            elif baseline_rows or prompt_len <= 2048:
                failures.append(
                    f"token exactness failed for prompt={prompt_len}; baseline tokens missing"
                )
            else:
                warnings.append(
                    f"token exactness skipped for prompt={prompt_len}; no baseline row"
                )
            continue
        for variant in sorted({variant for variant, prompt in grouped if prompt == prompt_len}):
            if variant == baseline_variant:
                continue
            for row in grouped[(variant, prompt_len)]:
                token_ids = row.get("token_ids")
                if token_ids is None:
                    failures.append(f"{variant} prompt={prompt_len} missing token IDs")
                    continue
                exact = token_ids == baseline_tokens
                exactness_checks.append(
                    {"variant": variant, "prompt_len": prompt_len, "exact": exact}
                )
                _require(
                    exact,
                    failures,
                    f"{variant} prompt={prompt_len} token IDs differ from baseline",
                )

    long_context_checks = []
    for prompt_len in (8192, 32768, 131072, 262144):
        matching = [
            row
            for row in gate_rows
            if int(row.get("target_prompt_tokens", -1)) == prompt_len
        ]
        if not matching:
            continue
        for row in matching:
            metrics = row.get("metrics") or {}
            attention_mask_path = metrics.get("cte_attention_mask_path")
            dense_mask_fallback = metrics.get("dense_cte_mask_fallback")
            attention_mask_path_observed = (
                attention_mask_path is not None or dense_mask_fallback is not None
            )
            avoids_dense_sxs_mask = (
                dense_mask_fallback is not True
                and attention_mask_path != "dense_4d_fallback"
            )
            long_context_checks.append(
                {
                    "variant": row.get("variant"),
                    "prompt_len": prompt_len,
                    "hbm_usage_present": metrics.get("hbm_usage") is not None,
                    "num_cte_chunks": metrics.get("num_cte_chunks"),
                    "compact_mask_enabled": metrics.get("compact_mask_enabled"),
                    "chunked_prefill_enabled": metrics.get("chunked_prefill_enabled"),
                    "cte_attention_mask_path": attention_mask_path,
                    "dense_cte_mask_fallback": dense_mask_fallback,
                    "attention_mask_path_observed": attention_mask_path_observed,
                    "avoids_dense_sxs_mask": avoids_dense_sxs_mask,
                }
            )
            _require(
                metrics.get("compact_mask_enabled") is True
                or metrics.get("chunked_prefill_enabled") is True,
                failures,
                (
                    f"{row.get('variant')} prompt={prompt_len} did not run with "
                    "compact mask or Neuron chunked-prefill path enabled"
                ),
            )
            if strict_final and not attention_mask_path_observed:
                failures.append(
                    (
                        f"{row.get('variant')} prompt={prompt_len} missing "
                        "CTE attention-mask path evidence"
                    )
                )
            _require(
                avoids_dense_sxs_mask,
                failures,
                (
                    f"{row.get('variant')} prompt={prompt_len} reported dense "
                    "SxS CTE attention-mask fallback"
                ),
            )
    if strict_final and not long_context_checks:
        failures.append("strict final acceptance requires at least one 8K+ prompt row")

    required_long_artifact_profiles = set()
    if require_128k:
        required_long_artifact_profiles.add(("H_128k_candidate", 131072))
    if require_262k:
        required_long_artifact_profiles.add(("I_262k_recovery_block256", 262144))
    if strict_final:
        required_long_artifact_profiles.add(("H_128k_candidate", 131072))
        required_long_artifact_profiles.add(("I_262k_recovery_block256", 262144))
        required_long_artifact_profiles.add(("J_262k_recovery_block128", 262144))

    long_artifact_profile_rows = rows if strict_final else gate_rows
    long_artifact_profile_checks = []
    for variant, prompt_len in sorted(required_long_artifact_profiles):
        expected_key = LONG_ARTIFACT_REQUIRED_TILE_CASES[(variant, prompt_len)]
        expected_cte_buckets = list(
            LONG_ARTIFACT_REQUIRED_CTE_BUCKETS[(variant, prompt_len)]
        )
        expected_tile_case = {
            field: expected_key[index]
            for index, field in enumerate(TILE_CASE_FIELDS)
        }
        matching_rows = [
            row
            for row in long_artifact_profile_rows
            if row.get("variant") == variant
            and int(row.get("target_prompt_tokens", -1)) == prompt_len
        ]
        for row in matching_rows:
            metrics = row.get("metrics") if isinstance(row.get("metrics"), dict) else {}
            actual_key = _tile_case_key(row)
            actual_tile_case = {
                field: actual_key[index]
                for index, field in enumerate(TILE_CASE_FIELDS)
            }
            actual_cte_buckets = _int_list_or_none(
                row.get("cte_buckets", metrics.get("cte_buckets"))
            )
            matches_profile = actual_key == expected_key
            matches_cte_buckets = actual_cte_buckets == expected_cte_buckets
            text_only_enabled = metrics.get("text_only_cte_enabled") is True
            compact_mask_enabled = metrics.get("compact_mask_enabled") is True
            matches_flags = text_only_enabled and compact_mask_enabled
            long_artifact_profile_checks.append(
                {
                    "variant": variant,
                    "prompt_len": prompt_len,
                    "max_tokens": row.get("max_tokens"),
                    "expected_tile_case": expected_tile_case,
                    "actual_tile_case": actual_tile_case,
                    "matches_profile": matches_profile,
                    "expected_cte_buckets": expected_cte_buckets,
                    "actual_cte_buckets": actual_cte_buckets,
                    "matches_cte_buckets": matches_cte_buckets,
                    "text_only_cte_enabled": text_only_enabled,
                    "compact_mask_enabled": compact_mask_enabled,
                    "matches_required_flags": matches_flags,
                }
            )
            if not matches_profile:
                failures.append(
                    (
                        f"{variant} prompt={prompt_len} max_tokens={row.get('max_tokens')} "
                        "did not use required long-artifact tile/block profile "
                        f"{expected_tile_case}"
                    )
                )
            if not matches_cte_buckets:
                failures.append(
                    (
                        f"{variant} prompt={prompt_len} max_tokens={row.get('max_tokens')} "
                        "did not use required long-artifact CTE buckets "
                        f"{expected_cte_buckets}"
                    )
                )
            if not matches_flags:
                failures.append(
                    (
                        f"{variant} prompt={prompt_len} max_tokens={row.get('max_tokens')} "
                        "did not enable text-only CTE and compact CTE mask"
                    )
                )

    feature_launch_profile_rows = rows if strict_final else gate_rows
    feature_launch_profile_checks = []
    for row in feature_launch_profile_rows:
        variant = row.get("variant")
        expected_profile = EXPECTED_FEATURE_LAUNCH_PROFILE_BY_VARIANT.get(variant)
        if expected_profile is None:
            continue
        metrics = row.get("metrics") if isinstance(row.get("metrics"), dict) else {}
        actual_profile = {
            "cte_buckets": _int_list_or_none(
                row.get("cte_buckets", metrics.get("cte_buckets"))
            ),
            "text_only_cte_enabled": _row_bool_or_none(row, "text_only_cte_enabled"),
            "compact_mask_enabled": _row_bool_or_none(row, "compact_mask_enabled"),
            "cold_zero_conv_fast_path_enabled": _row_bool_or_none(
                row,
                "cold_zero_conv_fast_path_enabled",
            ),
        }
        if not strict_final and actual_profile["cte_buckets"] is None:
            continue

        expected = {
            "cte_buckets": list(expected_profile["cte_buckets"]),
            "text_only_cte_enabled": expected_profile["text_only_cte_enabled"],
            "compact_mask_enabled": expected_profile["compact_mask_enabled"],
            "cold_zero_conv_fast_path_enabled": expected_profile[
                "cold_zero_conv_fast_path_enabled"
            ],
        }
        missing_fields = [
            name for name, value in actual_profile.items() if value is None
        ]
        mismatched_fields = [
            name
            for name, value in actual_profile.items()
            if value is not None and value != expected[name]
        ]
        matches_profile = not missing_fields and not mismatched_fields
        feature_launch_profile_checks.append(
            {
                "variant": variant,
                "prompt_len": row.get("target_prompt_tokens"),
                "max_tokens": row.get("max_tokens"),
                "expected_profile": expected,
                "actual_profile": actual_profile,
                "missing_fields": missing_fields,
                "mismatched_fields": mismatched_fields,
                "matches_profile": matches_profile,
            }
        )
        if strict_final and missing_fields:
            failures.append(
                (
                    f"{variant} prompt={row.get('target_prompt_tokens')} "
                    f"max_tokens={row.get('max_tokens')} missing A-G launch profile "
                    f"fields: {missing_fields}"
                )
            )
        if mismatched_fields:
            failures.append(
                (
                    f"{variant} prompt={row.get('target_prompt_tokens')} "
                    f"max_tokens={row.get('max_tokens')} did not use required A-G "
                    f"launch profile fields: {mismatched_fields}"
                )
            )

    kernel_profile_rows = rows if strict_final else gate_rows
    gdn_kernel_profile_checks = []
    for row in kernel_profile_rows:
        variant = row.get("variant")
        expected_kernel = EXPECTED_GDN_CTE_KERNEL_BY_VARIANT.get(variant)
        if expected_kernel is None:
            continue
        actual_kernel = _gdn_cte_kernel(row)
        matches_expected = actual_kernel == expected_kernel
        gdn_kernel_profile_checks.append(
            {
                "variant": variant,
                "prompt_len": row.get("target_prompt_tokens"),
                "max_tokens": row.get("max_tokens"),
                "expected_gdn_cte_kernel": expected_kernel,
                "actual_gdn_cte_kernel": actual_kernel,
                "matches_expected": matches_expected,
            }
        )
        if actual_kernel is None:
            if strict_final:
                failures.append(
                    (
                        f"{variant} prompt={row.get('target_prompt_tokens')} "
                        f"max_tokens={row.get('max_tokens')} missing GDN CTE kernel metric"
                    )
                )
            continue
        if not matches_expected:
            failures.append(
                (
                    f"{variant} prompt={row.get('target_prompt_tokens')} "
                    f"max_tokens={row.get('max_tokens')} used GDN CTE kernel "
                    f"{actual_kernel}; expected {expected_kernel}"
                )
            )

    long_artifact_requirements = {
        "require_128k": require_128k,
        "require_262k": require_262k,
        "saw_128k_candidate": any(
            row.get("variant") == "H_128k_candidate"
            and int(row.get("target_prompt_tokens", -1)) == 131072
            and row.get("returncode", 0) == 0
            and row.get("artifact_load_success") is not False
            for row in gate_rows
        ),
        "saw_262k_block256": any(
            row.get("variant") == "I_262k_recovery_block256"
            and int(row.get("target_prompt_tokens", -1)) == 262144
            and int(row.get("block_size", -1)) == 256
            and row.get("returncode", 0) == 0
            and row.get("artifact_load_success") is not False
            for row in gate_rows
        ),
        "saw_262k_block128": any(
            row.get("variant") == "J_262k_recovery_block128"
            and int(row.get("target_prompt_tokens", -1)) == 262144
            and int(row.get("block_size", -1)) == 128
            and row.get("returncode", 0) == 0
            and row.get("artifact_load_success") is not False
            for row in gate_rows
        ),
    }
    if require_128k:
        _require(
            long_artifact_requirements["saw_128k_candidate"],
            failures,
            "128K candidate artifact row missing or failed",
        )
    if require_262k:
        _require(
            long_artifact_requirements["saw_262k_block256"],
            failures,
            "262K block256 recovery artifact row missing or failed",
        )
    if strict_final:
        _require(
            long_artifact_requirements["saw_262k_block128"],
            failures,
            "strict final acceptance requires 262K block128 comparison artifact row",
        )

    gdn_state_checks = []
    for row in gate_rows:
        if row.get("variant") in (baseline_variant, DENSE_FALLBACK_VARIANT):
            continue
        recurrent_diff = _first_float(
            row,
            (
                "gdn_recurrent_state_max_abs_diff",
                "recurrent_state_max_abs_diff",
                "recurrent_max_abs_diff",
                "max_recurrent_diff",
            ),
        )
        conv_diff = _first_float(
            row,
            (
                "gdn_conv_state_max_abs_diff",
                "conv_state_max_abs_diff",
                "conv_max_abs_diff",
                "max_conv_diff",
            ),
        )
        if recurrent_diff is None and conv_diff is None:
            if strict_final:
                failures.append(
                    (
                        f"{row.get('variant')} prompt={row.get('target_prompt_tokens')} "
                        "missing GDN recurrent/conv state diff fields"
                    )
                )
            continue
        check = {
            "variant": row.get("variant"),
            "prompt_len": row.get("target_prompt_tokens"),
            "recurrent_state_max_abs_diff": recurrent_diff,
            "conv_state_max_abs_diff": conv_diff,
        }
        gdn_state_checks.append(check)
        _require(
            recurrent_diff is not None and recurrent_diff <= recurrent_state_tolerance,
            failures,
            (
                f"{row.get('variant')} prompt={row.get('target_prompt_tokens')} "
                "GDN recurrent_state diff missing or above tolerance"
            ),
        )
        _require(
            conv_diff is not None and conv_diff <= conv_state_tolerance,
            failures,
            (
                f"{row.get('variant')} prompt={row.get('target_prompt_tokens')} "
                "GDN conv_state diff missing or above tolerance"
            ),
        )
    if not gdn_state_checks:
        message = "GDN state-diff gate skipped; no recurrent/conv diff fields found"
        if require_gdn_state_diff:
            failures.append(message)
        else:
            warnings.append(message)

    hbm_checks = []
    hbm_usage_checks = []
    if hbm_regression_tolerance is not None or require_hbm_usage:
        hbm_presence_rows = rows if strict_final else gate_rows
        for row in hbm_presence_rows:
            used_bytes = _hbm_used_bytes(row)
            hbm_usage_checks.append(
                {
                    "variant": row.get("variant"),
                    "prompt_len": row.get("target_prompt_tokens"),
                    "max_tokens": row.get("max_tokens"),
                    "hbm_used_bytes": used_bytes,
                    "present": used_bytes is not None,
                }
            )
            if strict_final and used_bytes is None:
                failures.append(
                    (
                        f"{row.get('variant')} prompt={row.get('target_prompt_tokens')} "
                        f"max_tokens={row.get('max_tokens')} missing HBM usage"
                    )
                )
        for prompt_len in sorted({prompt for _variant, prompt in grouped if prompt > 0}):
            baseline_values = [
                value
                for row in grouped.get((baseline_variant, prompt_len), [])
                if (value := _hbm_used_bytes(row)) is not None
            ]
            baseline_hbm = _p50(baseline_values)
            if baseline_hbm is None:
                message = f"HBM gate skipped for prompt={prompt_len}; baseline HBM usage missing"
                if require_hbm_usage:
                    failures.append(message)
                else:
                    warnings.append(message)
                continue
            for variant in sorted({variant for variant, prompt in grouped if prompt == prompt_len}):
                if variant in (baseline_variant, DENSE_FALLBACK_VARIANT):
                    continue
                values = [
                    value
                    for row in grouped[(variant, prompt_len)]
                    if (value := _hbm_used_bytes(row)) is not None
                ]
                candidate_hbm = _p50(values)
                if candidate_hbm is None:
                    message = f"HBM gate skipped for {variant} prompt={prompt_len}; HBM usage missing"
                    if require_hbm_usage:
                        failures.append(message)
                    else:
                        warnings.append(message)
                    continue
                allowed = (
                    baseline_hbm * (1.0 + (hbm_regression_tolerance or 0.0))
                )
                hbm_checks.append(
                    {
                        "variant": variant,
                        "prompt_len": prompt_len,
                        "baseline_hbm_used_bytes": baseline_hbm,
                        "candidate_hbm_used_bytes": candidate_hbm,
                        "allowed_hbm_used_bytes": allowed,
                    }
                )
                if hbm_regression_tolerance is not None:
                    _require(
                        candidate_hbm <= allowed,
                        failures,
                        f"{variant} prompt={prompt_len} HBM usage regressed beyond tolerance",
                    )

    baseline_tokens_by_prompt = {
        prompt_len: next(
            (
                row.get("token_ids")
                for row in grouped.get((baseline_variant, prompt_len), [])
                if row.get("token_ids") is not None
            ),
            None,
        )
        for _variant, prompt_len in grouped
        if prompt_len > 0
    }
    feature_delta_pairs = [
        (
            "dynamic CTE buckets",
            "A_single512_old_chunked",
            "B_short_buckets_old_chunked",
        ),
        (
            "text-only CTE",
            "B_short_buckets_old_chunked",
            "C_short_text_only_old_chunked",
        ),
        (
            "compact CTE mask",
            "C_short_text_only_old_chunked",
            "D_short_text_compact_old_chunked",
        ),
        (
            "fused GDN CTE",
            "D_short_text_compact_old_chunked",
            "E_short_text_compact_fused",
        ),
        (
            "cold-zero conv fast path",
            "E_short_text_compact_fused",
            "F_short_text_compact_fused_cold_zero",
        ),
        (
            "tile/block sweep",
            "F_short_text_compact_fused_cold_zero",
            "G_tile_block_sweep",
        ),
    ]
    feature_delta_checks = []
    for name, source_variant, target_variant in feature_delta_pairs:
        prompt_checks = []
        tile_case_checks = []
        for prompt_len in sorted(
            {
                prompt
                for variant, prompt in grouped
                if prompt > 0
                and prompt <= SHORT_PROMPT_LIMIT
                and variant in (source_variant, target_variant)
            }
        ):
            source_rows = grouped.get((source_variant, prompt_len), [])
            target_rows = grouped.get((target_variant, prompt_len), [])
            if not source_rows or not target_rows:
                continue
            baseline_tokens = baseline_tokens_by_prompt.get(prompt_len)

            if name == "tile/block sweep":
                target_rows_by_tile = defaultdict(list)
                for row in target_rows:
                    target_rows_by_tile[_tile_case_key(row)].append(row)

                prompt_tile_checks = []
                for tile_key, tile_rows in sorted(target_rows_by_tile.items()):
                    target_tile_case = {
                        field: tile_key[index]
                        for index, field in enumerate(TILE_CASE_FIELDS)
                    }
                    check = _feature_delta_prompt_check(
                        source_variant=source_variant,
                        target_variant=target_variant,
                        prompt_len=prompt_len,
                        source_rows=source_rows,
                        target_rows=tile_rows,
                        baseline_tokens=baseline_tokens,
                        target_tile_case=target_tile_case,
                    )
                    prompt_tile_checks.append(check)
                    tile_case_checks.append(check)

                checks_with_latency = [
                    check
                    for check in prompt_tile_checks
                    if check["target_p50_ms"] is not None
                ]
                best_check = (
                    min(checks_with_latency, key=lambda check: check["target_p50_ms"])
                    if checks_with_latency
                    else (prompt_tile_checks[0] if prompt_tile_checks else None)
                )
                if best_check is not None:
                    best_check["is_best_tile_case"] = True
                    prompt_checks.append(best_check)
                continue

            prompt_checks.append(
                _feature_delta_prompt_check(
                    source_variant=source_variant,
                    target_variant=target_variant,
                    prompt_len=prompt_len,
                    source_rows=source_rows,
                    target_rows=target_rows,
                    baseline_tokens=baseline_tokens,
                )
            )
        status = "skip"
        if prompt_checks:
            status = (
                "pass"
                if all(_feature_delta_check_passed(check) for check in prompt_checks)
                else "regressed"
            )
        feature_delta_check = {
            "name": name,
            "source_variant": source_variant,
            "target_variant": target_variant,
            "status": status,
            "prompt_checks": prompt_checks,
        }
        if name == "tile/block sweep":
            feature_delta_check["tile_case_checks"] = tile_case_checks
        feature_delta_checks.append(feature_delta_check)
        if require_feature_deltas and status == "regressed":
            failures.append(f"{name} feature delta regressed")
        if strict_final and status == "skip":
            failures.append(f"{name} feature delta missing")

    short_speedup_passed = bool(short_results) and all(
        result["speedup"] >= short_latency_speedup for result in short_results
    )
    actual_tps_passed = bool(short_results) and all(
        (
            result.get("baseline_actual_tok_per_s") is not None
            and result.get("candidate_actual_tok_per_s") is not None
            and result["candidate_actual_tok_per_s"]
            > result["baseline_actual_tok_per_s"]
        )
        for result in short_results
    )
    bucket_tps_passed = bool(short_results) and all(
        (
            result.get("baseline_bucket_tok_per_s") is not None
            and result.get("candidate_bucket_tok_per_s") is not None
            and result["candidate_bucket_tok_per_s"]
            >= result["baseline_bucket_tok_per_s"]
            * (1.0 - bucket_tok_regression_tolerance)
        )
        for result in short_results
    )
    token_exactness_passed = bool(exactness_checks) and all(
        check["exact"] for check in exactness_checks
    )
    long_context_path_passed = bool(long_context_checks) and all(
        (
            check["compact_mask_enabled"] is True
            or check["chunked_prefill_enabled"] is True
        )
        and check["avoids_dense_sxs_mask"] is True
        and (not strict_final or check["attention_mask_path_observed"] is True)
        for check in long_context_checks
    )
    hbm_passed = bool(hbm_checks) and all(
        check["candidate_hbm_used_bytes"] <= check["allowed_hbm_used_bytes"]
        for check in hbm_checks
    )
    hbm_usage_present_passed = bool(hbm_usage_checks) and all(
        check["present"] for check in hbm_usage_checks
    )
    gdn_state_passed = bool(gdn_state_checks) and all(
        (
            check["recurrent_state_max_abs_diff"] is not None
            and check["recurrent_state_max_abs_diff"] <= recurrent_state_tolerance
            and check["conv_state_max_abs_diff"] is not None
            and check["conv_state_max_abs_diff"] <= conv_state_tolerance
        )
        for check in gdn_state_checks
    )
    controlled_inputs_passed = bool(controlled_input_checks) and all(
        check["matches_baseline"] for check in controlled_input_checks
    )
    prompt_suite_passed = bool(prompt_suite_checks) and all(
        check["present"] for check in prompt_suite_checks
    )
    feature_matrix_rows_passed = bool(feature_matrix_row_checks) and all(
        check["present"] for check in feature_matrix_row_checks
    )
    generation_rows_passed = bool(generation_row_checks) and all(
        check["present"] for check in generation_row_checks
    )
    generation_tile_cases_passed = bool(generation_tile_case_checks) and all(
        check["present"] for check in generation_tile_case_checks
    )
    generation_metrics_passed = bool(generation_metric_checks) and all(
        check["present"] for check in generation_metric_checks
    )
    generation_exactness_passed = bool(generation_exactness_checks) and all(
        check["exact"] is not False and check["exact"] is not None
        for check in generation_exactness_checks
    )
    strict_sample_counts_passed = bool(strict_sample_count_checks) and all(
        check["present"] for check in strict_sample_count_checks
    )
    prefill_latency_summaries_passed = bool(prefill_latency_summary_checks) and all(
        check["present"] for check in prefill_latency_summary_checks
    )
    tile_prefill_latency_summaries_passed = bool(
        tile_prefill_latency_summary_checks
    ) and all(check["present"] for check in tile_prefill_latency_summary_checks)
    tile_sweep_cases_passed = bool(tile_sweep_case_checks) and all(
        check["present"] for check in tile_sweep_case_checks
    )
    dense_fallback_exactness_passed = bool(dense_fallback_exactness_checks) and all(
        _dense_fallback_check_passed(check)
        for check in dense_fallback_exactness_checks
    )
    hybrid_apc_exactness_passed = bool(hybrid_apc_exactness_checks) and all(
        check.get("present") is True
        and check.get("full_prefix_exact") is True
        and check.get("partial_prefix_exact") is True
        and check.get("cold_full_metrics_present") is True
        and check.get("cold_partial_metrics_present") is True
        and check.get("strict_config_match") is True
        for check in hybrid_apc_exactness_checks
    )
    long_artifact_profiles_passed = bool(long_artifact_profile_checks) and all(
        check["matches_profile"]
        and check["matches_cte_buckets"]
        and check["matches_required_flags"]
        for check in long_artifact_profile_checks
    )
    feature_launch_profiles_passed = bool(feature_launch_profile_checks) and all(
        check["matches_profile"] for check in feature_launch_profile_checks
    )
    feature_launch_profiles_have_mismatch = any(
        check["mismatched_fields"] for check in feature_launch_profile_checks
    )
    gdn_kernel_profiles_passed = bool(gdn_kernel_profile_checks) and all(
        check["matches_expected"] for check in gdn_kernel_profile_checks
    )
    prompt_token_counts_passed = bool(prompt_token_checks) and all(
        check["within_tolerance"] for check in prompt_token_checks
    )
    sampling_config_passed = bool(sampling_config_checks) and all(
        check["matches_greedy_sampling"] for check in sampling_config_checks
    )
    baseline_cold_prompt_checks_passed = bool(baseline_cold_prompt_checks) and all(
        check["within_target_range"] for check in baseline_cold_prompt_checks
    )
    cold_prefill_metrics_passed = bool(cold_prefill_metric_checks) and all(
        check["present"] for check in cold_prefill_metric_checks
    )
    benchmark_row_provenance_passed = bool(benchmark_row_provenance_checks) and all(
        check["matches_provenance"] for check in benchmark_row_provenance_checks
    )
    strict_launch_shape_passed = bool(strict_launch_shape_checks) and all(
        check["matches_launch_shape"] for check in strict_launch_shape_checks
    )
    artifact_load_success_passed = bool(artifact_load_success_checks) and all(
        check["present"] for check in artifact_load_success_checks
    )
    artifact_path_evidence_passed = bool(artifact_path_checks) and all(
        check["present"] for check in artifact_path_checks
    )
    runtime_returncode_passed = bool(runtime_returncode_checks) and all(
        check["present"] for check in runtime_returncode_checks
    )
    runtime_output_tail_passed = bool(runtime_output_tail_checks) and all(
        check["present"] for check in runtime_output_tail_checks
    )
    artifact_consistency_passed = bool(artifact_consistency_checks) and all(
        check["present"]
        and check["all_rows_have_artifacts"]
        and check["single_artifact_for_seq_len"]
        for check in artifact_consistency_checks
    )

    audit_checklist = [
        _audit_item(
            "fixed prompt suite rows",
            "pass" if prompt_suite_passed else "skip" if not strict_final else "fail",
            prompt_suite_checks,
            None if prompt_suite_checks else "strict final mode not enabled",
        ),
        _audit_item(
            "A-G fixed prompt matrix rows",
            "pass"
            if feature_matrix_rows_passed
            else "skip"
            if not strict_final
            else "fail",
            feature_matrix_row_checks,
            None if feature_matrix_row_checks else "strict final mode not enabled",
        ),
        _audit_item(
            "A-G max_tokens=32 generation rows",
            "pass"
            if generation_rows_passed
            else "skip"
            if not strict_final
            else "fail",
            generation_row_checks,
            None if generation_row_checks else "strict final mode not enabled",
        ),
        _audit_item(
            "G max_tokens=32 tile-sweep generation cases",
            "pass"
            if generation_tile_cases_passed
            else "skip"
            if not strict_final
            else "fail",
            generation_tile_case_checks,
            None if generation_tile_case_checks else "strict final mode not enabled",
        ),
        _audit_item(
            "generation latency/decode metrics",
            "pass"
            if generation_metrics_passed
            else "skip"
            if not strict_final
            else "fail",
            generation_metric_checks,
            None if generation_metric_checks else "strict final mode not enabled",
        ),
        _audit_item(
            "cold-prefill instrumentation metrics",
            "pass"
            if cold_prefill_metrics_passed
            else "skip"
            if not strict_final
            else "fail",
            cold_prefill_metric_checks,
            None
            if cold_prefill_metric_checks
            else "strict final mode not enabled",
        ),
        _audit_item(
            "benchmark row provenance",
            "pass"
            if benchmark_row_provenance_passed
            else "skip"
            if not strict_final
            else "fail",
            benchmark_row_provenance_checks,
            None
            if benchmark_row_provenance_checks
            else "strict final mode not enabled",
        ),
        _audit_item(
            "strict-final launch shape",
            "pass"
            if strict_launch_shape_passed
            else "skip"
            if not strict_final
            else "fail",
            strict_launch_shape_checks,
            None
            if strict_launch_shape_checks
            else "strict final mode not enabled",
        ),
        _audit_item(
            "artifact load success evidence",
            "pass"
            if artifact_load_success_passed
            else "skip"
            if not strict_final
            else "fail",
            artifact_load_success_checks,
            None
            if artifact_load_success_checks
            else "strict final mode not enabled"
            if not strict_final
            else "no compiled-artifact rows present",
        ),
        _audit_item(
            "compiled artifact path existence evidence",
            "pass"
            if artifact_path_evidence_passed
            else "skip"
            if not strict_final
            else "fail",
            artifact_path_checks,
            None
            if artifact_path_checks
            else "strict final mode not enabled"
            if not strict_final
            else "no compiled-artifact rows present",
        ),
        _audit_item(
            "runtime returncode evidence",
            "pass"
            if runtime_returncode_passed
            else "skip"
            if not strict_final
            else "fail",
            runtime_returncode_checks,
            None
            if runtime_returncode_checks
            else "strict final mode not enabled",
        ),
        _audit_item(
            "runtime output tail DMA-spill evidence",
            "pass"
            if runtime_output_tail_passed
            else "skip"
            if not strict_final
            else "fail",
            runtime_output_tail_checks,
            None
            if runtime_output_tail_checks
            else "strict final mode not enabled",
        ),
        _audit_item(
            "compiled artifact consistency by seq_len",
            "pass"
            if artifact_consistency_passed
            else "skip"
            if not strict_final
            else "fail",
            artifact_consistency_checks,
            None
            if artifact_consistency_checks
            else "strict final mode not enabled",
        ),
        _audit_item(
            "max_tokens=32 token exactness vs baseline",
            "pass"
            if generation_exactness_passed
            else "skip"
            if not strict_final
            else "fail",
            generation_exactness_checks,
            None if generation_exactness_checks else "strict final mode not enabled",
        ),
        _audit_item(
            "strict-final repeated samples",
            "pass"
            if strict_sample_counts_passed
            else "skip"
            if not strict_final
            else "fail",
            strict_sample_count_checks,
            None if strict_sample_count_checks else "strict final mode not enabled",
        ),
        _audit_item(
            "prefill latency p50/p95 summaries",
            "pass"
            if prefill_latency_summaries_passed
            else "skip"
            if not strict_final
            else "fail",
            prefill_latency_summary_checks,
            None
            if prefill_latency_summary_checks
            else "strict final mode not enabled",
        ),
        _audit_item(
            "G tile-sweep latency p50/p95 summaries",
            "pass"
            if tile_prefill_latency_summaries_passed
            else "skip"
            if not strict_final
            else "fail",
            tile_prefill_latency_summary_checks,
            None
            if tile_prefill_latency_summary_checks
            else "strict final mode not enabled",
        ),
        _audit_item(
            "2K/8K tile sweep cases",
            "pass"
            if tile_sweep_cases_passed
            else "skip"
            if not strict_final
            else "fail",
            tile_sweep_case_checks,
            None if tile_sweep_case_checks else "strict final mode not enabled",
        ),
        _audit_item(
            "small dense mask fallback exactness",
            "pass"
            if dense_fallback_exactness_passed
            else "skip"
            if not strict_final
            else "fail",
            dense_fallback_exactness_checks,
            None
            if dense_fallback_exactness_checks
            else "strict final mode not enabled",
        ),
        _audit_item(
            "hybrid APC partial-prefix exactness",
            "pass"
            if hybrid_apc_exactness_passed
            else "skip"
            if not strict_final
            else "fail",
            hybrid_apc_exactness_checks,
            None
            if hybrid_apc_exactness_checks
            else "strict final mode not enabled",
        ),
        _audit_item(
            "greedy sampling config",
            "pass"
            if sampling_config_passed
            else "skip"
            if not strict_final
            else "fail",
            sampling_config_checks,
            None if sampling_config_checks else "strict final mode not enabled",
        ),
        _audit_item(
            "controlled prompt/model/artifact/length inputs",
            "pass" if controlled_inputs_passed else "fail",
            controlled_input_checks,
        ),
        _audit_item(
            "fixed prompt suite token counts",
            "pass"
            if prompt_token_counts_passed
            else "skip"
            if not require_prompt_token_counts and not prompt_token_checks
            else "fail",
            prompt_token_checks,
            None if prompt_token_checks else "no actual prompt token counts present",
        ),
        _audit_item(
            "baseline cold tok/s reproduction",
            (
                "pass"
                if baseline_cold_gate
                and baseline_cold_tok_per_s is not None
                and baseline_cold_gate["lower_tok_per_s"]
                <= baseline_cold_tok_per_s
                <= baseline_cold_gate["upper_tok_per_s"]
                else "skip"
                if baseline_cold_gate is None
                else "fail"
            ),
            baseline_cold_gate,
            None if baseline_cold_gate is not None else "no baseline target configured",
        ),
        _audit_item(
            "baseline short prompt tok/s reproduction",
            "pass"
            if baseline_cold_prompt_checks_passed
            else "skip"
            if not strict_final or baseline_cold_tok_per_s_target is None
            else "fail",
            baseline_cold_prompt_checks,
            None
            if baseline_cold_prompt_checks
            else "strict final mode or baseline target not enabled",
        ),
        _audit_item("short prompt p50 latency speedup", "pass" if short_speedup_passed else "fail", short_results),
        _audit_item("short prompt actual tok/s improves", "pass" if actual_tps_passed else "fail", short_results),
        _audit_item("bucket tok/s does not regress badly", "pass" if bucket_tps_passed else "fail", short_results),
        _audit_item(
            "2K no-regression",
            "pass"
            if baseline_2k_p50 is not None and candidate_2k_p50 is not None
            else "skip"
            if baseline_2k_p50 is None
            else "fail",
            {
                "baseline_p50_ms": baseline_2k_p50,
                "candidate_p50_ms": candidate_2k_p50 if baseline_2k_p50 is not None else None,
                "tolerance": two_k_regression_tolerance,
            },
        ),
        _audit_item("token exactness vs baseline", "pass" if token_exactness_passed else "fail", exactness_checks),
        _audit_item(
            "8K+ compact/chunked path avoids dense SxS fallback",
            "pass"
            if long_context_path_passed
            else "skip"
            if not strict_final
            else "fail",
            long_context_checks,
            None if long_context_checks else "no 8K+ rows present",
        ),
        _audit_item(
            "HBM usage present on every row",
            "pass"
            if hbm_usage_present_passed
            else "skip"
            if not require_hbm_usage
            else "fail",
            hbm_usage_checks,
            None if hbm_usage_checks else "no HBM usage rows present",
        ),
        _audit_item(
            "HBM usage regression",
            "pass" if hbm_passed else "skip" if not require_hbm_usage else "fail",
            hbm_checks,
            None if hbm_checks else "no HBM usage rows present",
        ),
        _audit_item(
            "GDN recurrent/conv state diff bounds",
            "pass" if gdn_state_passed else "skip" if not require_gdn_state_diff else "fail",
            gdn_state_checks,
            None if gdn_state_checks else "no GDN state diff fields present",
        ),
        _audit_item(
            "128K artifact load/run",
            "pass"
            if long_artifact_requirements["saw_128k_candidate"]
            else "skip"
            if not require_128k
            else "fail",
            long_artifact_requirements,
        ),
        _audit_item(
            "262K block256 artifact load/run",
            "pass"
            if long_artifact_requirements["saw_262k_block256"]
            else "skip"
            if not require_262k
            else "fail",
            long_artifact_requirements,
        ),
        _audit_item(
            "262K block128 comparison load/run",
            "pass"
            if long_artifact_requirements["saw_262k_block128"]
            else "skip"
            if not strict_final
            else "fail",
            long_artifact_requirements,
        ),
        _audit_item(
            "long-artifact tile/block launch profiles",
            "pass"
            if long_artifact_profiles_passed
            else "skip"
            if not required_long_artifact_profiles
            else "fail",
            long_artifact_profile_checks,
            None if long_artifact_profile_checks else "no required long-artifact rows present",
        ),
        _audit_item(
            "A-G launch profiles",
            "pass"
            if feature_launch_profiles_passed
            else "skip"
            if not strict_final and not feature_launch_profiles_have_mismatch
            else "fail",
            feature_launch_profile_checks,
            None
            if feature_launch_profile_checks
            else "no A-G rows with CTE bucket metrics present",
        ),
        _audit_item(
            "GDN CTE kernel launch profiles",
            "pass"
            if gdn_kernel_profiles_passed
            else "skip"
            if not strict_final and not gdn_kernel_profile_checks
            else "fail",
            gdn_kernel_profile_checks,
            None if gdn_kernel_profile_checks else "no GDN CTE kernel rows present",
        ),
    ]

    return {
        "passed": not failures,
        "failures": failures,
        "warnings": warnings,
        "audit_checklist": audit_checklist,
        "prefill_max_tokens": prefill_max_tokens,
        "bucket_tok_regression_tolerance": bucket_tok_regression_tolerance,
        "goal_baseline_cold_tok_per_s_target": GOAL_BASELINE_COLD_TOK_PER_S_TARGET,
        "baseline_cold_tok_per_s_gate": baseline_cold_gate,
        "baseline_cold_prompt_checks": baseline_cold_prompt_checks,
        "prompt_token_tolerance": prompt_token_tolerance,
        "mandatory_prompt_lengths": list(MANDATORY_PROMPT_LENGTHS),
        "feature_matrix_variants": list(FEATURE_MATRIX_VARIANTS),
        "expected_feature_launch_profile_by_variant": {
            variant: {
                "cte_buckets": list(profile["cte_buckets"]),
                "text_only_cte_enabled": profile["text_only_cte_enabled"],
                "compact_mask_enabled": profile["compact_mask_enabled"],
                "cold_zero_conv_fast_path_enabled": profile[
                    "cold_zero_conv_fast_path_enabled"
                ],
            }
            for variant, profile in EXPECTED_FEATURE_LAUNCH_PROFILE_BY_VARIANT.items()
        },
        "expected_gdn_cte_kernel_by_variant": EXPECTED_GDN_CTE_KERNEL_BY_VARIANT,
        "long_artifact_generation_targets": [
            {"variant": variant, "prompt_len": prompt_len}
            for variant, prompt_len in LONG_ARTIFACT_GENERATION_TARGETS
        ],
        "long_artifact_required_tile_cases": [
            {
                "variant": variant,
                "prompt_len": prompt_len,
                "tile_case": {
                    field: tile_case[index]
                    for index, field in enumerate(TILE_CASE_FIELDS)
                },
            }
            for (variant, prompt_len), tile_case in sorted(
                LONG_ARTIFACT_REQUIRED_TILE_CASES.items()
            )
        ],
        "long_artifact_required_cte_buckets": [
            {
                "variant": variant,
                "prompt_len": prompt_len,
                "cte_buckets": list(cte_buckets),
            }
            for (variant, prompt_len), cte_buckets in sorted(
                LONG_ARTIFACT_REQUIRED_CTE_BUCKETS.items()
            )
        ],
        "long_artifact_profile_checks": long_artifact_profile_checks,
        "feature_launch_profile_checks": feature_launch_profile_checks,
        "gdn_kernel_profile_checks": gdn_kernel_profile_checks,
        "required_tile_sweep_prompts": list(REQUIRED_TILE_SWEEP_PROMPTS),
        "required_tile_sweep_cases": [
            {
                field: tile_case[index]
                for index, field in enumerate(TILE_CASE_FIELDS)
            }
            for tile_case in REQUIRED_TILE_SWEEP_CASES
        ],
        "dense_fallback_variant": DENSE_FALLBACK_VARIANT,
        "dense_fallback_prompt_lengths": list(DENSE_FALLBACK_PROMPT_LENGTHS),
        "dense_fallback_exactness_checks": dense_fallback_exactness_checks,
        "hybrid_apc_exactness_checks": hybrid_apc_exactness_checks,
        "sampling_config_checks": sampling_config_checks,
        "prompt_suite_checks": prompt_suite_checks,
        "feature_matrix_row_checks": feature_matrix_row_checks,
        "generation_row_checks": generation_row_checks,
        "generation_tile_case_checks": generation_tile_case_checks,
        "generation_metric_checks": generation_metric_checks,
        "cold_prefill_metric_checks": cold_prefill_metric_checks,
        "benchmark_row_provenance_checks": benchmark_row_provenance_checks,
        "strict_launch_shape_checks": strict_launch_shape_checks,
        "artifact_load_success_checks": artifact_load_success_checks,
        "artifact_path_checks": artifact_path_checks,
        "runtime_returncode_checks": runtime_returncode_checks,
        "runtime_output_tail_checks": runtime_output_tail_checks,
        "artifact_consistency_checks": artifact_consistency_checks,
        "generation_exactness_checks": generation_exactness_checks,
        "strict_min_samples": strict_min_samples,
        "strict_sample_count_checks": strict_sample_count_checks,
        "prefill_latency_summary_checks": prefill_latency_summary_checks,
        "tile_prefill_latency_summary_checks": tile_prefill_latency_summary_checks,
        "tile_sweep_case_checks": tile_sweep_case_checks,
        "prompt_token_checks": prompt_token_checks,
        "require_controlled_inputs": require_controlled_inputs,
        "require_feature_deltas": require_feature_deltas,
        "strict_final": strict_final,
        "controlled_input_checks": controlled_input_checks,
        "short_results": short_results,
        "latency_summaries": latency_summaries,
        "generation_summaries": generation_summaries,
        "feature_delta_checks": feature_delta_checks,
        "exactness_checks": exactness_checks,
        "long_context_checks": long_context_checks,
        "long_artifact_requirements": long_artifact_requirements,
        "gdn_state_checks": gdn_state_checks,
        "hbm_usage_checks": hbm_usage_checks,
        "hbm_checks": hbm_checks,
    }


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("benchmark_json", nargs="+", type=Path)
    parser.add_argument("--baseline-variant", default=DEFAULT_BASELINE)
    parser.add_argument("--candidate-variant", default=DEFAULT_CANDIDATE)
    parser.add_argument("--short-latency-speedup", type=float, default=1.5)
    parser.add_argument("--two-k-regression-tolerance", type=float, default=0.0)
    parser.add_argument("--bucket-tok-regression-tolerance", type=float, default=0.20)
    parser.add_argument("--baseline-cold-tok-per-s-target", type=float, default=None)
    parser.add_argument("--baseline-cold-tok-per-s-tolerance", type=float, default=0.15)
    parser.add_argument("--prompt-token-tolerance", type=float, default=0.05)
    parser.add_argument("--require-prompt-token-counts", action="store_true")
    parser.add_argument("--prefill-max-tokens", type=int, default=1)
    parser.add_argument("--recurrent-state-tolerance", type=float, default=DEFAULT_GDN_DIFF_TOLERANCE)
    parser.add_argument("--conv-state-tolerance", type=float, default=DEFAULT_GDN_DIFF_TOLERANCE)
    parser.add_argument("--require-gdn-state-diff", action="store_true")
    parser.add_argument("--hbm-regression-tolerance", type=float, default=None)
    parser.add_argument("--require-hbm-usage", action="store_true")
    parser.add_argument(
        "--require-controlled-inputs",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Require baseline and candidate rows to share prompt/model/artifact/length metadata.",
    )
    parser.add_argument("--require-feature-deltas", action="store_true")
    parser.add_argument("--require-128k", action="store_true")
    parser.add_argument("--require-262k", action="store_true")
    parser.add_argument(
        "--hybrid-apc-report",
        type=Path,
        help=(
            "JSON report from qwen36_hybrid_apc_validation.py exactness. "
            "Required by --strict-final."
        ),
    )
    parser.add_argument(
        "--strict-final",
        action="store_true",
        help=(
            "Require all final goal gates: prompt counts, feature deltas, HBM, "
            "GDN state diff, 128K/262K artifacts, and an explicit baseline tok/s target."
        ),
    )
    parser.add_argument(
        "--strict-min-samples",
        type=int,
        default=DEFAULT_STRICT_MIN_SAMPLES,
        help="Minimum repeated samples per strict-final prefill/generation row.",
    )
    parser.add_argument("--output-json", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = evaluate(
        _load_all_rows(args.benchmark_json),
        baseline_variant=args.baseline_variant,
        candidate_variant=args.candidate_variant,
        short_latency_speedup=args.short_latency_speedup,
        two_k_regression_tolerance=args.two_k_regression_tolerance,
        bucket_tok_regression_tolerance=args.bucket_tok_regression_tolerance,
        baseline_cold_tok_per_s_target=args.baseline_cold_tok_per_s_target,
        baseline_cold_tok_per_s_tolerance=args.baseline_cold_tok_per_s_tolerance,
        prompt_token_tolerance=args.prompt_token_tolerance,
        require_prompt_token_counts=args.require_prompt_token_counts,
        prefill_max_tokens=args.prefill_max_tokens,
        recurrent_state_tolerance=args.recurrent_state_tolerance,
        conv_state_tolerance=args.conv_state_tolerance,
        require_gdn_state_diff=args.require_gdn_state_diff,
        hbm_regression_tolerance=args.hbm_regression_tolerance,
        require_hbm_usage=args.require_hbm_usage,
        require_controlled_inputs=args.require_controlled_inputs,
        require_feature_deltas=args.require_feature_deltas,
        require_128k=args.require_128k,
        require_262k=args.require_262k,
        strict_final=args.strict_final,
        strict_min_samples=args.strict_min_samples,
        hybrid_apc_report=_load_hybrid_apc_report(args.hybrid_apc_report),
    )
    payload = json.dumps(report, indent=2, sort_keys=True)
    if args.output_json:
        args.output_json.write_text(payload + "\n")
    print(payload)
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
