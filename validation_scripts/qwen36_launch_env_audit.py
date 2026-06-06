#!/usr/bin/env python3
"""Audit a Qwen3.6 launch env log against the compile env log."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


def parse_env_log(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for raw_line in path.read_text(errors="replace").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key] = value
    return values


def _ints(value: str | None) -> list[int]:
    if not value:
        return []
    return [int(item) for item in re.findall(r"\d+", value)]


def _int_set(value: str | None) -> set[int]:
    return set(_ints(value))


def _pairs(value: str | None) -> set[tuple[int, int]]:
    pairs: set[tuple[int, int]] = set()
    if not value:
        return pairs
    for raw in re.split(r"[\s,]+", value.strip()):
        if not raw:
            continue
        match = re.fullmatch(r"(\d+):(\d+)", raw)
        if match:
            pairs.add((int(match.group(1)), int(match.group(2))))
    return pairs


def _normal_dtype(value: str | None) -> str:
    raw = (value or "").strip().lower()
    if raw.startswith("torch."):
        raw = raw.split(".", 1)[1]
    if raw in {"bf16", "bfloat16"}:
        return "bfloat16"
    if raw in {"fp32", "float", "float32"}:
        return "float32"
    return raw


def _format_int_set(values: set[int]) -> list[int]:
    return sorted(values)


def _format_pairs(values: set[tuple[int, int]]) -> list[str]:
    return [f"{active}:{prefix}" for active, prefix in sorted(values)]


def _as_int(values: dict[str, str], key: str) -> int | None:
    raw = values.get(key)
    if raw is None or raw.strip() == "":
        return None
    try:
        return int(raw)
    except ValueError:
        return None


def _require_equal(
    errors: list[str],
    compile_env: dict[str, str],
    launch_env: dict[str, str],
    key: str,
    *,
    normalizer=lambda value: (value or "").strip(),
) -> None:
    compile_value = normalizer(compile_env.get(key))
    launch_value = normalizer(launch_env.get(key))
    if compile_value != launch_value:
        errors.append(
            f"{key} mismatch: compile={compile_env.get(key)!r} launch={launch_env.get(key)!r}"
        )


def _require_launch_value(
    errors: list[str],
    launch_env: dict[str, str],
    key: str,
    expected: str,
) -> None:
    actual = launch_env.get(key)
    if actual != expected:
        errors.append(f"{key} must be {expected!r}, got {actual!r}")


def audit(
    *,
    compile_env: dict[str, str],
    launch_env: dict[str, str],
    require_launch_dry_run: bool = False,
) -> dict[str, Any]:
    errors: list[str] = []
    warnings: list[str] = []

    _require_equal(errors, compile_env, launch_env, "ARTIFACT")
    for key in (
        "MAX_GDN_CHECKPOINT_SLOTS",
        "QWEN36_DELTANET_MULTIHEAD_CTE",
        "QWEN36_SPLIT_QKV_TKG_ROW_SCALES",
        "QWEN36_DELTANET_FUSED_SEGMENT_TOKENS",
        "QWEN36_DELTANET_SOLVE_MODE",
        "QWEN36_DELTANET_SOLVE_SCAN_STEPS",
    ):
        if key in compile_env or key in launch_env:
            _require_equal(errors, compile_env, launch_env, key)
    _require_equal(
        errors,
        compile_env,
        launch_env,
        "GDN_RECURRENT_CACHE_DTYPE",
        normalizer=_normal_dtype,
    )
    _require_equal(
        errors,
        compile_env,
        launch_env,
        "GDN_CONV_CACHE_DTYPE",
        normalizer=_normal_dtype,
    )

    compile_context = _as_int(compile_env, "MAX_CONTEXT_LENGTH")
    launch_max_model_len = _as_int(launch_env, "MAX_MODEL_LEN")
    if compile_context is None:
        errors.append("MAX_CONTEXT_LENGTH is missing or invalid in compile env log")
    if launch_max_model_len is None:
        errors.append("MAX_MODEL_LEN is missing or invalid in launch env log")
    if compile_context is not None and launch_max_model_len is not None:
        if launch_max_model_len != compile_context:
            errors.append(
                "MAX_MODEL_LEN must match compiled MAX_CONTEXT_LENGTH: "
                f"compile={compile_context} launch={launch_max_model_len}"
            )

    compile_seq = _as_int(compile_env, "SEQ_LEN")
    launch_seq = _as_int(launch_env, "SEQ_LEN")
    if compile_seq is None:
        errors.append("SEQ_LEN is missing or invalid in compile env log")
    if launch_seq is None:
        errors.append("SEQ_LEN is missing or invalid in launch env log")
    if compile_seq is not None and launch_seq is not None and launch_seq != compile_seq:
        errors.append(f"SEQ_LEN must match compiled SEQ_LEN: compile={compile_seq} launch={launch_seq}")

    compile_cte = _int_set(compile_env.get("CTE_BUCKETS"))
    launch_cte = _int_set(launch_env.get("CTE_BUCKETS"))
    if not compile_cte:
        errors.append("CTE_BUCKETS missing from compile env log")
    if not launch_cte:
        errors.append("CTE_BUCKETS missing from launch env log")
    if compile_cte and launch_cte and compile_cte != launch_cte:
        errors.append(
            "CTE_BUCKETS must match compiled buckets: "
            f"missing={_format_int_set(compile_cte - launch_cte)} "
            f"extra={_format_int_set(launch_cte - compile_cte)}"
        )

    compile_tkg = _int_set(compile_env.get("TOKEN_GENERATION_BUCKETS"))
    launch_tkg = _int_set(launch_env.get("TOKEN_GENERATION_BUCKETS"))
    if not compile_tkg:
        errors.append("TOKEN_GENERATION_BUCKETS missing from compile env log")
    if not launch_tkg:
        errors.append("TOKEN_GENERATION_BUCKETS missing from launch env log")
    if compile_tkg and launch_tkg and compile_tkg != launch_tkg:
        errors.append(
            "TOKEN_GENERATION_BUCKETS must match compiled buckets: "
            f"missing={_format_int_set(compile_tkg - launch_tkg)} "
            f"extra={_format_int_set(launch_tkg - compile_tkg)}"
        )

    compile_pairs = _pairs(compile_env.get("CONTEXT_ENCODING_BUCKET_PAIRS"))
    launch_pairs = _pairs(launch_env.get("CONTEXT_ENCODING_BUCKET_PAIRS"))
    if not compile_pairs:
        errors.append("CONTEXT_ENCODING_BUCKET_PAIRS missing from compile env log")
    if not launch_pairs:
        errors.append("CONTEXT_ENCODING_BUCKET_PAIRS missing from launch env log")
    if compile_pairs and launch_pairs:
        missing = compile_pairs - launch_pairs
        extra = launch_pairs - compile_pairs
        invalid_extra = {pair for pair in extra if pair[1] != 0}
        if missing or invalid_extra:
            errors.append(
                "CONTEXT_ENCODING_BUCKET_PAIRS must match compiled pairs "
                "except optional prefix-0 launch pairs: "
                f"missing={_format_pairs(missing)} "
                f"extra_not_prefix0={_format_pairs(invalid_extra)}"
            )
        allowed_extra = extra - invalid_extra
        if allowed_extra:
            warnings.append(
                "launch adds prefix-0 context pair(s), which are allowed: "
                + ", ".join(_format_pairs(allowed_extra))
            )

    for key, expected in (
        ("DISABLE_HYBRID_KV_CACHE_MANAGER", "0"),
        ("ENABLE_HYBRID_APC", "1"),
        ("ENABLE_PREFIX_CACHING", "1"),
        ("ENABLE_VLLM_CHUNKED_PREFILL", "1"),
        ("HYBRID_APC_REQUIRE_VLLM_METADATA", "1"),
        ("HYBRID_APC_ENABLE_BACKED_PREFIX_READS", "1"),
        ("BLOCK_SIZE", "256"),
        ("GDN_CHECKPOINT_INTERVAL", "256"),
    ):
        _require_launch_value(errors, launch_env, key, expected)

    if require_launch_dry_run:
        _require_launch_value(errors, launch_env, "LAUNCH_DRY_RUN", "1")

    return {
        "schema": "qwen36-launch-env-audit-v1",
        "passed": not errors,
        "errors": errors,
        "warnings": warnings,
        "compile": {
            "env_log": compile_env.get("ENVLOG"),
            "artifact": compile_env.get("ARTIFACT"),
            "source_commit": compile_env.get("SOURCE_COMMIT"),
            "source_branch": compile_env.get("SOURCE_BRANCH"),
            "max_context_length": compile_env.get("MAX_CONTEXT_LENGTH"),
            "seq_len": compile_env.get("SEQ_LEN"),
            "cte_buckets": _format_int_set(compile_cte),
            "token_generation_buckets": _format_int_set(compile_tkg),
            "context_encoding_bucket_pairs": _format_pairs(compile_pairs),
            "gdn_recurrent_cache_dtype": compile_env.get("GDN_RECURRENT_CACHE_DTYPE"),
            "gdn_conv_cache_dtype": compile_env.get("GDN_CONV_CACHE_DTYPE"),
        },
        "launch": {
            "env_log": launch_env.get("ENVLOG"),
            "artifact": launch_env.get("ARTIFACT"),
            "max_model_len": launch_env.get("MAX_MODEL_LEN"),
            "seq_len": launch_env.get("SEQ_LEN"),
            "cte_buckets": _format_int_set(launch_cte),
            "token_generation_buckets": _format_int_set(launch_tkg),
            "context_encoding_bucket_pairs": _format_pairs(launch_pairs),
            "gdn_recurrent_cache_dtype": launch_env.get("GDN_RECURRENT_CACHE_DTYPE"),
            "gdn_conv_cache_dtype": launch_env.get("GDN_CONV_CACHE_DTYPE"),
            "launch_dry_run": launch_env.get("LAUNCH_DRY_RUN"),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--compile-env-log", type=Path, required=True)
    parser.add_argument("--launch-env-log", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument("--require-launch-dry-run", action="store_true")
    args = parser.parse_args()

    compile_env = parse_env_log(args.compile_env_log.expanduser())
    launch_env = parse_env_log(args.launch_env_log.expanduser())
    payload = audit(
        compile_env=compile_env,
        launch_env=launch_env,
        require_launch_dry_run=args.require_launch_dry_run,
    )
    encoded = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if args.output_json is not None:
        args.output_json.expanduser().parent.mkdir(parents=True, exist_ok=True)
        args.output_json.expanduser().write_text(encoded)
    print(encoded, end="")
    return 0 if payload["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
