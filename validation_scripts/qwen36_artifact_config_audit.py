#!/usr/bin/env python3
"""Audit Qwen3.6 Neuron artifact config for APC/prefill A/B experiments."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


def _load_config(path: Path) -> dict[str, Any]:
    config_path = path
    if path.is_dir():
        config_path = path / "neuron_config.json"
    with config_path.open() as handle:
        return json.load(handle)


def _first_config_value(config: dict[str, Any], *keys: str, default: Any = None) -> Any:
    for key in keys:
        if key in config:
            return config[key]
    override = config.get("override_neuron_config")
    if isinstance(override, dict):
        for key in keys:
            if key in override:
                return override[key]
    nested = config.get("neuron_config")
    if isinstance(nested, dict):
        for key in keys:
            if key in nested:
                return nested[key]
        nested_override = nested.get("override_neuron_config")
        if isinstance(nested_override, dict):
            for key in keys:
                if key in nested_override:
                    return nested_override[key]
    return default


def _parse_env_log(path: Path | None) -> dict[str, str]:
    if path is None:
        return {}
    values: dict[str, str] = {}
    for raw_line in path.read_text(errors="replace").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key] = value
    return values


def _ints(value: Any) -> list[int]:
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        result: list[int] = []
        for item in value:
            result.extend(_ints(item))
        return result
    return [int(token) for token in str(value).replace(",", " ").split() if token]


def _pair_tokens(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        if len(value) == 2 and not any(isinstance(item, (list, tuple)) for item in value):
            return [f"{int(value[0])}:{int(value[1])}"]
        pairs: list[str] = []
        for item in value:
            pairs.extend(_pair_tokens(item))
        return pairs
    tokens = []
    for token in str(value).replace(",", " ").split():
        if ":" in token:
            active, prefix = token.split(":", 1)
        elif "x" in token:
            active, prefix = token.split("x", 1)
        else:
            continue
        tokens.append(f"{int(active)}:{int(prefix)}")
    return tokens


def _normal_dtype(value: Any) -> str:
    raw = str(value or "").strip().lower()
    if raw.startswith("torch."):
        raw = raw.split(".", 1)[1]
    return {
        "bf16": "bfloat16",
        "fp32": "float32",
        "float": "float32",
    }.get(raw, raw)


def _normal_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    raw = str(value or "").strip().lower()
    return raw in {"1", "true", "yes", "on"}


def _bool_config(config: dict[str, Any], *keys: str) -> bool:
    return bool(_first_config_value(config, *keys, default=False))


def _compile_backend_from_log(path: Path | None) -> str | None:
    if path is None or not path.exists():
        return None
    backend = None
    for line in path.read_text(errors="replace").splitlines():
        if line.startswith("DELTANET_CTE_BACKEND "):
            backend = line.split(maxsplit=1)[1].strip()
        elif " --deltanet-cte-backend " in line:
            backend = line.rsplit(" --deltanet-cte-backend ", 1)[1].split()[0]
    return backend


def _warning(
    warnings: list[dict[str, Any]],
    *,
    code: str,
    message: str,
    value: Any = None,
):
    warnings.append({"code": code, "message": message, "value": value})


def _policy_error(
    errors: list[dict[str, Any]],
    *,
    code: str,
    message: str,
    expected: Any = None,
    actual: Any = None,
):
    errors.append(
        {
            "code": code,
            "message": message,
            "expected": expected,
            "actual": actual,
        }
    )


def _require_equal(
    errors: list[dict[str, Any]],
    *,
    code: str,
    message: str,
    expected: Any,
    actual: Any,
):
    if expected != actual:
        _policy_error(
            errors,
            code=code,
            message=message,
            expected=expected,
            actual=actual,
        )


_SPEED_SLICE_COMMON_ENV = {
    "ENABLE_QKV_NKI_KERNELS": "1",
    "ENABLE_QKV_CTE_NKI_KERNEL_FUSE_QK_NORM": "1",
    "ENABLE_QKV_CTE_NKI_KERNEL_FUSE_ROPE": "0",
    "ENABLE_OUT_PROJ_NKI_KERNEL": "0",
    "ENABLE_KV_CACHE_QUANT": "0",
    "PREFIX_CTE_ATTENTION_BACKEND": "attention_cte",
    "QWEN36_DELTANET_FUSED_SEGMENT_TOKENS": "0",
    "QWEN36_DELTANET_MULTIHEAD_CTE": "0",
    "QWEN36_DELTANET_SOLVE_MODE": "direct",
    "QWEN36_DELTANET_SOLVE_SCAN_STEPS": "0",
    "GDN_RECURRENT_CACHE_DTYPE": "bfloat16",
    "GDN_CONV_CACHE_DTYPE": "bfloat16",
    "SEQ_LEN": "32768",
    "MAX_CONTEXT_LENGTH": "32768",
    "FP8_QUANTIZE_LINEAR_ATTN_GATES": "1",
}

_SPEED_SLICE_SPECIFIC_ENV = {
    "sampletokonly": {
        "DISABLE_ON_DEVICE_SAMPLING": "0",
        "OUTPUT_LOGITS_WITH_ON_DEVICE_SAMPLING": "0",
        "QUANTIZE_LM_HEAD": "1",
    },
    "hostlogits": {
        "DISABLE_ON_DEVICE_SAMPLING": "1",
        "OUTPUT_LOGITS_WITH_ON_DEVICE_SAMPLING": "0",
        "QUANTIZE_LM_HEAD": "1",
    },
    "hostlogits_lmheadbf16": {
        "DISABLE_ON_DEVICE_SAMPLING": "1",
        "OUTPUT_LOGITS_WITH_ON_DEVICE_SAMPLING": "0",
        "QUANTIZE_LM_HEAD": "0",
    },
}


def _speed_slice_from_env(env_values: dict[str, str]) -> tuple[str, str]:
    speed_slice = env_values.get("SPEED_SLICE", "").strip()
    if speed_slice and speed_slice != "none":
        return speed_slice, "speed_slice"
    base = env_values.get("BASE", "").lower()
    sampling = env_values.get("SAMPLING", "").lower()
    if "hostlogits_lmheadbf16" in base:
        return "hostlogits_lmheadbf16", "base"
    if "hostlogits" in base or sampling == "host_logits":
        return "hostlogits", "base" if "hostlogits" in base else "sampling"
    if "sampletokonly" in base or "sampletokonly" in sampling:
        return "sampletokonly", "base" if "sampletokonly" in base else "sampling"
    return "none", "none"


def _env_policy_value(env_values: dict[str, str], key: str) -> str | None:
    if key in env_values:
        return env_values[key].strip()
    if key == "DISABLE_ON_DEVICE_SAMPLING":
        sampling = env_values.get("SAMPLING", "").strip().lower()
        if sampling == "host_logits":
            return "1"
        if sampling.startswith("on_device"):
            return "0"
    return None


def _speed_slice_policy_errors(
    env_values: dict[str, str],
    *,
    speed_slice: str,
) -> list[dict[str, Any]]:
    errors: list[dict[str, Any]] = []
    if not speed_slice or speed_slice == "none":
        return errors
    specific = _SPEED_SLICE_SPECIFIC_ENV.get(speed_slice)
    if specific is None:
        _policy_error(
            errors,
            code="unknown_speed_slice",
            message="SPEED_SLICE is not a recognized coherent-speed slice",
            expected=sorted(_SPEED_SLICE_SPECIFIC_ENV),
            actual=speed_slice,
        )
        return errors

    for key, expected in {
        **_SPEED_SLICE_COMMON_ENV,
        **specific,
    }.items():
        actual = _env_policy_value(env_values, key)
        if actual is None or actual != expected:
            _policy_error(
                errors,
                code=f"speed_slice_{key.lower()}_mismatch",
                message=f"SPEED_SLICE={speed_slice} requires {key}={expected}",
                expected=expected,
                actual="<missing>" if actual is None else actual,
            )
    cte_buckets = _ints(env_values.get("CTE_BUCKETS"))
    if cte_buckets != [2048]:
        _policy_error(
            errors,
            code="speed_slice_cte_buckets_mismatch",
            message=f"SPEED_SLICE={speed_slice} requires CTE_BUCKETS=2048",
            expected=[2048],
            actual="<missing>" if "CTE_BUCKETS" not in env_values else cte_buckets,
        )
    return errors


def _policy_errors_from_env(
    *,
    config: dict[str, Any],
    env_values: dict[str, str],
    summary: dict[str, Any],
) -> list[dict[str, Any]]:
    errors: list[dict[str, Any]] = []
    if not env_values:
        return errors
    speed_slice, _ = _speed_slice_from_env(env_values)
    errors.extend(_speed_slice_policy_errors(env_values, speed_slice=speed_slice))

    int_fields = [
        ("SEQ_LEN", "seq_len", summary["seq_len"]),
        (
            "MAX_CONTEXT_LENGTH",
            "max_context_length",
            _first_config_value(config, "max_context_length", default=None),
        ),
        (
            "MAX_GDN_CHECKPOINT_SLOTS",
            "max_gdn_checkpoint_slots",
            summary["max_gdn_checkpoint_slots"],
        ),
    ]
    for env_key, config_key, actual in int_fields:
        if env_key in env_values and actual is not None:
            _require_equal(
                errors,
                code=f"{config_key}_mismatch",
                message=f"{config_key} does not match compile env {env_key}",
                expected=int(env_values[env_key]),
                actual=int(actual),
            )

    list_fields = [
        ("CTE_BUCKETS", "context_encoding_buckets", summary["context_encoding_buckets"]),
        (
            "TOKEN_GENERATION_BUCKETS",
            "token_generation_buckets",
            summary["token_generation_buckets"],
        ),
        ("PREFIX_BUCKETS", "prefix_buckets", summary["prefix_buckets"]),
    ]
    for env_key, config_key, actual in list_fields:
        if env_key in env_values:
            _require_equal(
                errors,
                code=f"{config_key}_mismatch",
                message=f"{config_key} does not match compile env {env_key}",
                expected=sorted(_ints(env_values[env_key])),
                actual=sorted(_ints(actual)),
            )

    if "CONTEXT_ENCODING_BUCKET_PAIRS" in env_values:
        actual_pairs = _first_config_value(
            config,
            "context_encoding_bucket_pairs",
            default=[],
        )
        _require_equal(
            errors,
            code="context_encoding_bucket_pairs_mismatch",
            message="context_encoding_bucket_pairs do not match compile env",
            expected=sorted(_pair_tokens(env_values["CONTEXT_ENCODING_BUCKET_PAIRS"])),
            actual=sorted(_pair_tokens(actual_pairs)),
        )

    dtype_fields = [
        (
            "GDN_RECURRENT_CACHE_DTYPE",
            "gdn_recurrent_cache_dtype",
            _first_config_value(
                config,
                "gdn_recurrent_cache_dtype",
                "hybrid_recurrent_cache_dtype",
                default=None,
            ),
        ),
        (
            "GDN_CONV_CACHE_DTYPE",
            "gdn_conv_cache_dtype",
            _first_config_value(
                config,
                "gdn_conv_cache_dtype",
                "hybrid_conv_cache_dtype",
                default=None,
            ),
        ),
    ]
    for env_key, config_key, actual in dtype_fields:
        if env_key in env_values:
            _require_equal(
                errors,
                code=f"{config_key}_mismatch",
                message=f"{config_key} does not match compile env {env_key}",
                expected=_normal_dtype(env_values[env_key]),
                actual=_normal_dtype(actual),
            )

    bool_fields = [
        (
            "ENABLE_QKV_NKI_KERNELS",
            "qkv_nki_kernel_enabled",
            _bool_config(config, "qkv_nki_kernel_enabled", "qkv_kernel_enabled"),
        ),
        (
            "ENABLE_QKV_CTE_NKI_KERNEL_FUSE_ROPE",
            "qkv_cte_nki_kernel_fuse_rope",
            _bool_config(config, "qkv_cte_nki_kernel_fuse_rope"),
        ),
        (
            "ENABLE_QKV_CTE_NKI_KERNEL_FUSE_QK_NORM",
            "qkv_cte_nki_kernel_fuse_qk_norm",
            _bool_config(config, "qkv_cte_nki_kernel_fuse_qk_norm"),
        ),
        (
            "ENABLE_OUT_PROJ_NKI_KERNEL",
            "out_proj_kernel_enabled",
            _bool_config(config, "out_proj_kernel_enabled"),
        ),
        (
            "ENABLE_KV_CACHE_QUANT",
            "kv_cache_quant",
            _bool_config(config, "kv_cache_quant"),
        ),
        (
            "DISABLE_CONTEXT_ENCODING_ARGMAX_KERNEL",
            "disable_context_encoding_argmax_kernel",
            _bool_config(config, "disable_context_encoding_argmax_kernel"),
        ),
    ]
    for env_key, config_key, actual in bool_fields:
        if env_key in env_values:
            _require_equal(
                errors,
                code=f"{config_key}_mismatch",
                message=f"{config_key} does not match compile env {env_key}",
                expected=_normal_bool(env_values[env_key]),
                actual=actual,
            )

    if "PREFIX_CTE_ATTENTION_BACKEND" in env_values:
        _require_equal(
            errors,
            code="prefix_cte_attention_backend_mismatch",
            message="prefix_cte_attention_backend does not match compile env",
            expected=env_values["PREFIX_CTE_ATTENTION_BACKEND"],
            actual=_first_config_value(config, "prefix_cte_attention_backend", default=None),
        )
    if "PREFIX_CTE_ATTENTION_SEGMENT_SIZE" in env_values:
        _require_equal(
            errors,
            code="prefix_cte_attention_segment_size_mismatch",
            message="prefix_cte_attention_segment_size does not match compile env",
            expected=int(env_values["PREFIX_CTE_ATTENTION_SEGMENT_SIZE"]),
            actual=int(
                _first_config_value(
                    config,
                    "prefix_cte_attention_segment_size",
                    default=0,
                )
                or 0
            ),
        )

    if "DISABLE_ON_DEVICE_SAMPLING" in env_values:
        disable_on_device = _normal_bool(env_values["DISABLE_ON_DEVICE_SAMPLING"])
        expected_on_device = not disable_on_device
        _require_equal(
            errors,
            code="on_device_sampling_mismatch",
            message="on-device sampling presence does not match compile env",
            expected=expected_on_device,
            actual=summary["on_device_sampling"],
        )
        expected_output_logits = (
            True
            if disable_on_device
            else _normal_bool(env_values.get("OUTPUT_LOGITS_WITH_ON_DEVICE_SAMPLING"))
        )
        _require_equal(
            errors,
            code="output_logits_mismatch",
            message="output_logits does not match sampling compile env",
            expected=expected_output_logits,
            actual=summary["output_logits"],
        )
        if expected_on_device:
            _require_equal(
                errors,
                code="vocab_parallel_mismatch",
                message="on-device sampling should compile vocab_parallel=true",
                expected=True,
                actual=_bool_config(config, "vocab_parallel"),
            )

    if "QUANTIZE_LM_HEAD" in env_values:
        modules_to_not_convert = _first_config_value(
            config,
            "modules_to_not_convert",
            default=[],
        )
        if isinstance(modules_to_not_convert, list):
            excludes_lm_head = any(
                str(item) in {"lm_head", "model.lm_head"}
                for item in modules_to_not_convert
            )
            _require_equal(
                errors,
                code="lm_head_quant_policy_mismatch",
                message="modules_to_not_convert lm_head policy does not match QUANTIZE_LM_HEAD",
                expected=not _normal_bool(env_values["QUANTIZE_LM_HEAD"]),
                actual=excludes_lm_head,
            )

    return errors


def audit(
    *,
    artifact: Path,
    compile_log: Path | None,
    env_log: Path | None = None,
    recommended_block_size: int,
    min_usable_headroom_blocks: int,
    strict_hybrid_gate: bool,
) -> dict[str, Any]:
    config = _load_config(artifact)
    env_values = _parse_env_log(env_log)
    seq_len = int(_first_config_value(config, "seq_len", "max_length", default=0) or 0)
    max_num_seqs = int(_first_config_value(config, "batch_size", default=1) or 1)
    ctx_batch_size = int(_first_config_value(config, "ctx_batch_size", default=1) or 1)
    block_size = int(_first_config_value(config, "pa_block_size", default=0) or 0)
    pa_num_blocks = int(_first_config_value(config, "pa_num_blocks", default=0) or 0)
    max_gdn_slots = int(
        _first_config_value(config, "max_gdn_checkpoint_slots", default=0) or 0
    )
    cte_buckets = _first_config_value(config, "context_encoding_buckets", default=[])
    token_generation_buckets = _first_config_value(
        config,
        "token_generation_buckets",
        default=[],
    )
    prefix_buckets = _first_config_value(config, "prefix_buckets", default=[])
    tkg_batch_size = int(_first_config_value(config, "tkg_batch_size", default=1) or 1)
    async_mode = _bool_config(config, "async_mode")
    output_logits = _bool_config(config, "output_logits")
    on_device_sampling_config = _first_config_value(
        config,
        "on_device_sampling_config",
        default=None,
    )
    min_blocks = (
        max(1, math.ceil(seq_len / block_size) * max_num_seqs)
        if seq_len > 0 and block_size > 0
        else 0
    )
    usable_headroom_blocks = pa_num_blocks - min_blocks if pa_num_blocks else None
    usable_headroom_blocks = (
        max(0, usable_headroom_blocks)
        if usable_headroom_blocks is not None
        else None
    )
    required_full_prompt_boundaries = (
        math.ceil(seq_len / block_size) if seq_len > 0 and block_size > 0 else 0
    )
    compile_backend = _compile_backend_from_log(compile_log)
    if compile_backend is None and "nki_chunked" in str(artifact):
        compile_backend = "nki_chunked_from_artifact_name"

    warnings: list[dict[str, Any]] = []
    if block_size and recommended_block_size and block_size != recommended_block_size:
        _warning(
            warnings,
            code="non_recommended_block_size",
            message=(
                "Artifact PA block size differs from the configured Neuron "
                "performance recommendation."
            ),
            value={"pa_block_size": block_size, "recommended": recommended_block_size},
        )
    if (
        usable_headroom_blocks is not None
        and usable_headroom_blocks < min_usable_headroom_blocks
    ):
        _warning(
            warnings,
            code="low_pa_headroom",
            message=(
                "PA block capacity has little usable residency headroom after "
                "minimum sequence capacity."
            ),
            value={
                "pa_num_blocks": pa_num_blocks,
                "min_blocks": min_blocks,
                "usable_headroom_blocks": usable_headroom_blocks,
                "minimum_expected": min_usable_headroom_blocks,
            },
        )
    if strict_hybrid_gate and max_gdn_slots and required_full_prompt_boundaries > max_gdn_slots:
        _warning(
            warnings,
            code="strict_gate_boundary_slots_exceed_gdn_slots",
            message=(
                "With the current disable-unbacked-prefix-reads gate, a full "
                "prompt can require more backed prefix boundaries than the GDN "
                "checkpoint slot budget can hold unless boundary chunk commits "
                "or a less conservative gate are used."
            ),
            value={
                "required_full_prompt_boundaries": required_full_prompt_boundaries,
                "max_gdn_checkpoint_slots": max_gdn_slots,
            },
        )
    if compile_backend and "nki_chunked" in compile_backend:
        _warning(
            warnings,
            code="nki_chunked_deltanet_cte",
            message=(
                "Compile log or artifact name indicates the nki_chunked DeltaNet "
                "CTE backend; compare against a fused-control artifact."
            ),
            value=compile_backend,
        )
    if (
        seq_len >= 32768
        and isinstance(token_generation_buckets, list)
        and token_generation_buckets == [seq_len]
    ):
        _warning(
            warnings,
            code="single_full_length_tkg_bucket",
            message=(
                "Decode has only a full-length token-generation bucket. Short "
                "generations will still use the largest TKG trace shape; compare "
                "against an artifact compiled with smaller TKG buckets such as "
                "8192,32768,seq_len."
            ),
            value=token_generation_buckets,
        )
    if not async_mode:
        _warning(
            warnings,
            code="sync_neuron_runtime_decode",
            message=(
                "Neuron async_mode is disabled. The previous fast decode control "
                "path used async runtime execution for token generation."
            ),
            value=False,
        )
    if tkg_batch_size <= 1:
        _warning(
            warnings,
            code="single_sequence_tkg_batch",
            message=(
                "tkg_batch_size is 1, so decode cannot amortize per-token runner "
                "overhead across concurrent sequences."
            ),
            value=tkg_batch_size,
        )

    summary = {
        "artifact": str(artifact),
        "compile_log": str(compile_log) if compile_log is not None else None,
        "env_log": str(env_log) if env_log is not None else None,
        "seq_len": seq_len,
        "max_context_length": int(
            _first_config_value(config, "max_context_length", default=0) or 0
        ),
        "max_num_seqs": max_num_seqs,
        "ctx_batch_size": ctx_batch_size,
        "pa_block_size": block_size,
        "pa_num_blocks": pa_num_blocks,
        "pa_min_blocks": min_blocks,
        "pa_usable_headroom_blocks": usable_headroom_blocks,
        "max_gdn_checkpoint_slots": max_gdn_slots,
        "required_full_prompt_boundaries": required_full_prompt_boundaries,
        "context_encoding_buckets": cte_buckets,
        "token_generation_buckets": token_generation_buckets,
        "tkg_batch_size": tkg_batch_size,
        "async_mode": async_mode,
        "output_logits": output_logits,
        "on_device_sampling": on_device_sampling_config is not None,
        "speed_slice": _speed_slice_from_env(env_values)[0] if env_values else None,
        "speed_slice_source": _speed_slice_from_env(env_values)[1]
        if env_values
        else None,
        "prefix_buckets": prefix_buckets,
        "is_prefix_caching": _bool_config(config, "is_prefix_caching"),
        "use_hybrid_apc_manager": _bool_config(config, "use_hybrid_apc_manager"),
        "use_qwen_hybrid_chunked_prefill": _bool_config(
            config,
            "use_qwen_hybrid_chunked_prefill",
        ),
        "deltanet_cte_backend": compile_backend,
        "warnings": warnings,
    }
    summary["policy_errors"] = _policy_errors_from_env(
        config=config,
        env_values=env_values,
        summary=summary,
    )
    summary["policy_error_count"] = len(summary["policy_errors"])
    summary["policy_passed"] = not summary["policy_errors"]
    summary["warning_count"] = len(warnings)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("artifact", help="Artifact directory or neuron_config.json path")
    parser.add_argument("--compile-log", type=Path, default=None)
    parser.add_argument("--env-log", type=Path, default=None)
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument("--recommended-block-size", type=int, default=32)
    parser.add_argument("--min-usable-headroom-blocks", type=int, default=8)
    parser.add_argument(
        "--no-strict-hybrid-gate",
        action="store_true",
        help="Do not warn when full-prompt boundary count exceeds GDN slot count.",
    )
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()

    summary = audit(
        artifact=Path(args.artifact).expanduser().resolve(),
        compile_log=args.compile_log.expanduser().resolve()
        if args.compile_log is not None
        else None,
        env_log=args.env_log.expanduser().resolve() if args.env_log is not None else None,
        recommended_block_size=args.recommended_block_size,
        min_usable_headroom_blocks=args.min_usable_headroom_blocks,
        strict_hybrid_gate=not args.no_strict_hybrid_gate,
    )
    encoded = json.dumps(summary, indent=2, sort_keys=True) + "\n"
    if args.output_json is not None:
        args.output_json.expanduser().parent.mkdir(parents=True, exist_ok=True)
        args.output_json.expanduser().write_text(encoded)
    print(encoded, end="")
    if summary["policy_errors"]:
        return 1
    return 1 if args.strict and summary["warnings"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
