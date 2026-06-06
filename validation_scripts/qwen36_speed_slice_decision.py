#!/usr/bin/env python3
"""Decide the next Qwen3.6 speed-slice action from validation outputs."""

from __future__ import annotations

import argparse
import json
import re
import shlex
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


_NEXT_SLICE = {
    "none": "sampletokonly",
    "sampletokonly": "hostlogits",
    "hostlogits": "hostlogits_lmheadbf16",
}
DEFAULT_BOUNDARY_LENGTHS = (
    "123,146,160,485,505,526,1225,1265,1346,2048,2049,2500,4092,4096"
)

_SLICE_FLAGS = {
    "sampletokonly": {
        "SPEED_SLICE": "sampletokonly",
        "OUTPUT_LOGITS_WITH_ON_DEVICE_SAMPLING": "0",
    },
    "hostlogits": {
        "SPEED_SLICE": "hostlogits",
        "DISABLE_ON_DEVICE_SAMPLING": "1",
        "OUTPUT_LOGITS_WITH_ON_DEVICE_SAMPLING": "0",
        "QUANTIZE_LM_HEAD": "1",
    },
    "hostlogits_lmheadbf16": {
        "SPEED_SLICE": "hostlogits_lmheadbf16",
        "DISABLE_ON_DEVICE_SAMPLING": "1",
        "OUTPUT_LOGITS_WITH_ON_DEVICE_SAMPLING": "0",
        "QUANTIZE_LM_HEAD": "0",
    },
}

_ANCHOR_FLAGS = {
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
    "CTE_BUCKETS_RAW": "2048",
    "SEQ_LEN": "32768",
    "MAX_CONTEXT_LENGTH": "32768",
    "FP8_QUANTIZE_LINEAR_ATTN_GATES": "1",
}

_INHERIT_ENV_KEYS = [
    "REPO",
    "MODEL",
    "ART_ROOT",
    "LOGDIR",
    "NKI_LIBRARY_SRC",
    "NEURON_PLATFORM_TARGET_OVERRIDE",
    "NEURON_CC_FLAGS",
    "MAX_GDN_CHECKPOINT_SLOTS",
]


def parse_env_log(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for raw_line in path.read_text(errors="replace").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key] = value
    return values


def _load_json(path: Path) -> dict[str, Any]:
    with path.open() as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def _normal_slice(values: dict[str, str]) -> str:
    raw = values.get("SPEED_SLICE", "none").strip()
    return raw if raw else "none"


def _speed_json_from_summary(
    summary: dict[str, Any],
    override: Path | None,
) -> Path | None:
    if override is not None:
        return override
    for row in summary.get("results", []):
        if not isinstance(row, dict):
            continue
        if row.get("name") == "raw_prefill_speed" and row.get("output_path"):
            return Path(str(row["output_path"]))
    output_dir = summary.get("output_dir")
    if isinstance(output_dir, str):
        return Path(output_dir) / "raw_prefill_speed.json"
    return None


def _speed_gate(speed: dict[str, Any] | None) -> dict[str, Any] | None:
    if speed is None:
        return None
    gate = speed.get("speed_gate")
    return gate if isinstance(gate, dict) else None


def _summary_results(summary: dict[str, Any]) -> list[dict[str, Any]]:
    rows = summary.get("results")
    return [row for row in rows if isinstance(row, dict)] if isinstance(rows, list) else []


def _failed_runtime_gate_rows(summary: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        row
        for row in _summary_results(summary)
        if row.get("phase") in {"coherence", "log_scan"}
        and not bool(row.get("skipped"))
        and not bool(row.get("passed"))
    ]


def _runtime_validation_gap_reason(summary: dict[str, Any]) -> str | None:
    if _failed_runtime_gate_rows(summary):
        return None
    gate_counts = summary.get("gate_counts")
    if isinstance(gate_counts, dict):
        total_gate_steps = int(gate_counts.get("total_gate_steps") or 0)
        total_log_scan_steps = int(gate_counts.get("total_log_scan_steps") or 0)
        passed_gate_steps = int(gate_counts.get("passed_gate_steps") or 0)
        if total_log_scan_steps == 0:
            return "runtime_log_scan_not_run"
        if passed_gate_steps < total_gate_steps:
            return "coherence_or_log_scan_not_completed"
    for row in _summary_results(summary):
        if row.get("phase") != "speed" or not bool(row.get("skipped")):
            continue
        reason = str(row.get("skip_reason") or "")
        if reason in {"runtime_log_scan_not_run", "coherence_or_log_scan_not_completed"}:
            return reason
    return "coherence_or_log_scan_not_completed"


def _required_flags(next_slice: str | None) -> dict[str, str]:
    if next_slice is None:
        return {}
    return dict(_SLICE_FLAGS.get(next_slice, {}))


def _quote_command(command: list[str]) -> str:
    return " ".join(shlex.quote(item) for item in command)


def _quote_env_command(env: dict[str, str], command: list[str]) -> str:
    parts = [f"{key}={shlex.quote(value)}" for key, value in env.items()]
    parts.extend(shlex.quote(item) for item in command)
    return " ".join(parts)


def _default_next_ts(next_slice: str) -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{stamp}_{next_slice}"


def _automation_name(next_slice: str, ts: str) -> str:
    slug = _slug_with_suffix(f"qwen-{next_slice}", ts)
    return f"monitor-{slug}-compile"


def _context_tokens_from_env(env_values: dict[str, str]) -> int:
    raw = env_values.get("CTE_BUCKETS_RAW") or env_values.get("CTE_BUCKETS") or "2048"
    buckets = [int(item) for item in re.findall(r"\d+", raw)]
    return max(buckets) if buckets else 2048


def profile_preflight(
    env_values: dict[str, str],
    *,
    speed_json_path: Path | None,
    target_prefill_tok_s: float,
    profile_ts: str | None = None,
) -> dict[str, Any]:
    ts = profile_ts or _default_next_ts("profile")
    workdir = env_values.get("WORKDIR")
    logdir = env_values.get("LOGDIR")
    context_neff_root = (
        str(Path(workdir) / "context_encoding_model")
        if workdir
        else "<WORKDIR>/context_encoding_model"
    )
    output_dir = (
        str(Path(logdir) / f"context_neff_profile_{ts}")
        if logdir
        else f"<PROFILE_OUTPUT_PARENT>/context_neff_profile_{ts}"
    )
    context_tokens = _context_tokens_from_env(env_values)
    profile_base_command = [
        "python3",
        "validation_scripts/qwen36_context_neff_profile.py",
        "--context-neff-root",
        context_neff_root,
        "--output-dir",
        output_dir,
        "--buckets",
        "0,7",
        "--enable-dge",
    ]
    profile_run_command = [*profile_base_command, "--run"]
    compare_command = [
        "python3",
        "validation_scripts/qwen36_profile_summary_compare.py",
        "--summary",
        "current=<SUMMARY_JSON_FROM_context_neff_profile_results>",
        "--prompt-tokens",
        "16384",
        "--context-tokens",
        str(context_tokens),
        "--target-prefill-tok-s",
        str(float(target_prefill_tok_s)),
        "--output-json",
        str(Path(output_dir) / "profile_summary_compare.json"),
    ]
    if speed_json_path is not None:
        compare_command.extend(["--speed-json", str(speed_json_path)])
    return {
        "ts": ts,
        "reason": "coherent_but_all_planned_speed_slices_below_target",
        "do_not_profile_live_vllm": True,
        "profile_requires_idle_neuron_cores": True,
        "context_neff_root": context_neff_root,
        "output_dir": output_dir,
        "buckets": [0, 7],
        "context_tokens": context_tokens,
        "target_prefill_tok_s": float(target_prefill_tok_s),
        "plan_command": _quote_command(profile_base_command),
        "run_command": _quote_command(profile_run_command),
        "compare_command_template": _quote_command(compare_command),
        "run_order": [
            "stop_vllm_or_use_idle_trainium_host",
            "run_plan_command_and_verify_targets",
            "run_profile_command",
            "choose_summary_json_from_context_neff_profile_results",
            "run_compare_command_template_with_summary_json",
        ],
    }


def _slug(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9]+", "-", value).strip("-").lower()


def _slug_with_suffix(head: str, suffix: str, max_len: int = 56) -> str:
    head_slug = _slug(head)
    suffix_slug = _slug(suffix)
    if not suffix_slug:
        return head_slug[:max_len]
    if len(suffix_slug) >= max_len:
        return suffix_slug[-max_len:]
    available_head = max_len - len(suffix_slug) - 1
    if available_head <= 0:
        return suffix_slug[-max_len:]
    joined = f"{head_slug[:available_head].strip('-')}-{suffix_slug}".strip("-")
    return joined[:max_len]


def next_preflight(
    env_values: dict[str, str],
    next_slice: str,
    *,
    next_ts: str | None = None,
    boundary_lengths: str | None = None,
) -> dict[str, Any]:
    env: dict[str, str] = {
        key: env_values[key] for key in _INHERIT_ENV_KEYS if key in env_values
    }
    ts = next_ts or _default_next_ts(next_slice)
    env["TS"] = ts
    env.update(_ANCHOR_FLAGS)
    env.update(_SLICE_FLAGS[next_slice])
    env["COMPILE_DRY_RUN"] = "1"
    env.setdefault("MAX_GDN_CHECKPOINT_SLOTS", "64")
    resolved_boundary_lengths = boundary_lengths or DEFAULT_BOUNDARY_LENGTHS
    boundary_arg = shlex.quote(resolved_boundary_lengths)
    dry_run_command = _quote_env_command(
        env,
        ["bash", "tmp_compile_qwen32k_segcte2048_gdnseg512.sh"],
    )
    launch_env = dict(env)
    launch_env["COMPILE_DRY_RUN"] = "0"
    automation_name = _automation_name(next_slice, ts)
    next_compile_preflight_command = (
        "python3 validation_scripts/qwen36_next_compile_preflight.py "
        "--decision-json <SPEED_SLICE_DECISION_JSON> "
        "--output-json <NEXT_COMPILE_PREFLIGHT_JSON> "
        "--automation-json-output <AUTOMATION_PAYLOAD_JSON> "
        f"--boundary-lengths {boundary_arg}"
    )
    compile_command_release_command = (
        "python3 validation_scripts/qwen36_next_compile_preflight.py "
        "--decision-json <SPEED_SLICE_DECISION_JSON> "
        "--output-json <NEXT_COMPILE_RELEASE_JSON> "
        "--automation-created-name <CREATED_AUTOMATION_NAME> "
        f"--boundary-lengths {boundary_arg}"
    )
    return {
        "ts": ts,
        "boundary_lengths": resolved_boundary_lengths,
        "compile_driver": "tmp_compile_qwen32k_segcte2048_gdnseg512.sh",
        "dry_run_env": dict(env),
        "dry_run_command": dry_run_command,
        "launch_env": dict(launch_env),
        "next_compile_preflight_command_template": next_compile_preflight_command,
        "compile_command_release_command_template": compile_command_release_command,
        "automation_payload_command_template": (
            "python3 validation_scripts/qwen36_compile_monitor_prompt.py "
            "--env-log <ENVLOG_FROM_DRY_RUN> "
            f"--automation-json --automation-name {shlex.quote(automation_name)}"
        ),
        "compile_command_after_automation": None,
        "launch_command_after_automation": None,
        "run_order": [
            "run_next_compile_preflight_command",
            "create_heartbeat_automation_from_payload_json",
            "run_compile_command_release_command",
            "run_compile_command_after_automation_exists",
        ],
    }


def decide(
    *,
    env_values: dict[str, str],
    runtime_summary: dict[str, Any],
    speed_output: dict[str, Any] | None,
    speed_json_path: Path | None,
    next_ts: str | None = None,
    boundary_lengths: str | None = None,
) -> dict[str, Any]:
    current_slice = _normal_slice(env_values)
    coherence_ok = bool(runtime_summary.get("coherence_and_log_scan_passed"))
    runtime_passed = bool(runtime_summary.get("passed"))
    speed_gate = _speed_gate(speed_output)
    mean_speed = (
        speed_output.get("prefill_tok_s_mean")
        if isinstance(speed_output, dict)
        else None
    )

    result: dict[str, Any] = {
        "schema": "qwen36-speed-slice-decision-v1",
        "artifact": env_values.get("ARTIFACT"),
        "base": env_values.get("BASE"),
        "current_speed_slice": current_slice,
        "runtime_summary_passed": runtime_passed,
        "coherence_and_log_scan_passed": coherence_ok,
        "failed_runtime_gate_count": len(_failed_runtime_gate_rows(runtime_summary)),
        "speed_json": str(speed_json_path) if speed_json_path else None,
        "speed_gate": speed_gate,
        "prefill_tok_s_mean": mean_speed,
        "decision": None,
        "next_speed_slice": None,
        "next_required_flags": {},
        "next_preflight": None,
        "profile_preflight": None,
        "reason": None,
    }

    if not coherence_ok:
        validation_gap = _runtime_validation_gap_reason(runtime_summary)
        if validation_gap is not None:
            result.update(
                {
                    "decision": "rerun_runtime_validation",
                    "reason": validation_gap,
                }
            )
            return result
        result.update(
            {
                "decision": "stop_incoherent",
                "reason": "coherence_or_runtime_log_scan_failed",
            }
        )
        return result

    if speed_output is None or speed_gate is None:
        result.update(
            {
                "decision": "rerun_speed_validation",
                "reason": "missing_or_unreadable_speed_output",
            }
        )
        return result

    if not bool(speed_output.get("row_gate_passed", False)):
        result.update(
            {
                "decision": "rerun_speed_validation",
                "reason": "speed_rows_failed_before_speed_gate",
            }
        )
        return result

    if bool(speed_gate.get("passed")):
        result.update(
            {
                "decision": "candidate_meets_target",
                "reason": "coherence_log_scan_and_speed_gate_passed",
            }
        )
        return result

    if speed_gate.get("failure_reason") == "no_valid_prefill_speed":
        result.update(
            {
                "decision": "rerun_speed_validation",
                "reason": "no_valid_prefill_speed",
            }
        )
        return result

    next_slice = _NEXT_SLICE.get(current_slice)
    if next_slice is None:
        result.update(
            {
                "decision": "profile_slow_coherent",
                "profile_preflight": profile_preflight(
                    env_values,
                    speed_json_path=speed_json_path,
                    target_prefill_tok_s=float(
                        speed_gate.get("min_prefill_tok_s") or 3000.0
                    ),
                    profile_ts=next_ts,
                ),
                "reason": "coherent_but_last_planned_speed_slice_is_still_slow",
            }
        )
        return result

    result.update(
        {
            "decision": "launch_next_speed_slice",
            "next_speed_slice": next_slice,
            "next_required_flags": _required_flags(next_slice),
            "next_preflight": next_preflight(
                env_values,
                next_slice,
                next_ts=next_ts,
                boundary_lengths=boundary_lengths,
            ),
            "reason": "coherent_but_prefill_speed_below_target",
        }
    )
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-log", type=Path, required=True)
    parser.add_argument("--runtime-summary", type=Path, required=True)
    parser.add_argument(
        "--speed-json",
        type=Path,
        default=None,
        help="Override raw_prefill_speed.json path; defaults to runtime summary output.",
    )
    parser.add_argument(
        "--next-ts",
        default=None,
        help="Timestamp/suffix to reuse for generated next-slice dry-run and launch commands.",
    )
    parser.add_argument(
        "--boundary-lengths",
        default=DEFAULT_BOUNDARY_LENGTHS,
        help="Exact boundary lengths to preserve in generated next-compile automation.",
    )
    parser.add_argument("--output-json", type=Path, default=None)
    args = parser.parse_args()

    env_values = parse_env_log(args.env_log)
    runtime_summary = _load_json(args.runtime_summary)
    speed_path = _speed_json_from_summary(runtime_summary, args.speed_json)
    speed_output = None
    if speed_path is not None and speed_path.exists():
        speed_output = _load_json(speed_path)
    decision = decide(
        env_values=env_values,
        runtime_summary=runtime_summary,
        speed_output=speed_output,
        speed_json_path=speed_path,
        next_ts=args.next_ts,
        boundary_lengths=args.boundary_lengths,
    )
    encoded = json.dumps(decision, indent=2, sort_keys=True) + "\n"
    if args.output_json is not None:
        args.output_json.expanduser().parent.mkdir(parents=True, exist_ok=True)
        args.output_json.expanduser().write_text(encoded)
    print(encoded, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
