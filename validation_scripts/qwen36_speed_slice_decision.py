#!/usr/bin/env python3
"""Decide the next Qwen3.6 speed-slice action from validation outputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


_NEXT_SLICE = {
    "none": "sampletokonly",
    "sampletokonly": "hostlogits",
    "hostlogits": "hostlogits_lmheadbf16",
}

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


def _required_flags(next_slice: str | None) -> dict[str, str]:
    if next_slice is None:
        return {}
    return dict(_SLICE_FLAGS.get(next_slice, {}))


def decide(
    *,
    env_values: dict[str, str],
    runtime_summary: dict[str, Any],
    speed_output: dict[str, Any] | None,
    speed_json_path: Path | None,
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
        "speed_json": str(speed_json_path) if speed_json_path else None,
        "speed_gate": speed_gate,
        "prefill_tok_s_mean": mean_speed,
        "decision": None,
        "next_speed_slice": None,
        "next_required_flags": {},
        "reason": None,
    }

    if not coherence_ok:
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
                "reason": "coherent_but_last_planned_speed_slice_is_still_slow",
            }
        )
        return result

    result.update(
        {
            "decision": "launch_next_speed_slice",
            "next_speed_slice": next_slice,
            "next_required_flags": _required_flags(next_slice),
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
    )
    encoded = json.dumps(decision, indent=2, sort_keys=True) + "\n"
    if args.output_json is not None:
        args.output_json.expanduser().parent.mkdir(parents=True, exist_ok=True)
        args.output_json.expanduser().write_text(encoded)
    print(encoded, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
