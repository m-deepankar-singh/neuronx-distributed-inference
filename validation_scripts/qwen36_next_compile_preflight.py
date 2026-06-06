#!/usr/bin/env python3
"""Prepare the next Qwen3.6 compile and its heartbeat automation payload."""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any


_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from validation_scripts import qwen36_compile_monitor_prompt as monitor_prompt


def _load_json(path: Path) -> dict[str, Any]:
    with path.open() as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.expanduser().parent.mkdir(parents=True, exist_ok=True)
    path.expanduser().write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _parse_assignments(text: str) -> dict[str, str]:
    values: dict[str, str] = {}
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key] = value
    return values


def _string_map(value: Any, *, key: str) -> dict[str, str]:
    if not isinstance(value, dict):
        raise ValueError(f"next_preflight.{key} must be an object")
    return {str(item_key): str(item_value) for item_key, item_value in value.items()}


def _quote_env_command(env: dict[str, str], command: list[str]) -> str:
    parts = [f"{key}={shlex.quote(value)}" for key, value in env.items()]
    parts.extend(shlex.quote(item) for item in command)
    return " ".join(parts)


def _quote_command(command: list[str]) -> str:
    return " ".join(shlex.quote(item) for item in command)


def _driver_path(repo_root: Path, driver: str) -> Path:
    raw = Path(driver)
    path = raw if raw.is_absolute() else repo_root / raw
    resolved = path.expanduser().resolve()
    repo_resolved = repo_root.expanduser().resolve()
    if resolved != repo_resolved and repo_resolved not in resolved.parents:
        raise ValueError(f"compile driver must be under repo root: {driver}")
    if not resolved.exists():
        raise ValueError(f"compile driver does not exist: {resolved}")
    return resolved


def _run_dry_run(
    *,
    repo_root: Path,
    driver: str,
    dry_run_env: dict[str, str],
) -> tuple[subprocess.CompletedProcess[str], dict[str, str], Path]:
    if dry_run_env.get("COMPILE_DRY_RUN") != "1":
        raise ValueError("next_preflight.dry_run_env must set COMPILE_DRY_RUN=1")

    driver_path = _driver_path(repo_root, driver)
    env = os.environ.copy()
    env.update(dry_run_env)
    completed = subprocess.run(
        ["bash", str(driver_path)],
        cwd=repo_root,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    stdout_values = _parse_assignments(completed.stdout)
    if completed.returncode != 0:
        stderr_tail = completed.stderr.strip().splitlines()[-20:]
        raise RuntimeError(
            "compile dry run failed with "
            f"exit_code={completed.returncode}; stderr_tail={stderr_tail}"
        )
    env_log = stdout_values.get("ENVLOG")
    if not env_log:
        raise ValueError("compile dry run did not print ENVLOG=<path>")
    env_log_path = Path(env_log).expanduser()
    if not env_log_path.exists():
        raise ValueError(f"compile dry run ENVLOG does not exist: {env_log_path}")
    return completed, stdout_values, env_log_path


def _source_commit(source_dir: str, override: str | None, values: dict[str, str]) -> str:
    if override:
        return override
    if values.get("SOURCE_COMMIT"):
        return values["SOURCE_COMMIT"]
    path = Path(source_dir) if source_dir != "unknown" else None
    return monitor_prompt._git_commit(path)


def build_preflight(
    *,
    decision: dict[str, Any],
    decision_path: Path,
    repo_root: Path,
    compile_host: str,
    runtime_host: str,
    source_dir: str | None,
    source_commit: str | None,
    automation_name: str | None,
    automation_created_name: str | None,
    automation_interval_minutes: int,
    launch_script: str,
    boundary_lengths: str,
) -> dict[str, Any]:
    if decision.get("decision") != "launch_next_speed_slice":
        raise ValueError(
            "decision JSON is not requesting a compile; "
            f"decision={decision.get('decision')!r}"
        )
    preflight = decision.get("next_preflight")
    if not isinstance(preflight, dict):
        raise ValueError("decision JSON is missing next_preflight")

    dry_run_env = _string_map(preflight.get("dry_run_env"), key="dry_run_env")
    launch_env = _string_map(preflight.get("launch_env"), key="launch_env")
    if launch_env.get("COMPILE_DRY_RUN") not in {"0", ""}:
        raise ValueError("next_preflight.launch_env must set COMPILE_DRY_RUN=0")

    driver = str(preflight.get("compile_driver") or "")
    if not driver:
        raise ValueError("next_preflight.compile_driver is required")

    completed, stdout_values, env_log_path = _run_dry_run(
        repo_root=repo_root,
        driver=driver,
        dry_run_env=dry_run_env,
    )
    env_values = monitor_prompt.parse_env_log(env_log_path)
    env_values.setdefault("ENVLOG", str(env_log_path))

    resolved_source_dir = source_dir or env_values.get("REPO", "unknown")
    resolved_source_commit = _source_commit(
        resolved_source_dir,
        source_commit,
        env_values,
    )
    prompt = monitor_prompt.render_prompt(
        env_values,
        compile_host=compile_host,
        runtime_host=runtime_host,
        source_dir=resolved_source_dir,
        source_commit=resolved_source_commit,
        launch_script=launch_script,
        boundary_lengths=boundary_lengths,
    )
    automation_payload = monitor_prompt.build_automation_payload(
        env_values,
        prompt=prompt,
        name=automation_name,
        interval_minutes=automation_interval_minutes,
    )
    compile_command = _quote_env_command(
        launch_env,
        ["bash", driver],
    )

    release_compile_command = automation_created_name is not None
    if release_compile_command and automation_created_name != automation_payload["name"]:
        raise ValueError(
            "automation-created-name does not match generated automation payload "
            f"name: {automation_created_name!r} != {automation_payload['name']!r}"
        )
    release_command = [
        "python3",
        "validation_scripts/qwen36_next_compile_preflight.py",
        "--decision-json",
        str(decision_path),
        "--repo-root",
        str(repo_root),
        "--output-json",
        "<NEXT_COMPILE_RELEASE_JSON>",
        "--compile-host",
        compile_host,
        "--runtime-host",
        runtime_host,
        "--source-dir",
        resolved_source_dir,
        "--source-commit",
        resolved_source_commit,
        "--automation-name",
        automation_payload["name"],
        "--automation-created-name",
        automation_payload["name"],
        "--automation-interval-minutes",
        str(automation_interval_minutes),
        "--launch-script",
        launch_script,
        "--boundary-lengths",
        boundary_lengths,
    ]

    result: dict[str, Any] = {
        "schema": "qwen36-next-compile-preflight-v1",
        "passed": True,
        "decision_json": str(decision_path),
        "next_speed_slice": decision.get("next_speed_slice"),
        "ts": preflight.get("ts"),
        "compile_driver": driver,
        "dry_run": {
            "command": preflight.get("dry_run_command"),
            "returncode": completed.returncode,
            "env_log": str(env_log_path),
            "stdout_assignments": stdout_values,
        },
        "automation_payload": automation_payload,
        "automation_ack": {
            "required": True,
            "created_name": automation_created_name,
            "matched_payload_name": release_compile_command,
        },
        "requires_automation_creation_before_compile": True,
        "compile_command_after_automation": compile_command
        if release_compile_command
        else None,
        "launch_command_after_automation": compile_command
        if release_compile_command
        else None,
        "compile_command_release_command_template": (
            _quote_command(release_command)
        ),
        "release_preflight_context": {
            "repo_root": str(repo_root),
            "compile_host": compile_host,
            "runtime_host": runtime_host,
            "source_dir": resolved_source_dir,
            "source_commit": resolved_source_commit,
            "automation_name": automation_payload["name"],
            "automation_interval_minutes": automation_interval_minutes,
            "launch_script": launch_script,
            "boundary_lengths": boundary_lengths,
        },
        "run_order": [
            "create_heartbeat_automation_from_automation_payload",
            "rerun_next_compile_preflight_with_automation_created_name",
            "run_compile_command_after_automation_exists",
        ],
    }
    if release_compile_command:
        result["run_order"] = ["run_compile_command_after_automation_exists"]
    return result


def _failure_payload(error: Exception) -> dict[str, Any]:
    return {
        "schema": "qwen36-next-compile-preflight-v1",
        "passed": False,
        "error_type": type(error).__name__,
        "error": str(error),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--decision-json", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument("--automation-json-output", type=Path, default=None)
    parser.add_argument("--compile-host", default="ubuntu@16.26.135.243")
    parser.add_argument("--runtime-host", default="ubuntu@16.26.184.190")
    parser.add_argument("--source-dir", default=None)
    parser.add_argument("--source-commit", default=None)
    parser.add_argument("--automation-name", default=None)
    parser.add_argument(
        "--automation-created-name",
        default=None,
        help=(
            "Name of the heartbeat automation after codex_app.automation_update "
            "succeeds. The live compile command is emitted only when this matches "
            "the generated payload name."
        ),
    )
    parser.add_argument(
        "--automation-interval-minutes",
        type=int,
        default=10,
    )
    parser.add_argument("--launch-script", default="tmp_launch_qwen36_segcte2048.sh")
    parser.add_argument(
        "--boundary-lengths",
        default="146,160,485,505,526,1225,2048,2049,2500,4092,4096",
    )
    args = parser.parse_args()

    try:
        decision_path = args.decision_json.expanduser()
        payload = build_preflight(
            decision=_load_json(decision_path),
            decision_path=decision_path,
            repo_root=args.repo_root.expanduser().resolve(),
            compile_host=args.compile_host,
            runtime_host=args.runtime_host,
            source_dir=args.source_dir,
            source_commit=args.source_commit,
            automation_name=args.automation_name,
            automation_created_name=args.automation_created_name,
            automation_interval_minutes=args.automation_interval_minutes,
            launch_script=args.launch_script,
            boundary_lengths=args.boundary_lengths,
        )
        if args.automation_json_output is not None:
            _write_json(args.automation_json_output, payload["automation_payload"])
            payload["automation_payload_path"] = str(args.automation_json_output)
        exit_code = 0
    except Exception as exc:
        payload = _failure_payload(exc)
        exit_code = 1

    if args.output_json is not None:
        _write_json(args.output_json, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
