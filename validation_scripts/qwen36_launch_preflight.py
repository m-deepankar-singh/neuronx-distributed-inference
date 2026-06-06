#!/usr/bin/env python3
"""Generate and optionally run Qwen3.6 launch preflight from a compile env log."""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from validation_scripts.qwen36_launch_env_audit import audit, parse_env_log


_INHERIT_KEYS = [
    "REPO",
    "MODEL",
    "MAX_GDN_CHECKPOINT_SLOTS",
    "GDN_RECURRENT_CACHE_DTYPE",
    "GDN_CONV_CACHE_DTYPE",
    "QWEN36_DELTANET_MULTIHEAD_CTE",
    "QWEN36_SPLIT_QKV_TKG_ROW_SCALES",
    "QWEN36_DELTANET_FUSED_SEGMENT_TOKENS",
    "QWEN36_DELTANET_SOLVE_MODE",
    "QWEN36_DELTANET_SOLVE_SCAN_STEPS",
]


def _env_required(values: dict[str, str], key: str) -> str:
    value = values.get(key)
    if value is None or value.strip() == "":
        raise ValueError(f"compile env log is missing required key {key!r}")
    return value


def _tokens(value: str) -> list[str]:
    return [item for item in value.split() if item]


def _context_pairs_with_prefix_zero(compile_env: dict[str, str]) -> str:
    cte_buckets = _tokens(_env_required(compile_env, "CTE_BUCKETS"))
    compiled_pairs = _tokens(_env_required(compile_env, "CONTEXT_ENCODING_BUCKET_PAIRS"))
    ordered: list[str] = []
    seen: set[str] = set()
    for bucket in cte_buckets:
        pair = f"{bucket}:0"
        if pair not in seen:
            ordered.append(pair)
            seen.add(pair)
    for pair in compiled_pairs:
        if pair not in seen:
            ordered.append(pair)
            seen.add(pair)
    return " ".join(ordered)


def launch_env_from_compile(compile_env: dict[str, str]) -> dict[str, str]:
    env = {key: compile_env[key] for key in _INHERIT_KEYS if key in compile_env}
    env.update(
        {
            "LAUNCH_DRY_RUN": "1",
            "MAX_MODEL_LEN": _env_required(compile_env, "MAX_CONTEXT_LENGTH"),
            "SEQ_LEN": _env_required(compile_env, "SEQ_LEN"),
            "CTE_BUCKETS": _env_required(compile_env, "CTE_BUCKETS"),
            "CONTEXT_ENCODING_BUCKET_PAIRS": _context_pairs_with_prefix_zero(
                compile_env
            ),
            "TOKEN_GENERATION_BUCKETS": _env_required(
                compile_env, "TOKEN_GENERATION_BUCKETS"
            ),
            "DISABLE_HYBRID_KV_CACHE_MANAGER": "0",
        }
    )
    if compile_env.get("ENABLE_KV_CACHE_QUANT", "0").strip() == "1":
        env["KV_CACHE_DTYPE"] = "fp8"
    return env


def _quote_env_command(env: dict[str, str], command: list[str]) -> str:
    parts = [f"{key}={shlex.quote(value)}" for key, value in env.items()]
    parts.extend(shlex.quote(item) for item in command)
    return " ".join(parts)


def _launch_env_log_path(serve_log: Path) -> Path:
    if serve_log.suffix == ".log":
        return serve_log.with_suffix("").with_name(serve_log.stem + "_env.txt")
    return serve_log.with_name(serve_log.name + "_env.txt")


def build_preflight(
    *,
    compile_env: dict[str, str],
    compile_env_log: Path,
    serve_log: Path,
    launch_script: str,
    audit_script: str = "validation_scripts/qwen36_launch_env_audit.py",
) -> dict[str, Any]:
    artifact = _env_required(compile_env, "ARTIFACT")
    launch_env = launch_env_from_compile(compile_env)
    dry_run_command = ["bash", launch_script, artifact, str(serve_log)]
    live_env = dict(launch_env)
    live_env["LAUNCH_DRY_RUN"] = "0"
    live_command = ["bash", launch_script, artifact, str(serve_log)]
    launch_env_log = _launch_env_log_path(serve_log)
    audit_command = [
        "python3",
        audit_script,
        "--compile-env-log",
        str(compile_env_log),
        "--launch-env-log",
        str(launch_env_log),
        "--require-launch-dry-run",
    ]
    return {
        "schema": "qwen36-launch-preflight-v1",
        "compile_env_log": str(compile_env_log),
        "artifact": artifact,
        "serve_log": str(serve_log),
        "launch_env_log": str(launch_env_log),
        "launch_env": launch_env,
        "dry_run_command": _quote_env_command(launch_env, dry_run_command),
        "audit_command": _quote_env_command({}, audit_command),
        "live_launch_env": live_env,
        "live_launch_command": _quote_env_command(live_env, live_command),
        "run_order": [
            "run_dry_run_command",
            "run_audit_command",
            "run_live_launch_command_only_if_audit_passes",
        ],
    }


def run_preflight(plan: dict[str, Any]) -> dict[str, Any]:
    env = os.environ.copy()
    env.update({str(key): str(value) for key, value in plan["launch_env"].items()})
    # Keep the executable command structured instead of reparsing the display string.
    launch_script = str(plan["_launch_script"])
    artifact = str(plan["artifact"])
    serve_log = str(plan["serve_log"])
    dry_run = subprocess.run(
        ["bash", launch_script, artifact, serve_log],
        check=False,
        capture_output=True,
        env=env,
        text=True,
    )
    payload = dict(plan)
    payload.pop("_launch_script", None)
    payload["dry_run"] = {
        "returncode": dry_run.returncode,
        "stdout": dry_run.stdout,
        "stderr": dry_run.stderr,
    }
    if dry_run.returncode != 0:
        payload["passed"] = False
        payload["failure_stage"] = "launch_dry_run"
        return payload

    launch_env = parse_env_log(Path(plan["launch_env_log"]))
    audit_result = audit(
        compile_env=parse_env_log(Path(plan["compile_env_log"])),
        launch_env=launch_env,
        require_launch_dry_run=True,
    )
    payload["audit"] = audit_result
    payload["passed"] = bool(audit_result["passed"])
    if not payload["passed"]:
        payload["failure_stage"] = "launch_env_audit"
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--compile-env-log", type=Path, required=True)
    parser.add_argument("--serve-log", type=Path, required=True)
    parser.add_argument("--launch-script", default="tmp_launch_qwen36_segcte2048.sh")
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args()

    compile_env_log = args.compile_env_log.expanduser().resolve()
    compile_env = parse_env_log(compile_env_log)
    try:
        plan = build_preflight(
            compile_env=compile_env,
            compile_env_log=compile_env_log,
            serve_log=args.serve_log.expanduser(),
            launch_script=args.launch_script,
        )
    except ValueError as exc:
        payload = {
            "schema": "qwen36-launch-preflight-v1",
            "passed": False,
            "failure_stage": "build_preflight",
            "errors": [str(exc)],
            "compile_env_log": str(compile_env_log),
        }
    else:
        plan["_launch_script"] = args.launch_script
        payload = run_preflight(plan) if args.run else {**plan, "passed": None}
    payload.pop("_launch_script", None)
    encoded = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if args.output_json is not None:
        args.output_json.expanduser().parent.mkdir(parents=True, exist_ok=True)
        args.output_json.expanduser().write_text(encoded)
    print(encoded, end="")
    return 0 if payload.get("passed") is not False else 1


if __name__ == "__main__":
    raise SystemExit(main())
