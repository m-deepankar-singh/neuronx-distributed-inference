#!/usr/bin/env python3
"""Check Qwen3.6 compile log and artifact readiness."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


DEFAULT_FAILURE_MARKERS = [
    "Traceback",
    "Exception",
    "RuntimeError",
    "ValueError",
    "NCC_",
    "NCC_INKI",
    "No space left on device",
    "OutOfMemory",
    "Compile failed",
    "COMPILE_FAILED",
]

_CHECKPOINT_RE = re.compile(
    r"CHECKPOINT_BANK_WEIGHTS_ADDED\s+"
    r"(?P<name>\S*tp(?P<rank>\d+)\S*)\s+"
    r"(?P<recurrent_count>\d+)\s+"
    r"(?P<conv_count>\d+)\s+"
    r"(?P<recurrent_dtype>\S+)\s+"
    r"(?P<conv_dtype>\S+)"
)


def parse_key_values(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for raw_line in path.read_text(errors="replace").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key] = value
    return values


def _pid_running(pid_file: Path | None) -> bool | None:
    if pid_file is None or not pid_file.exists():
        return None
    try:
        pid = int(pid_file.read_text().strip())
    except (OSError, ValueError):
        return None
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _artifact_files(artifact: Path) -> dict[str, dict[str, Any]]:
    files = {}
    for name in ["model.pt", "neuron_config.json"]:
        path = artifact / name
        try:
            stat = path.stat()
        except OSError:
            files[name] = {"exists": False, "size": None}
        else:
            files[name] = {"exists": True, "size": stat.st_size}
    return files


def _checkpoint_banks(log_text: str) -> dict[str, dict[str, Any]]:
    banks: dict[str, dict[str, Any]] = {}
    for match in _CHECKPOINT_RE.finditer(log_text):
        rank = f"tp{int(match.group('rank'))}"
        banks[rank] = {
            "name": match.group("name"),
            "recurrent_count": int(match.group("recurrent_count")),
            "conv_count": int(match.group("conv_count")),
            "recurrent_dtype": match.group("recurrent_dtype"),
            "conv_dtype": match.group("conv_dtype"),
        }
    return banks


def _failure_lines(log_text: str, markers: list[str]) -> list[dict[str, Any]]:
    failures: list[dict[str, Any]] = []
    for line_no, line in enumerate(log_text.splitlines(), start=1):
        for marker in markers:
            if marker in line:
                failures.append({"line": line_no, "marker": marker, "text": line})
                break
    return failures


def _parse_ranks(raw: str) -> list[str]:
    ranks = []
    for item in raw.replace(",", " ").split():
        item = item.strip()
        if not item:
            continue
        ranks.append(item if item.startswith("tp") else f"tp{int(item)}")
    return ranks


def _normalize_dtype(dtype: str | None) -> str | None:
    if dtype is None:
        return None
    normalized = dtype.strip().lower()
    if normalized.startswith("torch."):
        normalized = normalized.split(".", 1)[1]
    aliases = {
        "bf16": "bfloat16",
        "fp32": "float32",
        "fp16": "float16",
    }
    return aliases.get(normalized, normalized)


def _dtype_mismatches(
    banks: dict[str, dict[str, Any]],
    *,
    required_ranks: list[str],
    expected_recurrent_dtype: str | None,
    expected_conv_dtype: str | None,
) -> list[dict[str, Any]]:
    expected_recurrent = _normalize_dtype(expected_recurrent_dtype)
    expected_conv = _normalize_dtype(expected_conv_dtype)
    mismatches: list[dict[str, Any]] = []
    for rank in required_ranks:
        bank = banks.get(rank)
        if bank is None:
            continue
        actual_recurrent = _normalize_dtype(str(bank["recurrent_dtype"]))
        actual_conv = _normalize_dtype(str(bank["conv_dtype"]))
        if expected_recurrent is not None and actual_recurrent != expected_recurrent:
            mismatches.append(
                {
                    "rank": rank,
                    "field": "recurrent_dtype",
                    "expected": expected_recurrent,
                    "actual": actual_recurrent,
                    "raw_actual": bank["recurrent_dtype"],
                }
            )
        if expected_conv is not None and actual_conv != expected_conv:
            mismatches.append(
                {
                    "rank": rank,
                    "field": "conv_dtype",
                    "expected": expected_conv,
                    "actual": actual_conv,
                    "raw_actual": bank["conv_dtype"],
                }
            )
    return mismatches


def _artifact_policy_result(
    *,
    required: bool,
    basic_ready: bool,
    artifact: Path,
    log: Path,
    env_log: Path | None,
    output_path: Path | None,
    recommended_block_size: int,
    min_usable_headroom_blocks: int,
    strict_hybrid_gate: bool,
) -> dict[str, Any]:
    if not required:
        return {
            "required": False,
            "passed": None,
            "skipped_reason": "not_requested",
            "summary": None,
            "output_path": str(output_path) if output_path is not None else None,
        }
    if not basic_ready:
        return {
            "required": True,
            "passed": None,
            "skipped_reason": "compile_not_ready",
            "summary": None,
            "output_path": str(output_path) if output_path is not None else None,
        }

    from validation_scripts.qwen36_artifact_config_audit import audit

    summary = audit(
        artifact=artifact,
        compile_log=log,
        env_log=env_log,
        recommended_block_size=recommended_block_size,
        min_usable_headroom_blocks=min_usable_headroom_blocks,
        strict_hybrid_gate=strict_hybrid_gate,
    )
    encoded = json.dumps(summary, indent=2, sort_keys=True) + "\n"
    if output_path is not None:
        output_path.expanduser().parent.mkdir(parents=True, exist_ok=True)
        output_path.expanduser().write_text(encoded)
    return {
        "required": True,
        "passed": bool(summary.get("policy_passed")),
        "skipped_reason": None,
        "summary": summary,
        "output_path": str(output_path) if output_path is not None else None,
    }


def check_status(
    *,
    log: Path,
    artifact: Path,
    pid_file: Path | None,
    env_log: Path | None = None,
    required_ranks: list[str],
    expected_recurrent_dtype: str | None = None,
    expected_conv_dtype: str | None = None,
    source_commit: str | None = None,
    source_branch: str | None = None,
    failure_markers: list[str] = DEFAULT_FAILURE_MARKERS,
    require_artifact_policy: bool = False,
    artifact_policy_output: Path | None = None,
    artifact_policy_recommended_block_size: int = 256,
    artifact_policy_min_usable_headroom_blocks: int = 0,
    artifact_policy_strict_hybrid_gate: bool = False,
) -> dict[str, Any]:
    try:
        log_text = log.read_text(errors="replace")
        log_exists = True
    except OSError:
        log_text = ""
        log_exists = False

    finished_hlos = "Finished Compilation for all HLOs" in log_text
    compile_done = "COMPILE_DONE" in log_text
    failures = _failure_lines(log_text, failure_markers)
    banks = _checkpoint_banks(log_text)
    missing_ranks = [rank for rank in required_ranks if rank not in banks]
    dtype_mismatches = _dtype_mismatches(
        banks,
        required_ranks=required_ranks,
        expected_recurrent_dtype=expected_recurrent_dtype,
        expected_conv_dtype=expected_conv_dtype,
    )
    artifact_files = _artifact_files(artifact)
    missing_files = [
        name for name, info in artifact_files.items() if not bool(info["exists"])
    ]
    pid_running = _pid_running(pid_file)

    basic_ready = (
        log_exists
        and finished_hlos
        and compile_done
        and not failures
        and not missing_ranks
        and not dtype_mismatches
        and not missing_files
    )
    artifact_policy = _artifact_policy_result(
        required=require_artifact_policy,
        basic_ready=basic_ready,
        artifact=artifact,
        log=log,
        env_log=env_log,
        output_path=artifact_policy_output,
        recommended_block_size=artifact_policy_recommended_block_size,
        min_usable_headroom_blocks=artifact_policy_min_usable_headroom_blocks,
        strict_hybrid_gate=artifact_policy_strict_hybrid_gate,
    )
    ready = basic_ready and (
        not require_artifact_policy or bool(artifact_policy.get("passed"))
    )
    if ready:
        state = "ready"
    elif failures:
        state = "failed"
    elif basic_ready and require_artifact_policy and not artifact_policy.get("passed"):
        state = "failed"
    elif pid_running:
        state = "running"
    else:
        state = "incomplete"

    return {
        "state": state,
        "ready": ready,
        "basic_ready": basic_ready,
        "log": str(log),
        "artifact": str(artifact),
        "pid_file": str(pid_file) if pid_file is not None else None,
        "pid_running": pid_running,
        "source": {
            "commit": source_commit,
            "branch": source_branch,
        },
        "markers": {
            "finished_hlos": finished_hlos,
            "compile_done": compile_done,
        },
        "checkpoint_banks": banks,
        "required_checkpoint_ranks": required_ranks,
        "missing_checkpoint_ranks": missing_ranks,
        "expected_checkpoint_dtypes": {
            "recurrent_dtype": _normalize_dtype(expected_recurrent_dtype),
            "conv_dtype": _normalize_dtype(expected_conv_dtype),
        },
        "checkpoint_dtype_mismatches": dtype_mismatches,
        "artifact_files": artifact_files,
        "missing_artifact_files": missing_files,
        "artifact_policy": artifact_policy,
        "failure_lines": failures,
        "tail": log_text.splitlines()[-40:],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-log", type=Path)
    parser.add_argument("--log", type=Path)
    parser.add_argument("--artifact", type=Path)
    parser.add_argument("--pid-file", type=Path)
    parser.add_argument("--required-tp-ranks", default="0,1,2,3")
    parser.add_argument("--expected-recurrent-dtype")
    parser.add_argument("--expected-conv-dtype")
    parser.add_argument("--failure-marker", action="append", dest="failure_markers")
    parser.add_argument(
        "--require-artifact-policy",
        action="store_true",
        help="Require neuron_config.json policy to match the compile env log.",
    )
    parser.add_argument("--artifact-policy-output", type=Path, default=None)
    parser.add_argument("--artifact-policy-recommended-block-size", type=int, default=256)
    parser.add_argument(
        "--artifact-policy-min-usable-headroom-blocks",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--artifact-policy-strict-hybrid-gate",
        action="store_true",
    )
    parser.add_argument(
        "--zero-when-running",
        action="store_true",
        help="Exit 0 for state=running so heartbeat monitors can stay quiet.",
    )
    args = parser.parse_args()

    env = parse_key_values(args.env_log) if args.env_log is not None else {}
    log = args.log or (Path(env["LOG"]) if "LOG" in env else None)
    artifact = args.artifact or (Path(env["ARTIFACT"]) if "ARTIFACT" in env else None)
    pid_file = args.pid_file or (Path(env["PIDFILE"]) if "PIDFILE" in env else None)
    if log is None:
        raise SystemExit("--log or env-log LOG is required")
    if artifact is None:
        raise SystemExit("--artifact or env-log ARTIFACT is required")

    result = check_status(
        log=log,
        artifact=artifact,
        pid_file=pid_file,
        env_log=args.env_log,
        required_ranks=_parse_ranks(args.required_tp_ranks),
        expected_recurrent_dtype=(
            args.expected_recurrent_dtype or env.get("GDN_RECURRENT_CACHE_DTYPE")
        ),
        expected_conv_dtype=args.expected_conv_dtype or env.get("GDN_CONV_CACHE_DTYPE"),
        source_commit=env.get("SOURCE_COMMIT"),
        source_branch=env.get("SOURCE_BRANCH"),
        failure_markers=args.failure_markers or DEFAULT_FAILURE_MARKERS,
        require_artifact_policy=args.require_artifact_policy,
        artifact_policy_output=args.artifact_policy_output,
        artifact_policy_recommended_block_size=(
            args.artifact_policy_recommended_block_size
        ),
        artifact_policy_min_usable_headroom_blocks=(
            args.artifact_policy_min_usable_headroom_blocks
        ),
        artifact_policy_strict_hybrid_gate=args.artifact_policy_strict_hybrid_gate,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    if args.zero_when_running and result["state"] == "running":
        return 0
    return 0 if result["ready"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
