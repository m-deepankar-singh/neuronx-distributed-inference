#!/usr/bin/env python3
"""Strict-final cold-prefill benchmark orchestrator for Qwen3.6."""

from __future__ import annotations

import argparse
import json
import shutil
import shlex
import subprocess
import sys
import time
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
BENCHMARK = REPO_ROOT / "validation_scripts" / "qwen36_cold_prefill_benchmark.py"
ACCEPTANCE = REPO_ROOT / "validation_scripts" / "qwen36_cold_prefill_acceptance.py"
HYBRID_APC_VALIDATION = (
    REPO_ROOT / "validation_scripts" / "qwen36_hybrid_apc_validation.py"
)

FIXED_PROMPT_LENGTHS = (128, 256, 384, 512, 1024, 2048, 8192, 32768)
STRICT_MAX_TOKENS_VALUES = (1, 32)
LONG_CONTEXT_BASE_PROMPTS = (128, 256, 384, 512, 2048)
LONG_CONTEXT_VARIANTS = (
    "A_single512_old_chunked",
    "H_128k_candidate",
    "J_262k_recovery_block128",
)
RUN_PHASES = ("matrix", "long_context", "hybrid_apc", "acceptance")
DEFAULT_EXPECTED_BRANCH = "qwen36-cold-prefill-perf"


def _add_runtime_args(command: list[str], args: argparse.Namespace) -> None:
    command.extend(
        [
            "--tensor-parallel-size",
            str(args.tensor_parallel_size),
            "--logical-nc-config",
            str(args.logical_nc_config),
            "--max-num-seqs",
            str(args.max_num_seqs),
            "--ctx-batch-size",
            str(args.ctx_batch_size),
        ]
    )
    if args.num_gpu_blocks_override is not None:
        command.extend(
            ["--num-gpu-blocks-override", str(args.num_gpu_blocks_override)]
        )
    if args.num_gpu_blocks_override_by_len:
        command.append("--num-gpu-blocks-override-by-len")
        command.extend(args.num_gpu_blocks_override_by_len)


def _num_gpu_blocks_override_by_len(args: argparse.Namespace) -> dict[int, int]:
    mapping: dict[int, int] = {}
    for raw in args.num_gpu_blocks_override_by_len or []:
        if "=" not in raw:
            raise ValueError(
                "--num-gpu-blocks-override-by-len entries must be shaped LEN=BLOCKS"
            )
        seq_len_raw, blocks_raw = raw.split("=", 1)
        mapping[int(seq_len_raw)] = int(blocks_raw)
    return mapping


def _num_gpu_blocks_override_for_len(
    args: argparse.Namespace, seq_len: int
) -> int | None:
    return _num_gpu_blocks_override_by_len(args).get(seq_len) or args.num_gpu_blocks_override


def _add_cold_artifact_runtime_args(
    command: list[str], args: argparse.Namespace
) -> None:
    if args.enable_prefix_caching:
        command.append("--enable-prefix-caching")
    if args.enable_hybrid_apc:
        command.append("--enable-hybrid-apc")
    if args.hybrid_apc_require_vllm_metadata:
        command.append("--hybrid-apc-require-vllm-metadata")
    command.extend(
        [
            "--gdn-checkpoint-interval",
            str(args.gdn_checkpoint_interval),
            "--max-gdn-checkpoint-slots",
            str(args.max_gdn_checkpoint_slots),
            "--hybrid-cache-mode",
            args.hybrid_cache_mode,
            "--block-size",
            str(args.block_size),
            "--compiled-max-prompt-length",
            str(args.compiled_max_prompt_length),
        ]
    )


def _build_commands(args: argparse.Namespace) -> dict[str, list[str]]:
    output_dir = args.output_dir.expanduser()
    matrix_json = output_dir / f"{args.output_prefix}_matrix.json"
    long_context_json = output_dir / f"{args.output_prefix}_long_context.json"
    hybrid_apc_json = output_dir / f"{args.output_prefix}_hybrid_apc_exactness.json"
    acceptance_json = output_dir / f"{args.output_prefix}_strict_acceptance.json"

    matrix_cmd = [
        args.python,
        str(BENCHMARK),
        "--model-path",
        args.model_path,
        "--compiled-artifacts",
        args.artifacts_2k,
        "--compiled-artifacts-by-len",
        f"8192={args.artifacts_8k}",
        f"32768={args.artifacts_32k}",
        "--prompt-lengths",
        *[str(length) for length in FIXED_PROMPT_LENGTHS],
        "--include-dense-fallback",
        "--repetitions",
        str(args.repetitions),
        "--max-tokens-values",
        *[str(value) for value in STRICT_MAX_TOKENS_VALUES],
        "--output-json",
        str(matrix_json),
    ]
    _add_runtime_args(matrix_cmd, args)
    _add_cold_artifact_runtime_args(matrix_cmd, args)
    if args.fail_fast:
        matrix_cmd.append("--fail-fast")
    if args.gdn_state_diff_json is not None:
        matrix_cmd.extend(["--gdn-state-diff-json", str(args.gdn_state_diff_json)])

    long_context_cmd = [
        args.python,
        str(BENCHMARK),
        "--model-path",
        args.model_path,
        "--compiled-artifacts",
        args.artifacts_2k,
        "--compiled-artifacts-by-len",
        f"131072={args.artifacts_128k}",
        f"262144={args.artifacts_262k}",
        "--prompt-lengths",
        *[str(length) for length in LONG_CONTEXT_BASE_PROMPTS],
        "--include-128k",
        "--include-262k",
        "--variants",
        *LONG_CONTEXT_VARIANTS,
        "--repetitions",
        str(args.repetitions),
        "--max-tokens-values",
        *[str(value) for value in STRICT_MAX_TOKENS_VALUES],
        "--output-json",
        str(long_context_json),
    ]
    _add_runtime_args(long_context_cmd, args)
    _add_cold_artifact_runtime_args(long_context_cmd, args)
    if args.fail_fast:
        long_context_cmd.append("--fail-fast")
    if args.gdn_state_diff_json is not None:
        long_context_cmd.extend(
            ["--gdn-state-diff-json", str(args.gdn_state_diff_json)]
        )

    hybrid_apc_cmd = [
        args.python,
        str(HYBRID_APC_VALIDATION),
        "exactness",
        "--model-path",
        args.model_path,
        "--compiled-artifacts",
        args.artifacts_2k,
        "--max-model-len",
        "2048",
        "--seq-len",
        "2048",
        "--cte-bucket-profile",
        "short",
        "--tensor-parallel-size",
        str(args.tensor_parallel_size),
        "--logical-nc-config",
        str(args.logical_nc_config),
        "--ctx-batch-size",
        str(args.ctx_batch_size),
        "--max-num-seqs",
        str(args.max_num_seqs),
        "--block-size",
        str(args.block_size),
        "--gdn-checkpoint-interval",
        str(args.gdn_checkpoint_interval),
        "--enable-vllm-chunked-prefill",
        "--kernel-q-tile-size",
        "128",
        "--kernel-kv-tile-size",
        "1024",
        "--text-only-cte",
        "--compact-cte-attention-mask",
        "--hybrid-apc-require-vllm-metadata",
        "--max-tokens",
        "32",
        "--output-json",
        str(hybrid_apc_json),
    ]
    hybrid_num_gpu_blocks_override = _num_gpu_blocks_override_for_len(args, 2048)
    if hybrid_num_gpu_blocks_override is not None:
        hybrid_apc_cmd.extend(
            ["--num-gpu-blocks-override", str(hybrid_num_gpu_blocks_override)]
        )

    acceptance_cmd = [
        args.python,
        str(ACCEPTANCE),
        str(matrix_json),
        str(long_context_json),
        "--short-latency-speedup",
        str(args.short_latency_speedup),
        "--baseline-cold-tok-per-s-target",
        str(args.baseline_cold_tok_per_s_target),
        "--baseline-cold-tok-per-s-tolerance",
        str(args.baseline_cold_tok_per_s_tolerance),
        "--bucket-tok-regression-tolerance",
        str(args.bucket_tok_regression_tolerance),
        "--prompt-token-tolerance",
        str(args.prompt_token_tolerance),
        "--prefill-max-tokens",
        "1",
        "--strict-final",
        "--strict-min-samples",
        str(args.strict_min_samples),
        "--hybrid-apc-report",
        str(hybrid_apc_json),
        "--output-json",
        str(acceptance_json),
    ]
    return {
        "matrix": matrix_cmd,
        "long_context": long_context_cmd,
        "hybrid_apc": hybrid_apc_cmd,
        "acceptance": acceptance_cmd,
    }


def _phase_log_path(args: argparse.Namespace, phase: str) -> Path:
    return args.output_dir.expanduser() / f"{args.output_prefix}_{phase}.log"


def _phase_log_paths(args: argparse.Namespace) -> dict[str, str]:
    return {phase: str(_phase_log_path(args, phase)) for phase in RUN_PHASES}


def _run_manifest_path(args: argparse.Namespace) -> Path:
    return args.output_dir.expanduser() / f"{args.output_prefix}_run_manifest.json"


def _expected_output_paths(args: argparse.Namespace) -> dict[str, str]:
    output_dir = args.output_dir.expanduser()
    return {
        "matrix": str(output_dir / f"{args.output_prefix}_matrix.json"),
        "long_context": str(output_dir / f"{args.output_prefix}_long_context.json"),
        "hybrid_apc": str(
            output_dir / f"{args.output_prefix}_hybrid_apc_exactness.json"
        ),
        "acceptance": str(
            output_dir / f"{args.output_prefix}_strict_acceptance.json"
        ),
    }


def _file_status(path: str | Path) -> dict:
    path_obj = Path(path).expanduser()
    exists = path_obj.exists()
    is_file = exists and path_obj.is_file()
    size_bytes = path_obj.stat().st_size if is_file else None
    return {
        "path": str(path_obj),
        "resolved": str(path_obj.resolve()) if exists else None,
        "exists": exists,
        "is_file": is_file,
        "bytes": size_bytes,
        "nonempty": size_bytes is not None and size_bytes > 0,
    }


def _evidence_status(args: argparse.Namespace) -> dict:
    return {
        "preflight": _file_status(_preflight_json_path(args)),
        "outputs": {
            name: _file_status(path)
            for name, path in _expected_output_paths(args).items()
        },
        "logs": {
            name: _file_status(path)
            for name, path in _phase_log_paths(args).items()
        },
        "run_manifest": _file_status(_run_manifest_path(args)),
    }


def _evidence_check_json_path(args: argparse.Namespace) -> Path:
    if args.evidence_check_json is not None:
        return args.evidence_check_json.expanduser()
    return args.output_dir.expanduser() / f"{args.output_prefix}_evidence_check.json"


def _load_json_file(path: Path) -> tuple[object | None, str | None]:
    try:
        return json.loads(path.read_text(encoding="utf-8")), None
    except FileNotFoundError:
        return None, "file not found"
    except json.JSONDecodeError as exc:
        return None, f"invalid JSON: {exc}"
    except OSError as exc:
        return None, str(exc)


def _nonempty_file_checks(status: dict) -> list[dict]:
    checks: list[dict] = []
    singleton_groups = ("preflight", "run_manifest")
    for group in singleton_groups:
        file_status = status[group]
        checks.append(
            {
                "group": group,
                "name": group,
                "path": file_status["path"],
                "bytes": file_status["bytes"],
                "ok": file_status["exists"]
                and file_status["is_file"]
                and file_status["nonempty"],
            }
        )
    for group in ("outputs", "logs"):
        for name, file_status in status[group].items():
            checks.append(
                {
                    "group": group,
                    "name": name,
                    "path": file_status["path"],
                    "bytes": file_status["bytes"],
                    "ok": file_status["exists"]
                    and file_status["is_file"]
                    and file_status["nonempty"],
                }
            )
    return checks


def _manifest_phase_checks(manifest: object) -> list[dict]:
    phases = manifest.get("phases", {}) if isinstance(manifest, dict) else {}
    checks: list[dict] = []
    for phase in RUN_PHASES:
        phase_payload = phases.get(phase)
        returncode = (
            phase_payload.get("returncode")
            if isinstance(phase_payload, dict)
            else None
        )
        checks.append(
            {
                "phase": phase,
                "returncode": returncode,
                "ok": returncode == 0,
            }
        )
    return checks


def _manifest_expected_path_check(manifest: object, args: argparse.Namespace) -> dict:
    if not isinstance(manifest, dict):
        return {
            "expected_outputs_match": False,
            "logs_match": False,
            "ok": False,
        }
    expected_outputs_match = manifest.get("expected_outputs") == _expected_output_paths(
        args
    )
    logs_match = manifest.get("logs") == _phase_log_paths(args)
    return {
        "expected_outputs_match": expected_outputs_match,
        "logs_match": logs_match,
        "ok": expected_outputs_match and logs_match,
    }


def _preflight_expected_path_check(preflight: object, args: argparse.Namespace) -> dict:
    if not isinstance(preflight, dict):
        return {
            "expected_outputs_match": False,
            "expected_logs_match": False,
            "run_manifest_match": False,
            "output_dir_match": False,
            "output_prefix_match": False,
            "ok": False,
        }
    expected_outputs_match = preflight.get("expected_outputs") == _expected_output_paths(
        args
    )
    expected_logs_match = preflight.get("expected_logs") == _phase_log_paths(args)
    run_manifest_match = preflight.get("run_manifest") == str(_run_manifest_path(args))
    output_dir_match = preflight.get("output_dir") == str(args.output_dir.expanduser())
    output_prefix_match = preflight.get("output_prefix") == args.output_prefix
    return {
        "expected_outputs_match": expected_outputs_match,
        "expected_logs_match": expected_logs_match,
        "run_manifest_match": run_manifest_match,
        "output_dir_match": output_dir_match,
        "output_prefix_match": output_prefix_match,
        "ok": expected_outputs_match
        and expected_logs_match
        and run_manifest_match
        and output_dir_match
        and output_prefix_match,
    }


def _manifest_command_checks(manifest: object) -> list[dict]:
    commands = manifest.get("commands", {}) if isinstance(manifest, dict) else {}
    command_strings = (
        manifest.get("command_strings", {}) if isinstance(manifest, dict) else {}
    )
    checks: list[dict] = []
    for phase in RUN_PHASES:
        command = commands.get(phase)
        command_string = command_strings.get(phase)
        checks.append(
            {
                "phase": phase,
                "has_command": isinstance(command, list) and bool(command),
                "has_command_string": isinstance(command_string, str)
                and bool(command_string.strip()),
                "ok": isinstance(command, list)
                and bool(command)
                and isinstance(command_string, str)
                and bool(command_string.strip()),
            }
        )
    return checks


def _evidence_check_report(args: argparse.Namespace) -> dict:
    status = _evidence_status(args)
    file_checks = _nonempty_file_checks(status)
    preflight_path = _preflight_json_path(args)
    manifest_path = _run_manifest_path(args)
    acceptance_path = Path(_expected_output_paths(args)["acceptance"])
    preflight, preflight_error = _load_json_file(preflight_path)
    manifest, manifest_error = _load_json_file(manifest_path)
    acceptance, acceptance_error = _load_json_file(acceptance_path)
    preflight_passed = isinstance(preflight, dict) and preflight.get("passed") is True
    manifest_passed = (
        isinstance(manifest, dict)
        and manifest.get("strict_final") is True
        and manifest.get("passed") is True
        and manifest.get("returncode") == 0
    )
    acceptance_passed = isinstance(acceptance, dict) and acceptance.get("passed") is True
    acceptance_strict_final = (
        isinstance(acceptance, dict) and acceptance.get("strict_final") is True
    )
    audit_checklist = (
        acceptance.get("audit_checklist") if isinstance(acceptance, dict) else None
    )
    audit_failed_items = (
        [
            item
            for item in audit_checklist
            if isinstance(item, dict) and item.get("status") == "fail"
        ]
        if isinstance(audit_checklist, list)
        else None
    )
    audit_checklist_passed = (
        isinstance(audit_checklist, list)
        and bool(audit_checklist)
        and not audit_failed_items
    )
    preflight_path_check = _preflight_expected_path_check(preflight, args)
    phase_checks = _manifest_phase_checks(manifest)
    command_checks = _manifest_command_checks(manifest)
    manifest_path_check = _manifest_expected_path_check(manifest, args)
    passed = (
        all(check["ok"] for check in file_checks)
        and preflight_error is None
        and manifest_error is None
        and acceptance_error is None
        and preflight_passed
        and manifest_passed
        and acceptance_passed
        and acceptance_strict_final
        and audit_checklist_passed
        and preflight_path_check["ok"]
        and all(check["ok"] for check in phase_checks)
        and all(check["ok"] for check in command_checks)
        and manifest_path_check["ok"]
    )
    return {
        "passed": passed,
        "evidence_status": status,
        "file_checks": file_checks,
        "preflight": {
            "path": str(preflight_path),
            "load_error": preflight_error,
            "passed": preflight_passed,
            "expected_path_check": preflight_path_check,
            "failed_checks": (
                [
                    check
                    for check in preflight.get("checks", [])
                    if isinstance(check, dict) and not check.get("ok")
                ]
                if isinstance(preflight, dict)
                else None
            ),
        },
        "manifest": {
            "path": str(manifest_path),
            "load_error": manifest_error,
            "passed": manifest_passed,
            "expected_path_check": manifest_path_check,
            "phase_checks": phase_checks,
            "command_checks": command_checks,
        },
        "acceptance": {
            "path": str(acceptance_path),
            "load_error": acceptance_error,
            "passed": acceptance_passed,
            "strict_final": acceptance_strict_final,
            "audit_checklist_present": isinstance(audit_checklist, list)
            and bool(audit_checklist),
            "failed_audit_items": audit_failed_items,
            "failure_count": (
                len(acceptance.get("failures", []))
                if isinstance(acceptance, dict)
                else None
            ),
        },
    }


def _write_evidence_check_report(args: argparse.Namespace, report: dict) -> Path:
    path = _evidence_check_json_path(args)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _print_evidence_check(report: dict) -> None:
    for check in report["file_checks"]:
        status = "OK" if check["ok"] else "MISSING"
        print(
            f"{status} {check['group']}.{check['name']}: "
            f"{check['path']} bytes={check['bytes']}",
            flush=True,
        )
    manifest = report["manifest"]
    preflight = report["preflight"]
    preflight_status = "OK" if preflight["passed"] else "FAIL"
    failed_preflight_count = (
        len(preflight["failed_checks"])
        if isinstance(preflight["failed_checks"], list)
        else None
    )
    print(
        f"{preflight_status} preflight_passed: {preflight['path']} "
        f"failed_checks={failed_preflight_count}",
        flush=True,
    )
    preflight_path_status = "OK" if preflight["expected_path_check"]["ok"] else "FAIL"
    print(
        f"{preflight_path_status} preflight_expected_paths: "
        f"outputs={preflight['expected_path_check']['expected_outputs_match']} "
        f"logs={preflight['expected_path_check']['expected_logs_match']} "
        f"manifest={preflight['expected_path_check']['run_manifest_match']}",
        flush=True,
    )
    manifest_status = "OK" if manifest["passed"] else "FAIL"
    print(f"{manifest_status} manifest_passed: {manifest['path']}", flush=True)
    path_status = "OK" if manifest["expected_path_check"]["ok"] else "FAIL"
    print(
        f"{path_status} manifest_expected_paths: "
        f"outputs={manifest['expected_path_check']['expected_outputs_match']} "
        f"logs={manifest['expected_path_check']['logs_match']}",
        flush=True,
    )
    for check in manifest["phase_checks"]:
        status = "OK" if check["ok"] else "FAIL"
        print(
            f"{status} phase.{check['phase']}: returncode={check['returncode']}",
            flush=True,
        )
    for check in manifest["command_checks"]:
        status = "OK" if check["ok"] else "FAIL"
        print(
            f"{status} command.{check['phase']}: "
            f"command={check['has_command']} "
            f"command_string={check['has_command_string']}",
            flush=True,
        )
    acceptance = report["acceptance"]
    acceptance_status = "OK" if acceptance["passed"] else "FAIL"
    print(
        f"{acceptance_status} acceptance_passed: {acceptance['path']} "
        f"failures={acceptance['failure_count']}",
        flush=True,
    )
    strict_status = "OK" if acceptance["strict_final"] else "FAIL"
    print(
        f"{strict_status} acceptance_strict_final: {acceptance['path']}",
        flush=True,
    )
    failed_audit_count = (
        len(acceptance["failed_audit_items"])
        if isinstance(acceptance["failed_audit_items"], list)
        else None
    )
    audit_status = (
        "OK"
        if acceptance["audit_checklist_present"] and failed_audit_count == 0
        else "FAIL"
    )
    print(
        f"{audit_status} acceptance_audit_checklist: "
        f"present={acceptance['audit_checklist_present']} "
        f"failed_items={failed_audit_count}",
        flush=True,
    )


def _path_nonempty(path: Path) -> bool | None:
    if path.is_dir():
        return any(path.iterdir())
    if path.is_file():
        return path.stat().st_size > 0
    return None


def _path_check(
    label: str,
    raw_path: str | Path,
    *,
    kind: str,
    require_nonempty: bool = False,
) -> dict:
    path = Path(raw_path).expanduser()
    exists = path.exists()
    if kind == "dir":
        kind_ok = exists and path.is_dir()
    elif kind == "file":
        kind_ok = exists and path.is_file()
    else:
        raise ValueError(f"unsupported path check kind: {kind}")
    nonempty = _path_nonempty(path) if exists else None
    return {
        "label": label,
        "path": str(path),
        "resolved": str(path.resolve()) if exists else None,
        "kind": kind,
        "exists": exists,
        "nonempty": nonempty,
        "require_nonempty": require_nonempty,
        "ok": kind_ok and (not require_nonempty or nonempty is True),
    }


def _python_check(python: str) -> dict:
    if "/" in python:
        path = Path(python).expanduser()
        exists = path.exists()
        return {
            "label": "python",
            "path": str(path),
            "resolved": str(path.resolve()) if exists else None,
            "kind": "file",
            "exists": exists,
            "ok": exists and path.is_file(),
        }
    resolved = shutil.which(python)
    return {
        "label": "python",
        "path": python,
        "resolved": resolved,
        "kind": "executable",
        "exists": resolved is not None,
        "ok": resolved is not None,
    }


def _git_branch_check(expected_branch: str) -> dict:
    proc = subprocess.run(
        ["git", "-C", str(REPO_ROOT), "branch", "--show-current"],
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    actual_branch = proc.stdout.strip()
    return {
        "label": "git_branch",
        "path": str(REPO_ROOT),
        "kind": "git_branch",
        "expected_branch": expected_branch,
        "actual_branch": actual_branch or None,
        "returncode": proc.returncode,
        "stderr": proc.stderr.strip() or None,
        "ok": proc.returncode == 0 and actual_branch == expected_branch,
    }


def _preflight_checks(args: argparse.Namespace) -> list[dict]:
    checks = [
        _git_branch_check(args.expected_branch),
        _python_check(args.python),
        _path_check("model_path", args.model_path, kind="dir", require_nonempty=True),
        _path_check(
            "artifacts_2k",
            args.artifacts_2k,
            kind="dir",
            require_nonempty=True,
        ),
        _path_check(
            "artifacts_8k",
            args.artifacts_8k,
            kind="dir",
            require_nonempty=True,
        ),
        _path_check(
            "artifacts_32k",
            args.artifacts_32k,
            kind="dir",
            require_nonempty=True,
        ),
        _path_check(
            "artifacts_128k",
            args.artifacts_128k,
            kind="dir",
            require_nonempty=True,
        ),
        _path_check(
            "artifacts_262k",
            args.artifacts_262k,
            kind="dir",
            require_nonempty=True,
        ),
        _path_check("benchmark_script", BENCHMARK, kind="file"),
        _path_check("acceptance_script", ACCEPTANCE, kind="file"),
        _path_check("hybrid_apc_validation_script", HYBRID_APC_VALIDATION, kind="file"),
    ]
    output_parent = args.output_dir.expanduser().parent
    checks.append(_path_check("output_dir_parent", output_parent, kind="dir"))
    if args.gdn_state_diff_json is not None:
        checks.append(
            _path_check("gdn_state_diff_json", args.gdn_state_diff_json, kind="file")
        )
    return checks


def _print_preflight(checks: list[dict]) -> None:
    for check in checks:
        status = "OK" if check["ok"] else "MISSING"
        detail = check.get("resolved") or check.get("path") or check.get("label")
        if check.get("kind") == "git_branch":
            detail = (
                f"{detail} actual={check.get('actual_branch')} "
                f"expected={check.get('expected_branch')}"
            )
        if check.get("require_nonempty"):
            detail = f"{detail} nonempty={check.get('nonempty')}"
        print(f"{status} {check['label']}: {detail}", flush=True)


def _preflight_json_path(args: argparse.Namespace) -> Path:
    if args.preflight_json is not None:
        return args.preflight_json.expanduser()
    return args.output_dir.expanduser() / f"{args.output_prefix}_preflight.json"


def _preflight_report(args: argparse.Namespace, checks: list[dict]) -> dict:
    return {
        "passed": all(check["ok"] for check in checks),
        "checks": checks,
        "model_path": args.model_path,
        "artifacts": {
            "2k": args.artifacts_2k,
            "8k": args.artifacts_8k,
            "32k": args.artifacts_32k,
            "128k": args.artifacts_128k,
            "262k": args.artifacts_262k,
        },
        "output_dir": str(args.output_dir.expanduser()),
        "output_prefix": args.output_prefix,
        "expected_branch": args.expected_branch,
        "runtime_shape": {
            "tensor_parallel_size": args.tensor_parallel_size,
            "logical_nc_config": args.logical_nc_config,
            "max_num_seqs": args.max_num_seqs,
            "ctx_batch_size": args.ctx_batch_size,
            "num_gpu_blocks_override": args.num_gpu_blocks_override,
            "num_gpu_blocks_override_by_len": args.num_gpu_blocks_override_by_len,
            "enable_prefix_caching": args.enable_prefix_caching,
            "enable_hybrid_apc": args.enable_hybrid_apc,
            "hybrid_apc_require_vllm_metadata": (
                args.hybrid_apc_require_vllm_metadata
            ),
            "block_size": args.block_size,
            "compiled_max_prompt_length": args.compiled_max_prompt_length,
            "gdn_checkpoint_interval": args.gdn_checkpoint_interval,
            "max_gdn_checkpoint_slots": args.max_gdn_checkpoint_slots,
            "hybrid_cache_mode": args.hybrid_cache_mode,
        },
        "expected_outputs": _expected_output_paths(args),
        "expected_logs": _phase_log_paths(args),
        "run_manifest": str(_run_manifest_path(args)),
    }


def _write_preflight_report(args: argparse.Namespace, report: dict) -> Path:
    path = _preflight_json_path(args)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _run_manifest(args: argparse.Namespace, commands: dict[str, list[str]]) -> dict:
    return {
        "strict_final": True,
        "output_dir": str(args.output_dir.expanduser()),
        "output_prefix": args.output_prefix,
        "commands": commands,
        "command_strings": {
            phase: shlex.join(command) for phase, command in commands.items()
        },
        "expected_outputs": _expected_output_paths(args),
        "logs": _phase_log_paths(args),
        "evidence_status": _evidence_status(args),
        "phases": {},
    }


def _write_run_manifest(args: argparse.Namespace, manifest: dict) -> Path:
    path = _run_manifest_path(args)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _run_logged_command(name: str, command: list[str], log_path: Path) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"[{name}] log: {log_path}", flush=True)
    with log_path.open("w", encoding="utf-8") as log_file:
        log_file.write(f"$ {shlex.join(command)}\n\n")
        log_file.flush()
        proc = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        if proc.stdout is not None:
            for line in proc.stdout:
                sys.stdout.write(line)
                sys.stdout.flush()
                log_file.write(line)
                log_file.flush()
        returncode = proc.wait()
        footer = f"\n[{name}] returncode={returncode}\n"
        sys.stdout.write(footer)
        sys.stdout.flush()
        log_file.write(footer)
        log_file.flush()
        return returncode


def _should_stop_after_phase(args: argparse.Namespace, phase_returncode: int) -> bool:
    return bool(args.fail_fast and phase_returncode != 0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path")
    parser.add_argument("--artifacts-2k")
    parser.add_argument("--artifacts-8k")
    parser.add_argument("--artifacts-32k")
    parser.add_argument("--artifacts-128k")
    parser.add_argument("--artifacts-262k")
    parser.add_argument("--output-dir", type=Path, default=Path("/tmp"))
    parser.add_argument("--output-prefix", default="qwen36_cold_prefill")
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--strict-min-samples", type=int, default=3)
    parser.add_argument("--tensor-parallel-size", type=int, default=4)
    parser.add_argument("--logical-nc-config", type=int, default=2)
    parser.add_argument("--max-num-seqs", type=int, default=1)
    parser.add_argument("--ctx-batch-size", type=int, default=1)
    parser.add_argument("--num-gpu-blocks-override", type=int)
    parser.add_argument("--num-gpu-blocks-override-by-len", nargs="+", default=None)
    parser.add_argument("--compiled-max-prompt-length", type=int, default=1024)
    parser.add_argument("--block-size", type=int, default=128)
    parser.add_argument("--enable-prefix-caching", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--enable-hybrid-apc", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--hybrid-apc-require-vllm-metadata",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--gdn-checkpoint-interval", type=int, default=128)
    parser.add_argument("--max-gdn-checkpoint-slots", type=int, default=8)
    parser.add_argument("--hybrid-cache-mode", default="all")
    parser.add_argument(
        "--gdn-state-diff-json",
        type=Path,
        help=(
            "Optional sidecar consumed by benchmark phases for GDN recurrent/conv "
            "state-diff fields."
        ),
    )
    parser.add_argument("--short-latency-speedup", type=float, default=1.5)
    parser.add_argument("--baseline-cold-tok-per-s-target", type=float, default=420.0)
    parser.add_argument(
        "--baseline-cold-tok-per-s-tolerance",
        type=float,
        default=0.15,
    )
    parser.add_argument("--bucket-tok-regression-tolerance", type=float, default=0.20)
    parser.add_argument("--prompt-token-tolerance", type=float, default=0.05)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--expected-branch", default=DEFAULT_EXPECTED_BRANCH)
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--preflight",
        action="store_true",
        help="Check local model/artifact/script paths and exit before launching vLLM.",
    )
    parser.add_argument(
        "--preflight-json",
        type=Path,
        default=None,
        help=(
            "Path for the machine-readable preflight report. Defaults to "
            "--output-dir/OUTPUT_PREFIX_preflight.json when --preflight is used."
        ),
    )
    parser.add_argument(
        "--evidence-check",
        action="store_true",
        help=(
            "Inspect an existing strict-final output directory and verify the "
            "preflight JSON, run manifest, phase logs, benchmark outputs, and "
            "acceptance JSON are present and passing."
        ),
    )
    parser.add_argument(
        "--evidence-check-json",
        type=Path,
        default=None,
        help=(
            "Path for the machine-readable evidence-check report. Defaults to "
            "--output-dir/OUTPUT_PREFIX_evidence_check.json."
        ),
    )
    args = parser.parse_args()
    if args.repetitions < args.strict_min_samples:
        parser.error("--repetitions must be >= --strict-min-samples")
    if not args.evidence_check:
        required_paths = (
            ("model_path", "--model-path"),
            ("artifacts_2k", "--artifacts-2k"),
            ("artifacts_8k", "--artifacts-8k"),
            ("artifacts_32k", "--artifacts-32k"),
            ("artifacts_128k", "--artifacts-128k"),
            ("artifacts_262k", "--artifacts-262k"),
        )
        for attr, flag in required_paths:
            if getattr(args, attr) is None:
                parser.error(f"{flag} is required unless --evidence-check is used")
    return args


def main() -> int:
    args = parse_args()
    if args.evidence_check:
        report = _evidence_check_report(args)
        report_path = _write_evidence_check_report(args, report)
        _print_evidence_check(report)
        print(f"EVIDENCE_CHECK_JSON {report_path}", flush=True)
        return 0 if report["passed"] else 2

    if args.preflight:
        checks = _preflight_checks(args)
        report = _preflight_report(args, checks)
        report_path = _write_preflight_report(args, report)
        _print_preflight(checks)
        print(f"PREFLIGHT_JSON {report_path}", flush=True)
        return 0 if report["passed"] else 2

    commands = _build_commands(args)
    if args.dry_run:
        for name, command in commands.items():
            print(f"{name}: {shlex.join(command)}", flush=True)
        return 0

    args.output_dir.expanduser().mkdir(parents=True, exist_ok=True)
    manifest = _run_manifest(args, commands)
    manifest_path = _write_run_manifest(args, manifest)
    print(f"RUN_MANIFEST {manifest_path}", flush=True)
    returncode = 0
    for name in ("matrix", "long_context", "hybrid_apc"):
        start = time.perf_counter()
        phase_returncode = _run_logged_command(
            name,
            commands[name],
            _phase_log_path(args, name),
        )
        manifest["phases"][name] = {
            "returncode": phase_returncode,
            "elapsed_seconds": time.perf_counter() - start,
            "log": str(_phase_log_path(args, name)),
        }
        manifest["evidence_status"] = _evidence_status(args)
        _write_run_manifest(args, manifest)
        if phase_returncode != 0 and returncode == 0:
            returncode = phase_returncode
        if _should_stop_after_phase(args, phase_returncode):
            manifest["returncode"] = phase_returncode
            manifest["passed"] = False
            manifest["aborted_after_phase"] = name
            manifest["evidence_status"] = _evidence_status(args)
            _write_run_manifest(args, manifest)
            print(
                f"[{name}] fail-fast aborting strict-final before remaining phases",
                flush=True,
            )
            return phase_returncode

    start = time.perf_counter()
    acceptance_returncode = _run_logged_command(
        "acceptance",
        commands["acceptance"],
        _phase_log_path(args, "acceptance"),
    )
    manifest["phases"]["acceptance"] = {
        "returncode": acceptance_returncode,
        "elapsed_seconds": time.perf_counter() - start,
        "log": str(_phase_log_path(args, "acceptance")),
    }
    manifest["returncode"] = acceptance_returncode if acceptance_returncode != 0 else returncode
    manifest["passed"] = manifest["returncode"] == 0
    manifest["evidence_status"] = _evidence_status(args)
    _write_run_manifest(args, manifest)
    if acceptance_returncode != 0:
        return acceptance_returncode
    return returncode


if __name__ == "__main__":
    raise SystemExit(main())
