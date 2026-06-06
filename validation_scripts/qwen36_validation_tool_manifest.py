#!/usr/bin/env python3
"""Build or verify a manifest for Qwen3.6 validation/profiling tools."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any, Iterable


DEFAULT_FILES = [
    "tmp_compile_qwen32k_segcte2048_gdnseg512.sh",
    "tmp_launch_qwen36_segcte2048.sh",
    "validation_scripts/qwen36_compile_status.py",
    "validation_scripts/qwen36_compile_monitor_prompt.py",
    "validation_scripts/qwen36_artifact_config_audit.py",
    "validation_scripts/qwen36_runtime_validation_matrix.py",
    "validation_scripts/qwen36_runtime_log_scan.py",
    "validation_scripts/qwen36_raw_completion_prefill_bench.py",
    "validation_scripts/qwen36_openai_boundary_apc_probe.py",
    "validation_scripts/qwen36_chat_completion_context_bench.py",
    "validation_scripts/qwen36_context_neff_profile.py",
    "validation_scripts/qwen36_profile_summary_compare.py",
    "validation_scripts/qwen36_launch_env_audit.py",
    "validation_scripts/qwen36_launch_preflight.py",
    "validation_scripts/qwen36_speed_slice_decision.py",
    "validation_scripts/qwen36_next_compile_preflight.py",
    "validation_scripts/qwen36_validation_tool_manifest.py",
]

DEFAULT_TEST_FILES = [
    "contrib/models/Qwen3.6-27B/test/unit/test_qwen36_compile_driver.py",
    "contrib/models/Qwen3.6-27B/test/unit/test_qwen36_artifact_config_audit.py",
    "test/unit/scripts/test_qwen36_launch_driver.py",
    "test/unit/scripts/test_qwen36_compile_status.py",
    "test/unit/scripts/test_qwen36_compile_monitor_prompt.py",
    "test/unit/scripts/test_qwen36_runtime_validation_matrix.py",
    "test/unit/scripts/test_qwen36_runtime_log_scan.py",
    "test/unit/scripts/test_qwen36_raw_completion_prefill_bench.py",
    "test/unit/scripts/test_qwen36_context_neff_profile.py",
    "test/unit/scripts/test_qwen36_profile_summary_compare.py",
    "test/unit/scripts/test_qwen36_launch_env_audit.py",
    "test/unit/scripts/test_qwen36_launch_preflight.py",
    "test/unit/scripts/test_qwen36_speed_slice_decision.py",
    "test/unit/scripts/test_qwen36_next_compile_preflight.py",
    "test/unit/scripts/test_qwen36_validation_tool_manifest.py",
]


def _repo_commit(repo: Path) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=repo,
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _iter_files(files: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for raw in files:
        path = raw.strip()
        if not path or path in seen:
            continue
        seen.add(path)
        ordered.append(path)
    return ordered


def build_manifest(*, repo: Path, files: Iterable[str]) -> dict[str, Any]:
    entries = []
    for relative in _iter_files(files):
        path = repo / relative
        if not path.exists():
            entries.append(
                {
                    "path": relative,
                    "exists": False,
                    "sha256": None,
                    "size_bytes": None,
                }
            )
            continue
        entries.append(
            {
                "path": relative,
                "exists": True,
                "sha256": _sha256(path),
                "size_bytes": path.stat().st_size,
            }
        )
    return {
        "schema": "qwen36-validation-tool-manifest-v1",
        "repo": str(repo),
        "git_commit": _repo_commit(repo),
        "files": entries,
    }


def verify_manifest(*, repo: Path, manifest: dict[str, Any]) -> dict[str, Any]:
    mismatches = []
    files = manifest.get("files")
    if not isinstance(files, list):
        raise ValueError("manifest is missing a files list")
    for entry in files:
        if not isinstance(entry, dict):
            continue
        relative = str(entry.get("path", ""))
        expected_exists = bool(entry.get("exists"))
        expected_sha = entry.get("sha256")
        expected_size = entry.get("size_bytes")
        path = repo / relative
        actual_exists = path.exists()
        actual_sha = _sha256(path) if actual_exists else None
        actual_size = path.stat().st_size if actual_exists else None
        if (
            actual_exists != expected_exists
            or actual_sha != expected_sha
            or actual_size != expected_size
        ):
            mismatches.append(
                {
                    "path": relative,
                    "expected": {
                        "exists": expected_exists,
                        "sha256": expected_sha,
                        "size_bytes": expected_size,
                    },
                    "actual": {
                        "exists": actual_exists,
                        "sha256": actual_sha,
                        "size_bytes": actual_size,
                    },
                }
            )
    return {
        "schema": "qwen36-validation-tool-verify-v1",
        "repo": str(repo),
        "git_commit": _repo_commit(repo),
        "source_git_commit": manifest.get("git_commit"),
        "file_count": len(files),
        "mismatch_count": len(mismatches),
        "mismatches": mismatches,
        "passed": not mismatches,
    }


def _file_list(manifest: dict[str, Any]) -> list[str]:
    files = manifest.get("files", [])
    return [
        str(entry["path"])
        for entry in files
        if isinstance(entry, dict) and bool(entry.get("exists"))
    ]


def _load_manifest(path: Path) -> dict[str, Any]:
    with path.open() as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError("manifest JSON must be an object")
    return payload


def _files_from_args(args: argparse.Namespace) -> list[str]:
    files = list(DEFAULT_FILES)
    if args.include_tests:
        files.extend(DEFAULT_TEST_FILES)
    if args.file:
        files.extend(args.file)
    return files


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument("--include-tests", action="store_true")
    parser.add_argument(
        "--file",
        action="append",
        help="Additional repo-relative file to include in generated manifests.",
    )
    parser.add_argument(
        "--mode",
        choices=("build", "verify", "file-list"),
        default="build",
    )
    args = parser.parse_args()

    repo = args.repo.expanduser().resolve()
    if args.mode == "build":
        payload = build_manifest(repo=repo, files=_files_from_args(args))
        exit_code = 0 if all(entry["exists"] for entry in payload["files"]) else 1
    else:
        if args.manifest is None:
            raise SystemExit("--manifest is required for verify/file-list mode")
        manifest = _load_manifest(args.manifest.expanduser())
        if args.mode == "verify":
            payload = verify_manifest(repo=repo, manifest=manifest)
            exit_code = 0 if payload["passed"] else 1
        else:
            payload = {
                "schema": "qwen36-validation-tool-file-list-v1",
                "files": _file_list(manifest),
            }
            exit_code = 0

    encoded = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if args.output_json is not None:
        args.output_json.expanduser().parent.mkdir(parents=True, exist_ok=True)
        args.output_json.expanduser().write_text(encoded)
    print(encoded, end="")
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
