#!/usr/bin/env python3
"""Profile loose Qwen3.6 context-encoding NEFFs with neuron-explorer."""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence


_RESOURCE_MARKERS = (
    "NRT_RESOURCE",
    "out of memory",
    "insufficient",
    "allocated memory",
    "failed to allocated resource",
)


@dataclass(frozen=True)
class ProfileTarget:
    label: str
    tp_rank: int
    bucket: int
    prefix_tokens: int | None
    neff: Path


def _parse_prefix_map(raw: str) -> dict[int, int]:
    mapping: dict[int, int] = {}
    if not raw:
        return mapping
    for item in raw.split(","):
        part = item.strip()
        if not part:
            continue
        if ":" not in part:
            raise ValueError(f"prefix-map entry must be BUCKET:PREFIX, got {part!r}")
        bucket_raw, prefix_raw = part.split(":", 1)
        mapping[int(bucket_raw)] = int(prefix_raw)
    return mapping


def _default_prefix_for_bucket(bucket: int) -> int | None:
    if bucket == 0:
        return 0
    if bucket > 0:
        return 2 ** (bucket + 7)
    return None


def _bucket_from_path(path: Path) -> tuple[int, int] | None:
    match = re.fullmatch(r"_tp(\d+)_bk(\d+)", path.parent.name)
    if match is None:
        return None
    return int(match.group(1)), int(match.group(2))


def _label(bucket: int, prefix_tokens: int | None) -> str:
    if prefix_tokens is None:
        return f"context_bk{bucket}"
    return f"context_bk{bucket}_pfx{prefix_tokens}"


def discover_targets(
    *,
    context_neff_root: Path,
    tp_rank: int,
    buckets: set[int] | None,
    prefix_map: dict[int, int],
) -> list[ProfileTarget]:
    targets: list[ProfileTarget] = []
    for neff in sorted(context_neff_root.glob("_tp*_bk*/graph.neff")):
        parsed = _bucket_from_path(neff)
        if parsed is None:
            continue
        target_tp, bucket = parsed
        if target_tp != tp_rank:
            continue
        if buckets is not None and bucket not in buckets:
            continue
        prefix_tokens = prefix_map.get(bucket, _default_prefix_for_bucket(bucket))
        targets.append(
            ProfileTarget(
                label=_label(bucket, prefix_tokens),
                tp_rank=target_tp,
                bucket=bucket,
                prefix_tokens=prefix_tokens,
                neff=neff,
            )
        )
    return targets


def _capture_command(
    *,
    tool: Path,
    neff: Path,
    ntff_path: Path,
    collectives_worker_count: int,
    collectives_workers_per_node: int,
    num_exec: int,
    profile_nth_exec: int,
    enable_dge: bool,
    single_io: bool,
) -> list[str]:
    command = [
        str(tool),
        "capture",
        "-n",
        str(neff),
        "-s",
        str(ntff_path),
        "--collectives-worker-start-id=0",
        f"--collectives-worker-count={collectives_worker_count}",
        f"--collectives-workers-per-node={collectives_workers_per_node}",
        "--collectives-profile-id=0",
        f"--num-exec={num_exec}",
        f"--profile-nth-exec={profile_nth_exec}",
        "--ignore-exec-errors",
        "--io-from=neff",
    ]
    if single_io:
        command.append("--single-io")
    if enable_dge:
        command.append("--enable-dge-notifs")
    return command


def _view_command(
    *,
    tool: Path,
    neff: Path,
    ntff: Path,
    ignore_nc_buf_usage: bool,
) -> list[str]:
    command = [
        str(tool),
        "view",
        "-n",
        str(neff),
        "-s",
        str(ntff),
        "--output-format",
        "summary-json",
    ]
    if ignore_nc_buf_usage:
        command.append("--ignore-nc-buf-usage")
    return command


def _target_plan(
    *,
    target: ProfileTarget,
    output_dir: Path,
    tool: Path,
    collectives_worker_count: int,
    collectives_workers_per_node: int,
    num_exec: int,
    profile_nth_exec: int,
    enable_dge: bool,
) -> dict[str, Any]:
    target_dir = output_dir / target.label
    ntff_path = target_dir / "profile.ntff"
    return {
        "label": target.label,
        "tp_rank": target.tp_rank,
        "bucket": target.bucket,
        "prefix_tokens": target.prefix_tokens,
        "neff": str(target.neff),
        "output_dir": str(target_dir),
        "capture_command": _capture_command(
            tool=tool,
            neff=target.neff,
            ntff_path=ntff_path,
            collectives_worker_count=collectives_worker_count,
            collectives_workers_per_node=collectives_workers_per_node,
            num_exec=num_exec,
            profile_nth_exec=profile_nth_exec,
            enable_dge=enable_dge,
            single_io=False,
        ),
        "single_io_capture_command": _capture_command(
            tool=tool,
            neff=target.neff,
            ntff_path=target_dir / "profile_singleio.ntff",
            collectives_worker_count=collectives_worker_count,
            collectives_workers_per_node=collectives_workers_per_node,
            num_exec=num_exec,
            profile_nth_exec=profile_nth_exec,
            enable_dge=enable_dge,
            single_io=True,
        ),
    }


def build_plan(
    *,
    context_neff_root: Path,
    output_dir: Path,
    tool: Path,
    tp_rank: int,
    buckets: set[int] | None,
    prefix_map: dict[int, int],
    collectives_worker_count: int,
    collectives_workers_per_node: int,
    num_exec: int,
    profile_nth_exec: int,
    enable_dge: bool,
) -> dict[str, Any]:
    targets = discover_targets(
        context_neff_root=context_neff_root,
        tp_rank=tp_rank,
        buckets=buckets,
        prefix_map=prefix_map,
    )
    return {
        "context_neff_root": str(context_neff_root),
        "output_dir": str(output_dir),
        "tool": str(tool),
        "tp_rank": tp_rank,
        "buckets": sorted(buckets) if buckets is not None else None,
        "target_count": len(targets),
        "targets": [
            _target_plan(
                target=target,
                output_dir=output_dir,
                tool=tool,
                collectives_worker_count=collectives_worker_count,
                collectives_workers_per_node=collectives_workers_per_node,
                num_exec=num_exec,
                profile_nth_exec=profile_nth_exec,
                enable_dge=enable_dge,
            )
            for target in targets
        ],
    }


def _run_command(command: Sequence[str], *, stdout_path: Path, stderr_path: Path) -> int:
    with stdout_path.open("w") as stdout, stderr_path.open("w") as stderr:
        completed = subprocess.run(command, stdout=stdout, stderr=stderr)
    return completed.returncode


def _has_resource_error(path: Path) -> bool:
    if not path.exists():
        return False
    text = path.read_text(errors="replace")
    lowered = text.lower()
    return any(marker.lower() in lowered for marker in _RESOURCE_MARKERS)


def _find_ntff(directory: Path) -> Path | None:
    exec_2 = sorted(directory.glob("*exec_2*.ntff"))
    if exec_2:
        return exec_2[0]
    ntffs = sorted(directory.glob("*.ntff"))
    return ntffs[0] if ntffs else None


def _load_summary_metrics(summary_path: Path) -> dict[str, Any] | None:
    try:
        spec = importlib.util.spec_from_file_location(
            "qwen36_profile_summary_compare",
            Path(__file__).with_name("qwen36_profile_summary_compare.py"),
        )
        if spec is None or spec.loader is None:
            return None
        profile_compare = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(profile_compare)
    except (ImportError, OSError):
        return None
    try:
        return profile_compare.extract_summary_metrics(
            json.loads(summary_path.read_text())
        )
    except (OSError, ValueError, json.JSONDecodeError):
        return None


def run_target(plan_target: dict[str, Any]) -> dict[str, Any]:
    target_dir = Path(plan_target["output_dir"])
    target_dir.mkdir(parents=True, exist_ok=True)
    (target_dir / "target.json").write_text(
        json.dumps(plan_target, indent=2, sort_keys=True) + "\n"
    )
    start = time.perf_counter()
    capture_stdout = target_dir / "capture.stdout"
    capture_stderr = target_dir / "capture.stderr"
    capture_rc = _run_command(
        plan_target["capture_command"],
        stdout_path=capture_stdout,
        stderr_path=capture_stderr,
    )
    (target_dir / "capture.exit").write_text(f"{capture_rc}\n")

    single_io_rc: int | None = None
    if capture_rc != 0 and (
        _has_resource_error(capture_stdout) or _has_resource_error(capture_stderr)
    ):
        single_stdout = target_dir / "capture_singleio.stdout"
        single_stderr = target_dir / "capture_singleio.stderr"
        single_io_rc = _run_command(
            plan_target["single_io_capture_command"],
            stdout_path=single_stdout,
            stderr_path=single_stderr,
        )
        (target_dir / "capture_singleio.exit").write_text(f"{single_io_rc}\n")

    ntff = _find_ntff(target_dir)
    view_rc: int | None = None
    summary_path = target_dir / "summary.json"
    summary_metrics = None
    if ntff is not None:
        view_command = _view_command(
            tool=Path(plan_target["capture_command"][0]),
            neff=Path(plan_target["neff"]),
            ntff=ntff,
            ignore_nc_buf_usage=True,
        )
        with summary_path.open("w") as stdout, (target_dir / "summary.err").open(
            "w"
        ) as stderr:
            completed = subprocess.run(view_command, stdout=stdout, stderr=stderr)
        view_rc = completed.returncode
        if view_rc != 0:
            view_command = _view_command(
                tool=Path(plan_target["capture_command"][0]),
                neff=Path(plan_target["neff"]),
                ntff=ntff,
                ignore_nc_buf_usage=False,
            )
            with summary_path.open("w") as stdout, (
                target_dir / "summary_retry.err"
            ).open("w") as stderr:
                completed = subprocess.run(view_command, stdout=stdout, stderr=stderr)
            view_rc = completed.returncode
        (target_dir / "summary.exit").write_text(f"{view_rc}\n")
        if view_rc == 0:
            summary_metrics = _load_summary_metrics(summary_path)

    elapsed = time.perf_counter() - start
    result = {
        "label": plan_target["label"],
        "neff": plan_target["neff"],
        "output_dir": plan_target["output_dir"],
        "capture_returncode": capture_rc,
        "single_io_returncode": single_io_rc,
        "ntff": str(ntff) if ntff is not None else None,
        "view_returncode": view_rc,
        "summary_json": str(summary_path) if summary_path.exists() else None,
        "summary_metrics": summary_metrics,
        "elapsed_seconds": elapsed,
        "passed": capture_rc == 0 and ntff is not None and view_rc == 0,
    }
    (target_dir / "result.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n"
    )
    return result


def run_plan(plan: dict[str, Any]) -> dict[str, Any]:
    output_dir = Path(plan["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "context_neff_profile_plan.json").write_text(
        json.dumps(plan, indent=2, sort_keys=True) + "\n"
    )
    results = [run_target(target) for target in plan["targets"]]
    summary_paths = [
        result["summary_json"]
        for result in results
        if result.get("passed") and result.get("summary_json")
    ]
    report = {
        **plan,
        "results": results,
        "passed": bool(results) and all(bool(result["passed"]) for result in results),
        "summary_json_paths": summary_paths,
    }
    (output_dir / "context_neff_profile_results.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    return report


def _parse_bucket_filter(raw: str | None) -> set[int] | None:
    if raw is None or raw.strip() == "":
        return None
    return {int(item.strip()) for item in raw.split(",") if item.strip()}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--context-neff-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--tool", type=Path, default=Path("/opt/aws/neuron/bin/neuron-explorer"))
    parser.add_argument("--tp-rank", type=int, default=0)
    parser.add_argument("--buckets", default=None, help="Comma-separated bucket ids")
    parser.add_argument(
        "--prefix-map",
        default="",
        help="Optional BUCKET:PREFIX overrides, comma-separated",
    )
    parser.add_argument("--collectives-worker-count", type=int, default=4)
    parser.add_argument("--collectives-workers-per-node", type=int, default=4)
    parser.add_argument("--num-exec", type=int, default=2)
    parser.add_argument("--profile-nth-exec", type=int, default=2)
    parser.add_argument("--enable-dge", action="store_true")
    parser.add_argument(
        "--run",
        action="store_true",
        help="Execute captures. Without this, only print/write the plan JSON.",
    )
    args = parser.parse_args()

    plan = build_plan(
        context_neff_root=args.context_neff_root.expanduser(),
        output_dir=args.output_dir.expanduser(),
        tool=args.tool.expanduser(),
        tp_rank=args.tp_rank,
        buckets=_parse_bucket_filter(args.buckets),
        prefix_map=_parse_prefix_map(args.prefix_map),
        collectives_worker_count=args.collectives_worker_count,
        collectives_workers_per_node=args.collectives_workers_per_node,
        num_exec=args.num_exec,
        profile_nth_exec=args.profile_nth_exec,
        enable_dge=args.enable_dge,
    )
    if plan["target_count"] == 0:
        print(json.dumps(plan, indent=2, sort_keys=True))
        return 2
    if not args.run:
        args.output_dir.expanduser().mkdir(parents=True, exist_ok=True)
        (args.output_dir.expanduser() / "context_neff_profile_plan.json").write_text(
            json.dumps(plan, indent=2, sort_keys=True) + "\n"
        )
        print(json.dumps(plan, indent=2, sort_keys=True))
        return 0
    report = run_plan(plan)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
