#!/usr/bin/env python3
"""Run the Qwen3.6 coherence-first runtime validation matrix."""

from __future__ import annotations

import argparse
import importlib.util
import json
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence


_SCRIPT_DIR = Path(__file__).resolve().parent
_DECISION_SCRIPT = _SCRIPT_DIR / "qwen36_speed_slice_decision.py"


@dataclass(frozen=True)
class ValidationStep:
    name: str
    phase: str
    command: list[str]
    output_path: Path | None = None


def _csv_range(start: int, end: int) -> str:
    if end < start:
        raise ValueError("range end must be >= start")
    return ",".join(str(value) for value in range(start, end + 1))


def _script(name: str) -> str:
    return str(_SCRIPT_DIR / name)


def _common_model_args(args: argparse.Namespace) -> list[str]:
    return [
        "--base-url",
        args.base_url,
        "--model",
        args.model,
        "--model-path",
        args.model_path,
    ]


def build_steps(args: argparse.Namespace) -> list[ValidationStep]:
    out = Path(args.output_dir)
    steps = [
        ValidationStep(
            name="boundary_primary",
            phase="coherence",
            output_path=out / "boundary_primary.jsonl",
            command=[
                sys.executable,
                _script("qwen36_openai_boundary_apc_probe.py"),
                *_common_model_args(args),
                "--lengths",
                args.boundary_lengths,
                "--repeats",
                str(args.boundary_repeats),
                "--max-tokens",
                str(args.boundary_max_tokens),
                "--timeout",
                str(args.timeout),
                "--output-jsonl",
                str(out / "boundary_primary.jsonl"),
            ],
        ),
        ValidationStep(
            name="boundary_repeated",
            phase="coherence",
            output_path=out / "boundary_repeated.jsonl",
            command=[
                sys.executable,
                _script("qwen36_openai_boundary_apc_probe.py"),
                *_common_model_args(args),
                "--lengths",
                args.repeated_lengths,
                "--repeats",
                str(args.repeated_repeats),
                "--max-tokens",
                str(args.boundary_max_tokens),
                "--timeout",
                str(args.timeout),
                "--output-jsonl",
                str(out / "boundary_repeated.jsonl"),
            ],
        ),
        ValidationStep(
            name="boundary_4k_sweep",
            phase="coherence",
            output_path=out / "boundary_4k_sweep.jsonl",
            command=[
                sys.executable,
                _script("qwen36_openai_boundary_apc_probe.py"),
                *_common_model_args(args),
                "--lengths",
                _csv_range(args.sweep_start, args.sweep_end),
                "--repeats",
                "1",
                "--max-tokens",
                str(args.boundary_max_tokens),
                "--timeout",
                str(args.timeout),
                "--output-jsonl",
                str(out / "boundary_4k_sweep.jsonl"),
            ],
        ),
    ]
    if args.long_lengths:
        steps.append(
            ValidationStep(
                name="boundary_long",
                phase="coherence",
                output_path=out / "boundary_long.jsonl",
                command=[
                    sys.executable,
                    _script("qwen36_openai_boundary_apc_probe.py"),
                    *_common_model_args(args),
                    "--lengths",
                    args.long_lengths,
                    "--repeats",
                    "1",
                    "--max-tokens",
                    str(args.boundary_max_tokens),
                    "--timeout",
                    str(args.timeout),
                    "--output-jsonl",
                    str(out / "boundary_long.jsonl"),
                ],
            )
        )
    if not args.skip_chat:
        steps.append(
            ValidationStep(
                name="chat_multiturn",
                phase="coherence",
                output_path=out / "chat_multiturn.json",
                command=[
                    sys.executable,
                    _script("qwen36_chat_completion_context_bench.py"),
                    "--base-url",
                    args.base_url,
                    "--model",
                    args.chat_model,
                    "--model-path",
                    args.model_path,
                    "--lengths",
                    args.chat_lengths,
                    "--turns",
                    str(args.chat_turns),
                    "--repeats",
                    str(args.chat_repeats),
                    "--max-tokens",
                    str(args.chat_max_tokens),
                    "--timeout",
                    str(args.timeout),
                    "--unique-per-request",
                    "--output-json",
                    str(out / "chat_multiturn.json"),
                ],
            )
        )
    if not args.skip_log_scan:
        if not args.serve_log:
            raise ValueError("--serve-log is required unless --skip-log-scan is set")
        steps.append(
            ValidationStep(
                name="runtime_log_scan",
                phase="log_scan",
                output_path=out / "runtime_log_scan.json",
                command=[
                    sys.executable,
                    _script("qwen36_runtime_log_scan.py"),
                    "--json",
                    *args.serve_log,
                ],
            )
        )
    if not args.skip_speed:
        steps.append(
            ValidationStep(
                name="raw_prefill_speed",
                phase="speed",
                output_path=out / "raw_prefill_speed.json",
                command=[
                    sys.executable,
                    _script("qwen36_raw_completion_prefill_bench.py"),
                    *_common_model_args(args),
                    "--lengths",
                    args.speed_lengths,
                    "--repeats",
                    str(args.speed_repeats),
                    "--max-tokens",
                    str(args.speed_max_tokens),
                    "--timeout",
                    str(args.speed_timeout),
                    "--min-prefill-tok-s",
                    str(args.min_prefill_tok_s),
                    "--output-json",
                    str(out / "raw_prefill_speed.json"),
                ],
            )
        )
    return steps


def _run_step(step: ValidationStep, log_dir: Path) -> dict[str, object]:
    stdout_path = log_dir / f"{step.name}.stdout"
    stderr_path = log_dir / f"{step.name}.stderr"
    start = time.perf_counter()
    with stdout_path.open("w") as stdout, stderr_path.open("w") as stderr:
        completed = subprocess.run(step.command, stdout=stdout, stderr=stderr)
    elapsed = time.perf_counter() - start
    return {
        "name": step.name,
        "phase": step.phase,
        "returncode": completed.returncode,
        "elapsed_seconds": elapsed,
        "command": step.command,
        "output_path": str(step.output_path) if step.output_path is not None else None,
        "stdout": str(stdout_path),
        "stderr": str(stderr_path),
        "passed": completed.returncode == 0,
        "skipped": False,
    }


def run_matrix(
    steps: Sequence[ValidationStep],
    *,
    output_dir: Path,
    runner: Callable[[ValidationStep, Path], dict[str, object]] = _run_step,
) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    log_dir = output_dir / "step_logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    results: list[dict[str, object]] = []
    total_gate_steps = sum(1 for step in steps if step.phase in {"coherence", "log_scan"})
    total_log_scan_steps = sum(1 for step in steps if step.phase == "log_scan")
    passed_gate_steps = 0
    gate_failed = False
    for step in steps:
        speed_skip_reason = None
        if step.phase == "speed" and gate_failed:
            speed_skip_reason = "coherence_or_log_scan_failed"
        elif step.phase == "speed" and total_log_scan_steps == 0:
            speed_skip_reason = "runtime_log_scan_not_run"
        elif step.phase == "speed" and passed_gate_steps < total_gate_steps:
            speed_skip_reason = "coherence_or_log_scan_not_completed"
        if speed_skip_reason is not None:
            results.append(
                {
                    "name": step.name,
                    "phase": step.phase,
                    "returncode": None,
                    "elapsed_seconds": 0.0,
                    "command": step.command,
                    "output_path": str(step.output_path) if step.output_path else None,
                    "stdout": None,
                    "stderr": None,
                    "passed": False,
                    "skipped": True,
                    "skip_reason": speed_skip_reason,
                }
            )
            continue
        result = runner(step, log_dir)
        results.append(result)
        if step.phase in {"coherence", "log_scan"} and not bool(result["passed"]):
            gate_failed = True
        elif step.phase in {"coherence", "log_scan"}:
            passed_gate_steps += 1

    required = [row for row in results if not bool(row.get("skipped"))]
    passed = bool(required) and all(bool(row["passed"]) for row in required)
    passed = passed and not any(bool(row.get("skipped")) for row in results)
    coherence_and_log_scan_passed = (
        not gate_failed
        and total_gate_steps > 0
        and total_log_scan_steps > 0
        and passed_gate_steps == total_gate_steps
    )
    summary = {
        "passed": passed,
        "coherence_and_log_scan_passed": coherence_and_log_scan_passed,
        "gate_counts": {
            "total_gate_steps": total_gate_steps,
            "total_log_scan_steps": total_log_scan_steps,
            "passed_gate_steps": passed_gate_steps,
        },
        "output_dir": str(output_dir),
        "results": results,
    }
    (output_dir / "runtime_validation_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    return summary


def _load_speed_slice_decision_module():
    spec = importlib.util.spec_from_file_location(
        "qwen36_speed_slice_decision",
        _DECISION_SCRIPT,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {_DECISION_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _speed_json_from_summary(summary: dict[str, object]) -> Path | None:
    results = summary.get("results", [])
    if isinstance(results, list):
        for row in results:
            if not isinstance(row, dict):
                continue
            if row.get("name") == "raw_prefill_speed" and row.get("output_path"):
                return Path(str(row["output_path"]))
    output_dir = summary.get("output_dir")
    if isinstance(output_dir, str):
        return Path(output_dir) / "raw_prefill_speed.json"
    return None


def attach_speed_slice_decision(
    summary: dict[str, object],
    *,
    env_log: Path,
    output_path: Path,
    next_ts: str | None = None,
) -> dict[str, object]:
    decision_mod = _load_speed_slice_decision_module()
    speed_path = _speed_json_from_summary(summary)
    speed_output = None
    if speed_path is not None and speed_path.exists():
        with speed_path.open() as handle:
            speed_output = json.load(handle)
    decision = decision_mod.decide(
        env_values=decision_mod.parse_env_log(env_log),
        runtime_summary=summary,
        speed_output=speed_output,
        speed_json_path=speed_path,
        next_ts=next_ts,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(decision, indent=2, sort_keys=True) + "\n")
    summary["speed_slice_decision_path"] = str(output_path)
    summary["speed_slice_decision"] = {
        "decision": decision.get("decision"),
        "next_speed_slice": decision.get("next_speed_slice"),
        "reason": decision.get("reason"),
        "next_preflight": decision.get("next_preflight"),
        "profile_preflight": decision.get("profile_preflight"),
    }
    summary_path = Path(str(summary["output_dir"])) / "runtime_validation_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return decision


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8001")
    parser.add_argument("--model", default="auto")
    parser.add_argument("--chat-model", default="auto")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--serve-log", action="append", default=[])
    parser.add_argument(
        "--boundary-lengths",
        default="146,160,485,505,526,1225,2048,2049,2500,4092,4096",
    )
    parser.add_argument("--boundary-repeats", type=int, default=1)
    parser.add_argument("--boundary-max-tokens", type=int, default=1)
    parser.add_argument("--repeated-lengths", default="2500")
    parser.add_argument("--repeated-repeats", type=int, default=3)
    parser.add_argument("--sweep-start", type=int, default=4088)
    parser.add_argument("--sweep-end", type=int, default=4104)
    parser.add_argument("--long-lengths", default="8192,16384")
    parser.add_argument("--skip-chat", action="store_true")
    parser.add_argument("--chat-lengths", default="160,1225,2500")
    parser.add_argument("--chat-turns", type=int, default=8)
    parser.add_argument("--chat-repeats", type=int, default=1)
    parser.add_argument("--chat-max-tokens", type=int, default=16)
    parser.add_argument("--skip-log-scan", action="store_true")
    parser.add_argument("--skip-speed", action="store_true")
    parser.add_argument("--speed-lengths", default="16384")
    parser.add_argument("--speed-repeats", type=int, default=3)
    parser.add_argument("--speed-max-tokens", type=int, default=1)
    parser.add_argument("--speed-timeout", type=float, default=1200.0)
    parser.add_argument(
        "--min-prefill-tok-s",
        type=float,
        default=3000.0,
        help="Minimum mean 16k cold-prefill tok/s required by the speed gate.",
    )
    parser.add_argument(
        "--compile-env-log",
        type=Path,
        default=None,
        help="Compile env log used to emit speed_slice_decision.json after validation.",
    )
    parser.add_argument(
        "--speed-slice-decision-json",
        type=Path,
        default=None,
        help="Output path for speed-slice decision JSON. Defaults under output-dir.",
    )
    parser.add_argument(
        "--decision-next-ts",
        default=None,
        help="Optional fixed TS for generated next-slice preflight commands.",
    )
    parser.add_argument("--timeout", type=float, default=900.0)
    args = parser.parse_args()

    steps = build_steps(args)
    output_dir = Path(args.output_dir)
    summary = run_matrix(steps, output_dir=output_dir)
    if args.compile_env_log is not None:
        decision_output = args.speed_slice_decision_json or (
            output_dir / "speed_slice_decision.json"
        )
        attach_speed_slice_decision(
            summary,
            env_log=args.compile_env_log,
            output_path=decision_output,
            next_ts=args.decision_next_ts,
        )
    print(json.dumps(summary, sort_keys=True))
    return 0 if summary["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
