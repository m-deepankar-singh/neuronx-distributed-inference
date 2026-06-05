#!/usr/bin/env python3
"""Render a self-contained Qwen3.6 compile-monitor automation prompt."""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path


_LOG_SCAN_STRINGS = [
    "negative token_id",
    "out-of-vocab token_id",
    "fallback argmax",
    "finite=0",
    "nan=",
    "NaN",
    "NRT_RESOURCE",
    "Traceback",
    "RuntimeError",
    "Internal Server Error",
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


def _required(values: dict[str, str], key: str) -> str:
    try:
        return values[key]
    except KeyError as exc:
        raise ValueError(f"env log is missing required key {key!r}") from exc


def _git_commit(repo: Path | None) -> str:
    if repo is None:
        return "unknown"
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=repo,
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def render_prompt(
    values: dict[str, str],
    *,
    compile_host: str,
    runtime_host: str,
    source_dir: str,
    source_commit: str,
    launch_script: str,
    boundary_lengths: str,
) -> str:
    artifact = _required(values, "ARTIFACT")
    workdir = _required(values, "WORKDIR")
    log = _required(values, "LOG")
    envlog = _required(values, "ENVLOG")
    pidfile = _required(values, "PIDFILE")
    base = _required(values, "BASE")
    model_path = values.get("MODEL", "<model path>")

    flags = [
        "SAMPLING",
        "KERNELS",
        "MEMORY_FLAGS",
        "SEQ_LEN",
        "MAX_CONTEXT_LENGTH",
        "CTE_BUCKETS",
        "PREFIX_BUCKETS",
        "TOKEN_GENERATION_BUCKETS",
        "CONTEXT_ENCODING_BUCKET_PAIRS",
        "PREFIX_CTE_ATTENTION_BACKEND",
        "PREFIX_CTE_ATTENTION_SEGMENT_SIZE",
        "GDN_RECURRENT_CACHE_DTYPE",
        "GDN_CONV_CACHE_DTYPE",
        "QWEN36_DELTANET_FUSED_SEGMENT_TOKENS",
        "QWEN36_DELTANET_MULTIHEAD_CTE",
        "QWEN36_DELTANET_SOLVE_MODE",
        "QWEN36_DELTANET_SOLVE_SCAN_STEPS",
        "ENABLE_QKV_NKI_KERNELS",
        "ENABLE_QKV_CTE_NKI_KERNEL_FUSE_ROPE",
        "ENABLE_QKV_CTE_NKI_KERNEL_FUSE_QK_NORM",
        "ENABLE_OUT_PROJ_NKI_KERNEL",
        "ENABLE_KV_CACHE_QUANT",
        "QUANTIZE_LM_HEAD",
        "FP8_QUANTIZE_LINEAR_ATTN_GATES",
        "DISABLE_ON_DEVICE_SAMPLING",
        "OUTPUT_LOGITS_WITH_ON_DEVICE_SAMPLING",
    ]
    flag_lines = "\n".join(
        f"- {key}={values[key]}" for key in flags if key in values
    )
    scan = ", ".join(f"`{item}`" for item in _LOG_SCAN_STRINGS)

    return f"""Monitor the Qwen3.6 coherent-prefill compile on {compile_host}.
Source: {source_dir} at commit {source_commit}.
Base: {base}.
PID file: {pidfile}.
Compile log: {log}.
Env log: {envlog}.
Artifact: {artifact}.
Workdir: {workdir}.

Expected compile shape:
{flag_lines}

Success requires: compile process exited cleanly, `Finished Compilation for all HLOs`, no exception, `CHECKPOINT_BANK_WEIGHTS_ADDED` for tp0..tp3 with dtypes matching the env log, `COMPILE_DONE`, `{artifact}/model.pt`, and `{artifact}/neuron_config.json`. Use `validation_scripts/qwen36_compile_status.py --env-log {envlog}` for the structured readiness verdict before rsync.

If the compile is still running and the status helper or log scan shows no new failure signal, return a quiet/DONT_NOTIFY heartbeat status instead of starting a manual monitoring loop. Notify only for terminal compile failure, completed compile readiness, runtime validation failure, or coherent speed results.

If successful: verify artifact files and neuron_config, rsync EC2-to-EC2 from {compile_host} to {runtime_host}, launch on {runtime_host} with `{launch_script}` using the artifact and dtype settings from the env log, wait for `/health`, then run the coherence matrix before any speed claim.

Preferred validation driver after launch: `validation_scripts/qwen36_runtime_validation_matrix.py --base-url http://127.0.0.1:<port> --model-path {model_path} --serve-log <serve.log> --output-dir <validation-output-dir> --min-prefill-tok-s 3000`. It runs the coherence/log-scan/speed gates in order and skips speed automatically if coherence or log scan fails.

Coherence matrix: use the maintained `validation_scripts/qwen36_runtime_validation_matrix.py` driver, not the old ad hoc `/tmp/tmp_bisect_probe3.py` helper that has been missing on runtime hosts. It must cover exact boundary lengths {boundary_lengths}, the unique 4k sweep around 4088..4104, repeated 2500 prompt, multi-turn probes at 160, 1225, and 2500, and long probes at 8192 and 16384 where the artifact supports them. Runtime log scan must use `validation_scripts/qwen36_runtime_log_scan.py` and show zero matches for {scan}.

Speed validation only after coherence passes: run `validation_scripts/qwen36_raw_completion_prefill_bench.py --lengths 16384 --repeats 3 --max-tokens 1 --min-prefill-tok-s 3000`; it requests `stream_options.include_usage=true`, computes tok/s from `usage.prompt_tokens / TTFT`, and fails the speed gate when mean 16k cold-prefill throughput is below 3000 tok/s. Report coherence verdict, log-scan result, cold-prefill tok/s, and all JSON/log paths.

If failed: report the exact command/log path/error text, failing stage, disk usage, and best current root-cause hypothesis. Do not launch a duplicate compile if the PID in `{pidfile}` is still running."""


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-log", required=True, type=Path)
    parser.add_argument("--compile-host", default="ubuntu@16.26.135.243")
    parser.add_argument("--runtime-host", default="ubuntu@16.26.184.190")
    parser.add_argument("--source-dir", default=None)
    parser.add_argument("--source-commit", default=None)
    parser.add_argument(
        "--launch-script",
        default="tmp_launch_qwen36_segcte2048.sh",
    )
    parser.add_argument(
        "--boundary-lengths",
        default="146,160,485,505,526,1225,2048,2049,2500,4092,4096",
    )
    args = parser.parse_args()

    values = parse_env_log(args.env_log)
    values.setdefault("ENVLOG", str(args.env_log))
    source_dir = args.source_dir or values.get("REPO", "unknown")
    source_commit = args.source_commit or _git_commit(
        Path(source_dir) if source_dir != "unknown" else None
    )
    print(
        render_prompt(
            values,
            compile_host=args.compile_host,
            runtime_host=args.runtime_host,
            source_dir=source_dir,
            source_commit=source_commit,
            launch_script=args.launch_script,
            boundary_lengths=args.boundary_lengths,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
