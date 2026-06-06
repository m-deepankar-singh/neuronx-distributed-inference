import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path


_REPO_ROOT = Path(__file__).resolve().parents[3]
_SCRIPT_PATH = _REPO_ROOT / "validation_scripts/qwen36_launch_preflight.py"
_SPEC = importlib.util.spec_from_file_location("qwen36_launch_preflight", _SCRIPT_PATH)
_SCRIPT = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _SCRIPT
_SPEC.loader.exec_module(_SCRIPT)


def _compile_env(**overrides):
    values = {
        "ARTIFACT": "/artifacts/qwen",
        "REPO": "/home/ubuntu/repo",
        "MODEL": "/home/ubuntu/models/Qwen3.6-27B",
        "MAX_CONTEXT_LENGTH": "32768",
        "SEQ_LEN": "32768",
        "CTE_BUCKETS": "2048",
        "TOKEN_GENERATION_BUCKETS": "512 16384 16640 32768",
        "CONTEXT_ENCODING_BUCKET_PAIRS": "2048:256 2048:512 2048:1024 2048:2048 2048:4096 2048:8192 2048:16384 2048:32768",
        "MAX_GDN_CHECKPOINT_SLOTS": "64",
        "GDN_RECURRENT_CACHE_DTYPE": "bfloat16",
        "GDN_CONV_CACHE_DTYPE": "bfloat16",
        "QWEN36_DELTANET_MULTIHEAD_CTE": "0",
        "QWEN36_SPLIT_QKV_TKG_ROW_SCALES": "0",
        "QWEN36_DELTANET_FUSED_SEGMENT_TOKENS": "0",
        "QWEN36_DELTANET_SOLVE_MODE": "direct",
        "QWEN36_DELTANET_SOLVE_SCAN_STEPS": "0",
    }
    values.update(overrides)
    return values


def _write_env(path, values):
    path.write_text("".join(f"{key}={value}\n" for key, value in values.items()))


def _fake_launch_script(path):
    path.write_text(
        """#!/usr/bin/env bash
set -euo pipefail
ARTIFACT="$1"
LOG="$2"
ENVLOG="${LOG%.log}_env.txt"
mkdir -p "$(dirname "$LOG")"
printf "%s\\n" \
  "ARTIFACT=${ARTIFACT}" \
  "LOG=${LOG}" \
  "ENVLOG=${ENVLOG}" \
  "PIDFILE=${LOG%.log}.pid" \
  "LAUNCH_DRY_RUN=${LAUNCH_DRY_RUN}" \
  "MAX_MODEL_LEN=${MAX_MODEL_LEN}" \
  "SEQ_LEN=${SEQ_LEN}" \
  "CTE_BUCKETS=${CTE_BUCKETS}" \
  "CONTEXT_ENCODING_BUCKET_PAIRS=${CONTEXT_ENCODING_BUCKET_PAIRS}" \
  "TOKEN_GENERATION_BUCKETS=${TOKEN_GENERATION_BUCKETS}" \
  "MAX_GDN_CHECKPOINT_SLOTS=${MAX_GDN_CHECKPOINT_SLOTS}" \
  "GDN_RECURRENT_CACHE_DTYPE=${GDN_RECURRENT_CACHE_DTYPE}" \
  "GDN_CONV_CACHE_DTYPE=${GDN_CONV_CACHE_DTYPE}" \
  "QWEN36_DELTANET_MULTIHEAD_CTE=${QWEN36_DELTANET_MULTIHEAD_CTE}" \
  "QWEN36_SPLIT_QKV_TKG_ROW_SCALES=${QWEN36_SPLIT_QKV_TKG_ROW_SCALES}" \
  "QWEN36_DELTANET_FUSED_SEGMENT_TOKENS=${QWEN36_DELTANET_FUSED_SEGMENT_TOKENS}" \
  "QWEN36_DELTANET_SOLVE_MODE=${QWEN36_DELTANET_SOLVE_MODE}" \
  "QWEN36_DELTANET_SOLVE_SCAN_STEPS=${QWEN36_DELTANET_SOLVE_SCAN_STEPS}" \
  "DISABLE_HYBRID_KV_CACHE_MANAGER=${DISABLE_HYBRID_KV_CACHE_MANAGER}" \
  "ENABLE_HYBRID_APC=1" \
  "ENABLE_PREFIX_CACHING=1" \
  "ENABLE_VLLM_CHUNKED_PREFILL=1" \
  "HYBRID_APC_REQUIRE_VLLM_METADATA=1" \
  "HYBRID_APC_ENABLE_BACKED_PREFIX_READS=1" \
  "BLOCK_SIZE=256" \
  "GDN_CHECKPOINT_INTERVAL=256" \
  >"${ENVLOG}"
echo "ENVLOG=${ENVLOG}"
"""
    )
    path.chmod(0o755)


def test_build_preflight_derives_launch_contract_from_compile_env(tmp_path):
    compile_env_log = tmp_path / "compile_env.txt"
    serve_log = tmp_path / "serve.log"
    _write_env(compile_env_log, _compile_env())

    plan = _SCRIPT.build_preflight(
        compile_env=_compile_env(),
        compile_env_log=compile_env_log,
        serve_log=serve_log,
        launch_script="tmp_launch_qwen36_segcte2048.sh",
    )

    assert plan["schema"] == "qwen36-launch-preflight-v1"
    assert plan["launch_env"]["LAUNCH_DRY_RUN"] == "1"
    assert plan["launch_env"]["MAX_MODEL_LEN"] == "32768"
    assert plan["launch_env"]["SEQ_LEN"] == "32768"
    assert plan["launch_env"]["CONTEXT_ENCODING_BUCKET_PAIRS"].startswith("2048:0 ")
    assert "2048:32768" in plan["launch_env"]["CONTEXT_ENCODING_BUCKET_PAIRS"]
    assert "MAX_MODEL_LEN=32768" in plan["dry_run_command"]
    assert "LAUNCH_DRY_RUN=1" in plan["dry_run_command"]
    assert plan["live_launch_env"]["LAUNCH_DRY_RUN"] == "0"
    assert "qwen36_launch_env_audit.py" in plan["audit_command"]
    assert plan["run_order"] == [
        "run_dry_run_command",
        "run_audit_command",
        "run_live_launch_command_only_if_audit_passes",
    ]


def test_run_preflight_runs_launch_dry_run_and_audit(tmp_path):
    compile_env_log = tmp_path / "compile_env.txt"
    serve_log = tmp_path / "serve.log"
    launch_script = tmp_path / "fake_launch.sh"
    _write_env(compile_env_log, _compile_env(ARTIFACT=str(tmp_path / "artifact")))
    _fake_launch_script(launch_script)
    plan = _SCRIPT.build_preflight(
        compile_env=_SCRIPT.parse_env_log(compile_env_log),
        compile_env_log=compile_env_log,
        serve_log=serve_log,
        launch_script=str(launch_script),
    )
    plan["_launch_script"] = str(launch_script)

    result = _SCRIPT.run_preflight(plan)

    assert result["passed"]
    assert result["dry_run"]["returncode"] == 0
    assert result["audit"]["passed"]
    assert Path(result["launch_env_log"]).exists()


def test_cli_outputs_nonzero_when_compile_env_is_missing_required_key(tmp_path):
    compile_env_log = tmp_path / "compile_env.txt"
    compile_env_log.write_text("ARTIFACT=/artifact\n")
    completed = subprocess.run(
        [
            sys.executable,
            str(_SCRIPT_PATH),
            "--compile-env-log",
            str(compile_env_log),
            "--serve-log",
            str(tmp_path / "serve.log"),
        ],
        check=False,
        capture_output=True,
        text=True,
        cwd=_REPO_ROOT,
        env=os.environ.copy(),
    )

    assert completed.returncode != 0
    payload = json.loads(completed.stdout)
    assert payload["failure_stage"] == "build_preflight"
    assert "MAX_CONTEXT_LENGTH" in payload["errors"][0]
    assert completed.stderr == ""


def test_cli_run_writes_output_json(tmp_path):
    compile_env_log = tmp_path / "compile_env.txt"
    serve_log = tmp_path / "serve.log"
    launch_script = tmp_path / "fake_launch.sh"
    output_json = tmp_path / "preflight.json"
    _write_env(compile_env_log, _compile_env(ARTIFACT=str(tmp_path / "artifact")))
    _fake_launch_script(launch_script)

    completed = subprocess.run(
        [
            sys.executable,
            str(_SCRIPT_PATH),
            "--compile-env-log",
            str(compile_env_log),
            "--serve-log",
            str(serve_log),
            "--launch-script",
            str(launch_script),
            "--output-json",
            str(output_json),
            "--run",
        ],
        check=False,
        capture_output=True,
        text=True,
        cwd=_REPO_ROOT,
        env=os.environ.copy(),
    )

    assert completed.returncode == 0
    payload = json.loads(output_json.read_text())
    assert payload["passed"]
    assert payload == json.loads(completed.stdout)
