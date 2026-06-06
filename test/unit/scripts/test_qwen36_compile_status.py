import importlib.util
import json
import subprocess
import sys
from pathlib import Path


_REPO_ROOT = Path(__file__).resolve().parents[3]
_SCRIPT_PATH = _REPO_ROOT / "validation_scripts/qwen36_compile_status.py"
_SPEC = importlib.util.spec_from_file_location("qwen36_compile_status", _SCRIPT_PATH)
_SCRIPT = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_SCRIPT)


def _artifact(tmp_path):
    artifact = tmp_path / "artifact"
    artifact.mkdir()
    (artifact / "model.pt").write_bytes(b"model")
    (artifact / "neuron_config.json").write_text("{}\n")
    return artifact


def _success_log() -> str:
    return "\n".join(
        [
            "INFO:Neuron:Finished Compilation for all HLOs in 418.0 seconds",
            "CHECKPOINT_BANK_WEIGHTS_ADDED tp0_sharded_checkpoint.safetensors 48 48 torch.bfloat16 torch.bfloat16",
            "CHECKPOINT_BANK_WEIGHTS_ADDED tp1_sharded_checkpoint.safetensors 48 48 torch.bfloat16 torch.bfloat16",
            "CHECKPOINT_BANK_WEIGHTS_ADDED tp2_sharded_checkpoint.safetensors 48 48 torch.bfloat16 torch.bfloat16",
            "CHECKPOINT_BANK_WEIGHTS_ADDED tp3_sharded_checkpoint.safetensors 48 48 torch.bfloat16 torch.bfloat16",
            "COMPILE_DONE",
        ]
    )


def test_check_status_ready_when_all_compile_markers_and_files_exist(tmp_path):
    log = tmp_path / "compile.log"
    log.write_text(_success_log())

    result = _SCRIPT.check_status(
        log=log,
        artifact=_artifact(tmp_path),
        pid_file=None,
        required_ranks=["tp0", "tp1", "tp2", "tp3"],
        expected_recurrent_dtype="bf16",
        expected_conv_dtype="torch.bfloat16",
    )

    assert result["ready"]
    assert result["state"] == "ready"
    assert result["markers"]["finished_hlos"]
    assert result["markers"]["compile_done"]
    assert result["checkpoint_banks"]["tp0"]["recurrent_count"] == 48
    assert result["missing_checkpoint_ranks"] == []
    assert result["missing_artifact_files"] == []
    assert result["checkpoint_dtype_mismatches"] == []
    assert result["expected_checkpoint_dtypes"]["recurrent_dtype"] == "bfloat16"


def test_check_status_reports_missing_checkpoint_rank(tmp_path):
    log = tmp_path / "compile.log"
    log.write_text(_success_log().replace("CHECKPOINT_BANK_WEIGHTS_ADDED tp3", "SKIP tp3"))

    result = _SCRIPT.check_status(
        log=log,
        artifact=_artifact(tmp_path),
        pid_file=None,
        required_ranks=["tp0", "tp1", "tp2", "tp3"],
    )

    assert not result["ready"]
    assert result["state"] == "incomplete"
    assert result["missing_checkpoint_ranks"] == ["tp3"]


def test_check_status_reports_failure_marker(tmp_path):
    log = tmp_path / "compile.log"
    log.write_text("Traceback\nRuntimeError: boom\n")

    result = _SCRIPT.check_status(
        log=log,
        artifact=_artifact(tmp_path),
        pid_file=None,
        required_ranks=["tp0"],
    )

    assert not result["ready"]
    assert result["state"] == "failed"
    assert result["failure_lines"][0]["marker"] == "Traceback"


def test_check_status_reports_checkpoint_dtype_mismatch(tmp_path):
    log = tmp_path / "compile.log"
    log.write_text(_success_log())

    result = _SCRIPT.check_status(
        log=log,
        artifact=_artifact(tmp_path),
        pid_file=None,
        required_ranks=["tp0", "tp1"],
        expected_recurrent_dtype="float32",
        expected_conv_dtype="bfloat16",
    )

    assert not result["ready"]
    assert result["state"] == "incomplete"
    assert result["checkpoint_dtype_mismatches"] == [
        {
            "rank": "tp0",
            "field": "recurrent_dtype",
            "expected": "float32",
            "actual": "bfloat16",
            "raw_actual": "torch.bfloat16",
        },
        {
            "rank": "tp1",
            "field": "recurrent_dtype",
            "expected": "float32",
            "actual": "bfloat16",
            "raw_actual": "torch.bfloat16",
        },
    ]


def test_check_status_running_when_pid_is_live_and_not_ready(tmp_path):
    log = tmp_path / "compile.log"
    log.write_text("still compiling\n")
    pid_file = tmp_path / "compile.pid"
    proc = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(5)"])
    pid_file.write_text(str(proc.pid))
    try:
        result = _SCRIPT.check_status(
            log=log,
            artifact=_artifact(tmp_path),
            pid_file=pid_file,
            required_ranks=["tp0"],
        )
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=2)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=2)

    assert not result["ready"]
    assert result["state"] == "running"
    assert result["pid_running"] is True


def test_cli_zero_when_running_keeps_monitor_poll_green(tmp_path):
    artifact = _artifact(tmp_path)
    log = tmp_path / "compile.log"
    log.write_text("still compiling\n")
    env = tmp_path / "compile_env.txt"
    pid_file = tmp_path / "compile.pid"
    proc = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(5)"])
    pid_file.write_text(str(proc.pid))
    env.write_text(
        f"LOG={log}\n"
        f"ARTIFACT={artifact}\n"
        f"PIDFILE={pid_file}\n"
    )
    try:
        strict = subprocess.run(
            [
                sys.executable,
                str(_SCRIPT_PATH),
                "--env-log",
                str(env),
            ],
            check=False,
            capture_output=True,
            text=True,
        )
        polling = subprocess.run(
            [
                sys.executable,
                str(_SCRIPT_PATH),
                "--env-log",
                str(env),
                "--zero-when-running",
            ],
            check=False,
            capture_output=True,
            text=True,
        )
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=2)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=2)

    assert strict.returncode == 1
    assert polling.returncode == 0
    assert json.loads(polling.stdout)["state"] == "running"


def test_cli_zero_when_running_still_fails_on_failure_marker(tmp_path):
    artifact = _artifact(tmp_path)
    log = tmp_path / "compile.log"
    log.write_text("still compiling\nTraceback\nRuntimeError: boom\n")
    env = tmp_path / "compile_env.txt"
    pid_file = tmp_path / "compile.pid"
    proc = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(5)"])
    pid_file.write_text(str(proc.pid))
    env.write_text(
        f"LOG={log}\n"
        f"ARTIFACT={artifact}\n"
        f"PIDFILE={pid_file}\n"
    )
    try:
        completed = subprocess.run(
            [
                sys.executable,
                str(_SCRIPT_PATH),
                "--env-log",
                str(env),
                "--zero-when-running",
            ],
            check=False,
            capture_output=True,
            text=True,
        )
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=2)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=2)

    assert completed.returncode == 1
    payload = json.loads(completed.stdout)
    assert payload["state"] == "failed"
    assert payload["pid_running"] is True
    assert payload["failure_lines"][0]["marker"] == "Traceback"


def test_cli_reads_paths_from_env_log(tmp_path):
    artifact = _artifact(tmp_path)
    log = tmp_path / "compile.log"
    log.write_text(_success_log())
    env = tmp_path / "compile_env.txt"
    env.write_text(
        f"LOG={log}\n"
        f"ARTIFACT={artifact}\n"
        f"PIDFILE={tmp_path / 'compile.pid'}\n"
        "SOURCE_COMMIT=envcommit\n"
        "SOURCE_BRANCH=codex/qwen36-prefill-speed-coherent\n"
        "GDN_RECURRENT_CACHE_DTYPE=bfloat16\n"
        "GDN_CONV_CACHE_DTYPE=bfloat16\n"
    )

    completed = subprocess.run(
        [
            sys.executable,
            str(_SCRIPT_PATH),
            "--env-log",
            str(env),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    payload = json.loads(completed.stdout)
    assert payload["ready"]
    assert payload["log"] == str(log)
    assert payload["artifact"] == str(artifact)
    assert payload["expected_checkpoint_dtypes"] == {
        "conv_dtype": "bfloat16",
        "recurrent_dtype": "bfloat16",
    }
    assert payload["source"] == {
        "branch": "codex/qwen36-prefill-speed-coherent",
        "commit": "envcommit",
    }


def test_cli_fails_when_env_log_expected_dtype_disagrees(tmp_path):
    artifact = _artifact(tmp_path)
    log = tmp_path / "compile.log"
    log.write_text(_success_log())
    env = tmp_path / "compile_env.txt"
    env.write_text(
        f"LOG={log}\n"
        f"ARTIFACT={artifact}\n"
        f"PIDFILE={tmp_path / 'compile.pid'}\n"
        "GDN_RECURRENT_CACHE_DTYPE=float32\n"
        "GDN_CONV_CACHE_DTYPE=bfloat16\n"
    )

    completed = subprocess.run(
        [
            sys.executable,
            str(_SCRIPT_PATH),
            "--env-log",
            str(env),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 1
    payload = json.loads(completed.stdout)
    assert not payload["ready"]
    assert payload["checkpoint_dtype_mismatches"][0]["expected"] == "float32"
