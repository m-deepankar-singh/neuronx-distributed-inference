import argparse
import importlib.util
import json
import sys
from pathlib import Path


_REPO_ROOT = Path(__file__).resolve().parents[3]
_SCRIPT_PATH = _REPO_ROOT / "validation_scripts/qwen36_runtime_validation_matrix.py"
_SPEC = importlib.util.spec_from_file_location(
    "qwen36_runtime_validation_matrix",
    _SCRIPT_PATH,
)
_SCRIPT = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _SCRIPT
_SPEC.loader.exec_module(_SCRIPT)


def _args(**overrides):
    defaults = dict(
        base_url="http://127.0.0.1:8001",
        model="auto",
        chat_model="auto",
        model_path="/models/Qwen3.6-27B",
        output_dir="/tmp/qwen-runtime-validation",
        serve_log=["/logs/server.log"],
        boundary_lengths="146,160",
        boundary_repeats=1,
        boundary_max_tokens=1,
        repeated_lengths="2500",
        repeated_repeats=3,
        sweep_start=4088,
        sweep_end=4090,
        long_lengths="8192,16384",
        skip_chat=False,
        chat_lengths="160,1225,2500",
        chat_turns=8,
        chat_repeats=1,
        chat_max_tokens=16,
        skip_log_scan=False,
        skip_speed=False,
        speed_lengths="16384",
        speed_repeats=3,
        speed_max_tokens=1,
        speed_timeout=1200.0,
        min_prefill_tok_s=3000.0,
        timeout=900.0,
    )
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


def test_csv_range_includes_endpoints():
    assert _SCRIPT._csv_range(4088, 4090) == "4088,4089,4090"


def test_build_steps_orders_coherence_log_scan_then_speed():
    steps = _SCRIPT.build_steps(_args())

    assert [step.name for step in steps] == [
        "boundary_primary",
        "boundary_repeated",
        "boundary_4k_sweep",
        "boundary_long",
        "chat_multiturn",
        "runtime_log_scan",
        "raw_prefill_speed",
    ]
    assert steps[-2].phase == "log_scan"
    assert steps[-1].phase == "speed"
    assert "qwen36_runtime_log_scan.py" in steps[-2].command[1]
    assert "qwen36_raw_completion_prefill_bench.py" in steps[-1].command[1]
    chat_step = next(step for step in steps if step.name == "chat_multiturn")
    assert chat_step.command[chat_step.command.index("--model") + 1] == "auto"
    assert "--lengths" in steps[-1].command
    assert "16384" in steps[-1].command
    assert "--min-prefill-tok-s" in steps[-1].command
    assert "3000.0" in steps[-1].command


def test_build_steps_requires_log_scan_unless_explicitly_skipped():
    try:
        _SCRIPT.build_steps(_args(serve_log=[]))
    except ValueError as exc:
        assert "--serve-log" in str(exc)
    else:
        raise AssertionError("missing serve log should fail")

    steps = _SCRIPT.build_steps(_args(serve_log=[], skip_log_scan=True))
    assert "runtime_log_scan" not in [step.name for step in steps]


def test_run_matrix_skips_speed_after_coherence_failure(tmp_path):
    steps = [
        _SCRIPT.ValidationStep("boundary_primary", "coherence", ["false"]),
        _SCRIPT.ValidationStep("raw_prefill_speed", "speed", ["speed"]),
    ]

    def runner(step, log_dir):
        return {
            "name": step.name,
            "phase": step.phase,
            "returncode": 1 if step.name == "boundary_primary" else 0,
            "elapsed_seconds": 0.0,
            "command": step.command,
            "output_path": None,
            "stdout": str(log_dir / f"{step.name}.stdout"),
            "stderr": str(log_dir / f"{step.name}.stderr"),
            "passed": step.name != "boundary_primary",
            "skipped": False,
        }

    summary = _SCRIPT.run_matrix(steps, output_dir=tmp_path, runner=runner)

    assert not summary["passed"]
    assert not summary["coherence_and_log_scan_passed"]
    assert summary["results"][1]["skipped"]
    assert summary["results"][1]["skip_reason"] == "coherence_or_log_scan_failed"
    assert summary["gate_counts"] == {
        "total_gate_steps": 1,
        "total_log_scan_steps": 0,
        "passed_gate_steps": 0,
    }
    assert (tmp_path / "runtime_validation_summary.json").exists()


def test_run_matrix_skips_speed_when_log_scan_not_run(tmp_path):
    steps = [
        _SCRIPT.ValidationStep("boundary_primary", "coherence", ["ok"]),
        _SCRIPT.ValidationStep("raw_prefill_speed", "speed", ["speed"]),
    ]

    def runner(step, log_dir):
        return {
            "name": step.name,
            "phase": step.phase,
            "returncode": 0,
            "elapsed_seconds": 0.0,
            "command": step.command,
            "output_path": None,
            "stdout": str(log_dir / f"{step.name}.stdout"),
            "stderr": str(log_dir / f"{step.name}.stderr"),
            "passed": True,
            "skipped": False,
        }

    summary = _SCRIPT.run_matrix(steps, output_dir=tmp_path, runner=runner)

    assert not summary["passed"]
    assert not summary["coherence_and_log_scan_passed"]
    assert summary["results"][1]["skipped"]
    assert summary["results"][1]["skip_reason"] == "runtime_log_scan_not_run"
    assert summary["gate_counts"] == {
        "total_gate_steps": 1,
        "total_log_scan_steps": 0,
        "passed_gate_steps": 1,
    }


def test_run_matrix_passes_when_all_steps_pass(tmp_path):
    steps = [
        _SCRIPT.ValidationStep("boundary_primary", "coherence", ["ok"]),
        _SCRIPT.ValidationStep("runtime_log_scan", "log_scan", ["ok"]),
        _SCRIPT.ValidationStep("raw_prefill_speed", "speed", ["ok"]),
    ]

    def runner(step, log_dir):
        return {
            "name": step.name,
            "phase": step.phase,
            "returncode": 0,
            "elapsed_seconds": 0.0,
            "command": step.command,
            "output_path": None,
            "stdout": str(log_dir / f"{step.name}.stdout"),
            "stderr": str(log_dir / f"{step.name}.stderr"),
            "passed": True,
            "skipped": False,
        }

    summary = _SCRIPT.run_matrix(steps, output_dir=tmp_path, runner=runner)

    assert summary["passed"]
    assert summary["coherence_and_log_scan_passed"]
    assert summary["gate_counts"] == {
        "total_gate_steps": 2,
        "total_log_scan_steps": 1,
        "passed_gate_steps": 2,
    }


def test_attach_speed_slice_decision_writes_next_action(tmp_path):
    env_log = tmp_path / "compile_env.txt"
    speed_json = tmp_path / "raw_prefill_speed.json"
    decision_json = tmp_path / "speed_slice_decision.json"
    env_log.write_text(
        "SPEED_SLICE=sampletokonly\n"
        "ARTIFACT=/artifact\n"
        "BASE=qwen36_sampletokonly\n"
    )
    speed_json.write_text(
        json.dumps(
            {
                "row_gate_passed": True,
                "prefill_tok_s_mean": 640.0,
                "speed_gate": {
                    "passed": False,
                    "failure_reason": "mean_prefill_tok_s_below_threshold",
                },
            }
        )
        + "\n"
    )
    summary = {
        "passed": False,
        "coherence_and_log_scan_passed": True,
        "output_dir": str(tmp_path),
        "results": [
            {
                "name": "raw_prefill_speed",
                "phase": "speed",
                "output_path": str(speed_json),
                "passed": False,
                "skipped": False,
            }
        ],
    }

    decision = _SCRIPT.attach_speed_slice_decision(
        summary,
        env_log=env_log,
        output_path=decision_json,
        next_ts="20260606T010203Z_hostlogits",
    )

    assert decision["decision"] == "launch_next_speed_slice"
    assert decision["next_speed_slice"] == "hostlogits"
    assert decision_json.exists()
    rewritten = json.loads((tmp_path / "runtime_validation_summary.json").read_text())
    assert rewritten["speed_slice_decision_path"] == str(decision_json)
    assert rewritten["speed_slice_decision"]["decision"] == "launch_next_speed_slice"
    assert rewritten["speed_slice_decision"]["failed_runtime_gate_count"] == 0
    assert rewritten["speed_slice_decision"]["next_required_flags"] == {
        "DISABLE_ON_DEVICE_SAMPLING": "1",
        "OUTPUT_LOGITS_WITH_ON_DEVICE_SAMPLING": "0",
        "QUANTIZE_LM_HEAD": "1",
        "SPEED_SLICE": "hostlogits",
    }
    assert rewritten["speed_slice_decision"]["next_preflight"]["ts"] == (
        "20260606T010203Z_hostlogits"
    )
    assert rewritten["speed_slice_decision"]["profile_preflight"] is None


def test_attach_speed_slice_decision_surfaces_profile_preflight(tmp_path):
    env_log = tmp_path / "compile_env.txt"
    speed_json = tmp_path / "raw_prefill_speed.json"
    decision_json = tmp_path / "speed_slice_decision.json"
    env_log.write_text(
        "SPEED_SLICE=hostlogits_lmheadbf16\n"
        f"WORKDIR={tmp_path / 'work'}\n"
        f"LOGDIR={tmp_path / 'logs'}\n"
        "CTE_BUCKETS_RAW=2048\n"
    )
    speed_json.write_text(
        json.dumps(
            {
                "row_gate_passed": True,
                "prefill_tok_s_mean": 950.0,
                "speed_gate": {
                    "passed": False,
                    "min_prefill_tok_s": 3000.0,
                    "failure_reason": "mean_prefill_tok_s_below_threshold",
                },
            }
        )
        + "\n"
    )
    summary = {
        "passed": False,
        "coherence_and_log_scan_passed": True,
        "output_dir": str(tmp_path),
        "results": [
            {
                "name": "raw_prefill_speed",
                "phase": "speed",
                "output_path": str(speed_json),
                "passed": False,
                "skipped": False,
            }
        ],
    }

    decision = _SCRIPT.attach_speed_slice_decision(
        summary,
        env_log=env_log,
        output_path=decision_json,
        next_ts="20260606T010203Z_profile",
    )

    assert decision["decision"] == "profile_slow_coherent"
    rewritten = json.loads((tmp_path / "runtime_validation_summary.json").read_text())
    profile_preflight = rewritten["speed_slice_decision"]["profile_preflight"]
    assert profile_preflight["ts"] == "20260606T010203Z_profile"
    assert profile_preflight["do_not_profile_live_vllm"] is True
    assert profile_preflight["context_tokens"] == 2048
    assert "qwen36_context_neff_profile.py" in profile_preflight["run_command"]
    assert rewritten["speed_slice_decision"]["next_preflight"] is None


def test_attach_speed_slice_decision_handles_skipped_speed_after_failure(tmp_path):
    env_log = tmp_path / "compile_env.txt"
    decision_json = tmp_path / "speed_slice_decision.json"
    env_log.write_text("SPEED_SLICE=sampletokonly\n")
    summary = {
        "passed": False,
        "coherence_and_log_scan_passed": False,
        "output_dir": str(tmp_path),
        "results": [
            {
                "name": "raw_prefill_speed",
                "phase": "speed",
                "output_path": str(tmp_path / "raw_prefill_speed.json"),
                "passed": False,
                "skipped": True,
            }
        ],
    }

    decision = _SCRIPT.attach_speed_slice_decision(
        summary,
        env_log=env_log,
        output_path=decision_json,
    )

    assert decision["decision"] == "stop_incoherent"
    assert decision_json.exists()
    rewritten = json.loads((tmp_path / "runtime_validation_summary.json").read_text())
    assert rewritten["speed_slice_decision"]["failed_runtime_gate_count"] == 0
    assert rewritten["speed_slice_decision"]["next_required_flags"] == {}


def test_attach_speed_slice_decision_surfaces_failed_gate_count(tmp_path):
    env_log = tmp_path / "compile_env.txt"
    decision_json = tmp_path / "speed_slice_decision.json"
    env_log.write_text("SPEED_SLICE=sampletokonly\n")
    summary = {
        "passed": False,
        "coherence_and_log_scan_passed": False,
        "gate_counts": {
            "total_gate_steps": 2,
            "total_log_scan_steps": 1,
            "passed_gate_steps": 1,
        },
        "output_dir": str(tmp_path),
        "results": [
            {
                "name": "boundary_primary",
                "phase": "coherence",
                "passed": False,
                "skipped": False,
            },
            {
                "name": "runtime_log_scan",
                "phase": "log_scan",
                "passed": True,
                "skipped": False,
            },
            {
                "name": "raw_prefill_speed",
                "phase": "speed",
                "output_path": str(tmp_path / "raw_prefill_speed.json"),
                "passed": False,
                "skipped": True,
                "skip_reason": "coherence_or_log_scan_failed",
            },
        ],
    }

    decision = _SCRIPT.attach_speed_slice_decision(
        summary,
        env_log=env_log,
        output_path=decision_json,
    )

    assert decision["decision"] == "stop_incoherent"
    rewritten = json.loads((tmp_path / "runtime_validation_summary.json").read_text())
    assert rewritten["speed_slice_decision"]["failed_runtime_gate_count"] == 1
    assert rewritten["speed_slice_decision"]["reason"] == (
        "coherence_or_runtime_log_scan_failed"
    )
