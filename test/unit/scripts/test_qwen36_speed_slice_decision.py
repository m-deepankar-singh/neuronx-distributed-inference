import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path


_REPO_ROOT = Path(__file__).resolve().parents[3]
_SCRIPT_PATH = _REPO_ROOT / "validation_scripts/qwen36_speed_slice_decision.py"
_SPEC = importlib.util.spec_from_file_location(
    "qwen36_speed_slice_decision",
    _SCRIPT_PATH,
)
_SCRIPT = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _SCRIPT
_SPEC.loader.exec_module(_SCRIPT)


def _summary(*, coherence_ok=True, passed=False, speed_path="/tmp/raw_speed.json"):
    return {
        "passed": passed,
        "coherence_and_log_scan_passed": coherence_ok,
        "output_dir": "/tmp/validation",
        "results": [
            {
                "name": "raw_prefill_speed",
                "phase": "speed",
                "output_path": speed_path,
                "passed": passed,
                "skipped": False,
            }
        ],
    }


def _speed(*, passed=False, mean=628.0, row_gate_passed=True):
    return {
        "passed": passed,
        "row_gate_passed": row_gate_passed,
        "prefill_tok_s_mean": mean,
        "speed_gate": {
            "enabled": True,
            "passed": passed,
            "min_prefill_tok_s": 3000.0,
            "mean_prefill_tok_s": mean,
            "failure_reason": None if passed else "mean_prefill_tok_s_below_threshold",
        },
    }


def test_incoherent_result_stops_speed_work():
    decision = _SCRIPT.decide(
        env_values={"SPEED_SLICE": "sampletokonly"},
        runtime_summary={
            "passed": False,
            "coherence_and_log_scan_passed": False,
            "results": [
                {
                    "name": "boundary_primary",
                    "phase": "coherence",
                    "passed": False,
                    "skipped": False,
                }
            ],
        },
        speed_output=_speed(),
        speed_json_path=Path("/tmp/raw_speed.json"),
    )

    assert decision["decision"] == "stop_incoherent"
    assert decision["next_speed_slice"] is None


def test_ambiguous_legacy_summary_reruns_runtime_validation():
    decision = _SCRIPT.decide(
        env_values={"SPEED_SLICE": "sampletokonly"},
        runtime_summary=_summary(coherence_ok=False),
        speed_output=_speed(),
        speed_json_path=Path("/tmp/raw_speed.json"),
    )

    assert decision["decision"] == "rerun_runtime_validation"
    assert decision["reason"] == "coherence_or_log_scan_not_completed"
    assert decision["failed_runtime_gate_count"] == 0


def test_missing_log_scan_reruns_runtime_validation_not_incoherent():
    decision = _SCRIPT.decide(
        env_values={"SPEED_SLICE": "sampletokonly"},
        runtime_summary={
            "passed": False,
            "coherence_and_log_scan_passed": False,
            "gate_counts": {
                "total_gate_steps": 1,
                "total_log_scan_steps": 0,
                "passed_gate_steps": 1,
            },
            "results": [
                {
                    "name": "boundary_primary",
                    "phase": "coherence",
                    "passed": True,
                    "skipped": False,
                },
                {
                    "name": "raw_prefill_speed",
                    "phase": "speed",
                    "passed": False,
                    "skipped": True,
                    "skip_reason": "runtime_log_scan_not_run",
                },
            ],
        },
        speed_output=None,
        speed_json_path=Path("/tmp/raw_speed.json"),
    )

    assert decision["decision"] == "rerun_runtime_validation"
    assert decision["reason"] == "runtime_log_scan_not_run"
    assert decision["failed_runtime_gate_count"] == 0


def test_failed_runtime_gate_still_stops_speed_work():
    decision = _SCRIPT.decide(
        env_values={"SPEED_SLICE": "sampletokonly"},
        runtime_summary={
            "passed": False,
            "coherence_and_log_scan_passed": False,
            "gate_counts": {
                "total_gate_steps": 2,
                "total_log_scan_steps": 1,
                "passed_gate_steps": 1,
            },
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
            ],
        },
        speed_output=None,
        speed_json_path=Path("/tmp/raw_speed.json"),
    )

    assert decision["decision"] == "stop_incoherent"
    assert decision["reason"] == "coherence_or_runtime_log_scan_failed"
    assert decision["failed_runtime_gate_count"] == 1


def test_missing_speed_json_requests_speed_rerun_not_compile():
    decision = _SCRIPT.decide(
        env_values={"SPEED_SLICE": "sampletokonly"},
        runtime_summary=_summary(coherence_ok=True),
        speed_output=None,
        speed_json_path=Path("/tmp/missing.json"),
    )

    assert decision["decision"] == "rerun_speed_validation"
    assert decision["reason"] == "missing_or_unreadable_speed_output"


def test_speed_pass_keeps_candidate():
    decision = _SCRIPT.decide(
        env_values={"SPEED_SLICE": "hostlogits_lmheadbf16"},
        runtime_summary=_summary(coherence_ok=True, passed=True),
        speed_output=_speed(passed=True, mean=3475.0),
        speed_json_path=Path("/tmp/raw_speed.json"),
    )

    assert decision["decision"] == "candidate_meets_target"
    assert decision["prefill_tok_s_mean"] == 3475.0


def test_slow_sample_token_slice_advances_to_hostlogits():
    decision = _SCRIPT.decide(
        env_values={
            "SPEED_SLICE": "sampletokonly",
            "REPO": "/home/ubuntu/repo",
            "MODEL": "/home/ubuntu/model",
        },
        runtime_summary=_summary(coherence_ok=True),
        speed_output=_speed(passed=False, mean=640.0),
        speed_json_path=Path("/tmp/raw_speed.json"),
        next_ts="20260606T010203Z_hostlogits",
    )

    assert decision["decision"] == "launch_next_speed_slice"
    assert decision["next_speed_slice"] == "hostlogits"
    assert decision["next_required_flags"]["DISABLE_ON_DEVICE_SAMPLING"] == "1"
    assert decision["next_required_flags"]["QUANTIZE_LM_HEAD"] == "1"
    preflight = decision["next_preflight"]
    assert preflight["run_order"] == [
        "run_dry_run_command",
        "create_heartbeat_automation_from_template",
        "run_launch_command_after_automation_exists",
    ]
    assert preflight["ts"] == "20260606T010203Z_hostlogits"
    assert preflight["compile_driver"] == "tmp_compile_qwen32k_segcte2048_gdnseg512.sh"
    assert preflight["dry_run_env"]["COMPILE_DRY_RUN"] == "1"
    assert preflight["launch_env"]["COMPILE_DRY_RUN"] == "0"
    assert preflight["dry_run_env"]["TS"] == "20260606T010203Z_hostlogits"
    assert preflight["launch_env"]["TS"] == "20260606T010203Z_hostlogits"
    assert preflight["dry_run_env"]["PREFIX_CTE_ATTENTION_BACKEND"] == "attention_cte"
    assert preflight["dry_run_env"]["QWEN36_DELTANET_FUSED_SEGMENT_TOKENS"] == "0"
    assert preflight["dry_run_env"]["ENABLE_KV_CACHE_QUANT"] == "0"
    assert preflight["dry_run_env"]["SPEED_SLICE"] == "hostlogits"
    dry_run = preflight["dry_run_command"]
    launch = preflight["launch_command_after_automation"]
    assert "TS=20260606T010203Z_hostlogits" in dry_run
    assert "TS=20260606T010203Z_hostlogits" in launch
    assert "COMPILE_DRY_RUN=1" in dry_run
    assert "SPEED_SLICE=hostlogits" in dry_run
    assert "PREFIX_CTE_ATTENTION_BACKEND=attention_cte" in dry_run
    assert "QWEN36_DELTANET_FUSED_SEGMENT_TOKENS=0" in dry_run
    assert "ENABLE_KV_CACHE_QUANT=0" in dry_run
    assert "ENABLE_QKV_CTE_NKI_KERNEL_FUSE_ROPE=0" in dry_run
    assert "REPO=/home/ubuntu/repo" in dry_run
    assert "MODEL=/home/ubuntu/model" in dry_run
    assert "tmp_compile_qwen32k_segcte2048_gdnseg512.sh" in dry_run
    assert "<ENVLOG_FROM_DRY_RUN>" in preflight["automation_payload_command_template"]
    assert "monitor-qwen-hostlogits-20260606t010203z-hostlogits-compile" in preflight[
        "automation_payload_command_template"
    ]
    assert "COMPILE_DRY_RUN=0" in launch


def test_slow_hostlogits_advances_to_lmhead_bf16_only():
    decision = _SCRIPT.decide(
        env_values={"SPEED_SLICE": "hostlogits"},
        runtime_summary=_summary(coherence_ok=True),
        speed_output=_speed(passed=False, mean=900.0),
        speed_json_path=Path("/tmp/raw_speed.json"),
        next_ts="20260606T010203Z_lmhead",
    )

    assert decision["decision"] == "launch_next_speed_slice"
    assert decision["next_speed_slice"] == "hostlogits_lmheadbf16"
    assert decision["next_required_flags"]["QUANTIZE_LM_HEAD"] == "0"
    assert "QUANTIZE_LM_HEAD=0" in decision["next_preflight"]["dry_run_command"]
    assert "DISABLE_ON_DEVICE_SAMPLING=1" in decision["next_preflight"][
        "dry_run_command"
    ]
    assert "TS=20260606T010203Z_lmhead" in decision["next_preflight"][
        "launch_command_after_automation"
    ]
    assert decision["next_preflight"]["launch_env"]["QUANTIZE_LM_HEAD"] == "0"


def test_decision_automation_name_preserves_long_timestamp_suffix():
    name = _SCRIPT._automation_name(
        "hostlogits_lmheadbf16",
        "20260606T010203Z_hostlogits_lmheadbf16",
    )

    assert name.startswith("monitor-qwen-hostlogits-")
    assert name.endswith("20260606t010203z-hostlogits-lmheadbf16-compile")
    assert len(name) <= len("monitor-") + 56 + len("-compile")


def test_next_preflight_dry_run_env_matches_compile_driver(tmp_path):
    repo = tmp_path / "repo"
    model = tmp_path / "model"
    art_root = tmp_path / "artifacts"
    logdir = tmp_path / "logs"
    repo.mkdir()
    model.mkdir()
    decision = _SCRIPT.decide(
        env_values={
            "SPEED_SLICE": "sampletokonly",
            "REPO": str(repo),
            "MODEL": str(model),
            "ART_ROOT": str(art_root),
            "LOGDIR": str(logdir),
        },
        runtime_summary=_summary(coherence_ok=True),
        speed_output=_speed(passed=False, mean=640.0),
        speed_json_path=Path("/tmp/raw_speed.json"),
        next_ts="20260606T010203Z_driver",
    )
    env = os.environ.copy()
    env.update(decision["next_preflight"]["dry_run_env"])

    completed = subprocess.run(
        ["bash", str(_REPO_ROOT / decision["next_preflight"]["compile_driver"])],
        check=True,
        capture_output=True,
        env=env,
        text=True,
    )
    stdout = dict(
        line.split("=", 1)
        for line in completed.stdout.splitlines()
        if "=" in line
    )
    envlog = dict(
        line.split("=", 1)
        for line in Path(stdout["ENVLOG"]).read_text().splitlines()
        if "=" in line
    )

    assert stdout["TS"] == "20260606T010203Z_driver"
    assert envlog["TS"] == "20260606T010203Z_driver"
    assert envlog["SPEED_SLICE"] == "hostlogits"
    assert envlog["SAMPLING"] == "host_logits"
    assert envlog["PREFIX_CTE_ATTENTION_BACKEND"] == "attention_cte"
    assert envlog["QWEN36_DELTANET_FUSED_SEGMENT_TOKENS"] == "0"
    assert envlog["ENABLE_KV_CACHE_QUANT"] == "0"
    assert "hostlogits" in stdout["BASE"]
    assert "attention_cte512" in stdout["BASE"]
    assert "gdnseg0" in stdout["BASE"]


def test_slow_final_planned_slice_profiles_instead_of_branching():
    decision = _SCRIPT.decide(
        env_values={
            "SPEED_SLICE": "hostlogits_lmheadbf16",
            "WORKDIR": "/mnt/work",
            "LOGDIR": "/logs",
            "CTE_BUCKETS_RAW": "2048",
        },
        runtime_summary=_summary(coherence_ok=True),
        speed_output=_speed(passed=False, mean=950.0),
        speed_json_path=Path("/tmp/raw_speed.json"),
        next_ts="20260606T010203Z_profile",
    )

    assert decision["decision"] == "profile_slow_coherent"
    assert decision["next_speed_slice"] is None
    preflight = decision["profile_preflight"]
    assert preflight["do_not_profile_live_vllm"] is True
    assert preflight["context_neff_root"] == "/mnt/work/context_encoding_model"
    assert preflight["output_dir"] == (
        "/logs/context_neff_profile_20260606T010203Z_profile"
    )
    assert preflight["context_tokens"] == 2048
    assert preflight["run_order"][0] == "stop_vllm_or_use_idle_trainium_host"
    assert "--enable-dge" in preflight["run_command"]
    assert "--run" in preflight["run_command"]
    assert "qwen36_context_neff_profile.py" in preflight["run_command"]
    assert "qwen36_profile_summary_compare.py" in preflight["compare_command_template"]
    assert "--speed-json /tmp/raw_speed.json" in preflight[
        "compare_command_template"
    ]
    assert "'current=<SUMMARY_JSON_FROM_context_neff_profile_results>'" in preflight[
        "compare_command_template"
    ]


def test_main_finds_speed_path_from_runtime_summary(tmp_path, monkeypatch, capsys):
    env_log = tmp_path / "env.txt"
    summary_path = tmp_path / "runtime_validation_summary.json"
    speed_path = tmp_path / "raw_prefill_speed.json"
    env_log.write_text("SPEED_SLICE=sampletokonly\nARTIFACT=/artifact\n")
    summary_path.write_text(json.dumps(_summary(speed_path=str(speed_path))) + "\n")
    speed_path.write_text(json.dumps(_speed(mean=630.0)) + "\n")

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "qwen36_speed_slice_decision.py",
            "--env-log",
            str(env_log),
            "--runtime-summary",
            str(summary_path),
            "--next-ts",
            "20260606T010203Z_cli",
        ],
    )

    assert _SCRIPT.main() == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["decision"] == "launch_next_speed_slice"
    assert payload["next_speed_slice"] == "hostlogits"
    assert payload["next_preflight"]["ts"] == "20260606T010203Z_cli"
    assert payload["next_preflight"]["dry_run_env"]["TS"] == "20260606T010203Z_cli"
    assert "COMPILE_DRY_RUN=1" in payload["next_preflight"]["dry_run_command"]
