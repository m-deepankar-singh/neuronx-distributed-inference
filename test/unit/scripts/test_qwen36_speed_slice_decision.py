import importlib.util
import json
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
        runtime_summary=_summary(coherence_ok=False),
        speed_output=_speed(),
        speed_json_path=Path("/tmp/raw_speed.json"),
    )

    assert decision["decision"] == "stop_incoherent"
    assert decision["next_speed_slice"] is None


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
        env_values={"SPEED_SLICE": "sampletokonly"},
        runtime_summary=_summary(coherence_ok=True),
        speed_output=_speed(passed=False, mean=640.0),
        speed_json_path=Path("/tmp/raw_speed.json"),
    )

    assert decision["decision"] == "launch_next_speed_slice"
    assert decision["next_speed_slice"] == "hostlogits"
    assert decision["next_required_flags"]["DISABLE_ON_DEVICE_SAMPLING"] == "1"
    assert decision["next_required_flags"]["QUANTIZE_LM_HEAD"] == "1"


def test_slow_hostlogits_advances_to_lmhead_bf16_only():
    decision = _SCRIPT.decide(
        env_values={"SPEED_SLICE": "hostlogits"},
        runtime_summary=_summary(coherence_ok=True),
        speed_output=_speed(passed=False, mean=900.0),
        speed_json_path=Path("/tmp/raw_speed.json"),
    )

    assert decision["decision"] == "launch_next_speed_slice"
    assert decision["next_speed_slice"] == "hostlogits_lmheadbf16"
    assert decision["next_required_flags"]["QUANTIZE_LM_HEAD"] == "0"


def test_slow_final_planned_slice_profiles_instead_of_branching():
    decision = _SCRIPT.decide(
        env_values={"SPEED_SLICE": "hostlogits_lmheadbf16"},
        runtime_summary=_summary(coherence_ok=True),
        speed_output=_speed(passed=False, mean=950.0),
        speed_json_path=Path("/tmp/raw_speed.json"),
    )

    assert decision["decision"] == "profile_slow_coherent"
    assert decision["next_speed_slice"] is None


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
        ],
    )

    assert _SCRIPT.main() == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["decision"] == "launch_next_speed_slice"
    assert payload["next_speed_slice"] == "hostlogits"
