import importlib.util
import json
import sys
from pathlib import Path


_REPO_ROOT = Path(__file__).resolve().parents[3]
_SCRIPT_PATH = _REPO_ROOT / "validation_scripts/qwen36_next_compile_preflight.py"
_SPEED_SCRIPT_PATH = _REPO_ROOT / "validation_scripts/qwen36_speed_slice_decision.py"

_SPEC = importlib.util.spec_from_file_location(
    "qwen36_next_compile_preflight",
    _SCRIPT_PATH,
)
_SCRIPT = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _SCRIPT
_SPEC.loader.exec_module(_SCRIPT)

_SPEED_SPEC = importlib.util.spec_from_file_location(
    "qwen36_speed_slice_decision_for_next_preflight",
    _SPEED_SCRIPT_PATH,
)
_SPEED_SCRIPT = importlib.util.module_from_spec(_SPEED_SPEC)
sys.modules[_SPEED_SPEC.name] = _SPEED_SCRIPT
_SPEED_SPEC.loader.exec_module(_SPEED_SCRIPT)


def _runtime_summary():
    return {
        "passed": False,
        "coherence_and_log_scan_passed": True,
        "results": [
            {
                "name": "raw_prefill_speed",
                "phase": "speed",
                "output_path": "/tmp/raw_speed.json",
                "passed": False,
                "skipped": False,
            }
        ],
    }


def _speed():
    return {
        "passed": False,
        "row_gate_passed": True,
        "prefill_tok_s_mean": 640.0,
        "speed_gate": {
            "enabled": True,
            "passed": False,
            "min_prefill_tok_s": 3000.0,
            "mean_prefill_tok_s": 640.0,
            "failure_reason": "mean_prefill_tok_s_below_threshold",
        },
    }


def _decision(tmp_path):
    model = tmp_path / "model"
    art_root = tmp_path / "artifacts"
    logdir = tmp_path / "logs"
    model.mkdir()
    art_root.mkdir()
    logdir.mkdir()
    return _SPEED_SCRIPT.decide(
        env_values={
            "SPEED_SLICE": "sampletokonly",
            "REPO": str(_REPO_ROOT),
            "MODEL": str(model),
            "ART_ROOT": str(art_root),
            "LOGDIR": str(logdir),
        },
        runtime_summary=_runtime_summary(),
        speed_output=_speed(),
        speed_json_path=Path("/tmp/raw_speed.json"),
        next_ts="20260606T010203Z_hostlogits",
    )


def test_build_preflight_runs_dry_run_and_builds_automation_payload(tmp_path):
    decision = _decision(tmp_path)
    decision_path = tmp_path / "speed_slice_decision.json"
    decision_path.write_text(json.dumps(decision) + "\n")

    result = _SCRIPT.build_preflight(
        decision=decision,
        decision_path=decision_path,
        repo_root=_REPO_ROOT,
        compile_host="ubuntu@compile",
        runtime_host="ubuntu@runtime",
        source_dir=None,
        source_commit=None,
        automation_name=None,
        automation_created_name=None,
        automation_interval_minutes=7,
        launch_script="tmp_launch_qwen36_segcte2048.sh",
        boundary_lengths="146,160",
    )

    assert result["passed"] is True
    assert result["next_speed_slice"] == "hostlogits"
    assert result["ts"] == "20260606T010203Z_hostlogits"
    assert result["requires_automation_creation_before_compile"] is True
    assert result["run_order"] == [
        "create_heartbeat_automation_from_automation_payload",
        "rerun_next_compile_preflight_with_automation_created_name",
        "run_compile_command_after_automation_exists",
    ]
    assert result["compile_command_after_automation"] is None
    assert result["launch_command_after_automation"] is None
    assert result["automation_ack"] == {
        "required": True,
        "created_name": None,
        "matched_payload_name": False,
    }
    assert result["dry_run"]["returncode"] == 0
    assert Path(result["dry_run"]["env_log"]).exists()

    payload = result["automation_payload"]
    assert payload["mode"] == "create"
    assert payload["kind"] == "heartbeat"
    assert payload["destination"] == "thread"
    assert payload["rrule"] == "FREQ=MINUTELY;INTERVAL=7"
    assert payload["name"].startswith("monitor-qwen-hostlogits-")
    assert "20260606t010203z-hostlogits" in payload["name"]
    assert "Compile log:" in payload["prompt"]
    assert result["dry_run"]["env_log"] in payload["prompt"]
    assert "ubuntu@compile" in payload["prompt"]
    assert "ubuntu@runtime" in payload["prompt"]
    release = result["compile_command_release_command_template"]
    assert payload["name"] in release
    assert f"--repo-root {str(_REPO_ROOT)}" in release
    assert "--compile-host ubuntu@compile" in release
    assert "--runtime-host ubuntu@runtime" in release
    assert "--source-dir" in release
    assert "--source-commit" in release
    assert "--automation-interval-minutes 7" in release
    assert "--launch-script tmp_launch_qwen36_segcte2048.sh" in release
    assert "--boundary-lengths 146,160" in release
    assert "NEXT_COMPILE_RELEASE_JSON" in release
    assert result["release_preflight_context"]["repo_root"] == str(_REPO_ROOT)
    assert result["release_preflight_context"]["automation_name"] == payload["name"]
    assert result["release_preflight_context"]["compile_host"] == "ubuntu@compile"
    assert result["release_preflight_context"]["runtime_host"] == "ubuntu@runtime"


def test_build_preflight_releases_compile_command_after_matching_automation_ack(
    tmp_path,
):
    decision = _decision(tmp_path)
    decision_path = tmp_path / "speed_slice_decision.json"
    decision_path.write_text(json.dumps(decision) + "\n")
    automation_name = "monitor-explicit-hostlogits-compile"

    result = _SCRIPT.build_preflight(
        decision=decision,
        decision_path=decision_path,
        repo_root=_REPO_ROOT,
        compile_host="ubuntu@compile",
        runtime_host="ubuntu@runtime",
        source_dir=None,
        source_commit=None,
        automation_name=automation_name,
        automation_created_name=automation_name,
        automation_interval_minutes=7,
        launch_script="tmp_launch_qwen36_segcte2048.sh",
        boundary_lengths="146,160",
    )

    assert result["automation_payload"]["name"] == automation_name
    assert result["automation_ack"] == {
        "required": True,
        "created_name": automation_name,
        "matched_payload_name": True,
    }
    assert result["run_order"] == ["run_compile_command_after_automation_exists"]
    assert result["compile_command_after_automation"] == result[
        "launch_command_after_automation"
    ]
    assert "COMPILE_DRY_RUN=0" in result["compile_command_after_automation"]


def test_build_preflight_rejects_mismatched_automation_ack(tmp_path):
    decision = _decision(tmp_path)
    decision_path = tmp_path / "speed_slice_decision.json"
    decision_path.write_text(json.dumps(decision) + "\n")

    try:
        _SCRIPT.build_preflight(
            decision=decision,
            decision_path=decision_path,
            repo_root=_REPO_ROOT,
            compile_host="ubuntu@compile",
            runtime_host="ubuntu@runtime",
            source_dir=None,
            source_commit=None,
            automation_name="monitor-expected",
            automation_created_name="monitor-wrong",
            automation_interval_minutes=7,
            launch_script="tmp_launch_qwen36_segcte2048.sh",
            boundary_lengths="146,160",
        )
    except ValueError as exc:
        assert "automation-created-name does not match" in str(exc)
    else:
        raise AssertionError("mismatched automation ack should fail")


def test_cli_writes_preflight_and_automation_payload_json(tmp_path, monkeypatch, capsys):
    decision = _decision(tmp_path)
    decision_path = tmp_path / "speed_slice_decision.json"
    output_path = tmp_path / "next_compile_preflight.json"
    automation_payload_path = tmp_path / "automation_payload.json"
    decision_path.write_text(json.dumps(decision) + "\n")

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "qwen36_next_compile_preflight.py",
            "--decision-json",
            str(decision_path),
            "--repo-root",
            str(_REPO_ROOT),
            "--output-json",
            str(output_path),
            "--automation-json-output",
            str(automation_payload_path),
        ],
    )

    assert _SCRIPT.main() == 0
    printed = json.loads(capsys.readouterr().out)
    written = json.loads(output_path.read_text())
    automation_payload = json.loads(automation_payload_path.read_text())

    assert printed["passed"] is True
    assert written["passed"] is True
    assert printed["automation_payload_path"] == str(automation_payload_path)
    assert (
        "--boundary-lengths "
        "123,146,160,485,505,526,1225,1265,1346,2048,2049,2500,4092,4096"
    ) in printed["compile_command_release_command_template"]
    assert automation_payload["mode"] == "create"
    assert automation_payload["name"] == printed["automation_payload"]["name"]
    assert printed["run_order"][0] == "create_heartbeat_automation_from_automation_payload"
    assert printed["compile_command_after_automation"] is None


def test_cli_releases_compile_command_after_automation_ack(
    tmp_path,
    monkeypatch,
    capsys,
):
    decision = _decision(tmp_path)
    decision_path = tmp_path / "speed_slice_decision.json"
    output_path = tmp_path / "next_compile_release.json"
    decision_path.write_text(json.dumps(decision) + "\n")

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "qwen36_next_compile_preflight.py",
            "--decision-json",
            str(decision_path),
            "--repo-root",
            str(_REPO_ROOT),
            "--output-json",
            str(output_path),
            "--automation-name",
            "monitor-cli-release",
            "--automation-created-name",
            "monitor-cli-release",
        ],
    )

    assert _SCRIPT.main() == 0
    printed = json.loads(capsys.readouterr().out)
    written = json.loads(output_path.read_text())

    assert printed["automation_ack"]["matched_payload_name"] is True
    assert written["compile_command_after_automation"] == printed[
        "compile_command_after_automation"
    ]
    assert "COMPILE_DRY_RUN=0" in printed["compile_command_after_automation"]


def test_cli_rejects_decision_that_does_not_request_compile(tmp_path, monkeypatch, capsys):
    decision_path = tmp_path / "speed_slice_decision.json"
    output_path = tmp_path / "next_compile_preflight.json"
    decision_path.write_text(json.dumps({"decision": "candidate_meets_target"}) + "\n")

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "qwen36_next_compile_preflight.py",
            "--decision-json",
            str(decision_path),
            "--repo-root",
            str(_REPO_ROOT),
            "--output-json",
            str(output_path),
        ],
    )

    assert _SCRIPT.main() == 1
    printed = json.loads(capsys.readouterr().out)
    written = json.loads(output_path.read_text())

    assert printed["passed"] is False
    assert written["passed"] is False
    assert printed["error_type"] == "ValueError"
    assert "not requesting a compile" in printed["error"]
