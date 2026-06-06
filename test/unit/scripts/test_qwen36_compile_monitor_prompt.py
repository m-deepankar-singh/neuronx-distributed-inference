import importlib.util
import json
import sys
from pathlib import Path


_REPO_ROOT = Path(__file__).resolve().parents[3]
_SCRIPT_PATH = _REPO_ROOT / "validation_scripts/qwen36_compile_monitor_prompt.py"
_SPEC = importlib.util.spec_from_file_location(
    "qwen36_compile_monitor_prompt",
    _SCRIPT_PATH,
)
_SCRIPT = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_SCRIPT)


def test_parse_env_log_skips_comments_and_blank_lines(tmp_path):
    env_log = tmp_path / "env.txt"
    env_log.write_text(
        "\n# ignored\nBASE=test\nARTIFACT=/artifact\nWORKDIR=/work\n"
        "LOG=/compile.log\nENVLOG=/env.txt\nPIDFILE=/compile.pid\n"
    )

    values = _SCRIPT.parse_env_log(env_log)

    assert values["BASE"] == "test"
    assert values["ARTIFACT"] == "/artifact"
    assert "#" not in values


def test_render_prompt_contains_compile_validation_and_runtime_gates():
    values = {
        "BASE": "qwen36_hostlogits_unittest",
        "ARTIFACT": "/mnt/artifacts/qwen36_hostlogits_unittest",
        "WORKDIR": "/mnt/artifacts/_work_qwen36_hostlogits_unittest",
        "LOG": "/logs/compile.log",
        "ENVLOG": "/logs/env.txt",
        "PIDFILE": "/logs/compile.pid",
        "SAMPLING": "host_logits",
        "KERNELS": "decode_deltanet,qkvnki_qknorm,outprojstd,segmented_attention_cte",
        "SOURCE_COMMIT": "envc0de",
        "SOURCE_BRANCH": "codex/qwen36-prefill-speed-coherent",
        "SEQ_LEN": "32768",
        "MAX_CONTEXT_LENGTH": "32768",
        "TS": "20260606T010203Z_hostlogits",
        "CTE_BUCKETS": "2048",
        "GDN_RECURRENT_CACHE_DTYPE": "bfloat16",
        "GDN_CONV_CACHE_DTYPE": "bfloat16",
        "DISABLE_ON_DEVICE_SAMPLING": "1",
        "OUTPUT_LOGITS_WITH_ON_DEVICE_SAMPLING": "0",
        "SPEED_SLICE": "hostlogits",
    }

    prompt = _SCRIPT.render_prompt(
        values,
        compile_host="ubuntu@compile",
        runtime_host="ubuntu@runtime",
        source_dir="/home/ubuntu/repo",
        source_commit="abc1234",
        launch_script="tmp_launch_qwen36_segcte2048.sh",
        boundary_lengths="146,160",
    )

    assert "ubuntu@compile" in prompt
    assert "ubuntu@runtime" in prompt
    assert "abc1234" in prompt
    assert "- SOURCE_COMMIT=envc0de" in prompt
    assert "- SOURCE_BRANCH=codex/qwen36-prefill-speed-coherent" in prompt
    assert "PID file: /logs/compile.pid" in prompt
    assert "- SAMPLING=host_logits" in prompt
    assert "- SPEED_SLICE=hostlogits" in prompt
    assert "- TS=20260606T010203Z_hostlogits" in prompt
    assert "- DISABLE_ON_DEVICE_SAMPLING=1" in prompt
    assert "`Finished Compilation for all HLOs`" in prompt
    assert "`CHECKPOINT_BANK_WEIGHTS_ADDED` for tp0..tp3" in prompt
    assert "`COMPILE_DONE`" in prompt
    assert "qwen36_compile_status.py" in prompt
    assert "return a quiet/DONT_NOTIFY heartbeat status" in prompt
    assert "rsync EC2-to-EC2" in prompt
    assert "copy the compile env log to a runtime-local path" in prompt
    assert "qwen36_launch_preflight.py" in prompt
    assert "LAUNCH_DRY_RUN=1" in prompt
    assert "qwen36_launch_env_audit.py" in prompt
    assert "live_launch_command" in prompt
    assert "A launch audit failure is a validation failure" in prompt
    assert "qwen36_runtime_validation_matrix.py" in prompt
    assert "--compile-env-log <runtime-compile-env-log>" in prompt
    assert "speed_slice_decision.json" in prompt
    assert "qwen36_validation_tool_manifest.py" in prompt
    assert "not the old ad hoc `/tmp/tmp_bisect_probe3.py`" in prompt
    assert "exact boundary lengths 146,160" in prompt
    assert "`negative token_id`" in prompt
    assert "`fallback argmax`" in prompt
    assert "usage.prompt_tokens / TTFT" in prompt
    assert "--min-prefill-tok-s 3000" in prompt
    assert "do not launch that compile directly from this monitor" in prompt
    assert "Do not launch a duplicate compile" in prompt


def test_cli_prefers_env_log_source_commit(tmp_path, monkeypatch, capsys):
    env_log = tmp_path / "env.txt"
    env_log.write_text(
        "BASE=qwen36_source_identity\n"
        "ARTIFACT=/artifact\n"
        "WORKDIR=/work\n"
        "LOG=/compile.log\n"
        f"ENVLOG={env_log}\n"
        "PIDFILE=/compile.pid\n"
        "REPO=/missing/repo\n"
        "SOURCE_COMMIT=envcommit\n"
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "qwen36_compile_monitor_prompt.py",
            "--env-log",
            str(env_log),
        ],
    )

    assert _SCRIPT.main() == 0
    prompt = capsys.readouterr().out
    assert "Source: /missing/repo at commit envcommit" in prompt


def test_render_prompt_rejects_missing_required_env_key():
    values = {
        "BASE": "qwen36_missing",
        "ARTIFACT": "/artifact",
        "WORKDIR": "/work",
        "LOG": "/log",
        "ENVLOG": "/env",
    }

    try:
        _SCRIPT.render_prompt(
            values,
            compile_host="ubuntu@compile",
            runtime_host="ubuntu@runtime",
            source_dir="/repo",
            source_commit="abc1234",
            launch_script="tmp_launch_qwen36_segcte2048.sh",
            boundary_lengths="146",
        )
    except ValueError as exc:
        assert "PIDFILE" in str(exc)
    else:
        raise AssertionError("missing PIDFILE should fail")


def test_build_automation_payload_uses_speed_slice_name_and_prompt():
    values = {
        "BASE": "qwen36_very_long_candidate_name_that_should_be_safely_shortened",
        "SPEED_SLICE": "hostlogits_lmheadbf16",
        "TS": "20260606T010203Z_lmhead",
    }

    payload = _SCRIPT.build_automation_payload(
        values,
        prompt="Monitor this compile",
        interval_minutes=7,
    )

    assert payload["mode"] == "create"
    assert payload["kind"] == "heartbeat"
    assert payload["destination"] == "thread"
    assert payload["status"] == "ACTIVE"
    assert payload["rrule"] == "FREQ=MINUTELY;INTERVAL=7"
    assert payload["name"] == (
        "monitor-qwen-hostlogits-lmheadbf16-20260606t010203z-lmhead-compile"
    )
    assert payload["prompt"] == "Monitor this compile"


def test_build_automation_payload_rejects_non_positive_interval():
    try:
        _SCRIPT.build_automation_payload({}, prompt="x", interval_minutes=0)
    except ValueError as exc:
        assert "interval_minutes" in str(exc)
    else:
        raise AssertionError("non-positive interval should fail")


def test_automation_name_preserves_timestamp_when_base_is_long():
    payload = _SCRIPT.build_automation_payload(
        {
            "BASE": "qwen36_" + ("verylongcomponent_" * 8),
            "SPEED_SLICE": "none",
            "TS": "20260606T010203Z_hostlogits",
        },
        prompt="Monitor this compile",
    )

    assert payload["name"].startswith("monitor-qwen36-")
    assert payload["name"].endswith("20260606t010203z-hostlogits-compile")
    assert len(payload["name"]) <= len("monitor-") + 56 + len("-compile")


def test_main_uses_env_log_argument_when_envlog_key_is_missing(
    tmp_path,
    capsys,
    monkeypatch,
):
    env_log = tmp_path / "compile_env.txt"
    env_log.write_text(
        "BASE=qwen36_old_env\n"
        "ARTIFACT=/artifact\n"
        "WORKDIR=/work\n"
        "LOG=/compile.log\n"
        "PIDFILE=/compile.pid\n"
    )

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "qwen36_compile_monitor_prompt.py",
            "--env-log",
            str(env_log),
            "--source-dir",
            "/repo",
            "--source-commit",
            "abc1234",
        ],
    )

    assert _SCRIPT.main() == 0
    assert f"Env log: {env_log}" in capsys.readouterr().out


def test_main_can_emit_automation_json_payload(tmp_path, capsys, monkeypatch):
    env_log = tmp_path / "compile_env.txt"
    env_log.write_text(
        "BASE=qwen36_hostlogits_test\n"
        "ARTIFACT=/artifact\n"
        "WORKDIR=/work\n"
        "LOG=/compile.log\n"
        "ENVLOG=/env.txt\n"
        "PIDFILE=/compile.pid\n"
        "SPEED_SLICE=hostlogits\n"
    )

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "qwen36_compile_monitor_prompt.py",
            "--env-log",
            str(env_log),
            "--source-dir",
            "/repo",
            "--source-commit",
            "abc1234",
            "--automation-json",
            "--automation-name",
            "monitor-explicit-name",
            "--automation-interval-minutes",
            "12",
        ],
    )

    assert _SCRIPT.main() == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["name"] == "monitor-explicit-name"
    assert payload["rrule"] == "FREQ=MINUTELY;INTERVAL=12"
    assert payload["kind"] == "heartbeat"
    assert "PID file: /compile.pid" in payload["prompt"]
