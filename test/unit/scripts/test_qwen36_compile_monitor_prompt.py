import importlib.util
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
        "SEQ_LEN": "32768",
        "MAX_CONTEXT_LENGTH": "32768",
        "CTE_BUCKETS": "2048",
        "GDN_RECURRENT_CACHE_DTYPE": "bfloat16",
        "GDN_CONV_CACHE_DTYPE": "bfloat16",
        "DISABLE_ON_DEVICE_SAMPLING": "1",
        "OUTPUT_LOGITS_WITH_ON_DEVICE_SAMPLING": "0",
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
    assert "PID file: /logs/compile.pid" in prompt
    assert "- SAMPLING=host_logits" in prompt
    assert "- DISABLE_ON_DEVICE_SAMPLING=1" in prompt
    assert "`Finished Compilation for all HLOs`" in prompt
    assert "`CHECKPOINT_BANK_WEIGHTS_ADDED` for tp0..tp3" in prompt
    assert "`COMPILE_DONE`" in prompt
    assert "qwen36_compile_status.py" in prompt
    assert "return a quiet/DONT_NOTIFY heartbeat status" in prompt
    assert "rsync EC2-to-EC2" in prompt
    assert "qwen36_runtime_validation_matrix.py" in prompt
    assert "not the old ad hoc `/tmp/tmp_bisect_probe3.py`" in prompt
    assert "exact boundary lengths 146,160" in prompt
    assert "`negative token_id`" in prompt
    assert "`fallback argmax`" in prompt
    assert "usage.prompt_tokens / TTFT" in prompt
    assert "Do not launch a duplicate compile" in prompt


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
