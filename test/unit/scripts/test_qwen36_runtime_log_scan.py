import importlib.util
import json
import sys
from pathlib import Path


_REPO_ROOT = Path(__file__).resolve().parents[3]
_SCRIPT_PATH = _REPO_ROOT / "validation_scripts/qwen36_runtime_log_scan.py"
_SPEC = importlib.util.spec_from_file_location("qwen36_runtime_log_scan", _SCRIPT_PATH)
_SCRIPT = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_SCRIPT)


def test_scan_text_passes_clean_log():
    result = _SCRIPT.scan_text("server ready\nordinary decode\n")

    assert result["ok"]
    assert result["match_count"] == 0
    assert result["matches"] == {}


def test_scan_text_reports_exact_marker_lines():
    text = (
        "stage=sample_on_device negative token_id=-1054818384\n"
        "logits summary finite=0 nan=248320\n"
    )

    result = _SCRIPT.scan_text(text)

    assert not result["ok"]
    assert result["match_count"] == 3
    assert result["matches"]["negative token_id"][0]["line"] == 1
    assert result["matches"]["finite=0"][0]["line"] == 2
    assert result["matches"]["nan="][0]["line"] == 2


def test_scan_files_aggregates_multiple_logs(tmp_path):
    clean = tmp_path / "clean.log"
    bad = tmp_path / "bad.log"
    clean.write_text("HEALTH_OK\n")
    bad.write_text("fallback argmax -> token_id=0\n")

    result = _SCRIPT.scan_files([clean, bad])

    assert not result["ok"]
    assert result["match_count"] == 1
    assert result["files"][0]["ok"]
    assert not result["files"][1]["ok"]


def test_cli_json_exits_nonzero_on_marker(tmp_path, capsys, monkeypatch):
    log = tmp_path / "serve.log"
    log.write_text("NRT_RESOURCE allocation failed\n")
    monkeypatch.setattr(
        sys,
        "argv",
        ["qwen36_runtime_log_scan.py", "--json", str(log)],
    )

    assert _SCRIPT.main() == 1
    payload = json.loads(capsys.readouterr().out)
    assert payload["match_count"] == 1
    assert payload["files"][0]["matches"]["NRT_RESOURCE"][0]["line"] == 1


def test_cli_custom_marker_replaces_default_markers(tmp_path, capsys, monkeypatch):
    log = tmp_path / "serve.log"
    log.write_text("fallback argmax\ncustom badness\n")
    monkeypatch.setattr(
        sys,
        "argv",
        ["qwen36_runtime_log_scan.py", "--marker", "custom badness", str(log)],
    )

    assert _SCRIPT.main() == 1
    output = capsys.readouterr().out
    assert "custom badness" in output
    assert "fallback argmax" not in output
