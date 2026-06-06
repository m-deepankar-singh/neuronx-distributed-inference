import importlib.util
import json
import sys
from pathlib import Path


_REPO_ROOT = Path(__file__).resolve().parents[3]
_SCRIPT_PATH = _REPO_ROOT / "validation_scripts/qwen36_validation_tool_manifest.py"
_SPEC = importlib.util.spec_from_file_location(
    "qwen36_validation_tool_manifest",
    _SCRIPT_PATH,
)
_SCRIPT = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _SCRIPT
_SPEC.loader.exec_module(_SCRIPT)


def test_default_manifest_covers_compile_driver_and_validation_tools():
    assert "tmp_compile_qwen32k_segcte2048_gdnseg512.sh" in _SCRIPT.DEFAULT_FILES
    assert "tmp_launch_qwen36_segcte2048.sh" in _SCRIPT.DEFAULT_FILES
    assert "validation_scripts/qwen36_runtime_validation_matrix.py" in _SCRIPT.DEFAULT_FILES
    assert "validation_scripts/qwen36_launch_env_audit.py" in _SCRIPT.DEFAULT_FILES
    assert "validation_scripts/qwen36_launch_preflight.py" in _SCRIPT.DEFAULT_FILES
    assert "validation_scripts/qwen36_speed_slice_decision.py" in _SCRIPT.DEFAULT_FILES
    assert (
        "contrib/models/Qwen3.6-27B/test/unit/test_qwen36_compile_driver.py"
        in _SCRIPT.DEFAULT_TEST_FILES
    )
    assert (
        "test/unit/scripts/test_qwen36_launch_driver.py"
        in _SCRIPT.DEFAULT_TEST_FILES
    )
    assert (
        "test/unit/scripts/test_qwen36_launch_env_audit.py"
        in _SCRIPT.DEFAULT_TEST_FILES
    )
    assert (
        "test/unit/scripts/test_qwen36_launch_preflight.py"
        in _SCRIPT.DEFAULT_TEST_FILES
    )
    assert (
        "test/unit/scripts/test_qwen36_speed_slice_decision.py"
        in _SCRIPT.DEFAULT_TEST_FILES
    )


def test_default_manifest_paths_exist_in_repo():
    missing = [
        path
        for path in [*_SCRIPT.DEFAULT_FILES, *_SCRIPT.DEFAULT_TEST_FILES]
        if not (_REPO_ROOT / path).exists()
    ]

    assert missing == []


def test_build_manifest_records_hashes_and_missing_files(tmp_path):
    existing = tmp_path / "validation_scripts" / "tool.py"
    existing.parent.mkdir(parents=True)
    existing.write_text("print('ok')\n")

    manifest = _SCRIPT.build_manifest(
        repo=tmp_path,
        files=[
            "validation_scripts/tool.py",
            "validation_scripts/missing.py",
        ],
    )

    assert manifest["schema"] == "qwen36-validation-tool-manifest-v1"
    assert manifest["files"][0]["exists"]
    assert manifest["files"][0]["size_bytes"] == len("print('ok')\n")
    assert manifest["files"][0]["sha256"]
    assert not manifest["files"][1]["exists"]
    assert manifest["files"][1]["sha256"] is None


def test_verify_manifest_detects_hash_mismatch(tmp_path):
    script = tmp_path / "validation_scripts" / "tool.py"
    script.parent.mkdir(parents=True)
    script.write_text("before\n")
    manifest = _SCRIPT.build_manifest(repo=tmp_path, files=["validation_scripts/tool.py"])

    script.write_text("after\n")
    result = _SCRIPT.verify_manifest(repo=tmp_path, manifest=manifest)

    assert not result["passed"]
    assert result["mismatch_count"] == 1
    assert result["mismatches"][0]["path"] == "validation_scripts/tool.py"
    assert result["mismatches"][0]["expected"]["sha256"] != result["mismatches"][0][
        "actual"
    ]["sha256"]


def test_verify_manifest_passes_for_matching_checkout(tmp_path):
    script = tmp_path / "validation_scripts" / "tool.py"
    script.parent.mkdir(parents=True)
    script.write_text("same\n")
    manifest = _SCRIPT.build_manifest(repo=tmp_path, files=["validation_scripts/tool.py"])

    result = _SCRIPT.verify_manifest(repo=tmp_path, manifest=manifest)

    assert result["passed"]
    assert result["mismatch_count"] == 0


def test_file_list_includes_existing_files_only(tmp_path):
    script = tmp_path / "validation_scripts" / "tool.py"
    script.parent.mkdir(parents=True)
    script.write_text("same\n")
    manifest = _SCRIPT.build_manifest(
        repo=tmp_path,
        files=["validation_scripts/tool.py", "validation_scripts/missing.py"],
    )

    assert _SCRIPT._file_list(manifest) == ["validation_scripts/tool.py"]


def test_main_build_and_verify_roundtrip(tmp_path, monkeypatch, capsys):
    script = tmp_path / "validation_scripts" / "tool.py"
    script.parent.mkdir(parents=True)
    script.write_text("same\n")
    manifest_path = tmp_path / "manifest.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "qwen36_validation_tool_manifest.py",
            "--repo",
            str(tmp_path),
            "--file",
            "validation_scripts/tool.py",
            "--output-json",
            str(manifest_path),
        ],
    )

    assert _SCRIPT.main() == 1
    payload = json.loads(capsys.readouterr().out)
    # Default files are absent in this temp repo, so build exits nonzero.
    assert any(not entry["exists"] for entry in payload["files"])
    assert manifest_path.exists()

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "qwen36_validation_tool_manifest.py",
            "--repo",
            str(tmp_path),
            "--manifest",
            str(manifest_path),
            "--mode",
            "verify",
        ],
    )

    assert _SCRIPT.main() == 0
    verify_payload = json.loads(capsys.readouterr().out)
    assert verify_payload["passed"]
