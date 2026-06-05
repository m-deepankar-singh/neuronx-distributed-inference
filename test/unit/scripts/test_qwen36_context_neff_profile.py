import importlib.util
import json
import os
import sys
from pathlib import Path


_REPO_ROOT = Path(__file__).resolve().parents[3]
_SCRIPT_PATH = _REPO_ROOT / "validation_scripts/qwen36_context_neff_profile.py"
_SPEC = importlib.util.spec_from_file_location(
    "qwen36_context_neff_profile",
    _SCRIPT_PATH,
)
_SCRIPT = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _SCRIPT
_SPEC.loader.exec_module(_SCRIPT)


def _touch(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"neff")
    return path


def test_discover_targets_selects_tp_rank_and_default_prefixes(tmp_path):
    root = tmp_path / "context_encoding_model"
    _touch(root / "_tp0_bk0" / "graph.neff")
    _touch(root / "_tp0_bk7" / "graph.neff")
    _touch(root / "_tp1_bk7" / "graph.neff")

    targets = _SCRIPT.discover_targets(
        context_neff_root=root,
        tp_rank=0,
        buckets=None,
        prefix_map={},
    )

    assert [target.label for target in targets] == [
        "context_bk0_pfx0",
        "context_bk7_pfx16384",
    ]
    assert [target.prefix_tokens for target in targets] == [0, 16384]


def test_build_plan_filters_buckets_and_renders_capture_command(tmp_path):
    root = tmp_path / "context_encoding_model"
    _touch(root / "_tp0_bk0" / "graph.neff")
    _touch(root / "_tp0_bk7" / "graph.neff")

    plan = _SCRIPT.build_plan(
        context_neff_root=root,
        output_dir=tmp_path / "profiles",
        tool=Path("/opt/aws/neuron/bin/neuron-explorer"),
        tp_rank=0,
        buckets={7},
        prefix_map={7: 12000},
        collectives_worker_count=4,
        collectives_workers_per_node=4,
        num_exec=2,
        profile_nth_exec=2,
        enable_dge=True,
    )

    assert plan["target_count"] == 1
    target = plan["targets"][0]
    assert target["label"] == "context_bk7_pfx12000"
    command = target["capture_command"]
    assert "capture" in command
    assert "--io-from=neff" in command
    assert "--profile-nth-exec=2" in command
    assert "--enable-dge-notifs" in command
    assert not any("pkill" in part for part in command)


def test_parse_prefix_map_rejects_malformed_entries():
    try:
        _SCRIPT._parse_prefix_map("7=16384")
    except ValueError as exc:
        assert "BUCKET:PREFIX" in str(exc)
    else:
        raise AssertionError("malformed prefix map should fail")


def test_run_plan_with_fake_neuron_explorer(tmp_path, monkeypatch):
    root = tmp_path / "context_encoding_model"
    _touch(root / "_tp0_bk0" / "graph.neff")
    fake_tool = tmp_path / "neuron-explorer"
    fake_tool.write_text(
        "#!/usr/bin/env python3\n"
        "import json, pathlib, sys\n"
        "if sys.argv[1] == 'capture':\n"
        "    out = pathlib.Path(sys.argv[sys.argv.index('-s') + 1])\n"
        "    out.parent.mkdir(parents=True, exist_ok=True)\n"
        "    out.write_text('ntff')\n"
        "    sys.exit(0)\n"
        "if sys.argv[1] == 'view':\n"
        "    print(json.dumps({'total_time': 1.25, 'total_active_time': 1.0}))\n"
        "    sys.exit(0)\n"
        "sys.exit(9)\n"
    )
    fake_tool.chmod(fake_tool.stat().st_mode | 0o111)
    monkeypatch.setenv("PATH", os.environ.get("PATH", ""))

    plan = _SCRIPT.build_plan(
        context_neff_root=root,
        output_dir=tmp_path / "profiles",
        tool=fake_tool,
        tp_rank=0,
        buckets=None,
        prefix_map={},
        collectives_worker_count=4,
        collectives_workers_per_node=4,
        num_exec=2,
        profile_nth_exec=2,
        enable_dge=False,
    )
    report = _SCRIPT.run_plan(plan)

    assert report["passed"]
    assert report["summary_json_paths"] == [
        str(tmp_path / "profiles" / "context_bk0_pfx0" / "summary.json")
    ]
    result = report["results"][0]
    assert result["capture_returncode"] == 0
    assert result["view_returncode"] == 0
    assert result["summary_metrics"]["total_time"] == 1.25
    assert (tmp_path / "profiles" / "context_neff_profile_results.json").exists()


def test_main_writes_dry_run_plan(tmp_path, monkeypatch, capsys):
    root = tmp_path / "context_encoding_model"
    _touch(root / "_tp0_bk0" / "graph.neff")
    out = tmp_path / "profiles"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "qwen36_context_neff_profile.py",
            "--context-neff-root",
            str(root),
            "--output-dir",
            str(out),
        ],
    )

    assert _SCRIPT.main() == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["target_count"] == 1
    assert (out / "context_neff_profile_plan.json").exists()
