import importlib.util
import json
import sys
from pathlib import Path


_REPO_ROOT = Path(__file__).resolve().parents[3]
_SCRIPT_PATH = _REPO_ROOT / "validation_scripts/qwen36_profile_summary_compare.py"
_SPEC = importlib.util.spec_from_file_location(
    "qwen36_profile_summary_compare",
    _SCRIPT_PATH,
)
_SCRIPT = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _SCRIPT
_SPEC.loader.exec_module(_SCRIPT)


def _write_json(path: Path, payload: object) -> Path:
    path.write_text(json.dumps(payload) + "\n")
    return path


def _speed_payload(*, prompt_tokens: int = 16384, repeats: int = 3, mean: float = 628.0):
    return {
        "passed": True,
        "lengths": [prompt_tokens],
        "repeats": repeats,
        "max_tokens": 1,
        "allow_usage_fallback": False,
        "require_text": False,
        "min_prefill_tok_s": 3000.0,
        "row_gate_passed": True,
        "prefill_tokens_all_match_actual": True,
        "prefill_tok_s_mean": mean,
        "speed_gate": {
            "enabled": True,
            "passed": True,
            "min_prefill_tok_s": 3000.0,
            "mean_prefill_tok_s": mean,
            "failure_reason": None,
        },
        "results": [
            {
                "target_prompt_tokens": prompt_tokens,
                "actual_prompt_tokens": prompt_tokens,
                "repeat": repeat,
                "status": 200,
                "ttft_seconds": prompt_tokens / mean,
                "usage": {"prompt_tokens": prompt_tokens, "completion_tokens": 1},
                "prefill_tokens": prompt_tokens,
                "prefill_token_source": "usage",
                "prefill_tokens_match_actual": True,
                "prefill_tok_s": mean,
            }
            for repeat in range(repeats)
        ],
    }


def test_extract_summary_metrics_accepts_flat_neuron_summary():
    metrics = _SCRIPT.extract_summary_metrics(
        {
            "total_time": "3.105",
            "total_active_time": 1.24,
            "hbm_read_bytes": 10,
            "not_numeric": "ignored",
        }
    )

    assert metrics["total_time"] == 3.105
    assert metrics["total_active_time"] == 1.24
    assert metrics["hbm_read_bytes"] == 10.0
    assert "not_numeric" not in metrics


def test_extract_summary_metrics_accepts_nested_summary_rows():
    metrics = _SCRIPT.extract_summary_metrics(
        {
            "Summary": [
                {
                    "total_time": 0.5,
                    "tensor_engine_active_time": 0.2,
                }
            ]
        }
    )

    assert metrics["total_time"] == 0.5
    assert metrics["tensor_engine_active_time"] == 0.2


def test_build_report_estimates_prefill_speed_and_target_gap(tmp_path):
    summary = _write_json(
        tmp_path / "summary.json",
        {
            "total_time": 3.2,
            "total_active_time": 1.6,
            "tensor_engine_active_time": 0.8,
            "vector_engine_active_time": 0.4,
            "dma_active_time": 0.3,
            "hbm_read_bytes": 100,
            "hbm_write_bytes": 50,
        },
    )

    report = _SCRIPT.build_report(
        summaries=[f"current={summary}"],
        prompt_tokens=16384,
        context_tokens=2048,
        target_prefill_tok_s=3000.0,
    )
    profile = report["profiles"][0]

    assert report["chunks_for_prompt"] == 8
    assert profile["estimated_prompt_seconds"] == 25.6
    assert profile["estimated_prefill_tok_s"] == 640.0
    assert round(profile["estimated_speedup_needed"], 4) == 4.6875
    assert round(profile["target_context_time_seconds"], 6) == round(
        16384 / 3000 / 8,
        6,
    )
    assert profile["dominant_active_engine"] == "tensor"
    assert profile["hbm_total_bytes"] == 150.0


def test_build_report_compares_candidate_to_baseline(tmp_path):
    baseline = _write_json(tmp_path / "baseline.json", {"total_time": 3.0})
    candidate = _write_json(tmp_path / "candidate.json", {"total_time": 1.5})

    report = _SCRIPT.build_report(
        summaries=[f"baseline={baseline}", f"candidate={candidate}"],
        prompt_tokens=4096,
        context_tokens=2048,
        target_prefill_tok_s=3000.0,
    )

    comparison = report["comparisons"][0]
    assert comparison["baseline"] == "baseline"
    assert comparison["candidate"] == "candidate"
    assert comparison["context_time_ratio_candidate_over_baseline"] == 0.5
    assert comparison["context_time_speedup_baseline_over_candidate"] == 2.0
    assert abs(comparison["estimated_prefill_tok_s_delta"] - 682.6666666666667) < 1e-9


def test_speed_summary_parses_raw_prefill_benchmark_rows(tmp_path):
    speed = _write_json(
        tmp_path / "speed.json",
        {
            "passed": True,
            "prefill_tok_s_mean": 628.0,
            "results": [
                {"prefill_tok_s": 627.0, "ttft_seconds": 26.1, "prefill_tokens": 16384},
                {"prefill_tok_s": 629.0, "ttft_seconds": 26.0, "prefill_tokens": 16384},
            ],
        },
    )
    summary = _SCRIPT.extract_speed_summary(json.loads(speed.read_text()))

    assert summary["path_prefill_tok_s_mean"] == 628.0
    assert summary["row_prefill_tok_s_mean"] == 628.0
    assert summary["ttft_seconds_mean"] == 26.05
    assert summary["prefill_tokens_mean"] == 16384.0
    assert summary["run_count"] == 2
    assert summary["passed"] is True


def test_build_report_accepts_usage_accounted_speed_json(tmp_path):
    profile = _write_json(tmp_path / "profile.json", {"total_time": 3.2})
    speed = _write_json(tmp_path / "speed.json", _speed_payload())

    report = _SCRIPT.build_report(
        summaries=[f"current={profile}"],
        prompt_tokens=16384,
        context_tokens=2048,
        target_prefill_tok_s=3000.0,
        speed_json=speed,
    )

    assert report["speed_summary"]["run_count"] == 3
    assert report["speed_summary"]["prefill_tokens_mean"] == 16384.0


def test_build_report_rejects_stale_or_weak_speed_json(tmp_path):
    profile = _write_json(tmp_path / "profile.json", {"total_time": 3.2})
    speed = _write_json(
        tmp_path / "speed.json",
        _speed_payload(prompt_tokens=8192, repeats=1),
    )

    try:
        _SCRIPT.build_report(
            summaries=[f"current={profile}"],
            prompt_tokens=16384,
            context_tokens=2048,
            target_prefill_tok_s=3000.0,
            speed_json=speed,
        )
    except ValueError as exc:
        text = str(exc)
        assert "speed JSON does not match" in text
        assert "lengths" in text
        assert "repeats" in text
    else:
        raise AssertionError("weak speed JSON should be rejected")
