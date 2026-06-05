import importlib.util
from pathlib import Path


_REPO_ROOT = Path(__file__).resolve().parents[3]
_SCRIPT_PATH = _REPO_ROOT / "validation_scripts/qwen36_raw_completion_prefill_bench.py"
_SPEC = importlib.util.spec_from_file_location(
    "qwen36_raw_completion_prefill_bench",
    _SCRIPT_PATH,
)
_SCRIPT = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_SCRIPT)


def test_parse_lengths_accepts_commas_and_spaces():
    assert _SCRIPT._parse_lengths("160,1225 2500") == [160, 1225, 2500]


def test_sse_json_payloads_ignore_done_and_blank_lines():
    events = _SCRIPT._sse_json_payloads(
        [
            b"",
            b"data: {\"choices\":[{\"text\":\"A\"}]}",
            b"data: {\"usage\":{\"prompt_tokens\":16384}}",
            b"data: [DONE]",
            b"data: {\"choices\":[{\"text\":\"late\"}]}",
        ]
    )

    assert events == [
        {"choices": [{"text": "A"}]},
        {"usage": {"prompt_tokens": 16384}},
    ]


def test_prefill_token_count_prefers_usage_tokens():
    tokens, source = _SCRIPT._prefill_token_count(
        usage={"prompt_tokens": "16384"},
        actual_prompt_tokens=999,
        allow_usage_fallback=False,
    )

    assert tokens == 16384
    assert source == "usage"


def test_prefill_token_count_labels_fallback_explicitly():
    tokens, source = _SCRIPT._prefill_token_count(
        usage=None,
        actual_prompt_tokens=4096,
        allow_usage_fallback=True,
    )

    assert tokens == 4096
    assert source == "actual_prompt_tokens"


def test_prefill_token_count_missing_usage_fails_without_fallback():
    tokens, source = _SCRIPT._prefill_token_count(
        usage=None,
        actual_prompt_tokens=4096,
        allow_usage_fallback=False,
    )

    assert tokens is None
    assert source == "missing_usage"


def test_row_gate_requires_status_ttft_and_prefill_tokens():
    good = {"status": 200, "ttft_seconds": 1.0, "prefill_tokens": 16, "text": "x"}
    assert _SCRIPT._row_passed(good, require_text=True)

    missing_usage = dict(good, prefill_tokens=None)
    assert not _SCRIPT._row_passed(missing_usage, require_text=False)

    empty_text = dict(good, text="")
    assert not _SCRIPT._row_passed(empty_text, require_text=True)
    assert _SCRIPT._row_passed(empty_text, require_text=False)
