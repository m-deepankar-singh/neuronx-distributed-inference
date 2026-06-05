#!/usr/bin/env python3
"""Usage-accounted raw completion cold-prefill benchmark for Qwen3.6."""

from __future__ import annotations

import argparse
import json
import statistics
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Iterable


def _parse_lengths(raw: str) -> list[int]:
    lengths = [int(item) for item in raw.replace(",", " ").split()]
    if not lengths:
        raise ValueError("at least one prompt length is required")
    return lengths


def _load_json_from_url(url: str, timeout: float) -> tuple[int, dict[str, Any]]:
    request = urllib.request.Request(url, method="GET")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.status, json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        try:
            payload = json.loads(exc.read().decode("utf-8"))
        except Exception:
            payload = {"error": {"message": str(exc)}}
        return exc.code, payload


def _detect_model(base_url: str, fallback: str, timeout: float) -> str:
    status, payload = _load_json_from_url(base_url.rstrip("/") + "/v1/models", timeout)
    if status < 400:
        data = payload.get("data") if isinstance(payload, dict) else None
        if isinstance(data, list) and data:
            model_id = data[0].get("id")
            if isinstance(model_id, str) and model_id:
                return model_id
    return fallback


def _exact_token_ids(tokenizer: Any, length: int, salt: str) -> list[int]:
    filler = tokenizer.encode(
        " cold raw completion prefill benchmark coherent validation",
        add_special_tokens=False,
    )
    if not filler:
        raise RuntimeError("tokenizer produced no filler tokens")
    token_ids = tokenizer.encode(f"Raw prefill benchmark {salt}. ", add_special_tokens=False)
    while len(token_ids) < length:
        token_ids.extend(filler)
    return token_ids[:length]


def _sse_json_payloads(lines: Iterable[bytes | str]) -> list[dict[str, Any]]:
    payloads: list[dict[str, Any]] = []
    for raw_line in lines:
        line = (
            raw_line.decode("utf-8", errors="replace")
            if isinstance(raw_line, bytes)
            else raw_line
        ).strip()
        if not line or not line.startswith("data:"):
            continue
        data = line[len("data:") :].strip()
        if data == "[DONE]":
            break
        payloads.append(json.loads(data))
    return payloads


def _usage_prompt_tokens(usage: Any) -> int | None:
    if not isinstance(usage, dict):
        return None
    value = usage.get("prompt_tokens")
    try:
        return None if value is None else int(value)
    except (TypeError, ValueError):
        return None


def _prefill_token_count(
    *,
    usage: Any,
    actual_prompt_tokens: int,
    allow_usage_fallback: bool,
) -> tuple[int | None, str]:
    usage_tokens = _usage_prompt_tokens(usage)
    if usage_tokens is not None:
        return usage_tokens, "usage"
    if allow_usage_fallback:
        return actual_prompt_tokens, "actual_prompt_tokens"
    return None, "missing_usage"


def _stream_completion(
    *,
    endpoint: str,
    model: str,
    prompt_token_ids: list[int],
    max_tokens: int,
    timeout: float,
) -> dict[str, Any]:
    payload = {
        "model": model,
        "prompt": prompt_token_ids,
        "max_tokens": max_tokens,
        "temperature": 0,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    request = urllib.request.Request(
        endpoint,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    start = time.perf_counter()
    first_event_seconds = None
    usage = None
    chunks = 0
    text_parts: list[str] = []
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            status = response.status
            for raw_line in response:
                line = raw_line.decode("utf-8", errors="replace").strip()
                if not line or not line.startswith("data:"):
                    continue
                data = line[len("data:") :].strip()
                if data == "[DONE]":
                    break
                obj = json.loads(data)
                chunks += 1
                if first_event_seconds is None and obj.get("choices"):
                    first_event_seconds = time.perf_counter() - start
                if obj.get("usage") is not None:
                    usage = obj["usage"]
                for choice in obj.get("choices", []):
                    text = choice.get("text") or ""
                    if text:
                        text_parts.append(str(text))
            return {
                "status": status,
                "ttft_seconds": first_event_seconds,
                "total_seconds": time.perf_counter() - start,
                "chunk_count": chunks,
                "text": "".join(text_parts),
                "usage": usage,
                "error": None,
            }
    except urllib.error.HTTPError as exc:
        try:
            error = json.loads(exc.read().decode("utf-8"))
        except Exception:
            error = {"error": {"message": str(exc)}}
        return {
            "status": exc.code,
            "ttft_seconds": first_event_seconds,
            "total_seconds": time.perf_counter() - start,
            "chunk_count": chunks,
            "text": "".join(text_parts),
            "usage": usage,
            "error": error,
        }


def _row_passed(row: dict[str, Any], *, require_text: bool) -> bool:
    if int(row["status"]) >= 400:
        return False
    if row.get("ttft_seconds") is None:
        return False
    if row.get("prefill_tokens") is None:
        return False
    if require_text and not row.get("text"):
        return False
    return True


def _mean(values: list[float]) -> float | None:
    return statistics.fmean(values) if values else None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--model", default="auto")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--lengths", default="16384")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--max-tokens", type=int, default=1)
    parser.add_argument("--timeout", type=float, default=1200.0)
    parser.add_argument("--allow-usage-fallback", action="store_true")
    parser.add_argument("--require-text", action="store_true")
    parser.add_argument("--output-json", required=True)
    args = parser.parse_args()

    from transformers import AutoTokenizer  # noqa: WPS433

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    base_url = args.base_url.rstrip("/")
    model = (
        _detect_model(base_url, "Qwen3.6-27B", args.timeout)
        if args.model == "auto"
        else args.model
    )
    endpoint = base_url + "/v1/completions"
    stamp = time.time_ns()
    rows: list[dict[str, Any]] = []
    for length in _parse_lengths(args.lengths):
        for repeat in range(args.repeats):
            prompt_ids = _exact_token_ids(
                tokenizer,
                length,
                f"length-{length}-repeat-{repeat}-{stamp}",
            )
            result = _stream_completion(
                endpoint=endpoint,
                model=model,
                prompt_token_ids=prompt_ids,
                max_tokens=args.max_tokens,
                timeout=args.timeout,
            )
            prefill_tokens, token_source = _prefill_token_count(
                usage=result.get("usage"),
                actual_prompt_tokens=len(prompt_ids),
                allow_usage_fallback=args.allow_usage_fallback,
            )
            ttft = result.get("ttft_seconds")
            row = {
                "target_prompt_tokens": length,
                "actual_prompt_tokens": len(prompt_ids),
                "repeat": repeat,
                **result,
                "prefill_tokens": prefill_tokens,
                "prefill_token_source": token_source,
                "prefill_tok_s": (
                    prefill_tokens / ttft
                    if prefill_tokens is not None and ttft and ttft > 0
                    else None
                ),
            }
            rows.append(row)
            print(json.dumps(row, sort_keys=True), flush=True)

    speeds = [
        float(row["prefill_tok_s"])
        for row in rows
        if row.get("prefill_tok_s") is not None
    ]
    output = {
        "base_url": base_url,
        "model": model,
        "lengths": _parse_lengths(args.lengths),
        "repeats": args.repeats,
        "max_tokens": args.max_tokens,
        "allow_usage_fallback": args.allow_usage_fallback,
        "require_text": args.require_text,
        "passed": all(_row_passed(row, require_text=args.require_text) for row in rows),
        "prefill_tok_s_mean": _mean(speeds),
        "prefill_tok_s_min": min(speeds) if speeds else None,
        "prefill_tok_s_max": max(speeds) if speeds else None,
        "results": rows,
    }
    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    return 0 if output["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
