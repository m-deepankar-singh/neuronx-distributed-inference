#!/usr/bin/env python3
"""Benchmark OpenAI-compatible chat completions across context lengths.

The benchmark builds deterministic multi-turn chat histories, sends
``/v1/chat/completions`` requests with ``max_tokens=1``, and records wall/TTFT
latency. Streaming is used by default when the server supports it.
"""

from __future__ import annotations

import argparse
import json
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


def _parse_lengths(raw: str) -> list[int]:
    lengths = [int(item) for item in raw.replace(",", " ").split()]
    if not lengths:
        raise ValueError("at least one context length is required")
    return lengths


def _chat_token_count(tokenizer: Any, messages: list[dict[str, str]]) -> int:
    try:
        token_ids = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            enable_thinking=False,
        )
    except TypeError:
        token_ids = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
        )
    return len(token_ids)


def _base_messages(turns: int, *, salt: str = "") -> list[dict[str, str]]:
    messages = [
        {
            "role": "system",
            "content": (
                "You are a deterministic latency benchmark assistant. "
                "Reply with one concise token. "
                f"Benchmark salt: {salt}."
            ),
        }
    ]
    for idx in range(max(1, turns)):
        messages.append(
            {
                "role": "user",
                "content": f"Turn {idx}: remember benchmark key {idx}.",
            }
        )
        messages.append(
            {
                "role": "assistant",
                "content": f"ack {idx}",
            }
        )
    messages.append({"role": "user", "content": "Return the next benchmark token."})
    return messages


def _make_messages(
    tokenizer: Any,
    target_tokens: int,
    turns: int,
    *,
    salt: str = "",
) -> tuple[list[dict[str, str]], int]:
    messages = _base_messages(turns, salt=salt)
    filler_phrase = (
        " latency-prefix alpha beta gamma delta epsilon zeta eta theta iota kappa"
    )

    def set_repeats(repeats: int) -> None:
        messages[-1]["content"] = (
            "Return the next benchmark token."
            + (filler_phrase * max(0, repeats))
        )

    low = 0
    high = 1
    set_repeats(high)
    while _chat_token_count(tokenizer, messages) <= target_tokens:
        low = high
        high *= 2
        set_repeats(high)

    while low + 1 < high:
        mid = (low + high) // 2
        set_repeats(mid)
        if _chat_token_count(tokenizer, messages) <= target_tokens:
            low = mid
        else:
            high = mid

    set_repeats(low)
    return messages, _chat_token_count(tokenizer, messages)


def _post_json(url: str, payload: dict[str, Any], timeout: float) -> tuple[int, dict[str, Any]]:
    body = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.status, json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        try:
            payload = json.loads(exc.read().decode("utf-8"))
        except Exception:
            payload = {"error": {"message": str(exc)}}
        return exc.code, payload


def _stream_chat(
    url: str,
    payload: dict[str, Any],
    timeout: float,
) -> tuple[int, float | None, float, list[str], dict[str, Any] | None]:
    payload = dict(payload)
    payload["stream"] = True
    body = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    start = time.perf_counter()
    chunks: list[str] = []
    first_chunk_seconds = None
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
                if first_chunk_seconds is None:
                    first_chunk_seconds = time.perf_counter() - start
                chunks.append(data)
            total_seconds = time.perf_counter() - start
            return status, first_chunk_seconds, total_seconds, chunks, None
    except urllib.error.HTTPError as exc:
        total_seconds = time.perf_counter() - start
        try:
            error_payload = json.loads(exc.read().decode("utf-8"))
        except Exception:
            error_payload = {"error": {"message": str(exc)}}
        return exc.code, first_chunk_seconds, total_seconds, chunks, error_payload


def _run_one(
    *,
    url: str,
    model: str,
    messages: list[dict[str, str]],
    max_tokens: int,
    timeout: float,
    stream: bool,
) -> dict[str, Any]:
    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": 0,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    if stream:
        status, first_chunk_seconds, total_seconds, chunks, error_payload = _stream_chat(
            url,
            payload,
            timeout,
        )
        if status < 400:
            return {
                "status": status,
                "stream": True,
                "ttft_seconds": first_chunk_seconds,
                "total_seconds": total_seconds,
                "chunk_count": len(chunks),
                "error": None,
            }
        return {
            "status": status,
            "stream": True,
            "ttft_seconds": first_chunk_seconds,
            "total_seconds": total_seconds,
            "chunk_count": len(chunks),
            "error": error_payload,
        }

    start = time.perf_counter()
    status, response = _post_json(url, payload, timeout)
    total_seconds = time.perf_counter() - start
    return {
        "status": status,
        "stream": False,
        "ttft_seconds": None,
        "total_seconds": total_seconds,
        "chunk_count": None,
        "error": None if status < 400 else response,
        "usage": response.get("usage") if isinstance(response, dict) else None,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--model", default="Qwen3.6-27B")
    parser.add_argument("--model-path", required=True)
    parser.add_argument(
        "--lengths",
        default="1024,2048,4096,8192,16384,32768",
        help="Comma or space separated target chat-template token lengths.",
    )
    parser.add_argument("--turns", type=int, default=8)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--max-tokens", type=int, default=1)
    parser.add_argument("--timeout", type=float, default=900.0)
    parser.add_argument("--no-stream", action="store_true")
    parser.add_argument(
        "--unique-per-request",
        action="store_true",
        help="Add a unique system-message salt for each length/repeat.",
    )
    parser.add_argument("--output-json", required=True)
    args = parser.parse_args()

    from transformers import AutoTokenizer  # noqa: WPS433

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    endpoint = args.base_url.rstrip("/") + "/v1/chat/completions"
    results = []
    for target_tokens in _parse_lengths(args.lengths):
        for repeat_idx in range(args.repeats):
            salt = (
                f"target={target_tokens};repeat={repeat_idx};unique=1"
                if args.unique_per_request
                else ""
            )
            messages, prompt_tokens = _make_messages(
                tokenizer,
                target_tokens=target_tokens,
                turns=args.turns,
                salt=salt,
            )
            result = _run_one(
                url=endpoint,
                model=args.model,
                messages=messages,
                max_tokens=args.max_tokens,
                timeout=args.timeout,
                stream=not args.no_stream,
            )
            row = {
                "target_tokens": target_tokens,
                "prompt_tokens": prompt_tokens,
                "repeat": repeat_idx,
                **result,
            }
            print(json.dumps(row, sort_keys=True), flush=True)
            results.append(row)

    output = {
        "base_url": args.base_url,
        "model": args.model,
        "lengths": _parse_lengths(args.lengths),
        "turns": args.turns,
        "repeats": args.repeats,
        "max_tokens": args.max_tokens,
        "results": results,
    }
    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as f:
        json.dump(output, f, indent=2, sort_keys=True)
    return 0 if all(int(row["status"]) < 400 for row in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
