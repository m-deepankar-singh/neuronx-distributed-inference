#!/usr/bin/env python3
"""Concurrency smoke/throughput eval for the Qwen3.6 vLLM proxy."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import statistics
import time
import urllib.request
from datetime import datetime, timezone
from typing import Any


def post_json(url: str, payload: dict[str, Any], timeout: int):
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    started = time.perf_counter()
    with urllib.request.urlopen(req, timeout=timeout) as response:
        data = json.loads(response.read().decode("utf-8"))
    return data, time.perf_counter() - started


def make_messages(
    request_id: int,
    prompt_tokens_hint: int,
    response_mode: str,
) -> list[dict[str, str]]:
    filler = (
        "This request validates concurrent Qwen serving on Trainium. "
        "Keep the answer concise and include the request id. "
    )
    repeats = max(1, prompt_tokens_hint // 18)
    if response_mode == "long":
        return [
            {
                "role": "system",
                "content": (
                    "You are a deterministic load-test assistant. Write the requested "
                    "numbered list and include the request id in every item."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Request id {request_id}. "
                    + filler * repeats
                    + (
                        "Write 80 short numbered items. Every item must contain "
                        f"REQUEST_ID={request_id}."
                    )
                ),
            },
        ]
    return [
        {
            "role": "system",
            "content": "You are a concise assistant. Answer in one sentence.",
        },
        {
            "role": "user",
            "content": (
                f"Request id {request_id}. "
                + filler * repeats
                + f"End by writing REQUEST_ID={request_id}."
            ),
        },
    ]


def run_one(
    base_url: str,
    model: str,
    request_id: int,
    prompt_tokens_hint: int,
    max_tokens: int,
    response_mode: str,
    timeout: int,
):
    payload = {
        "model": model,
        "messages": make_messages(request_id, prompt_tokens_hint, response_mode),
        "max_tokens": max_tokens,
        "temperature": 0,
        "top_p": 1,
        "top_k": 1,
    }
    try:
        data, wall_s = post_json(f"{base_url}/v1/chat/completions", payload, timeout)
        text = data.get("choices", [{}])[0].get("message", {}).get("content", "")
        usage = data.get("usage") or {}
        return {
            "request_id": request_id,
            "ok": f"REQUEST_ID={request_id}" in text,
            "wall_s": wall_s,
            "usage": usage,
            "text_prefix": text[:240],
            "error": None,
        }
    except Exception as exc:  # noqa: BLE001
        return {
            "request_id": request_id,
            "ok": False,
            "wall_s": None,
            "usage": {},
            "text_prefix": "",
            "error": repr(exc),
        }


def percentile(values: list[float], pct: float) -> float | None:
    if not values:
        return None
    if len(values) == 1:
        return values[0]
    ordered = sorted(values)
    idx = min(len(ordered) - 1, max(0, round((pct / 100) * (len(ordered) - 1))))
    return ordered[idx]


def summarize(level: int, results: list[dict[str, Any]], total_wall_s: float):
    walls = [r["wall_s"] for r in results if isinstance(r.get("wall_s"), float)]
    prompt_tokens = sum((r.get("usage") or {}).get("prompt_tokens") or 0 for r in results)
    completion_tokens = sum(
        (r.get("usage") or {}).get("completion_tokens") or 0 for r in results
    )
    return {
        "concurrency": level,
        "total_wall_s": total_wall_s,
        "successes": sum(1 for r in results if r.get("ok")),
        "requests": len(results),
        "p50_wall_s": statistics.median(walls) if walls else None,
        "p95_wall_s": percentile(walls, 95),
        "max_wall_s": max(walls) if walls else None,
        "aggregate_prompt_tokens": prompt_tokens,
        "aggregate_completion_tokens": completion_tokens,
        "aggregate_prompt_tok_s": prompt_tokens / total_wall_s if total_wall_s else None,
        "aggregate_completion_tok_s": completion_tokens / total_wall_s if total_wall_s else None,
        "per_request": results,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--model", default="/opt/dlami/nvme/models/Qwen3.6-27B")
    parser.add_argument("--out-dir", default="/opt/dlami/nvme")
    parser.add_argument("--levels", default="1,2,4")
    parser.add_argument("--prompt-tokens-hint", type=int, default=2048)
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--response-mode", choices=["short", "long"], default="short")
    parser.add_argument("--timeout", type=int, default=1800)
    args = parser.parse_args()

    levels = [int(x.strip()) for x in args.levels.split(",") if x.strip()]
    report: dict[str, Any] = {
        "metadata": {
            "started_utc": datetime.now(timezone.utc).isoformat(),
            "base_url": args.base_url,
            "model": args.model,
            "levels": levels,
            "prompt_tokens_hint": args.prompt_tokens_hint,
            "max_tokens": args.max_tokens,
            "response_mode": args.response_mode,
        },
        "runs": [],
    }

    request_offset = 1000
    for level in levels:
        started = time.perf_counter()
        with concurrent.futures.ThreadPoolExecutor(max_workers=level) as executor:
            futures = [
                executor.submit(
                    run_one,
                    args.base_url,
                    args.model,
                    request_offset + i,
                    args.prompt_tokens_hint,
                    args.max_tokens,
                    args.response_mode,
                    args.timeout,
                )
                for i in range(level)
            ]
            results = [future.result() for future in futures]
        total_wall_s = time.perf_counter() - started
        run = summarize(level, results, total_wall_s)
        report["runs"].append(run)
        print("CONCURRENCY_RUN", json.dumps(run, sort_keys=True), flush=True)
        request_offset += 1000

    report["finished_utc"] = datetime.now(timezone.utc).isoformat()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_path = os.path.join(args.out_dir, f"qwen36_27b_vllm_concurrency_eval_{stamp}.json")
    with open(out_path, "w") as handle:
        json.dump(report, handle, indent=2)
    print("WROTE", out_path, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
