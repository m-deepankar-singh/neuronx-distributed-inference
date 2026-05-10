#!/usr/bin/env python3
"""Server-side APC hardening eval for Qwen3.6 vLLM proxy."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import statistics
import time
import urllib.request
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Any


@dataclass
class ChatResult:
    name: str
    wall_s: float
    text: str
    prompt_tokens: int | None
    completion_tokens: int | None
    expected: str | None
    ok: bool


def post_chat(base_url: str, payload: dict[str, Any], timeout: int) -> tuple[dict[str, Any], float]:
    req = urllib.request.Request(
        base_url.rstrip("/") + "/v1/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    start = time.perf_counter()
    with urllib.request.urlopen(req, timeout=timeout) as response:
        data = json.loads(response.read().decode("utf-8"))
    return data, time.perf_counter() - start


def shared_prefix(label: str, repeats: int) -> str:
    block = (
        f"Prefix label {label}. Tokyo hosted the postponed 2020 Summer Olympics in 2021. "
        "The Games were held under COVID-19 restrictions with limited spectators. "
        "This repeated block is intentionally stable for APC validation. "
    )
    return block * repeats


def payload(model: str, prompt: str, max_tokens: int) -> dict[str, Any]:
    return {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": "You are deterministic. Follow the requested answer format exactly.",
            },
            {"role": "user", "content": prompt},
        ],
        "max_tokens": max_tokens,
        "temperature": 0,
        "top_p": 1,
        "top_k": 1,
    }


def parse_result(name: str, data: dict[str, Any], wall_s: float, expected: str | None) -> ChatResult:
    choice = (data.get("choices") or [{}])[0]
    text = ((choice.get("message") or {}).get("content") or "").strip()
    usage = data.get("usage") or {}
    ok = True if expected is None else expected in text
    result = ChatResult(
        name=name,
        wall_s=wall_s,
        text=text,
        prompt_tokens=usage.get("prompt_tokens"),
        completion_tokens=usage.get("completion_tokens"),
        expected=expected,
        ok=ok,
    )
    print("APC_HARDENING_CASE " + json.dumps(asdict(result), sort_keys=True), flush=True)
    return result


def run_named(
    base_url: str,
    model: str,
    name: str,
    prompt: str,
    expected: str,
    max_tokens: int,
    timeout: int,
) -> ChatResult:
    data, wall_s = post_chat(base_url, payload(model, prompt, max_tokens), timeout)
    return parse_result(name, data, wall_s, expected)


def percentile(values: list[float], pct: float) -> float | None:
    if not values:
        return None
    if len(values) == 1:
        return values[0]
    ordered = sorted(values)
    idx = min(len(ordered) - 1, max(0, round((pct / 100) * (len(ordered) - 1))))
    return ordered[idx]


def run_concurrent_shared_prefix(args: argparse.Namespace) -> list[dict[str, Any]]:
    prefix = shared_prefix("CONCURRENT", args.prefix_repeats)
    warm_expected = "APC_WARMUP_OK"
    run_named(
        args.base_url,
        args.model,
        "concurrency_prefix_warmup",
        prefix + f"\nReply with exactly {warm_expected}.",
        warm_expected,
        args.max_tokens,
        args.timeout,
    )

    summaries: list[dict[str, Any]] = []
    for level in args.concurrency_levels:
        started = time.perf_counter()
        with concurrent.futures.ThreadPoolExecutor(max_workers=level) as executor:
            futures = []
            for offset in range(level):
                expected = f"APC_REQ_{level}_{offset}"
                futures.append(
                    executor.submit(
                        run_named,
                        args.base_url,
                        args.model,
                        f"concurrent_l{level}_r{offset}",
                        prefix + f"\nReply with exactly {expected}.",
                        expected,
                        args.max_tokens,
                        args.timeout,
                    )
                )
            results = [future.result() for future in futures]
        total_wall_s = time.perf_counter() - started
        walls = [result.wall_s for result in results]
        prompt_tokens = sum(result.prompt_tokens or 0 for result in results)
        completion_tokens = sum(result.completion_tokens or 0 for result in results)
        summary = {
            "concurrency": level,
            "requests": len(results),
            "successes": sum(1 for result in results if result.ok),
            "total_wall_s": total_wall_s,
            "p50_wall_s": statistics.median(walls) if walls else None,
            "p95_wall_s": percentile(walls, 95),
            "aggregate_prompt_tokens": prompt_tokens,
            "aggregate_completion_tokens": completion_tokens,
            "aggregate_prompt_tok_s": prompt_tokens / total_wall_s if total_wall_s else None,
            "aggregate_completion_tok_s": completion_tokens / total_wall_s if total_wall_s else None,
            "cases": [asdict(result) for result in results],
        }
        print("APC_CONCURRENCY_SUMMARY " + json.dumps(summary, sort_keys=True), flush=True)
        summaries.append(summary)
    return summaries


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--model", default="/opt/dlami/nvme/models/Qwen3.6-27B")
    parser.add_argument("--out-dir", default="/opt/dlami/nvme")
    parser.add_argument("--prefix-repeats", type=int, default=220)
    parser.add_argument("--max-tokens", type=int, default=32)
    parser.add_argument("--timeout", type=int, default=1800)
    parser.add_argument("--min-repeat-speedup", type=float, default=5.0)
    parser.add_argument("--concurrency-levels", default="1,2,4")
    args = parser.parse_args()
    args.concurrency_levels = [
        int(item.strip()) for item in args.concurrency_levels.split(",") if item.strip()
    ]

    cases: list[ChatResult] = []
    repeat_prompt = (
        shared_prefix("EXACT_REPEAT", args.prefix_repeats)
        + "\nReply with exactly TOKYO_2020_OK."
    )
    cases.append(
        run_named(
            args.base_url,
            args.model,
            "exact_repeat_cold",
            repeat_prompt,
            "TOKYO_2020_OK",
            args.max_tokens,
            args.timeout,
        )
    )
    cases.append(
        run_named(
            args.base_url,
            args.model,
            "exact_repeat_warm",
            repeat_prompt,
            "TOKYO_2020_OK",
            args.max_tokens,
            args.timeout,
        )
    )

    cross_a = (
        shared_prefix("CROSS_A", args.prefix_repeats)
        + "\nReply with exactly CROSS_A_OK."
    )
    cross_b = (
        shared_prefix("CROSS_B", args.prefix_repeats)
        + "\nReply with exactly CROSS_B_OK."
    )
    cases.append(
        run_named(
            args.base_url,
            args.model,
            "cross_prefix_a_cold",
            cross_a,
            "CROSS_A_OK",
            args.max_tokens,
            args.timeout,
        )
    )
    cases.append(
        run_named(
            args.base_url,
            args.model,
            "cross_prefix_b_cold",
            cross_b,
            "CROSS_B_OK",
            args.max_tokens,
            args.timeout,
        )
    )
    cases.append(
        run_named(
            args.base_url,
            args.model,
            "cross_prefix_a_warm_after_b",
            cross_a,
            "CROSS_A_OK",
            args.max_tokens,
            args.timeout,
        )
    )

    concurrency = run_concurrent_shared_prefix(args)

    exact_speedup = cases[0].wall_s / cases[1].wall_s if cases[1].wall_s else 0.0
    exact_text_match = cases[0].text == cases[1].text
    cross_a_match = cases[2].text == cases[4].text
    all_cases_ok = all(case.ok for case in cases)
    all_concurrency_ok = all(run["successes"] == run["requests"] for run in concurrency)
    summary = {
        "metadata": {
            "started_utc": datetime.now(timezone.utc).isoformat(),
            "base_url": args.base_url,
            "model": args.model,
            "prefix_repeats": args.prefix_repeats,
            "concurrency_levels": args.concurrency_levels,
        },
        "cases": [asdict(case) for case in cases],
        "concurrency": concurrency,
        "exact_repeat_speedup": exact_speedup,
        "exact_repeat_text_match": exact_text_match,
        "cross_prefix_a_text_match": cross_a_match,
        "all_cases_ok": all_cases_ok,
        "all_concurrency_ok": all_concurrency_ok,
    }
    print("APC_HARDENING_SUMMARY " + json.dumps(summary, sort_keys=True), flush=True)

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_path = os.path.join(args.out_dir, f"qwen36_vllm_apc_hardening_eval_{stamp}.json")
    with open(out_path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
    print("WROTE", out_path, flush=True)

    if not all_cases_ok:
        raise SystemExit("one or more APC hardening cases missed expected text")
    if not exact_text_match:
        raise SystemExit("exact-repeat APC changed output text")
    if not cross_a_match:
        raise SystemExit("cross-prefix reuse changed output text")
    if exact_speedup < args.min_repeat_speedup:
        raise SystemExit(
            f"exact-repeat speedup {exact_speedup:.2f}x below {args.min_repeat_speedup:.2f}x"
        )
    if not all_concurrency_ok:
        raise SystemExit("one or more shared-prefix concurrent requests failed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
