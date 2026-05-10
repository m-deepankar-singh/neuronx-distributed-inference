#!/usr/bin/env python3
"""Validate native vLLM prefix caching for Qwen3.6 over OpenAI chat API.

This intentionally tests vLLM APC as vLLM exposes it: repeated long prompts
should reuse cached full blocks, preserve greedy output text exactly, and reduce
prefill latency. Token IDs are not exposed by the OpenAI server, so token-exact
validation belongs in the offline vLLM harness.
"""

from __future__ import annotations

import argparse
import json
import re
import time
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass
from typing import Any


@dataclass
class CaseResult:
    name: str
    wall_s: float
    prompt_tokens: int | None
    completion_tokens: int | None
    text: str
    finish_reason: str | None
    prefix_metrics: dict[str, float]


def _read_url(url: str, timeout: int) -> str:
    req = urllib.request.Request(url, method="GET")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read().decode("utf-8", errors="replace")


def fetch_prefix_metrics(base_url: str, timeout: int = 30) -> dict[str, float]:
    """Return numeric Prometheus metrics containing 'prefix_cache', if present."""
    try:
        text = _read_url(base_url.rstrip("/") + "/metrics", timeout)
    except (urllib.error.URLError, TimeoutError):
        return {}

    metrics: dict[str, float] = {}
    for line in text.splitlines():
        if not line or line.startswith("#") or "prefix_cache" not in line:
            continue
        match = re.match(r"^([A-Za-z_:][A-Za-z0-9_:{}=\",.-]*)\s+([-+0-9.eE]+)$", line)
        if match:
            try:
                metrics[match.group(1)] = float(match.group(2))
            except ValueError:
                pass
    return metrics


def post_chat(base_url: str, payload: dict[str, Any], timeout: int) -> tuple[dict[str, Any], float]:
    req = urllib.request.Request(
        base_url.rstrip("/") + "/v1/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    start = time.perf_counter()
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        body = json.loads(resp.read().decode("utf-8"))
    return body, time.perf_counter() - start


def make_prompt(repeats: int, suffix: str) -> str:
    prefix = (
        "Reference dossier: Tokyo hosted the postponed 2020 Summer Olympics in 2021. "
        "The Games were held under COVID-19 restrictions, with limited spectators. "
        "This reference sentence is intentionally repeated to create a long shared prefix. "
    )
    return (prefix * repeats) + suffix


def make_payload(model: str, prompt: str, max_tokens: int) -> dict[str, Any]:
    return {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0,
        "top_p": 1,
        "top_k": 1,
        "chat_template_kwargs": {"enable_thinking": False},
    }


def summarize(name: str, response: dict[str, Any], wall_s: float, metrics: dict[str, float]) -> CaseResult:
    choice = (response.get("choices") or [{}])[0]
    message = choice.get("message") or {}
    usage = response.get("usage") or {}
    result = CaseResult(
        name=name,
        wall_s=wall_s,
        prompt_tokens=usage.get("prompt_tokens"),
        completion_tokens=usage.get("completion_tokens"),
        text=message.get("content") or "",
        finish_reason=choice.get("finish_reason"),
        prefix_metrics=metrics,
    )
    print("PREFIX_CACHE_CASE " + json.dumps(asdict(result), sort_keys=True), flush=True)
    return result


def metric_delta(before: dict[str, float], after: dict[str, float]) -> dict[str, float]:
    keys = set(before) | set(after)
    return {key: after.get(key, 0.0) - before.get(key, 0.0) for key in sorted(keys)}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--model", default="qwen3.6-27b-neuron-128k-fp8-mlp")
    parser.add_argument("--prefix-repeats", type=int, default=220)
    parser.add_argument("--max-tokens", type=int, default=32)
    parser.add_argument("--timeout", type=int, default=1800)
    parser.add_argument("--min-speedup", type=float, default=1.25)
    parser.add_argument("--no-fail-on-speed", action="store_true")
    args = parser.parse_args()

    prompt = make_prompt(
        args.prefix_repeats,
        "\nAnswer in one concise sentence: where were the 2020 Olympics held?",
    )
    payload = make_payload(args.model, prompt, args.max_tokens)

    metrics_0 = fetch_prefix_metrics(args.base_url)
    cold_body, cold_wall = post_chat(args.base_url, payload, args.timeout)
    metrics_1 = fetch_prefix_metrics(args.base_url)
    warm_body, warm_wall = post_chat(args.base_url, payload, args.timeout)
    metrics_2 = fetch_prefix_metrics(args.base_url)

    cold = summarize("cold_fill", cold_body, cold_wall, metric_delta(metrics_0, metrics_1))
    warm = summarize("warm_hit", warm_body, warm_wall, metric_delta(metrics_1, metrics_2))

    text_match = cold.text == warm.text
    speedup = cold.wall_s / warm.wall_s if warm.wall_s > 0 else 0.0
    prefix_metric_delta = metric_delta(metrics_0, metrics_2)
    saw_prefix_metric = any(value > 0 for value in prefix_metric_delta.values())

    summary = {
        "text_match": text_match,
        "cold_wall_s": cold.wall_s,
        "warm_wall_s": warm.wall_s,
        "speedup": speedup,
        "min_speedup": args.min_speedup,
        "saw_prefix_metric": saw_prefix_metric,
        "prefix_metric_delta": prefix_metric_delta,
        "cold_prompt_tokens": cold.prompt_tokens,
        "warm_prompt_tokens": warm.prompt_tokens,
    }
    print("PREFIX_CACHE_SUMMARY " + json.dumps(summary, sort_keys=True), flush=True)

    if not text_match:
        raise SystemExit("prefix cache changed greedy output text")
    if not args.no_fail_on_speed and speedup < args.min_speedup:
        raise SystemExit(f"prefix cache speedup {speedup:.2f}x below {args.min_speedup:.2f}x")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
