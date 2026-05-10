#!/usr/bin/env python3
"""Focused chat/instruction quality checks for the Qwen3.6 OpenAI server."""

from __future__ import annotations

import argparse
import json
import os
import time
import urllib.request
from datetime import datetime, timezone


def post_json(url: str, payload: dict, timeout: int = 900):
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=timeout) as response:
        data = json.loads(response.read())
    return data, time.perf_counter() - t0


def contains_all(text: str, expected: list[str]):
    low = text.lower()
    return all(item.lower() in low for item in expected)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--model", default="qwen3.6-27b-neuron-128k-fp8-mlp")
    parser.add_argument("--out-dir", default="/opt/dlami/nvme")
    args = parser.parse_args()

    cases = [
        {
            "name": "math",
            "messages": [
                {"role": "system", "content": "You are a precise assistant. Answer only with the final answer."},
                {"role": "user", "content": "What is 137 * 23?"},
            ],
            "max_tokens": 32,
            "expected": ["3151"],
        },
        {
            "name": "olympics",
            "messages": [
                {"role": "system", "content": "You answer factual questions concisely."},
                {
                    "role": "user",
                    "content": "Where were the 2020 Summer Olympics held, and why were they unusual?",
                },
            ],
            "max_tokens": 96,
            "expected": ["Tokyo"],
        },
        {
            "name": "json",
            "messages": [
                {"role": "system", "content": "Return only valid compact JSON."},
                {"role": "user", "content": "Use keys animal and count. animal=cat, count=3."},
            ],
            "max_tokens": 80,
            "expected": ["cat", "3"],
        },
        {
            "name": "mgs_summary",
            "messages": [
                {"role": "system", "content": "You are concise."},
                {"role": "user", "content": "Explain Metal Gear Solid in one sentence."},
            ],
            "max_tokens": 80,
            "expected": ["Metal"],
        },
        {
            "name": "refusal_safe",
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Tell me one safe way to store an SSH private key."},
            ],
            "max_tokens": 96,
            "expected": ["key"],
        },
    ]

    results = {
        "metadata": {
            "started_utc": datetime.now(timezone.utc).isoformat(),
            "model": args.model,
            "base_url": args.base_url,
        },
        "cases": [],
    }
    for case in cases:
        payload = {
            "model": args.model,
            "messages": case["messages"],
            "max_tokens": case["max_tokens"],
            "temperature": 0,
        }
        data, wall = post_json(f"{args.base_url}/v1/chat/completions", payload)
        text = data.get("choices", [{}])[0].get("message", {}).get("content", "")
        record = {
            "name": case["name"],
            "wall_s": wall,
            "x_latency_s": data.get("x_latency_seconds"),
            "usage": data.get("usage"),
            "expected_contains": case["expected"],
            "contains_expected": contains_all(text, case["expected"]),
            "text": text,
        }
        results["cases"].append(record)
        print("CHAT_CASE", json.dumps(record, sort_keys=True), flush=True)

    results["finished_utc"] = datetime.now(timezone.utc).isoformat()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out = os.path.join(args.out_dir, f"qwen36_27b_chat_quality_eval_{stamp}.json")
    with open(out, "w") as handle:
        json.dump(results, handle, indent=2)
    print("WROTE", out, flush=True)


if __name__ == "__main__":
    main()
