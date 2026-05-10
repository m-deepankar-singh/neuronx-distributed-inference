#!/usr/bin/env python3
"""Long-context chat eval for Qwen3.6 vLLM proxy/API."""

from __future__ import annotations

import argparse
import json
import os
import time
import urllib.request
from datetime import datetime, timezone
from typing import Any

from transformers import AutoTokenizer


def post_json(url: str, payload: dict[str, Any], timeout: int = 1800):
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    start = time.perf_counter()
    with urllib.request.urlopen(req, timeout=timeout) as response:
        data = json.loads(response.read())
    return data, time.perf_counter() - start


def contains_all(text: str, expected: list[str]) -> bool:
    lowered = text.lower()
    return all(item.lower() in lowered for item in expected)


def make_needle_prompt(tokenizer, target_tokens: int) -> str:
    intro = "BEGIN FACTS. START_CODE=ORCHID-17. "
    middle = " MIDDLE_CODE=TITAN-42. "
    end = " END_CODE=VIOLET-91. Reply with all three codes only."
    filler = (
        "The document discusses model serving, data centers, logistics, sports "
        "history, tactical games, and routine operational notes. "
    )
    intro_ids = tokenizer(intro, add_special_tokens=False).input_ids
    middle_ids = tokenizer(middle, add_special_tokens=False).input_ids
    end_ids = tokenizer(end, add_special_tokens=False).input_ids
    filler_ids = tokenizer(filler, add_special_tokens=False).input_ids
    remaining = max(16, target_tokens - len(intro_ids) - len(middle_ids) - len(end_ids))
    left = remaining // 2
    right = remaining - left
    ids = (
        intro_ids
        + (filler_ids * (left // len(filler_ids) + 2))[:left]
        + middle_ids
        + (filler_ids * (right // len(filler_ids) + 2))[:right]
        + end_ids
    )
    return tokenizer.decode(ids, skip_special_tokens=False)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--model", default="/opt/dlami/nvme/models/Qwen3.6-27B")
    parser.add_argument("--model-path", default="/opt/dlami/nvme/models/Qwen3.6-27B")
    parser.add_argument("--out-dir", default="/opt/dlami/nvme")
    parser.add_argument("--skip-64k", action="store_true")
    parser.add_argument("--skip-120k", action="store_true")
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    cases: list[dict[str, Any]] = [
        {
            "name": "math_no_kwargs_proxy_injection",
            "messages": [
                {
                    "role": "system",
                    "content": "You are a precise assistant. Answer only with the final answer.",
                },
                {"role": "user", "content": "What is 137 * 23?"},
            ],
            "max_tokens": 48,
            "expected": ["3151"],
        },
        {
            "name": "olympics_no_kwargs_proxy_injection",
            "messages": [
                {"role": "system", "content": "You answer factual questions concisely."},
                {
                    "role": "user",
                    "content": (
                        "Where were the 2020 Summer Olympics held, and why were "
                        "they unusual? Answer in one sentence."
                    ),
                }
            ],
            "max_tokens": 96,
            "expected": ["Tokyo"],
        },
        {
            "name": "needle_32k_chat",
            "messages": [
                {
                    "role": "system",
                    "content": "You are a retrieval assistant. Return only requested codes.",
                },
                {
                    "role": "user",
                    "content": make_needle_prompt(tokenizer, 32000),
                }
            ],
            "max_tokens": 96,
            "expected": ["ORCHID-17", "TITAN-42", "VIOLET-91"],
        },
    ]
    if not args.skip_64k:
        cases.append(
            {
                "name": "needle_64k_chat",
                "messages": [
                    {
                        "role": "system",
                        "content": "You are a retrieval assistant. Return only requested codes.",
                    },
                    {"role": "user", "content": make_needle_prompt(tokenizer, 64000)}
                ],
                "max_tokens": 96,
                "expected": ["ORCHID-17", "TITAN-42", "VIOLET-91"],
            }
        )
    if not args.skip_120k:
        cases.append(
            {
                "name": "needle_120k_chat",
                "messages": [
                    {
                        "role": "system",
                        "content": "You are a retrieval assistant. Return only requested codes.",
                    },
                    {"role": "user", "content": make_needle_prompt(tokenizer, 120000)}
                ],
                "max_tokens": 96,
                "expected": ["ORCHID-17", "TITAN-42", "VIOLET-91"],
            }
        )

    results = {
        "metadata": {
            "started_utc": datetime.now(timezone.utc).isoformat(),
            "base_url": args.base_url,
            "model": args.model,
            "model_path": args.model_path,
        },
        "cases": [],
    }
    for case in cases:
        payload = {
            "model": args.model,
            "messages": case["messages"],
            "max_tokens": case["max_tokens"],
            "temperature": 0,
            "top_k": 1,
            "top_p": 1,
        }
        data, wall = post_json(f"{args.base_url}/v1/chat/completions", payload)
        text = data.get("choices", [{}])[0].get("message", {}).get("content", "")
        usage = data.get("usage") or {}
        record = {
            "name": case["name"],
            "wall_s": wall,
            "usage": usage,
            "expected_contains": case["expected"],
            "contains_expected": contains_all(text, case["expected"]),
            "text_prefix": text[:1200],
        }
        results["cases"].append(record)
        print("LONG_CHAT_CASE", json.dumps(record, sort_keys=True), flush=True)

    results["finished_utc"] = datetime.now(timezone.utc).isoformat()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_path = os.path.join(args.out_dir, f"qwen36_27b_vllm_long_chat_eval_{stamp}.json")
    with open(out_path, "w") as handle:
        json.dump(results, handle, indent=2)
    print("WROTE", out_path, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
