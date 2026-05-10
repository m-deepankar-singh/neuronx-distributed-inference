#!/usr/bin/env python3
"""Probe Qwen3.6 OpenAI-compatible quality with deterministic requests."""

from __future__ import annotations

import argparse
import json
import urllib.request


def post_json(base_url: str, path: str, payload: dict) -> dict:
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        base_url.rstrip("/") + path,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=600) as resp:
        return json.loads(resp.read().decode("utf-8"))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--model", default="qwen3.6-27b-neuron-128k-fp8-mlp")
    parser.add_argument("--max-tokens", type=int, default=48)
    args = parser.parse_args()

    completion = post_json(
        args.base_url,
        "/v1/completions",
        {
            "model": args.model,
            "prompt": "Question: What is 137 * 23? Answer with only the integer.\nAnswer:",
            "max_tokens": args.max_tokens,
            "temperature": 0,
        },
    )
    print("=== completion ===")
    print(json.dumps(completion, indent=2, ensure_ascii=False))

    for enable_thinking in (False, True):
        chat = post_json(
            args.base_url,
            "/v1/chat/completions",
            {
                "model": args.model,
                "messages": [
                    {
                        "role": "user",
                        "content": "What is 137 * 23? Answer with only the integer.",
                    }
                ],
                "max_tokens": args.max_tokens,
                "temperature": 0,
                "enable_thinking": enable_thinking,
            },
        )
        print(f"=== chat enable_thinking={enable_thinking} ===")
        print(json.dumps(chat, indent=2, ensure_ascii=False))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
