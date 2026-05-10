#!/usr/bin/env python3
"""Phase-1 API probe for Qwen3.6 direct-RHS validation.

This intentionally checks the OpenAI logprobs path before running longer
quality suites. The direct-RHS validation plan requires top-logprob vectors for
baseline-vs-candidate cosine; if the server cannot return logprobs, the gate is
not covered and the artifact cannot be marked shippable from API evidence.
"""

from __future__ import annotations

import argparse
import json
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path


def post_json(base_url: str, path: str, payload: dict, timeout: int = 300) -> dict:
    req = urllib.request.Request(
        base_url.rstrip("/") + path,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    started = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            raw = response.read().decode("utf-8")
            status = response.status
    except urllib.error.HTTPError as exc:
        raw = exc.read().decode("utf-8", errors="replace")
        status = exc.code
    elapsed = time.perf_counter() - started
    try:
        body = json.loads(raw)
    except json.JSONDecodeError:
        body = {"raw": raw}
    return {"status": status, "elapsed_s": elapsed, "body": body}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8001")
    parser.add_argument("--model", default="/opt/dlami/nvme/models/Qwen3.6-27B")
    parser.add_argument("--out", default="")
    args = parser.parse_args()

    prompt = "What is 17 * 23? Answer with only the number."
    probes = [
        {
            "name": "chat_logprobs",
            "path": "/v1/chat/completions",
            "payload": {
                "model": args.model,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 2,
                "temperature": 1,
                "top_p": 1,
                "logprobs": True,
                "top_logprobs": 20,
            },
        },
        {
            "name": "completion_logprobs",
            "path": "/v1/completions",
            "payload": {
                "model": args.model,
                "prompt": f"Question: {prompt}\nAnswer:",
                "max_tokens": 1,
                "temperature": 1,
                "top_p": 1,
                "logprobs": 20,
            },
        },
        {
            "name": "completion_prompt_logprobs",
            "path": "/v1/completions",
            "payload": {
                "model": args.model,
                "prompt": f"Question: {prompt}\nAnswer:",
                "max_tokens": 1,
                "temperature": 1,
                "top_p": 1,
                "prompt_logprobs": 20,
            },
        },
    ]

    result = {
        "metadata": {
            "started_utc": datetime.now(timezone.utc).isoformat(),
            "base_url": args.base_url,
            "model": args.model,
        },
        "probes": [],
    }
    for probe in probes:
        record = dict(probe)
        record.pop("payload")
        record["response"] = post_json(args.base_url, probe["path"], probe["payload"])
        result["probes"].append(record)
        print("PROBE", json.dumps(record, sort_keys=True), flush=True)

    result["finished_utc"] = datetime.now(timezone.utc).isoformat()
    if args.out:
        path = Path(args.out)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(f"WROTE {path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
