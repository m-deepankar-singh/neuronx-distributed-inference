#!/usr/bin/env python3
"""OpenAI-compatible API eval harness for Qwen3.6-27B 128K FP8 CTE=512.

Runs throughput, long-context, chat, and lightweight instruction/reasoning
checks against the already-running local server on the Trainium instance.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import time
import urllib.request
from datetime import datetime, timezone
from typing import Any

from transformers import AutoTokenizer


def request_json(url: str, payload: dict[str, Any] | None = None, timeout: int = 1800):
    if payload is None:
        req = urllib.request.Request(url)
    else:
        req = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=timeout) as response:
        data = json.loads(response.read())
    return data, time.perf_counter() - t0


def completion(base_url: str, model: str, prompt: str, max_tokens: int, timeout: int = 1800):
    data, wall = request_json(
        f"{base_url}/v1/completions",
        {
            "model": model,
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": 0,
            "top_k": 1,
        },
        timeout=timeout,
    )
    return {
        "wall_s": wall,
        "x_latency_s": data.get("x_latency_seconds"),
        "usage": data.get("usage", {}),
        "text": data.get("choices", [{}])[0].get("text", ""),
        "finish_reason": data.get("choices", [{}])[0].get("finish_reason"),
    }


def chat(base_url: str, model: str, messages: list[dict[str, str]], max_tokens: int):
    data, wall = request_json(
        f"{base_url}/v1/chat/completions",
        {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": 0,
        },
        timeout=900,
    )
    return {
        "wall_s": wall,
        "x_latency_s": data.get("x_latency_seconds"),
        "usage": data.get("usage", {}),
        "text": data.get("choices", [{}])[0].get("message", {}).get("content", ""),
    }


def make_prompt(tokenizer, target_tokens: int, tail: str, filler: str | None = None):
    filler = filler or (
        "Metal Gear Solid is a stealth action game series about espionage, nuclear "
        "deterrence, soldiers, politics, identity, and tactical decision making. "
    )
    tail_ids = tokenizer(tail, add_special_tokens=False).input_ids
    filler_ids = tokenizer(filler, add_special_tokens=False).input_ids
    body_len = max(1, target_tokens - len(tail_ids))
    ids = (filler_ids * (body_len // len(filler_ids) + 2))[:body_len] + tail_ids
    return tokenizer.decode(ids, skip_special_tokens=False)


def make_needle_prompt(tokenizer, target_tokens: int):
    intro = "BEGIN IMPORTANT FACTS. START_CODE=ORCHID-17. "
    middle = " MIDDLE_CODE=TITAN-42. "
    end = " END_CODE=VIOLET-91. Reply with START_CODE, MIDDLE_CODE, and END_CODE only."
    filler = (
        "The archive discusses logistics, sports history, character motivation, "
        "mission planning, and ordinary background details. "
    )
    intro_ids = tokenizer(intro, add_special_tokens=False).input_ids
    middle_ids = tokenizer(middle, add_special_tokens=False).input_ids
    end_ids = tokenizer(end, add_special_tokens=False).input_ids
    filler_ids = tokenizer(filler, add_special_tokens=False).input_ids
    remaining = max(10, target_tokens - len(intro_ids) - len(middle_ids) - len(end_ids))
    first = remaining // 2
    second = remaining - first
    ids = (
        intro_ids
        + (filler_ids * (first // len(filler_ids) + 2))[:first]
        + middle_ids
        + (filler_ids * (second // len(filler_ids) + 2))[:second]
        + end_ids
    )
    return tokenizer.decode(ids, skip_special_tokens=False)


def parse_monitor(path: str):
    peak_device = 0
    peak_host = 0
    peak_core_util = 0.0
    samples = 0
    if not os.path.exists(path):
        return {"error": "monitor file missing"}
    with open(path, "r", errors="ignore") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            samples += 1
            for runtime in obj.get("neuron_runtime_data", []):
                used = (
                    runtime.get("report", {})
                    .get("memory_used", {})
                    .get("neuron_runtime_used_bytes", {})
                )
                peak_device = max(peak_device, int(used.get("neuron_device", 0) or 0))
                peak_host = max(peak_host, int(used.get("host", 0) or 0))
                counters = (
                    runtime.get("report", {})
                    .get("neuroncore_counters", {})
                    .get("neuroncores_in_use", {})
                )
                for core in counters.values():
                    peak_core_util = max(
                        peak_core_util,
                        float(core.get("neuroncore_utilization", 0) or 0),
                    )
    return {
        "samples": samples,
        "peak_neuron_device_bytes": peak_device,
        "peak_neuron_device_gb_decimal": peak_device / 1e9,
        "peak_neuron_device_gib": peak_device / (1024**3),
        "peak_host_bytes": peak_host,
        "peak_host_gb_decimal": peak_host / 1e9,
        "peak_neuroncore_utilization_percent": peak_core_util,
    }


def contains_all(text: str, expected: list[str]):
    low = text.lower()
    return all(item.lower() in low for item in expected)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--model", default="qwen3.6-27b-neuron-128k-fp8-mlp")
    parser.add_argument("--model-path", default="/opt/dlami/nvme/models/Qwen3.6-27B")
    parser.add_argument("--out-dir", default="/opt/dlami/nvme")
    parser.add_argument("--skip-120k", action="store_true")
    args = parser.parse_args()

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_path = os.path.join(args.out_dir, f"qwen36_27b_128k_fp8_cte512_eval_{stamp}.json")
    monitor_path = out_path.replace(".json", "_neuron_monitor.jsonl")

    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    health, _ = request_json(f"{args.base_url}/health")
    models, _ = request_json(f"{args.base_url}/v1/models")
    results: dict[str, Any] = {
        "metadata": {
            "started_utc": datetime.now(timezone.utc).isoformat(),
            "model": args.model,
            "server_health": health,
            "models": models,
            "artifact": "/opt/dlami/nvme/qwen_artifacts/qwen36_27b_128k_fp8_mlp_only_run1",
            "seq_len": 131072,
            "cte_chunk": 512,
            "api": "minimal OpenAI-compatible non-streaming server",
        },
        "throughput": [],
        "intelligence": [],
        "chat": [],
        "errors": [],
    }
    print("METADATA", json.dumps(results["metadata"], sort_keys=True), flush=True)

    monitor_handle = open(monitor_path, "w")
    monitor = subprocess.Popen(
        ["neuron-monitor"],
        stdout=monitor_handle,
        stderr=subprocess.DEVNULL,
        text=True,
    )

    try:
        bench_specs = [
            (512, 129),
            (2048, 129),
            (6000, 129),
            (16000, 65),
            (32000, 65),
            (64000, 33),
        ]
        if not args.skip_120k:
            bench_specs.append((120000, 17))

        for target_tokens, max_tokens_many in bench_specs:
            prompt = make_prompt(
                tokenizer,
                target_tokens,
                "\nContinue this analysis in one concise paragraph:",
            )
            print(f"BENCH target={target_tokens} max_tokens={max_tokens_many}", flush=True)
            one = completion(args.base_url, args.model, prompt, 1)
            many = completion(args.base_url, args.model, prompt, max_tokens_many)
            prompt_tokens = one["usage"].get("prompt_tokens", 0)
            ttft_proxy = one["x_latency_s"] or one["wall_s"]
            total_s = many["x_latency_s"] or many["wall_s"]
            completion_tokens = many["usage"].get("completion_tokens", 0)
            decode_steps = max(0, completion_tokens - 1)
            decode_s = max(0.0, total_s - ttft_proxy)
            record = {
                "target_prompt_tokens": target_tokens,
                "measured_prompt_tokens": prompt_tokens,
                "max_tokens_many": max_tokens_many,
                "completion_tokens_many": completion_tokens,
                "ttft_proxy_s_max_tokens_1": ttft_proxy,
                "prefill_tok_s_prompt_over_ttft_proxy": (
                    prompt_tokens / ttft_proxy if ttft_proxy else None
                ),
                "decode_steps_paired": decode_steps,
                "decode_s_paired": decode_s,
                "tpot_s_paired": decode_s / decode_steps if decode_steps else None,
                "decode_tok_s_paired": decode_steps / decode_s if decode_s else None,
                "request_total_latency_s": total_s,
                "request_completion_goodput_tok_s": (
                    completion_tokens / total_s if total_s else None
                ),
                "request_total_goodput_tok_s": (
                    many["usage"].get("total_tokens", 0) / total_s if total_s else None
                ),
                "wall_s_many": many["wall_s"],
                "text_prefix": many["text"][:400],
            }
            results["throughput"].append(record)
            print("BENCH_RESULT", json.dumps(record, sort_keys=True), flush=True)

        intelligence_specs = [
            (
                "arithmetic_exact",
                "Question: What is 137 * 23? Answer with only the integer.\nAnswer:",
                32,
                ["3151"],
            ),
            (
                "factual_olympics_2020",
                "In one sentence, where were the 2020 Summer Olympics held and why were they unusual?\nAnswer:",
                96,
                ["Tokyo"],
            ),
            (
                "instruction_json",
                "Return valid compact JSON with keys animal and count. animal must be cat and count must be 3. No extra text.\nJSON:",
                80,
                ["cat", "3"],
            ),
            (
                "needle_32k",
                make_needle_prompt(tokenizer, 32000),
                80,
                ["ORCHID-17", "TITAN-42", "VIOLET-91"],
            ),
        ]
        if not args.skip_120k:
            intelligence_specs.append(
                (
                    "needle_120k",
                    make_needle_prompt(tokenizer, 120000),
                    80,
                    ["ORCHID-17", "TITAN-42", "VIOLET-91"],
                )
            )

        for name, prompt, max_tokens, expected in intelligence_specs:
            print(f"INTELLIGENCE {name}", flush=True)
            response = completion(args.base_url, args.model, prompt, max_tokens)
            record = {
                "name": name,
                "prompt_tokens": response["usage"].get("prompt_tokens"),
                "completion_tokens": response["usage"].get("completion_tokens"),
                "latency_s": response["x_latency_s"],
                "expected_contains": expected,
                "contains_expected": contains_all(response["text"], expected),
                "text": response["text"][:1200],
            }
            results["intelligence"].append(record)
            print("INT_RESULT", json.dumps(record, sort_keys=True), flush=True)

        chat_response = chat(
            args.base_url,
            args.model,
            [
                {"role": "system", "content": "You are concise."},
                {"role": "user", "content": "Give two bullet points about AWS Trainium."},
            ],
            128,
        )
        results["chat"].append(chat_response)
        print(
            "CHAT_RESULT",
            json.dumps(
                {k: (v[:400] if k == "text" else v) for k, v in chat_response.items()},
                sort_keys=True,
            ),
            flush=True,
        )
    except Exception:
        import traceback

        tb = traceback.format_exc()
        results["errors"].append(tb)
        print("ERROR", tb, flush=True)
    finally:
        try:
            monitor.send_signal(signal.SIGINT)
            monitor.wait(timeout=5)
        except Exception:
            try:
                monitor.kill()
            except Exception:
                pass
        monitor_handle.close()
        results["monitor"] = parse_monitor(monitor_path)
        results["finished_utc"] = datetime.now(timezone.utc).isoformat()
        with open(out_path, "w") as handle:
            json.dump(results, handle, indent=2)
        print("WROTE", out_path, flush=True)
        print("MONITOR", monitor_path, flush=True)
        print(
            "SUMMARY",
            json.dumps(
                {
                    "throughput_cases": len(results["throughput"]),
                    "intelligence_cases": len(results["intelligence"]),
                    "chat_cases": len(results["chat"]),
                    "errors": len(results["errors"]),
                    "monitor": results["monitor"],
                },
                indent=2,
            ),
            flush=True,
        )


if __name__ == "__main__":
    main()
