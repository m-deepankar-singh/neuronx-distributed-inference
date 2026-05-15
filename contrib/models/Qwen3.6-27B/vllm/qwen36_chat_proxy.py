#!/usr/bin/env python3
"""Small OpenAI-compatible guard proxy for Qwen3.6 vLLM serving.

The upstream Qwen3.6 chat template defaults to thinking mode. For this Neuron
artifact the production-safe chat path is non-thinking mode, so this proxy
injects ``chat_template_kwargs={"enable_thinking": false}`` for chat requests.
It also blocks raw completions by default because they are not chat-templated.
"""

from __future__ import annotations

import argparse
import json
import os
import time
import uuid
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any


def _json_response(handler: BaseHTTPRequestHandler, status: int, payload: dict[str, Any]):
    body = json.dumps(payload).encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json")
    handler.send_header("Content-Length", str(len(body)))
    handler.end_headers()
    handler.wfile.write(body)


def _message_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict):
                text = item.get("text")
                if isinstance(text, str):
                    parts.append(text)
            elif isinstance(item, str):
                parts.append(item)
        return "\n".join(parts)
    return str(content)


def _normalize_messages_for_qwen(messages: Any) -> Any:
    """Make common OpenAI message layouts acceptable to the Qwen chat template."""
    if not isinstance(messages, list):
        return messages

    system_parts: list[str] = []
    normal_messages: list[Any] = []
    for message in messages:
        if not isinstance(message, dict):
            normal_messages.append(message)
            continue

        role = message.get("role")
        if role in {"system", "developer"}:
            system_parts.append(_message_text(message.get("content", "")))
        else:
            normal_messages.append(message)

    if not system_parts:
        return messages

    system_message = {
        "role": "system",
        "content": "\n\n".join(part for part in system_parts if part),
    }
    return [system_message, *normal_messages]


def _first_text_prompt(prompt: Any) -> str:
    if isinstance(prompt, str):
        return prompt
    if isinstance(prompt, list) and prompt:
        return str(prompt[0])
    return str(prompt)


def _load_cold_prefill_config() -> dict[str, Any]:
    raw = os.getenv("QWEN36_COLD_PREFILL_CONFIG", "{}")
    try:
        config = json.loads(raw)
    except json.JSONDecodeError:
        return {}
    return config if isinstance(config, dict) else {}


def _selected_cte_bucket(actual_prompt_len: int, buckets: list[int]) -> int:
    for bucket in buckets:
        if actual_prompt_len <= bucket:
            return bucket
    return buckets[-1]


def _bucketed_prefill_work(
    actual_prompt_len: int,
    buckets: list[int],
    *,
    chunked_prefill_enabled: bool,
) -> dict[str, Any]:
    if not buckets:
        return {}
    max_bucket = buckets[-1]
    if actual_prompt_len <= max_bucket or not chunked_prefill_enabled:
        selected = _selected_cte_bucket(actual_prompt_len, buckets)
        return {
            "selected_cte_bucket": selected,
            "selected_cte_buckets": [selected],
            "num_cte_chunks": 1,
            "bucket_work_tokens": selected,
            "padding_tokens": max(selected - actual_prompt_len, 0),
        }

    remaining = actual_prompt_len
    selected_buckets: list[int] = []
    bucket_work_tokens = 0
    while remaining > 0:
        chunk_tokens = min(remaining, max_bucket)
        selected = _selected_cte_bucket(chunk_tokens, buckets)
        selected_buckets.append(selected)
        bucket_work_tokens += selected
        remaining -= chunk_tokens
    return {
        "selected_cte_bucket": selected_buckets[-1],
        "selected_cte_buckets": selected_buckets,
        "num_cte_chunks": len(selected_buckets),
        "bucket_work_tokens": bucket_work_tokens,
        "padding_tokens": max(bucket_work_tokens - actual_prompt_len, 0),
    }


def _cold_prefill_request_metrics(
    *,
    request_id: str,
    route: str,
    prompt_text: str,
    tokenizer,
    config: dict[str, Any],
) -> dict[str, Any]:
    if tokenizer is not None:
        actual_prompt_len = len(tokenizer.encode(prompt_text, add_special_tokens=False))
        prompt_len_source = "tokenizer"
    else:
        actual_prompt_len = len(prompt_text.split())
        prompt_len_source = "whitespace"
    buckets = [int(bucket) for bucket in config.get("cte_buckets", [])]
    bucket_work = _bucketed_prefill_work(
        actual_prompt_len,
        buckets,
        chunked_prefill_enabled=bool(config.get("chunked_prefill_enabled", False)),
    )
    bucket_work_tokens = bucket_work.get("bucket_work_tokens", 0)
    padding_tokens = bucket_work.get("padding_tokens", 0)
    return {
        "request_id": request_id,
        "route": route,
        "actual_prompt_len": actual_prompt_len,
        "actual_prompt_len_source": prompt_len_source,
        **bucket_work,
        "padding_ratio": (
            padding_tokens / bucket_work_tokens if bucket_work_tokens else None
        ),
        "ctx_batch_size": config.get("ctx_batch_size"),
        "block_size": config.get("block_size"),
        "kernel_q_tile_size": config.get("kernel_q_tile_size"),
        "kernel_kv_tile_size": config.get("kernel_kv_tile_size"),
        "text_only_cte_enabled": config.get("text_only_cte_enabled"),
        "compact_mask_enabled": config.get("compact_mask_enabled"),
        "chunked_prefill_enabled": config.get("chunked_prefill_enabled"),
        "cold_zero_conv_fast_path_enabled": config.get(
            "cold_zero_conv_fast_path_enabled"
        ),
        "use_nki_fused": config.get("use_nki_fused"),
        "gdn_cte_kernel": config.get("gdn_cte_kernel"),
        "max_model_len": config.get("max_model_len"),
        "seq_len": config.get("seq_len"),
    }


def _load_tokenizer(model_path: str | None):
    if not model_path:
        return None
    try:
        from transformers import AutoTokenizer

        return AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    except Exception as exc:
        print(f"COLD_PREFILL_PROXY_TOKENIZER_LOAD_FAILED {exc}", flush=True)
        return None


class Qwen36ProxyHandler(BaseHTTPRequestHandler):
    backend_url: str = "http://127.0.0.1:8001"
    force_disable_thinking: bool = True
    allow_completions: bool = False
    cold_prefill_config: dict[str, Any] = {}
    tokenizer = None

    def log_message(self, fmt: str, *args):  # noqa: D401
        print(f"{self.address_string()} - {fmt % args}", flush=True)

    def _forward(self, method: str, body: bytes | None = None):
        headers = {
            key: value
            for key, value in self.headers.items()
            if key.lower() not in {"host", "content-length", "connection"}
        }
        url = self.backend_url.rstrip("/") + self.path
        req = urllib.request.Request(url, data=body, headers=headers, method=method)
        start = time.perf_counter()
        try:
            with urllib.request.urlopen(req, timeout=None) as resp:
                response_body = resp.read()
                self.send_response(resp.status)
                for key, value in resp.headers.items():
                    if key.lower() in {"transfer-encoding", "connection"}:
                        continue
                    self.send_header(key, value)
                self.end_headers()
                self.wfile.write(response_body)
                return {
                    "status": resp.status,
                    "backend_latency_ms": (time.perf_counter() - start) * 1000.0,
                }
        except urllib.error.HTTPError as exc:
            error_body = exc.read()
            self.send_response(exc.code)
            for key, value in exc.headers.items():
                if key.lower() in {"transfer-encoding", "connection"}:
                    continue
                self.send_header(key, value)
            self.end_headers()
            self.wfile.write(error_body)
            return {
                "status": exc.code,
                "backend_latency_ms": (time.perf_counter() - start) * 1000.0,
            }

    def do_GET(self):  # noqa: N802
        self._forward("GET")

    def do_POST(self):  # noqa: N802
        length = int(self.headers.get("Content-Length", "0") or "0")
        raw_body = self.rfile.read(length) if length else b""
        request_metrics = None

        if self.path == "/v1/completions" and not self.allow_completions:
            _json_response(
                self,
                400,
                {
                    "error": {
                        "message": (
                            "Raw /v1/completions is disabled for Qwen3.6. "
                            "Use /v1/chat/completions so the Qwen chat template "
                            "and non-thinking mode are applied."
                        ),
                        "type": "invalid_request_error",
                        "code": "qwen36_chat_required",
                    }
                },
            )
            return

        if self.path == "/v1/chat/completions" and raw_body:
            try:
                payload = json.loads(raw_body)
            except json.JSONDecodeError:
                self._forward("POST", raw_body)
                return

            template_kwargs = payload.get("chat_template_kwargs")
            if not isinstance(template_kwargs, dict):
                template_kwargs = {}
            if self.force_disable_thinking:
                template_kwargs["enable_thinking"] = False
            else:
                template_kwargs.setdefault("enable_thinking", False)
            payload["chat_template_kwargs"] = template_kwargs
            payload["messages"] = _normalize_messages_for_qwen(payload.get("messages"))
            prompt_text = json.dumps(payload["messages"], ensure_ascii=False)
            if self.tokenizer is not None:
                try:
                    prompt_text = self.tokenizer.apply_chat_template(
                        payload["messages"],
                        tokenize=False,
                        add_generation_prompt=True,
                        **template_kwargs,
                    )
                except Exception:
                    pass
            request_metrics = _cold_prefill_request_metrics(
                request_id=f"qwen36-{uuid.uuid4().hex}",
                route=self.path,
                prompt_text=prompt_text,
                tokenizer=self.tokenizer,
                config=self.cold_prefill_config,
            )
            raw_body = json.dumps(payload).encode("utf-8")
        elif self.path == "/v1/completions" and raw_body:
            try:
                payload = json.loads(raw_body)
            except json.JSONDecodeError:
                payload = {}
            request_metrics = _cold_prefill_request_metrics(
                request_id=f"qwen36-{uuid.uuid4().hex}",
                route=self.path,
                prompt_text=_first_text_prompt(payload.get("prompt", "")),
                tokenizer=self.tokenizer,
                config=self.cold_prefill_config,
            )

        if request_metrics is not None:
            print(
                "COLD_PREFILL_PROXY_REQUEST",
                json.dumps(request_metrics, sort_keys=True),
                flush=True,
            )
        response_metrics = self._forward("POST", raw_body)
        if request_metrics is not None and response_metrics is not None:
            complete_metrics = {**request_metrics, **response_metrics}
            print(
                "COLD_PREFILL_PROXY_RESPONSE",
                json.dumps(complete_metrics, sort_keys=True),
                flush=True,
            )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--backend-url", default=os.getenv("QWEN36_BACKEND_URL", "http://127.0.0.1:8001"))
    parser.add_argument("--model-path", default=os.getenv("QWEN36_MODEL_PATH"))
    parser.add_argument("--allow-completions", action="store_true")
    parser.add_argument("--allow-thinking", action="store_true")
    args = parser.parse_args()

    Qwen36ProxyHandler.backend_url = args.backend_url
    Qwen36ProxyHandler.allow_completions = args.allow_completions
    Qwen36ProxyHandler.force_disable_thinking = not args.allow_thinking
    Qwen36ProxyHandler.cold_prefill_config = _load_cold_prefill_config()
    Qwen36ProxyHandler.tokenizer = _load_tokenizer(args.model_path)

    server = ThreadingHTTPServer((args.host, args.port), Qwen36ProxyHandler)
    print(
        "Qwen3.6 proxy listening on "
        f"{args.host}:{args.port}, backend={args.backend_url}, "
        f"allow_completions={args.allow_completions}, "
        f"force_disable_thinking={not args.allow_thinking}",
        flush=True,
    )
    server.serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
