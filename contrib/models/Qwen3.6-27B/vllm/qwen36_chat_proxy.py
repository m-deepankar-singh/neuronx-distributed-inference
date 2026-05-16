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


class Qwen36ProxyHandler(BaseHTTPRequestHandler):
    backend_url: str = "http://127.0.0.1:8001"
    logits_backend_url: str | None = None
    fastpath_routing: bool = False
    force_disable_thinking: bool = True
    allow_completions: bool = False

    def log_message(self, fmt: str, *args):  # noqa: D401
        print(f"{self.address_string()} - {fmt % args}", flush=True)

    @staticmethod
    def _as_float(value: Any, default: float | None = None) -> float | None:
        if value is None:
            return default
        try:
            return float(value)
        except (TypeError, ValueError):
            return default

    @staticmethod
    def _as_int(value: Any, default: int | None = None) -> int | None:
        if value is None:
            return default
        try:
            return int(value)
        except (TypeError, ValueError):
            return default

    @classmethod
    def _requires_logits_path(cls, payload: dict[str, Any]) -> bool:
        """Return True for requests that cannot use the greedy token fast path."""
        if cls._as_float(payload.get("temperature"), 0.0) != 0.0:
            return True
        if cls._as_int(payload.get("top_k"), 1) != 1:
            return True
        if cls._as_float(payload.get("top_p"), 1.0) != 1.0:
            return True

        logits_fields = {
            "logprobs",
            "top_logprobs",
            "prompt_logprobs",
            "logit_bias",
            "logits_processors",
        }
        if any(payload.get(field) for field in logits_fields):
            return True

        guided_fields = {
            "guided_choice",
            "guided_decoding_backend",
            "guided_grammar",
            "guided_json",
            "guided_regex",
        }
        if any(
            field in payload and payload[field] is not None
            for field in guided_fields
        ):
            return True

        response_format = payload.get("response_format")
        if isinstance(response_format, dict):
            return response_format.get("type", "text") != "text"
        return bool(response_format)

    def _select_backend(self, payload: dict[str, Any] | None) -> str | None:
        if not self.fastpath_routing or not isinstance(payload, dict):
            return self.backend_url
        if not self._requires_logits_path(payload):
            return self.backend_url
        return self.logits_backend_url

    def _forward(
        self,
        method: str,
        body: bytes | None = None,
        backend_url: str | None = None,
    ):
        headers = {
            key: value
            for key, value in self.headers.items()
            if key.lower() not in {"host", "content-length", "connection"}
        }
        target = backend_url or self.backend_url
        url = target.rstrip("/") + self.path
        req = urllib.request.Request(url, data=body, headers=headers, method=method)
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
        except urllib.error.HTTPError as exc:
            error_body = exc.read()
            self.send_response(exc.code)
            for key, value in exc.headers.items():
                if key.lower() in {"transfer-encoding", "connection"}:
                    continue
                self.send_header(key, value)
            self.end_headers()
            self.wfile.write(error_body)

    def do_GET(self):  # noqa: N802
        self._forward("GET")

    def do_POST(self):  # noqa: N802
        length = int(self.headers.get("Content-Length", "0") or "0")
        raw_body = self.rfile.read(length) if length else b""
        target_backend = self.backend_url

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
            target_backend = self._select_backend(payload)
            if target_backend is None:
                _json_response(
                    self,
                    400,
                    {
                        "error": {
                            "message": (
                                "This request needs logits or sampling support, "
                                "but no logits backend is configured. Start a "
                                "logits-path backend and pass --logits-backend-url."
                            ),
                            "type": "invalid_request_error",
                            "code": "qwen36_logits_backend_required",
                        }
                    },
                )
                return
            raw_body = json.dumps(payload).encode("utf-8")

        self._forward("POST", raw_body, backend_url=target_backend)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--backend-url", default=os.getenv("QWEN36_BACKEND_URL", "http://127.0.0.1:8001"))
    parser.add_argument("--logits-backend-url", default=os.getenv("QWEN36_LOGITS_BACKEND_URL"))
    parser.add_argument("--fastpath-routing", action="store_true")
    parser.add_argument("--allow-completions", action="store_true")
    parser.add_argument("--allow-thinking", action="store_true")
    args = parser.parse_args()

    Qwen36ProxyHandler.backend_url = args.backend_url
    Qwen36ProxyHandler.logits_backend_url = args.logits_backend_url
    Qwen36ProxyHandler.fastpath_routing = args.fastpath_routing
    Qwen36ProxyHandler.allow_completions = args.allow_completions
    Qwen36ProxyHandler.force_disable_thinking = not args.allow_thinking

    server = ThreadingHTTPServer((args.host, args.port), Qwen36ProxyHandler)
    print(
        "Qwen3.6 proxy listening on "
        f"{args.host}:{args.port}, backend={args.backend_url}, "
        f"logits_backend={args.logits_backend_url}, "
        f"fastpath_routing={args.fastpath_routing}, "
        f"allow_completions={args.allow_completions}, "
        f"force_disable_thinking={not args.allow_thinking}",
        flush=True,
    )
    server.serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
