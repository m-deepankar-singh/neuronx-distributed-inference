#!/usr/bin/env python3
"""Minimal OpenAI-compatible HTTP server for the Qwen3.6-27B NxDI artifact.

This intentionally avoids uvicorn/fastapi runtime dependencies so it can run in
the stock Neuron inference venv. It supports non-streaming:
  - GET  /health
  - GET  /v1/models
  - POST /v1/completions
  - POST /v1/chat/completions
"""

import argparse
import json
import sys
import threading
import time
import traceback
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Dict, List


def _json_response(handler: BaseHTTPRequestHandler, status: int, payload: Dict[str, Any]):
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json")
    handler.send_header("Content-Length", str(len(body)))
    handler.send_header("Access-Control-Allow-Origin", "*")
    handler.send_header("Access-Control-Allow-Headers", "authorization,content-type")
    handler.send_header("Access-Control-Allow-Methods", "GET,POST,OPTIONS")
    handler.end_headers()
    handler.wfile.write(body)


def _error(handler: BaseHTTPRequestHandler, status: int, message: str):
    _json_response(
        handler,
        status,
        {"error": {"message": message, "type": "server_error", "code": status}},
    )


def _first_text_prompt(prompt: Any) -> str:
    if isinstance(prompt, str):
        return prompt
    if isinstance(prompt, list) and prompt:
        return str(prompt[0])
    return str(prompt)


class QwenOpenAIServer:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.model_id = args.model_id
        self.lock = threading.Lock()
        self._load_model()

    def _load_model(self):
        if self.args.contrib_root not in sys.path:
            sys.path.insert(0, self.args.contrib_root)

        import transformers
        from transformers import AutoTokenizer, GenerationConfig
        from neuronx_distributed_inference.utils.hf_adapter import HuggingFaceGenerationAdapter
        from src.modeling_qwen35 import NeuronQwen35ForCausalLM

        print("Loading tokenizer from", self.args.model_path, flush=True)
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.args.model_path,
            padding_side="right",
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        print("Loading NxDI artifact from", self.args.compiled_path, flush=True)
        t0 = time.perf_counter()
        self.model = NeuronQwen35ForCausalLM(self.args.compiled_path)
        self.model.load(self.args.compiled_path)
        self.adapter = HuggingFaceGenerationAdapter(self.model)
        self.adapter.generation_config.transformers_version = transformers.__version__
        self.GenerationConfig = GenerationConfig
        self.transformers_version = transformers.__version__
        print(f"Model loaded in {time.perf_counter() - t0:.2f}s", flush=True)

    def _generation_config(self, body: Dict[str, Any]):
        temperature = float(body.get("temperature", 0) or 0)
        top_p = float(body.get("top_p", 1) or 1)
        return self.GenerationConfig(
            do_sample=temperature > 0,
            temperature=temperature if temperature > 0 else 1.0,
            top_p=top_p,
            num_beams=1,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )

    def _chat_prompt(self, messages: List[Dict[str, Any]]) -> str:
        try:
            return self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        except Exception:
            lines = []
            for msg in messages:
                role = msg.get("role", "user")
                content = msg.get("content", "")
                lines.append(f"{role}: {content}")
            lines.append("assistant:")
            return "\n".join(lines)

    def _generate(self, prompt: str, body: Dict[str, Any]) -> Dict[str, Any]:
        max_tokens = int(body.get("max_tokens", body.get("max_completion_tokens", 128)) or 128)
        if max_tokens <= 0:
            raise ValueError("max_tokens must be positive")
        if max_tokens > self.args.max_new_tokens_limit:
            raise ValueError(
                f"max_tokens={max_tokens} exceeds server limit {self.args.max_new_tokens_limit}"
            )

        inputs = self.tokenizer(prompt, padding=True, return_tensors="pt")
        prompt_tokens = int(inputs.input_ids.shape[1])
        if prompt_tokens + max_tokens > self.args.seq_len:
            raise ValueError(
                f"prompt_tokens + max_tokens = {prompt_tokens + max_tokens} exceeds "
                f"seq_len={self.args.seq_len}"
            )

        gen_cfg = self._generation_config(body)
        gen_cfg.transformers_version = self.transformers_version

        with self.lock:
            if hasattr(self.model, "reset"):
                self.model.reset()
            t0 = time.perf_counter()
            output = self.adapter.generate(
                inputs.input_ids,
                generation_config=gen_cfg,
                attention_mask=inputs.attention_mask,
                max_new_tokens=max_tokens,
            )
            elapsed = time.perf_counter() - t0

        new_ids = output[0, prompt_tokens:].tolist()
        invalid = [tok for tok in new_ids if tok < 0 or tok >= len(self.tokenizer)]
        if invalid:
            raise RuntimeError(f"model generated invalid token ids: {invalid[:8]}")

        text = self.tokenizer.decode(new_ids, skip_special_tokens=True)
        for stop in body.get("stop") or []:
            if isinstance(stop, str) and stop in text:
                text = text.split(stop, 1)[0]

        return {
            "text": text,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": len(new_ids),
            "elapsed": elapsed,
            "tokens": new_ids,
        }


def make_handler(server_state: QwenOpenAIServer):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, fmt, *args):
            print(f"{self.address_string()} - {fmt % args}", flush=True)

        def do_OPTIONS(self):
            _json_response(self, 200, {})

        def do_GET(self):
            if self.path == "/health":
                _json_response(self, 200, {"status": "ok", "model": server_state.model_id})
            elif self.path == "/v1/models":
                _json_response(
                    self,
                    200,
                    {
                        "object": "list",
                        "data": [
                            {
                                "id": server_state.model_id,
                                "object": "model",
                                "created": int(time.time()),
                                "owned_by": "local",
                            }
                        ],
                    },
                )
            else:
                _error(self, 404, f"unknown route: {self.path}")

        def do_POST(self):
            try:
                length = int(self.headers.get("content-length", "0"))
                body = json.loads(self.rfile.read(length).decode("utf-8") or "{}")
                if body.get("stream"):
                    raise ValueError("stream=true is not supported by this minimal server yet")

                if self.path == "/v1/completions":
                    result = server_state._generate(_first_text_prompt(body.get("prompt", "")), body)
                    _json_response(
                        self,
                        200,
                        {
                            "id": f"cmpl-{uuid.uuid4().hex}",
                            "object": "text_completion",
                            "created": int(time.time()),
                            "model": server_state.model_id,
                            "choices": [
                                {
                                    "index": 0,
                                    "text": result["text"],
                                    "finish_reason": "length",
                                }
                            ],
                            "usage": {
                                "prompt_tokens": result["prompt_tokens"],
                                "completion_tokens": result["completion_tokens"],
                                "total_tokens": result["prompt_tokens"]
                                + result["completion_tokens"],
                            },
                            "x_latency_seconds": result["elapsed"],
                        },
                    )
                elif self.path == "/v1/chat/completions":
                    messages = body.get("messages") or []
                    if not isinstance(messages, list):
                        raise ValueError("messages must be a list")
                    result = server_state._generate(server_state._chat_prompt(messages), body)
                    _json_response(
                        self,
                        200,
                        {
                            "id": f"chatcmpl-{uuid.uuid4().hex}",
                            "object": "chat.completion",
                            "created": int(time.time()),
                            "model": server_state.model_id,
                            "choices": [
                                {
                                    "index": 0,
                                    "message": {
                                        "role": "assistant",
                                        "content": result["text"],
                                    },
                                    "finish_reason": "length",
                                }
                            ],
                            "usage": {
                                "prompt_tokens": result["prompt_tokens"],
                                "completion_tokens": result["completion_tokens"],
                                "total_tokens": result["prompt_tokens"]
                                + result["completion_tokens"],
                            },
                            "x_latency_seconds": result["elapsed"],
                        },
                    )
                else:
                    _error(self, 404, f"unknown route: {self.path}")
            except Exception as exc:
                traceback.print_exc()
                _error(self, 500, str(exc))

    return Handler


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--model-id", default="qwen3.6-27b-neuron")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--compiled-path", required=True)
    parser.add_argument("--contrib-root", required=True)
    parser.add_argument("--seq-len", type=int, default=65536)
    parser.add_argument("--max-new-tokens-limit", type=int, default=512)
    args = parser.parse_args()

    state = QwenOpenAIServer(args)
    httpd = ThreadingHTTPServer((args.host, args.port), make_handler(state))
    print(f"Serving {args.model_id} on http://{args.host}:{args.port}", flush=True)
    httpd.serve_forever()


if __name__ == "__main__":
    raise SystemExit(main())
