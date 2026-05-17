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
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import traceback
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Dict, List

import torch


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


def _token_scalar(tokens: Any) -> int:
    if hasattr(tokens, "detach"):
        tokens = tokens.detach().cpu()
    if tokens.ndim == 0:
        return int(tokens.item())
    return int(tokens.reshape(-1)[0].item())


def _load_cold_prefill_config() -> Dict[str, Any]:
    raw = os.getenv("QWEN36_COLD_PREFILL_CONFIG", "{}")
    try:
        config = json.loads(raw)
    except json.JSONDecodeError:
        return {}
    return config if isinstance(config, dict) else {}


def _selected_cte_bucket(actual_prompt_len: int, buckets: List[int]) -> int:
    for bucket in buckets:
        if actual_prompt_len <= bucket:
            return bucket
    return buckets[-1]


def _bucketed_prefill_work(
    actual_prompt_len: int,
    buckets: List[int],
    *,
    chunked_prefill_enabled: bool,
) -> Dict[str, Any]:
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
    selected_buckets = []
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


def _hbm_usage_if_available():
    probe_mode = os.environ.get("QWEN36_HBM_PROBE", "neuron-monitor").lower()
    if probe_mode in {"0", "false", "none", "off"}:
        return None
    if probe_mode != "xla":
        usage = _hbm_usage_from_neuron_monitor()
        if usage is not None or probe_mode == "neuron-monitor":
            return usage
    return _xla_hbm_usage_if_available() if probe_mode == "xla" else None


def _xla_hbm_usage_if_available():
    try:
        import torch_xla.core.xla_model as xm

        device = xm.xla_device()
        return xm.get_memory_info(device)
    except Exception:
        return None


def _safe_int(value) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def _parse_neuron_monitor_hbm(stdout: str) -> Dict[str, Any] | None:
    peak_device_bytes = 0
    peak_tensor_bytes = 0
    samples = 0

    for line in stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            report = json.loads(line)
        except json.JSONDecodeError:
            continue
        for runtime in report.get("neuron_runtime_data", []):
            memory_used = runtime.get("report", {}).get("memory_used", {})
            used = memory_used.get("neuron_runtime_used_bytes", {})
            peak_device_bytes = max(
                peak_device_bytes,
                _safe_int(used.get("neuron_device")),
            )
            nc_usage = (
                used.get("usage_breakdown", {}).get("neuroncore_memory_usage", {})
            )
            tensor_bytes = sum(
                _safe_int(core.get("tensors"))
                for core in nc_usage.values()
                if isinstance(core, dict)
            )
            peak_tensor_bytes = max(peak_tensor_bytes, tensor_bytes)
            samples += 1

    bytes_used = peak_device_bytes or peak_tensor_bytes
    if samples == 0 or bytes_used <= 0:
        return None
    return {
        "source": "neuron-monitor",
        "bytes_used": bytes_used,
        "neuron_device_bytes_used": peak_device_bytes,
        "tensor_bytes": peak_tensor_bytes,
        "samples": samples,
    }


def _hbm_usage_from_neuron_monitor() -> Dict[str, Any] | None:
    neuron_monitor = shutil.which("neuron-monitor")
    if neuron_monitor is None:
        return None
    timeout_seconds = float(os.environ.get("QWEN36_HBM_MONITOR_SECONDS", "1.0"))
    config = {
        "period": "0.25s",
        "neuron_runtimes": [
            {
                "tag_filter": ".*",
                "metrics": [{"type": "memory_used", "period": "0.25s"}],
            }
        ],
    }
    config_path = None
    proc = None
    try:
        with tempfile.NamedTemporaryFile(
            "w",
            encoding="utf-8",
            suffix=".json",
            delete=False,
        ) as handle:
            json.dump(config, handle)
            config_path = handle.name
        proc = subprocess.Popen(
            [neuron_monitor, "--config-file", config_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        time.sleep(max(timeout_seconds, 0.1))
        proc.terminate()
        stdout, _stderr = proc.communicate(timeout=5)
        return _parse_neuron_monitor_hbm(stdout)
    except Exception:
        if proc is not None and proc.poll() is None:
            proc.kill()
            proc.communicate()
        return None
    finally:
        if config_path is not None:
            try:
                os.unlink(config_path)
            except OSError:
                pass


def _cold_prefill_metrics(
    *,
    request_id: str,
    route: str,
    actual_prompt_len: int,
    prefill_elapsed_seconds: float,
    config: Dict[str, Any],
    fallback_chunk_size: int,
) -> Dict[str, Any]:
    buckets = [int(bucket) for bucket in config.get("cte_buckets") or []]
    if not buckets and fallback_chunk_size > 0:
        buckets = [int(fallback_chunk_size)]
    bucket_work = _bucketed_prefill_work(
        actual_prompt_len,
        buckets,
        chunked_prefill_enabled=bool(config.get("chunked_prefill_enabled", True)),
    )
    bucket_work_tokens = bucket_work.get("bucket_work_tokens", 0)
    padding_tokens = bucket_work.get("padding_tokens", 0)
    return {
        "request_id": request_id,
        "route": route,
        "actual_prompt_len": actual_prompt_len,
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
        "chunked_prefill_enabled": config.get("chunked_prefill_enabled", True),
        "cold_zero_conv_fast_path_enabled": config.get(
            "cold_zero_conv_fast_path_enabled"
        ),
        "use_nki_fused": config.get("use_nki_fused"),
        "gdn_cte_kernel": config.get("gdn_cte_kernel"),
        "max_model_len": config.get("max_model_len"),
        "seq_len": config.get("seq_len"),
        "prefill_latency_ms": prefill_elapsed_seconds * 1000.0,
        "actual_tok_per_s": (
            actual_prompt_len / prefill_elapsed_seconds
            if prefill_elapsed_seconds > 0
            else None
        ),
        "bucket_tok_per_s": (
            bucket_work_tokens / prefill_elapsed_seconds
            if prefill_elapsed_seconds > 0 and bucket_work_tokens
            else None
        ),
        "hbm_usage": _hbm_usage_if_available(),
    }


def _generation_metrics(
    *,
    request_id: str,
    route: str,
    prompt_tokens: int,
    completion_tokens: int,
    max_tokens: int,
    prefill_elapsed_seconds: float,
    request_elapsed_seconds: float,
    first_token_id: int | None,
    finish_reason: str,
) -> Dict[str, Any]:
    decode_elapsed_seconds = max(request_elapsed_seconds - prefill_elapsed_seconds, 0.0)
    decode_tokens = max(completion_tokens - 1, 0)
    return {
        "request_id": request_id,
        "route": route,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "max_tokens": max_tokens,
        "first_token_id": first_token_id,
        "finish_reason": finish_reason,
        "request_latency_ms": request_elapsed_seconds * 1000.0,
        "first_token_latency_ms": prefill_elapsed_seconds * 1000.0,
        "prefill_latency_ms": prefill_elapsed_seconds * 1000.0,
        "decode_latency_ms": decode_elapsed_seconds * 1000.0,
        "decode_tokens": decode_tokens,
        "decode_tok_per_s": (
            decode_tokens / decode_elapsed_seconds
            if decode_elapsed_seconds > 0 and decode_tokens
            else None
        ),
        "end_to_end_generated_tok_per_s": (
            completion_tokens / request_elapsed_seconds
            if request_elapsed_seconds > 0 and completion_tokens
            else None
        ),
    }


class QwenOpenAIServer:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.model_id = args.model_id
        self.lock = threading.Lock()
        self.cold_prefill_config = _load_cold_prefill_config()
        self._load_model()

    def _load_model(self):
        if self.args.contrib_root not in sys.path:
            sys.path.insert(0, self.args.contrib_root)

        from transformers import AutoTokenizer, GenerationConfig
        from neuronx_distributed_inference.modules.generation.sampling import (
            prepare_sampling_params,
        )
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
        self.model.reset()
        self.prepare_sampling_params = prepare_sampling_params
        self.GenerationConfig = GenerationConfig
        print(f"Model loaded in {time.perf_counter() - t0:.2f}s", flush=True)

    def _chat_prompt(self, messages: List[Dict[str, Any]], enable_thinking: bool = False) -> str:
        try:
            return self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=enable_thinking,
            )
        except TypeError:
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

    def _generate(self, prompt: str, body: Dict[str, Any], route: str) -> Dict[str, Any]:
        max_tokens = int(body.get("max_tokens", body.get("max_completion_tokens", 128)) or 128)
        if max_tokens <= 0:
            raise ValueError("max_tokens must be positive")
        if max_tokens > self.args.max_new_tokens_limit:
            raise ValueError(
                f"max_tokens={max_tokens} exceeds server limit {self.args.max_new_tokens_limit}"
            )

        input_ids = torch.tensor(
            [self.tokenizer(prompt, add_special_tokens=False).input_ids],
            dtype=torch.long,
        )
        prompt_tokens = int(input_ids.shape[1])
        if prompt_tokens <= 0:
            raise ValueError("prompt must contain at least one token")
        if prompt_tokens + max_tokens > self.args.seq_len:
            raise ValueError(
                f"prompt_tokens + max_tokens = {prompt_tokens + max_tokens} exceeds "
                f"seq_len={self.args.seq_len}"
            )

        temperature = float(body.get("temperature", 0.0) or 0.0)
        top_p = float(body.get("top_p", 1.0) or 1.0)
        top_k = int(body.get("top_k", 1) or 1)
        # NxDI's traced on-device sampler for this artifact uses do_sample=True.
        # OpenAI temperature=0 means greedy, but passing literal 0 into that
        # sampler divides logits by zero. top_k=1 with temperature=1 is the
        # deterministic greedy path used by the validated HF adapter tests.
        sampler_temperature = temperature
        if temperature <= 0.0:
            sampler_temperature = 1.0
            top_p = 1.0
            top_k = 1
        sampling_params = self.prepare_sampling_params(
            batch_size=1,
            top_k=[top_k],
            top_p=[top_p],
            temperature=[sampler_temperature],
        )
        seq_ids = torch.tensor([0], dtype=torch.int32)

        with self.lock:
            if hasattr(self.model, "reset"):
                self.model.reset()
            t0 = time.perf_counter()
            prefill_t0 = time.perf_counter()
            first_token = None
            for start in range(0, prompt_tokens, self.args.chunk_size):
                end = min(start + self.args.chunk_size, prompt_tokens)
                valid = end - start
                chunk_ids = input_ids[:, start:end]
                attention_mask = torch.ones((1, valid), dtype=torch.long)
                position_ids = torch.arange(
                    start,
                    end,
                    dtype=torch.long,
                ).unsqueeze(0)

                with torch.no_grad():
                    out = self.model(
                        input_ids=chunk_ids,
                        attention_mask=attention_mask,
                        position_ids=position_ids,
                        seq_ids=seq_ids,
                        sampling_params=sampling_params,
                        return_dict=True,
                    )
                first_token = _token_scalar(out.tokens)
            prefill_elapsed = time.perf_counter() - prefill_t0

            if first_token is None:
                raise RuntimeError("prefill produced no token")
            request_id = f"nxdi-{uuid.uuid4().hex}"
            cold_metrics = _cold_prefill_metrics(
                request_id=request_id,
                route=route,
                actual_prompt_len=prompt_tokens,
                prefill_elapsed_seconds=prefill_elapsed,
                config=self.cold_prefill_config,
                fallback_chunk_size=self.args.chunk_size,
            )
            print(
                "COLD_PREFILL_METRICS",
                json.dumps(cold_metrics, sort_keys=True),
                flush=True,
            )

            new_ids = []
            current_token = first_token
            vocab_size = len(self.tokenizer)
            raw_eos_id = self.tokenizer.eos_token_id
            eos_ids = (
                set(raw_eos_id)
                if isinstance(raw_eos_id, (list, tuple, set))
                else {raw_eos_id}
            )
            decode_ids = torch.empty((1, 1), dtype=torch.int32)
            decode_position_ids = torch.empty((1, 1), dtype=torch.int32)
            decode_attention_mask = torch.ones(
                (1, prompt_tokens + max_tokens),
                dtype=torch.int32,
            )
            finish_reason = "length"
            with torch.no_grad():
                for step in range(max_tokens):
                    if current_token in eos_ids:
                        finish_reason = "stop"
                        break
                    if current_token < 0 or current_token >= vocab_size:
                        raise RuntimeError(f"model generated invalid token id: {current_token}")
                    new_ids.append(current_token)
                    if step == max_tokens - 1:
                        break

                    pos_value = prompt_tokens + step
                    decode_ids[0, 0] = current_token
                    decode_position_ids[0, 0] = pos_value
                    active_attention_mask = decode_attention_mask[:, : pos_value + 1]
                    out = self.model(
                        input_ids=decode_ids,
                        attention_mask=active_attention_mask,
                        position_ids=decode_position_ids,
                        seq_ids=seq_ids,
                        sampling_params=sampling_params,
                        return_dict=True,
                    )
                    current_token = _token_scalar(out.tokens)
            elapsed = time.perf_counter() - t0
            generation_metrics = _generation_metrics(
                request_id=request_id,
                route=route,
                prompt_tokens=prompt_tokens,
                completion_tokens=len(new_ids),
                max_tokens=max_tokens,
                prefill_elapsed_seconds=prefill_elapsed,
                request_elapsed_seconds=elapsed,
                first_token_id=first_token,
                finish_reason=finish_reason,
            )
            print(
                "GENERATION_METRICS",
                json.dumps(generation_metrics, sort_keys=True),
                flush=True,
            )

        invalid = [tok for tok in new_ids if tok < 0 or tok >= vocab_size]
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
            "prefill_elapsed": prefill_elapsed,
            "generation_metrics": generation_metrics,
            "tokens": new_ids,
            "finish_reason": finish_reason,
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
                    result = server_state._generate(
                        _first_text_prompt(body.get("prompt", "")),
                        body,
                        self.path,
                    )
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
                                    "finish_reason": result["finish_reason"],
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
                    result = server_state._generate(
                        server_state._chat_prompt(
                            messages,
                            enable_thinking=bool(body.get("enable_thinking", False)),
                        ),
                        body,
                        self.path,
                    )
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
                                    "finish_reason": result["finish_reason"],
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
    parser.add_argument("--chunk-size", type=int, default=512)
    parser.add_argument("--max-new-tokens-limit", type=int, default=512)
    args = parser.parse_args()

    state = QwenOpenAIServer(args)
    httpd = ThreadingHTTPServer((args.host, args.port), make_handler(state))
    print(f"Serving {args.model_id} on http://{args.host}:{args.port}", flush=True)
    httpd.serve_forever()


if __name__ == "__main__":
    raise SystemExit(main())
