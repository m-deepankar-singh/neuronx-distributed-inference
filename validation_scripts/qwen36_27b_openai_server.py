#!/usr/bin/env python3
"""Minimal OpenAI-compatible HTTP server for the Qwen3.6-27B NxDI artifact.

This is intentionally dependency-free: it uses the Python standard library
instead of FastAPI/uvicorn so it can run inside the Neuron venv without extra
packages. It supports the subset needed by OpenAI SDK clients:

  GET  /v1/models
  GET  /health
  POST /v1/chat/completions

Only batch=1 greedy generation is supported. Requests are serialized because
the NxDI model instance owns one mutable cache.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import threading
import time
import traceback
import uuid
from collections import OrderedDict
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import SimpleNamespace
from typing import Any

import torch
from transformers import AutoTokenizer


def token_scalar(tokens: Any) -> int:
    if hasattr(tokens, "detach"):
        tokens = tokens.detach().cpu()
    if tokens.ndim == 0:
        return int(tokens.item())
    return int(tokens.reshape(-1)[0].item())


def normalize_messages(messages: list[dict[str, Any]]) -> list[dict[str, str]]:
    normalized: list[dict[str, str]] = []
    for message in messages:
        role = message.get("role", "user")
        content = message.get("content", "")
        if isinstance(content, str):
            text = content
        elif isinstance(content, list):
            parts: list[str] = []
            for item in content:
                if isinstance(item, dict):
                    if item.get("type") == "text" and isinstance(item.get("text"), str):
                        parts.append(item["text"])
                    elif isinstance(item.get("text"), str):
                        parts.append(item["text"])
            text = "\n".join(parts)
        else:
            text = str(content)
        normalized.append({"role": role, "content": text})
    return normalized


class QwenEngine:
    def __init__(
        self,
        compiled_path: str,
        model_path: str,
        contrib_root: str,
        chunk_size: int,
        seq_len: int,
        model_id: str,
        prefix_cache_entries: int,
    ) -> None:
        sys.path.insert(0, contrib_root)
        from src.modeling_qwen35 import NeuronQwen35ForCausalLM
        from neuronx_distributed_inference.utils.hf_adapter import (
            HuggingFaceGenerationAdapter,
        )
        from neuronx_distributed_inference.modules.generation.sampling import (
            prepare_sampling_params,
        )

        self.model_id = model_id
        self.chunk_size = chunk_size
        self.seq_len = seq_len
        self.prefix_cache_entries = max(0, prefix_cache_entries)
        self.prefix_cache: OrderedDict[str, dict[str, Any]] = OrderedDict()
        self.lock = threading.Lock()
        self.prepare_sampling_params = prepare_sampling_params

        self.tokenizer = AutoTokenizer.from_pretrained(model_path, padding_side="right")
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.pad_id = self.tokenizer.pad_token_id
        self.eos_ids = {
            token_id
            for token_id in [self.tokenizer.eos_token_id, self.tokenizer.convert_tokens_to_ids("<|im_end|>")]
            if token_id is not None and token_id >= 0
        }

        print("LOAD_START", flush=True)
        t0 = time.perf_counter()
        self.model = NeuronQwen35ForCausalLM(compiled_path)
        self.model.load(compiled_path)
        self.model.reset()
        self.fused_spec_enabled = bool(
            getattr(self.model.neuron_config, "enable_fused_speculation", False)
        )
        self.generation_adapter = (
            HuggingFaceGenerationAdapter(self.model)
            if self.fused_spec_enabled
            else None
        )
        print(f"LOAD_DONE seconds={time.perf_counter() - t0:.3f}", flush=True)
        print(
            "CONFIG "
            f"hybrid={getattr(self.model.config, 'use_hybrid_cache_manager', None)} "
            f"chunked={getattr(self.model.config, 'use_qwen_hybrid_chunked_prefill', None)} "
            f"nki={getattr(self.model.config, 'use_qwen_hybrid_chunked_prefill_nki', None)} "
            f"seq_len={self.model.config.neuron_config.seq_len} "
            f"ctx_buckets={self.model.config.neuron_config.context_encoding_buckets}",
            flush=True,
        )

    def _first_generated_token(self, outputs: Any) -> int:
        if self.fused_spec_enabled:
            return token_scalar(outputs.fused_outputs[0][:, 0])
        return token_scalar(outputs.tokens)

    def _decode_fused_spec(
        self,
        prefill_outputs: Any,
        first_token: int,
        prompt_len: int,
        max_tokens: int,
        sampling_params: torch.Tensor,
        stop_strings: list[str],
        on_token: Any | None,
    ) -> tuple[list[int], str, str, list[float]]:
        assert self.generation_adapter is not None

        generated: list[int] = []
        completion_text = ""
        finish_reason = "length"
        decode_times: list[float] = []

        if first_token in self.eos_ids:
            return generated, completion_text, "stop", decode_times
        if first_token < 0 or first_token >= len(self.tokenizer):
            raise RuntimeError(f"invalid token id generated: {first_token}")

        generated.append(first_token)
        piece = self.tokenizer.decode([first_token], skip_special_tokens=True)
        completion_text += piece
        if on_token is not None and piece:
            on_token(piece)
        if stop_strings:
            matched = next((s for s in stop_strings if s and s in completion_text), None)
            if matched:
                completion_text = completion_text.split(matched, 1)[0]
                return generated, completion_text, "stop", decode_times
        if len(generated) >= max_tokens:
            return generated, completion_text, finish_reason, decode_times

        returned_ids = torch.tensor([[first_token]], dtype=torch.long)
        outputs = prefill_outputs
        incremental_len = 0
        model_kwargs: dict[str, Any] = {
            "attention_mask": torch.ones((1, prompt_len), dtype=torch.long),
            "sampling_params": sampling_params,
        }

        while len(generated) < max_tokens:
            model_kwargs = self.generation_adapter._update_model_kwargs_for_fused_generation(
                outputs,
                model_kwargs,
                incremental_len,
            )
            model_inputs = self.generation_adapter.prepare_inputs_for_generation(
                returned_ids,
                **model_kwargs,
            )

            t0 = time.perf_counter()
            with torch.no_grad():
                outputs = self.generation_adapter(**model_inputs)
            decode_times.append(time.perf_counter() - t0)

            accepted_with_padding = outputs.fused_outputs[0]
            next_pos_ids = outputs.fused_outputs[3]
            n_matches_tensor = next_pos_ids - model_inputs["position_ids"]
            n_matches = int(n_matches_tensor.reshape(-1)[0].item())
            if n_matches <= 0:
                raise RuntimeError(f"fused speculation returned n_matches={n_matches}")

            accepted = accepted_with_padding[:, :n_matches].to(torch.long)
            accepted_ids = accepted.reshape(-1).detach().cpu().tolist()
            incremental_len = n_matches

            for token_id in accepted_ids:
                if len(generated) >= max_tokens:
                    break
                if token_id in self.eos_ids:
                    finish_reason = "stop"
                    break
                if token_id < 0 or token_id >= len(self.tokenizer):
                    raise RuntimeError(f"invalid token id generated: {token_id}")

                generated.append(token_id)
                piece = self.tokenizer.decode([token_id], skip_special_tokens=True)
                completion_text += piece
                if on_token is not None and piece:
                    on_token(piece)

                if stop_strings:
                    matched = next((s for s in stop_strings if s and s in completion_text), None)
                    if matched:
                        completion_text = completion_text.split(matched, 1)[0]
                        finish_reason = "stop"
                        break

            returned_ids = torch.cat((returned_ids, accepted), dim=1)
            if finish_reason == "stop":
                break

        return generated, completion_text, finish_reason, decode_times

    def _hash_tokens(self, input_ids: torch.Tensor, prefix_len: int) -> str:
        token_bytes = (
            input_ids[0, :prefix_len]
            .to(torch.int32)
            .contiguous()
            .numpy()
            .tobytes()
        )
        return hashlib.sha256(token_bytes).hexdigest()

    def _state_tensors(self, module_model: Any) -> list[torch.Tensor]:
        tensors: list[torch.Tensor] = []
        kv_mgr = getattr(module_model, "kv_mgr", None)
        if kv_mgr is not None:
            tensors.extend(list(kv_mgr.past_key_values))
        tensors.extend(list(getattr(module_model, "_deltanet_state_params", [])))
        return tensors

    def _snapshot_state(self) -> list[torch.Tensor]:
        # The token-generation model is the canonical runtime cache for decode.
        # The context model is restored from the same snapshot before suffix CTE.
        source = self.model.token_generation_model.model
        return [tensor.detach().cpu().clone() for tensor in self._state_tensors(source)]

    def _restore_state(self, snapshot: list[torch.Tensor]) -> None:
        for module_model in [
            self.model.context_encoding_model.model,
            self.model.token_generation_model.model,
        ]:
            targets = self._state_tensors(module_model)
            if len(targets) != len(snapshot):
                raise RuntimeError(
                    f"prefix cache tensor count mismatch: target={len(targets)} snapshot={len(snapshot)}"
                )
            for target, cached in zip(targets, snapshot):
                target.data = cached.clone()

    def _cache_get(
        self,
        cache_key: str | None,
        input_ids: torch.Tensor,
        prompt_len: int,
        enabled: bool,
    ) -> dict[str, Any] | None:
        if not enabled or not cache_key or self.prefix_cache_entries <= 0:
            return None
        entry = self.prefix_cache.get(cache_key)
        if entry is None:
            return None
        prefix_len = int(entry["prefix_len"])
        if prefix_len <= 0 or prefix_len > prompt_len:
            return None
        if self._hash_tokens(input_ids, prefix_len) != entry["token_hash"]:
            print(
                f"PREFIX_CACHE_MISS key={cache_key} reason=hash_mismatch prefix_tokens={prefix_len}",
                flush=True,
            )
            return None
        self.prefix_cache.move_to_end(cache_key)
        entry["hits"] = int(entry.get("hits", 0)) + 1
        print(f"PREFIX_CACHE_HIT key={cache_key} prefix_tokens={prefix_len}", flush=True)
        return entry

    def _cache_put(
        self,
        cache_key: str | None,
        input_ids: torch.Tensor,
        prefix_len: int,
        next_token: int,
        enabled: bool,
    ) -> bool:
        if not enabled or not cache_key or self.prefix_cache_entries <= 0:
            return False
        if prefix_len <= 0 or prefix_len % self.chunk_size != 0:
            return False
        t0 = time.perf_counter()
        entry = {
            "prefix_len": prefix_len,
            "token_hash": self._hash_tokens(input_ids, prefix_len),
            "snapshot": self._snapshot_state(),
            "next_token": next_token,
            "created": time.time(),
            "hits": 0,
        }
        self.prefix_cache[cache_key] = entry
        self.prefix_cache.move_to_end(cache_key)
        while len(self.prefix_cache) > self.prefix_cache_entries:
            evicted_key, _ = self.prefix_cache.popitem(last=False)
            print(f"PREFIX_CACHE_EVICT key={evicted_key}", flush=True)
        print(
            f"PREFIX_CACHE_STORE key={cache_key} prefix_tokens={prefix_len} "
            f"seconds={time.perf_counter() - t0:.3f} entries={len(self.prefix_cache)}",
            flush=True,
        )
        return True

    def _cache_cutoff(self, body: dict[str, Any], prompt_len: int) -> int:
        raw = body.get("prefix_cache_cutoff_tokens")
        if raw is None:
            cutoff = prompt_len
        else:
            cutoff = int(raw)
        cutoff = max(0, min(cutoff, prompt_len))
        return (cutoff // self.chunk_size) * self.chunk_size

    def render_prompt(self, body: dict[str, Any]) -> str:
        messages = normalize_messages(body.get("messages", []))
        if not messages:
            raise ValueError("messages must be a non-empty list")
        enable_thinking = bool(body.get("enable_thinking", False))
        try:
            return self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=enable_thinking,
            )
        except TypeError:
            rendered = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            if not enable_thinking and rendered.endswith("<think>\n"):
                rendered = rendered[: -len("<think>\n")] + "<think>\n\n</think>\n\n"
            return rendered

    def generate(
        self,
        body: dict[str, Any],
        on_token: Any | None = None,
    ) -> dict[str, Any]:
        prompt = self.render_prompt(body)
        input_ids = torch.tensor(
            [self.tokenizer(prompt, add_special_tokens=False).input_ids],
            dtype=torch.long,
        )
        prompt_len = input_ids.shape[1]
        max_tokens = int(body.get("max_tokens", body.get("max_completion_tokens", 256)))
        max_tokens = max(1, min(max_tokens, self.seq_len - prompt_len))
        if prompt_len + max_tokens > self.seq_len:
            raise ValueError(
                f"prompt plus completion exceeds seq_len: {prompt_len} + {max_tokens} > {self.seq_len}"
            )
        stop = body.get("stop")
        stop_strings = [stop] if isinstance(stop, str) else list(stop or [])
        temperature = float(body.get("temperature", 0.0))
        top_p = float(body.get("top_p", 1.0))
        top_k = int(body.get("top_k", 1))
        # OpenAI temperature=0 means greedy. The Qwen3.6 artifact's NxDI
        # sampler is traced in do_sample=True mode, so literal temperature 0
        # corrupts sampling. top_k=1, temperature=1 is deterministic greedy.
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

        with self.lock:
            self.model.reset()
            seq_ids = torch.tensor([0], dtype=torch.int32)
            generated: list[int] = []
            prefill_times: list[float] = []
            decode_times: list[float] = []
            cache_key = body.get("prefix_cache_key")
            if cache_key is not None:
                cache_key = str(cache_key)
            cache_read = bool(body.get("prefix_cache_read", True))
            cache_write = bool(body.get("prefix_cache_write", cache_key is not None))
            cache_status: dict[str, Any] = {
                "enabled": self.prefix_cache_entries > 0,
                "key": cache_key,
                "hit": False,
                "restored_tokens": 0,
                "stored": False,
                "stored_tokens": 0,
                "entries": len(self.prefix_cache),
            }

            restored_entry = self._cache_get(cache_key, input_ids, prompt_len, cache_read)
            start_offset = 0
            cached_next_token: int | None = None
            if restored_entry is not None:
                self._restore_state(restored_entry["snapshot"])
                start_offset = int(restored_entry["prefix_len"])
                cached_next_token = int(restored_entry["next_token"])
                cache_status["hit"] = True
                cache_status["restored_tokens"] = start_offset

            write_cutoff = self._cache_cutoff(body, prompt_len) if cache_key else 0
            if write_cutoff <= start_offset:
                cache_write = False

            first_token: int | None = None
            out: Any | None = None
            for start in range(start_offset, prompt_len, self.chunk_size):
                end = min(start + self.chunk_size, prompt_len)
                valid = end - start
                chunk_ids = input_ids[:, start:end]
                attn_mask = torch.ones((1, valid), dtype=torch.long)
                pos = torch.arange(start, end, dtype=torch.long).unsqueeze(0)

                t0 = time.perf_counter()
                with torch.no_grad():
                    out = self.model(
                        input_ids=chunk_ids,
                        attention_mask=attn_mask,
                        position_ids=pos,
                        seq_ids=seq_ids,
                        sampling_params=sampling_params,
                        return_dict=True,
                    )
                prefill_times.append(time.perf_counter() - t0)
                tok = self._first_generated_token(out)
                if cache_write and end == write_cutoff and valid == self.chunk_size:
                    cache_status["stored"] = self._cache_put(
                        cache_key=cache_key,
                        input_ids=input_ids,
                        prefix_len=write_cutoff,
                        next_token=tok,
                        enabled=True,
                    )
                    cache_status["stored_tokens"] = write_cutoff if cache_status["stored"] else 0
                if end == prompt_len:
                    first_token = tok

            if first_token is None:
                if cached_next_token is None:
                    raise RuntimeError("prefill produced no token")
                first_token = cached_next_token

            current_token = first_token
            finish_reason = "length"
            completion_text = ""
            if self.fused_spec_enabled:
                generated, completion_text, finish_reason, decode_times = (
                    self._decode_fused_spec(
                        prefill_outputs=out if out is not None else SimpleNamespace(state=None),
                        first_token=first_token,
                        prompt_len=prompt_len,
                        max_tokens=max_tokens,
                        sampling_params=sampling_params,
                        stop_strings=stop_strings,
                        on_token=on_token,
                    )
                )
            else:
                for step in range(max_tokens):
                    if current_token in self.eos_ids:
                        finish_reason = "stop"
                        break
                    if current_token < 0 or current_token >= len(self.tokenizer):
                        raise RuntimeError(f"invalid token id generated: {current_token}")

                    generated.append(current_token)
                    piece = self.tokenizer.decode([current_token], skip_special_tokens=True)
                    completion_text += piece
                    if on_token is not None and piece:
                        on_token(piece)

                    if stop_strings:
                        matched = next((s for s in stop_strings if s and s in completion_text), None)
                        if matched:
                            completion_text = completion_text.split(matched, 1)[0]
                            finish_reason = "stop"
                            break

                    if step == max_tokens - 1:
                        break

                    pos_value = prompt_len + step
                    ids = torch.tensor([[current_token]], dtype=torch.long)
                    pos = torch.tensor([[pos_value]], dtype=torch.long)
                    attn_mask = torch.zeros((1, self.seq_len), dtype=torch.long)
                    attn_mask[:, : pos_value + 1] = 1

                    t0 = time.perf_counter()
                    with torch.no_grad():
                        out = self.model(
                            input_ids=ids,
                            attention_mask=attn_mask,
                            position_ids=pos,
                            seq_ids=seq_ids,
                            sampling_params=sampling_params,
                            return_dict=True,
                        )
                    decode_times.append(time.perf_counter() - t0)
                    current_token = token_scalar(out.tokens)

            usage = {
                "prompt_tokens": prompt_len,
                "completion_tokens": len(generated),
                "total_tokens": prompt_len + len(generated),
            }
            decode_token_count = max(0, len(generated) - 1)
            timings = {
                "prefill_seconds": sum(prefill_times),
                "prefill_tok_s": prompt_len / sum(prefill_times) if prefill_times else 0.0,
                "decode_seconds": sum(decode_times),
                "decode_tok_s": decode_token_count / sum(decode_times) if decode_times else 0.0,
                "chunks": len(prefill_times),
                "prefix_cache": cache_status,
            }
            return {
                "text": completion_text,
                "finish_reason": finish_reason,
                "usage": usage,
                "timings": timings,
            }


class Handler(BaseHTTPRequestHandler):
    engine: QwenEngine

    protocol_version = "HTTP/1.1"

    def log_message(self, fmt: str, *args: Any) -> None:
        print(f"{self.address_string()} - {fmt % args}", flush=True)

    def send_json(self, status: int, payload: dict[str, Any]) -> None:
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "authorization,content-type")
        self.end_headers()
        self.wfile.write(data)

    def do_OPTIONS(self) -> None:
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "authorization,content-type")
        self.send_header("Access-Control-Allow-Methods", "GET,POST,OPTIONS")
        self.end_headers()

    def do_GET(self) -> None:
        if self.path == "/health":
            self.send_json(200, {"status": "ok", "model": self.engine.model_id})
            return
        if self.path == "/v1/models":
            self.send_json(
                200,
                {
                    "object": "list",
                    "data": [
                        {
                            "id": self.engine.model_id,
                            "object": "model",
                            "created": 0,
                            "owned_by": "local",
                        }
                    ],
                },
            )
            return
        self.send_json(404, {"error": {"message": "not found", "type": "not_found"}})

    def do_POST(self) -> None:
        if self.path != "/v1/chat/completions":
            self.send_json(404, {"error": {"message": "not found", "type": "not_found"}})
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            body = json.loads(self.rfile.read(length).decode("utf-8"))
            stream = bool(body.get("stream", False))
            request_id = f"chatcmpl-{uuid.uuid4().hex}"
            created = int(time.time())
            model = body.get("model") or self.engine.model_id

            if stream:
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Cache-Control", "no-cache")
                self.send_header("Connection", "keep-alive")
                self.send_header("Access-Control-Allow-Origin", "*")
                self.end_headers()

                def emit(payload: dict[str, Any]) -> None:
                    self.wfile.write(f"data: {json.dumps(payload, ensure_ascii=False)}\n\n".encode("utf-8"))
                    self.wfile.flush()

                emit(
                    {
                        "id": request_id,
                        "object": "chat.completion.chunk",
                        "created": created,
                        "model": model,
                        "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}],
                    }
                )

                def on_token(piece: str) -> None:
                    emit(
                        {
                            "id": request_id,
                            "object": "chat.completion.chunk",
                            "created": created,
                            "model": model,
                            "choices": [{"index": 0, "delta": {"content": piece}, "finish_reason": None}],
                        }
                    )

                result = self.engine.generate(body, on_token=on_token)
                emit(
                    {
                        "id": request_id,
                        "object": "chat.completion.chunk",
                        "created": created,
                        "model": model,
                        "choices": [{"index": 0, "delta": {}, "finish_reason": result["finish_reason"]}],
                    }
                )
                self.wfile.write(b"data: [DONE]\n\n")
                self.wfile.flush()
                print("REQUEST_DONE " + json.dumps(result["timings"]), flush=True)
                return

            result = self.engine.generate(body)
            payload = {
                "id": request_id,
                "object": "chat.completion",
                "created": created,
                "model": model,
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": result["text"]},
                        "finish_reason": result["finish_reason"],
                    }
                ],
                "usage": result["usage"],
                "timings": result["timings"],
            }
            print("REQUEST_DONE " + json.dumps(result["timings"]), flush=True)
            self.send_json(200, payload)
        except Exception as exc:
            traceback.print_exc()
            self.send_json(500, {"error": {"message": str(exc), "type": exc.__class__.__name__}})


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--compiled-path", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--contrib-root", required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--chunk-size", type=int, default=512)
    parser.add_argument("--seq-len", type=int, default=65536)
    parser.add_argument("--model-id", default="qwen3.6-27b-trainium")
    parser.add_argument("--prefix-cache-entries", type=int, default=1)
    args = parser.parse_args()

    Handler.engine = QwenEngine(
        compiled_path=args.compiled_path,
        model_path=args.model_path,
        contrib_root=args.contrib_root,
        chunk_size=args.chunk_size,
        seq_len=args.seq_len,
        model_id=args.model_id,
        prefix_cache_entries=args.prefix_cache_entries,
    )
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"SERVER_READY http://{args.host}:{args.port}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
