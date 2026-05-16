#!/usr/bin/env python3
"""Offline vLLM smoke runner for Qwen3.6-27B on Neuron."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path


def _contrib_root(repo_root: str | None) -> Path:
    if repo_root:
        return Path(repo_root).expanduser().resolve() / "contrib" / "models" / "Qwen3.6-27B"
    return Path(__file__).resolve().parents[1]


def _parse_int_list(value: str) -> list[int]:
    try:
        items = [int(item.strip()) for item in value.split(",") if item.strip()]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "expected comma-separated positive integers"
        ) from exc
    if not items or any(item < 1 for item in items):
        raise argparse.ArgumentTypeError("expected comma-separated positive integers")
    return items


def _parse_bool(value: str) -> bool:
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise argparse.ArgumentTypeError("expected true or false")


def _select_tkg_bucket(prompt_tokens: int, buckets: list[int]) -> int | None:
    if not buckets:
        return None
    for bucket in sorted(buckets):
        if prompt_tokens <= bucket:
            return bucket
    return buckets[-1]


def _metric_value(metrics, *names: str):
    if metrics is None:
        return None
    for name in names:
        if hasattr(metrics, name):
            value = getattr(metrics, name)
            if value is not None:
                return value
    if isinstance(metrics, dict):
        for name in names:
            value = metrics.get(name)
            if value is not None:
                return value
    return None


def _to_float(value) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _completion_token_ids(request_output) -> list[int]:
    if not request_output.outputs:
        return []
    return list(request_output.outputs[0].token_ids or [])


def _build_generation_metrics(
    *,
    label: str,
    request_output,
    elapsed_s: float,
    prompt_tokens: int,
    token_generation_buckets: list[int],
    args: argparse.Namespace,
) -> dict:
    token_ids = _completion_token_ids(request_output)
    completion_tokens = len(token_ids)
    metrics = getattr(request_output, "metrics", None)
    arrival_time = _to_float(
        _metric_value(metrics, "arrival_time", "request_arrival_time")
    )
    first_token_time = _to_float(
        _metric_value(metrics, "first_token_time", "first_token_timestamp")
    )
    finished_time = _to_float(
        _metric_value(metrics, "finished_time", "last_token_time", "finish_time")
    )
    num_cached_tokens = _metric_value(
        metrics, "num_cached_tokens", "cached_tokens", "prefix_cache_hit_tokens"
    )

    first_token_latency_s = None
    if arrival_time is not None and first_token_time is not None:
        first_token_latency_s = max(0.0, first_token_time - arrival_time)

    steady_decode_seconds = None
    if first_token_time is not None and finished_time is not None:
        steady_decode_seconds = max(0.0, finished_time - first_token_time)

    decode_tok_s = None
    if steady_decode_seconds and steady_decode_seconds > 0:
        decode_tokens = max(completion_tokens - 1, 0)
        decode_tok_s = decode_tokens / steady_decode_seconds

    prefix_cache_hit = None
    if num_cached_tokens is not None:
        try:
            prefix_cache_hit = int(num_cached_tokens) > 0
        except (TypeError, ValueError):
            prefix_cache_hit = bool(num_cached_tokens)

    return {
        "label": label,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "first_token_latency_s": first_token_latency_s,
        "steady_decode_seconds": steady_decode_seconds,
        "decode_tok_s": decode_tok_s,
        "end_to_end_tok_s": completion_tokens / elapsed_s if elapsed_s > 0 else None,
        "elapsed_s": elapsed_s,
        "selected_tkg_bucket": _select_tkg_bucket(
            prompt_tokens,
            token_generation_buckets,
        ),
        "output_logits": args.output_logits,
        "on_device_sampling": args.enable_on_device_sampling,
        "async_mode": args.async_mode,
        "max_num_seqs": args.max_num_seqs,
        "block_size": args.block_size,
        "prefix_cache_hit": prefix_cache_hit,
        "num_cached_tokens": num_cached_tokens,
        "metrics_source": "vllm_request_metrics"
        if metrics is not None
        else "wall_time_only",
    }


def _override_config(args: argparse.Namespace) -> dict:
    token_generation_buckets = args.token_generation_buckets or [args.seq_len]
    token_generation_buckets = sorted(token_generation_buckets)
    if token_generation_buckets[-1] > args.seq_len:
        raise ValueError("token generation buckets cannot exceed --seq-len")
    token_generation_batches = (
        sorted(args.token_generation_batches)
        if args.token_generation_batches is not None
        else None
    )
    if token_generation_batches and token_generation_batches[-1] > args.max_num_seqs:
        raise ValueError("token generation batches cannot exceed --max-num-seqs")

    neuron_config = {
        "tp_degree": args.tensor_parallel_size,
        "batch_size": args.max_num_seqs,
        "ctx_batch_size": 1,
        "tkg_batch_size": args.max_num_seqs,
        "max_batch_size": args.max_batch_size,
        "kv_cache_batch_size": args.max_num_seqs,
        "seq_len": args.seq_len,
        "max_length": args.seq_len,
        "max_context_length": args.cte_bucket,
        "context_encoding_buckets": [args.cte_bucket],
        "token_generation_buckets": token_generation_buckets,
        "enable_bucketing": args.token_generation_buckets is not None,
        "logical_nc_config": args.logical_nc_config,
        "torch_dtype": "bfloat16",
        "output_logits": args.output_logits,
        "save_sharded_checkpoint": True,
    }
    if args.enable_on_device_sampling:
        neuron_config["on_device_sampling_config"] = {
            "do_sample": False,
            "top_k": 1,
            "top_p": 1.0,
            "temperature": 1.0,
        }
    if args.async_mode:
        neuron_config["async_mode"] = True
    if token_generation_batches is not None:
        neuron_config["token_generation_batches"] = token_generation_batches
    if args.enable_vllm_chunked_prefill:
        neuron_config.update(
            {
                "is_block_kv_layout": True,
                "chunked_prefill_config": {
                    "max_num_seqs": args.max_num_seqs,
                    "tkg_model_enabled": True,
                    "kernel_q_tile_size": 128,
                    "kernel_kv_tile_size": 1024,
                },
            }
        )
    return {
        "max_prompt_length": args.cte_bucket,
        "override_neuron_config": neuron_config,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", default=None)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--compiled-artifacts", default=None)
    parser.add_argument("--prompt", default="What is 17 * 23? Answer with the number only.")
    parser.add_argument("--chat", action="store_true")
    parser.add_argument("--enable-vllm-chunked-prefill", action="store_true")
    parser.add_argument("--enable-prefix-caching", action="store_true")
    parser.add_argument("--mamba-cache-mode", default=None)
    parser.add_argument("--mamba-cache-dtype", default=None)
    parser.add_argument("--mamba-ssm-cache-dtype", default=None)
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-k", type=int, default=1)
    parser.add_argument("--tensor-parallel-size", type=int, default=4)
    parser.add_argument("--logical-nc-config", type=int, default=2)
    parser.add_argument("--max-num-seqs", type=int, default=1)
    parser.add_argument("--max-batch-size", type=int, default=None)
    parser.add_argument("--max-model-len", type=int, default=512)
    parser.add_argument("--seq-len", type=int, default=512)
    parser.add_argument("--cte-bucket", type=int, default=512)
    parser.add_argument("--block-size", type=int, default=256)
    parser.add_argument("--enable-on-device-sampling", action="store_true")
    parser.add_argument("--async-mode", action="store_true")
    parser.add_argument("--output-logits", type=_parse_bool, default=False)
    parser.add_argument("--token-generation-buckets", type=_parse_int_list, default=None)
    parser.add_argument("--token-generation-batches", type=_parse_int_list, default=None)
    parser.add_argument("--benchmark-repeats", type=int, default=1)
    parser.add_argument("--warm-apc", action="store_true")
    parser.add_argument("--ignore-eos", action="store_true")
    args = parser.parse_args()
    if args.max_batch_size is None:
        args.max_batch_size = args.max_num_seqs
    if args.benchmark_repeats < 1:
        parser.error("--benchmark-repeats must be >= 1")

    contrib_root = _contrib_root(args.repo_root)
    script_dir = Path(__file__).resolve().parent
    sys.path.insert(0, str(script_dir))
    sys.path.insert(0, str(contrib_root))
    os.environ["PYTHONPATH"] = (
        f"{script_dir}:{contrib_root}:{os.environ.get('PYTHONPATH', '')}"
    )
    os.environ.setdefault("VLLM_NEURON_FRAMEWORK", "neuronx-distributed-inference")
    os.environ.setdefault("VLLM_PLUGINS", "neuron")
    if args.enable_vllm_chunked_prefill:
        os.environ["DISABLE_NEURON_CUSTOM_SCHEDULER"] = "1"
    if args.compiled_artifacts:
        os.environ["NEURON_COMPILED_ARTIFACTS"] = str(
            Path(args.compiled_artifacts).expanduser().resolve()
        )

    from hf_qwen35_config import register_qwen35_config  # noqa: WPS433

    register_qwen35_config()

    from transformers import AutoTokenizer  # noqa: WPS433
    from vllm import LLM, SamplingParams  # noqa: WPS433

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path,
        trust_remote_code=True,
    )
    prompt = args.prompt
    if args.chat:
        prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": args.prompt}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )

    additional_config = _override_config(args)
    print("VLLM_QWEN36_CONFIG", json.dumps(additional_config, sort_keys=True), flush=True)

    llm_kwargs = {
        "model": str(Path(args.model_path).expanduser().resolve()),
        "trust_remote_code": True,
        "dtype": "bfloat16",
        "tensor_parallel_size": args.tensor_parallel_size,
        "max_num_seqs": args.max_num_seqs,
        "max_model_len": args.max_model_len,
        "enable_prefix_caching": args.enable_prefix_caching,
        "enable_chunked_prefill": args.enable_vllm_chunked_prefill,
        "additional_config": additional_config,
    }
    if args.mamba_cache_mode is not None:
        llm_kwargs["mamba_cache_mode"] = args.mamba_cache_mode
    if args.mamba_cache_dtype is not None:
        llm_kwargs["mamba_cache_dtype"] = args.mamba_cache_dtype
    if args.mamba_ssm_cache_dtype is not None:
        llm_kwargs["mamba_ssm_cache_dtype"] = args.mamba_ssm_cache_dtype
    if args.enable_vllm_chunked_prefill:
        llm_kwargs["max_num_batched_tokens"] = args.cte_bucket
        llm_kwargs["block_size"] = args.block_size
    llm = LLM(**llm_kwargs)

    sampling = SamplingParams(
        temperature=args.temperature,
        top_k=args.top_k,
        max_tokens=args.max_tokens,
        ignore_eos=args.ignore_eos,
    )
    prompt_tokens = len(tokenizer.encode(prompt))
    token_generation_buckets = args.token_generation_buckets or [args.seq_len]
    labels: list[str] = []
    if args.warm_apc:
        labels.extend(["cold", "warm_apc"])
    else:
        labels.extend(f"run_{idx + 1}" for idx in range(args.benchmark_repeats))

    last_output = None
    for label in labels:
        start = time.perf_counter()
        outputs = llm.generate([prompt], sampling)
        elapsed = time.perf_counter() - start
        last_output = outputs[0]
        generation_metrics = _build_generation_metrics(
            label=label,
            request_output=last_output,
            elapsed_s=elapsed,
            prompt_tokens=prompt_tokens,
            token_generation_buckets=token_generation_buckets,
            args=args,
        )
        print(
            "GENERATION_METRICS",
            json.dumps(generation_metrics, sort_keys=True),
            flush=True,
        )

    if last_output is None:
        raise RuntimeError("no generation was run")

    text = last_output.outputs[0].text
    token_ids = _completion_token_ids(last_output)

    print("PROMPT", prompt)
    print("OUTPUT", text)
    print("TOKENS", token_ids)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
