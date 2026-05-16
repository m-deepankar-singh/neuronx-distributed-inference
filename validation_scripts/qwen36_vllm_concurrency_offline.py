#!/usr/bin/env python3
"""Offline vLLM concurrency sweep for Qwen3.6-27B on Neuron."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path


def _parse_int_list(value: str) -> list[int]:
    items = [int(item.strip()) for item in value.split(",") if item.strip()]
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


def _build_prompt(tokenizer, target_tokens: int, request_idx: int) -> tuple[str, int]:
    unit = (
        f"Request {request_idx} marker R{request_idx:03d}. "
        "Neuron runtime profiling measures dispatch gaps, device activity, "
        "HBM traffic, scheduler behavior, and end-to-end request latency. "
    )
    text = "Summarize the following profiling notes in two precise bullet points.\n\n"
    while True:
        candidate = text + (unit * max(1, len(text) // max(1, len(unit)) + 1))
        token_count = len(tokenizer.encode(candidate))
        if token_count >= target_tokens:
            return candidate, token_count
        text = candidate


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--compiled-artifacts", required=True)
    parser.add_argument("--seq-len", type=int, default=262144)
    parser.add_argument("--cte-bucket", type=int, default=512)
    parser.add_argument("--tensor-parallel-size", type=int, default=4)
    parser.add_argument("--logical-nc-config", type=int, default=2)
    parser.add_argument("--max-num-seqs", type=int, default=1)
    parser.add_argument("--max-batch-size", type=int, default=None)
    parser.add_argument("--concurrency", type=_parse_int_list, default=[1, 2, 4, 8])
    parser.add_argument("--prompt-tokens", type=int, default=2048)
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--block-size", type=int, default=256)
    parser.add_argument("--enable-prefix-caching", action="store_true")
    parser.add_argument("--enable-on-device-sampling", action="store_true")
    parser.add_argument("--async-mode", action="store_true")
    parser.add_argument("--output-logits", type=_parse_bool, default=False)
    parser.add_argument("--token-generation-buckets", type=_parse_int_list, default=None)
    parser.add_argument("--token-generation-batches", type=_parse_int_list, default=None)
    parser.add_argument("--mamba-cache-mode", default=None)
    args = parser.parse_args()
    if args.max_batch_size is None:
        args.max_batch_size = args.max_num_seqs
    token_generation_buckets = args.token_generation_buckets or [args.seq_len]
    token_generation_buckets = sorted(token_generation_buckets)
    if token_generation_buckets[-1] > args.seq_len:
        parser.error("--token-generation-buckets cannot contain values greater than --seq-len")
    token_generation_batches = (
        sorted(args.token_generation_batches)
        if args.token_generation_batches is not None
        else None
    )
    if token_generation_batches and token_generation_batches[-1] > args.max_num_seqs:
        parser.error("--token-generation-batches cannot contain values greater than --max-num-seqs")

    repo = Path(args.repo_root).expanduser().resolve()
    model_path = Path(args.model_path).expanduser().resolve()
    artifact = Path(args.compiled_artifacts).expanduser().resolve()
    contrib_root = repo / "contrib" / "models" / "Qwen3.6-27B"
    script_dir = contrib_root / "vllm"

    sys.path.insert(0, str(script_dir))
    sys.path.insert(0, str(contrib_root))
    os.environ["PYTHONPATH"] = f"{script_dir}:{contrib_root}:{os.environ.get('PYTHONPATH', '')}"
    os.environ.setdefault("VLLM_NEURON_FRAMEWORK", "neuronx-distributed-inference")
    os.environ.setdefault("VLLM_PLUGINS", "neuron")
    os.environ["DISABLE_NEURON_CUSTOM_SCHEDULER"] = "1"
    os.environ["NEURON_COMPILED_ARTIFACTS"] = str(artifact)

    from hf_qwen35_config import register_qwen35_config  # noqa: WPS433
    from transformers import AutoTokenizer  # noqa: WPS433
    from vllm import LLM, SamplingParams  # noqa: WPS433

    register_qwen35_config()

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
        "is_block_kv_layout": True,
        "chunked_prefill_config": {
            "max_num_seqs": args.max_num_seqs,
            "tkg_model_enabled": True,
            "kernel_q_tile_size": 128,
            "kernel_kv_tile_size": 1024,
        },
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
    additional_config = {
        "max_prompt_length": args.cte_bucket,
        "override_neuron_config": neuron_config,
    }
    print("CONCURRENCY_CONFIG", json.dumps(additional_config, sort_keys=True), flush=True)

    tokenizer = AutoTokenizer.from_pretrained(str(model_path), trust_remote_code=True)
    max_concurrency = max(args.concurrency)
    prompts: list[str] = []
    input_token_counts: list[int] = []
    for idx in range(max_concurrency):
        user_prompt, _ = _build_prompt(tokenizer, args.prompt_tokens, idx)
        prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": user_prompt}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        prompts.append(prompt)
        input_token_counts.append(len(tokenizer.encode(prompt)))

    print(
        "CONCURRENCY_PROMPTS",
        json.dumps(
            {
                "count": len(prompts),
                "input_tokens_min": min(input_token_counts),
                "input_tokens_max": max(input_token_counts),
                "input_tokens": input_token_counts,
            },
            sort_keys=True,
        ),
        flush=True,
    )

    llm_kwargs = {
        "model": str(model_path),
        "trust_remote_code": True,
        "dtype": "bfloat16",
        "tensor_parallel_size": args.tensor_parallel_size,
        "max_num_seqs": args.max_num_seqs,
        "max_model_len": args.seq_len,
        "enable_prefix_caching": args.enable_prefix_caching,
        "enable_chunked_prefill": True,
        "max_num_batched_tokens": args.cte_bucket,
        "block_size": args.block_size,
        "additional_config": additional_config,
    }
    if args.mamba_cache_mode is not None:
        llm_kwargs["mamba_cache_mode"] = args.mamba_cache_mode

    print("CONCURRENCY_LLM_KWARGS", json.dumps({k: str(v) for k, v in llm_kwargs.items()}, sort_keys=True), flush=True)
    llm = LLM(**llm_kwargs)
    sampling = SamplingParams(
        temperature=0.0,
        top_k=1,
        max_tokens=args.max_tokens,
        ignore_eos=True,
    )

    warmup_start = time.perf_counter()
    warmup_outputs = llm.generate([prompts[0]], SamplingParams(temperature=0.0, top_k=1, max_tokens=8, ignore_eos=True))
    warmup_elapsed = time.perf_counter() - warmup_start
    print(
        "CONCURRENCY_WARMUP",
        json.dumps(
            {
                "elapsed_s": warmup_elapsed,
                "output_tokens": len(warmup_outputs[0].outputs[0].token_ids),
            },
            sort_keys=True,
        ),
        flush=True,
    )

    for concurrency in args.concurrency:
        batch_prompts = prompts[:concurrency]
        total_input_tokens = sum(input_token_counts[:concurrency])
        start = time.perf_counter()
        outputs = llm.generate(batch_prompts, sampling)
        elapsed = time.perf_counter() - start
        output_tokens = [len(output.outputs[0].token_ids) for output in outputs]
        result = {
            "concurrency": concurrency,
            "elapsed_s": elapsed,
            "request_per_s": concurrency / elapsed if elapsed > 0 else None,
            "total_input_tokens": total_input_tokens,
            "total_output_tokens": sum(output_tokens),
            "output_tokens_per_s": sum(output_tokens) / elapsed if elapsed > 0 else None,
            "input_tokens_per_s": total_input_tokens / elapsed if elapsed > 0 else None,
            "output_tokens": output_tokens,
        }
        print("CONCURRENCY_RESULT", json.dumps(result, sort_keys=True), flush=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
