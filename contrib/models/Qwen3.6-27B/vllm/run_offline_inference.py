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


def _parse_int_list(values: list[str] | None) -> list[int] | None:
    if values is None:
        return None
    tokens: list[str] = []
    for value in values:
        tokens.extend(value.replace(",", " ").split())
    return [int(token) for token in tokens]


def _cte_buckets(args: argparse.Namespace) -> list[int]:
    profile_buckets = {
        "short": [128, 256, 512, 1024],
        "general": [256, 512, 1024, 2048],
        "long": [4096, 8192, 16384, 32768],
        "262k": [256],
    }
    if args.cte_bucket_profile != "single":
        buckets = list(profile_buckets[args.cte_bucket_profile])
    else:
        buckets = _parse_int_list(args.cte_buckets) or [args.cte_bucket]
    buckets = sorted(set(buckets))
    if not buckets:
        raise ValueError("At least one CTE bucket is required")
    for bucket in buckets:
        if bucket <= 0:
            raise ValueError(f"CTE buckets must be positive, got {bucket}")
        if bucket % 128 != 0:
            raise ValueError(
                f"CTE bucket {bucket} is not 128-aligned; DeltaNet CTE uses 128-token chunks"
            )
    if buckets[-1] > args.seq_len:
        raise ValueError(
            f"Largest CTE bucket {buckets[-1]} exceeds --seq-len {args.seq_len}"
        )
    return buckets


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
) -> dict:
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


def _gdn_cte_kernel_from_env(env: dict[str, str] | None = None) -> str:
    env = env or os.environ
    if env.get("USE_PYTORCH_CHUNK") == "1":
        return "pytorch_chunk"
    if env.get("USE_NKI_FUSED", "1") != "0":
        return "fused_initial_state"
    if env.get("USE_NKI_CHUNKED") == "1":
        return "nki_chunked"
    if env.get("USE_NKI") == "1":
        return "nki_recurrent"
    return "fused_initial_state"


def _use_nki_fused_from_env(env: dict[str, str] | None = None) -> bool:
    return _gdn_cte_kernel_from_env(env) == "fused_initial_state"


def _hbm_usage_if_available():
    try:
        import torch_xla.core.xla_model as xm  # noqa: WPS433

        device = xm.xla_device()
        return xm.get_memory_info(device)
    except Exception:
        return None


def _prompt_token_count(model_path: str, prompt: str, tokenizer=None) -> int:
    try:
        if tokenizer is None:
            from transformers import AutoTokenizer  # noqa: WPS433

            tokenizer = AutoTokenizer.from_pretrained(
                model_path,
                trust_remote_code=True,
            )
        try:
            return len(tokenizer.encode(prompt, add_special_tokens=False))
        except TypeError:
            encoded = tokenizer(prompt, add_special_tokens=False)
            return len(encoded["input_ids"])
    except Exception:
        return len(prompt.split())


def _cte_attention_mask_path(args: argparse.Namespace, actual_prompt_len: int) -> str:
    if args.enable_vllm_chunked_prefill:
        return "neuron_chunked_prefill"
    if args.compact_cte_attention_mask:
        return "compact_2d"
    if actual_prompt_len > 2048:
        return "invalid_dense_long_guard"
    return "dense_4d_fallback"


def _cold_prefill_metrics(
    args: argparse.Namespace,
    actual_prompt_len: int,
    elapsed_seconds: float,
    generated_token_count: int | None = None,
    hbm_usage=None,
) -> dict:
    cte_buckets = _cte_buckets(args)
    bucket_work = _bucketed_prefill_work(
        actual_prompt_len,
        cte_buckets,
        chunked_prefill_enabled=args.enable_vllm_chunked_prefill,
    )
    selected_bucket = bucket_work["selected_cte_bucket"]
    bucket_work_tokens = bucket_work["bucket_work_tokens"]
    padding_tokens = bucket_work["padding_tokens"]
    latency_ms = elapsed_seconds * 1000.0
    actual_tok_per_s = (
        actual_prompt_len / elapsed_seconds if elapsed_seconds > 0 else None
    )
    bucket_tok_per_s = (
        bucket_work_tokens / elapsed_seconds if elapsed_seconds > 0 else None
    )
    generated_tok_per_s = (
        generated_token_count / elapsed_seconds
        if generated_token_count is not None and elapsed_seconds > 0
        else None
    )
    attention_mask_path = _cte_attention_mask_path(args, actual_prompt_len)

    return {
        "actual_prompt_len": actual_prompt_len,
        "selected_cte_bucket": selected_bucket,
        "selected_cte_buckets": bucket_work["selected_cte_buckets"],
        "num_cte_chunks": bucket_work["num_cte_chunks"],
        "bucket_work_tokens": bucket_work_tokens,
        "cte_buckets": cte_buckets,
        "cte_bucket_profile": args.cte_bucket_profile,
        "padding_tokens": padding_tokens,
        "padding_ratio": padding_tokens / bucket_work_tokens if bucket_work_tokens else 0.0,
        "tensor_parallel_size": args.tensor_parallel_size,
        "logical_nc_config": args.logical_nc_config,
        "max_num_seqs": args.max_num_seqs,
        "ctx_batch_size": args.ctx_batch_size,
        "block_size": args.block_size,
        "kernel_q_tile_size": args.kernel_q_tile_size,
        "kernel_kv_tile_size": args.kernel_kv_tile_size,
        "text_only_cte_enabled": bool(args.text_only_cte),
        "compact_mask_enabled": bool(args.compact_cte_attention_mask),
        "chunked_prefill_enabled": bool(args.enable_vllm_chunked_prefill),
        "cte_attention_mask_path": attention_mask_path,
        "dense_cte_mask_fallback": attention_mask_path == "dense_4d_fallback",
        "cold_zero_conv_fast_path_enabled": bool(args.cold_zero_conv_fast_path),
        "use_nki_fused": _use_nki_fused_from_env(),
        "gdn_cte_kernel": _gdn_cte_kernel_from_env(),
        "max_model_len": args.max_model_len,
        "seq_len": args.seq_len,
        "max_tokens": args.max_tokens,
        "prefill_latency_ms": latency_ms,
        "request_latency_ms": latency_ms,
        "first_token_latency_ms": latency_ms if args.max_tokens == 1 else None,
        "decode_tok_per_s": None,
        "generated_tokens": generated_token_count,
        "end_to_end_generated_tok_per_s": generated_tok_per_s,
        "actual_tok_per_s": actual_tok_per_s,
        "bucket_tok_per_s": bucket_tok_per_s,
        "hbm_usage": hbm_usage,
    }


def _float_attr(obj, *names: str) -> float | None:
    if obj is None:
        return None
    for name in names:
        value = getattr(obj, name, None)
        if value is not None:
            return float(value)
    return None


def _generation_metrics_from_vllm_output(
    output,
    *,
    request_start_time: float,
    elapsed_seconds: float,
    generated_token_count: int,
    max_tokens: int,
) -> dict:
    request_metrics = getattr(output, "metrics", None)
    arrival_time = _float_attr(
        request_metrics,
        "arrival_time",
        "request_start_time",
        "created_time",
    )
    first_token_time = _float_attr(request_metrics, "first_token_time")
    finished_time = _float_attr(
        request_metrics,
        "finished_time",
        "last_token_time",
        "last_step_time",
    )

    first_token_latency_s = None
    if first_token_time is not None:
        first_token_latency_s = first_token_time - (
            arrival_time if arrival_time is not None else request_start_time
        )
        first_token_latency_s = max(first_token_latency_s, 0.0)
    elif max_tokens == 1:
        first_token_latency_s = elapsed_seconds

    decode_tokens = max(generated_token_count - 1, 0)
    decode_latency_s = None
    if decode_tokens > 0:
        if first_token_time is not None and finished_time is not None:
            decode_latency_s = max(finished_time - first_token_time, 0.0)
        elif first_token_latency_s is not None:
            decode_latency_s = max(elapsed_seconds - first_token_latency_s, 0.0)

    return {
        "generated_tokens": generated_token_count,
        "max_tokens": max_tokens,
        "request_latency_ms": elapsed_seconds * 1000.0,
        "first_token_latency_ms": (
            first_token_latency_s * 1000.0
            if first_token_latency_s is not None
            else None
        ),
        "prefill_latency_ms": (
            first_token_latency_s * 1000.0
            if first_token_latency_s is not None
            else None
        ),
        "decode_tokens": decode_tokens,
        "decode_latency_ms": (
            decode_latency_s * 1000.0 if decode_latency_s is not None else None
        ),
        "decode_tok_per_s": (
            decode_tokens / decode_latency_s
            if decode_tokens > 0 and decode_latency_s and decode_latency_s > 0
            else None
        ),
        "end_to_end_generated_tok_per_s": (
            generated_token_count / elapsed_seconds if elapsed_seconds > 0 else None
        ),
        "metrics_source": (
            "vllm_request_metrics"
            if request_metrics is not None
            else "elapsed_single_token_fallback"
            if max_tokens == 1
            else "missing_vllm_request_metrics"
        ),
    }


def _validate_hybrid_apc_args(args: argparse.Namespace):
    if not args.enable_hybrid_apc:
        return
    if args.hybrid_cache_mode != "all":
        raise ValueError("--enable-hybrid-apc requires --hybrid-cache-mode all")
    if args.gdn_checkpoint_interval != args.block_size:
        raise ValueError(
            "--enable-hybrid-apc v0 requires --gdn-checkpoint-interval "
            "to equal --block-size"
        )
    args.enable_prefix_caching = True


def _override_config(args: argparse.Namespace) -> dict:
    _validate_hybrid_apc_args(args)
    cte_buckets = _cte_buckets(args)
    max_cte_bucket = cte_buckets[-1]
    recurrent_cache_dtype = (
        args.hybrid_gdn_recurrent_cache_dtype or args.gdn_recurrent_cache_dtype
    )
    conv_cache_dtype = args.hybrid_gdn_conv_cache_dtype or args.gdn_conv_cache_dtype
    neuron_config = {
        "tp_degree": args.tensor_parallel_size,
        "batch_size": args.max_num_seqs,
        "ctx_batch_size": args.ctx_batch_size,
        "tkg_batch_size": args.max_num_seqs,
        "seq_len": args.seq_len,
        "max_length": args.seq_len,
        "max_context_length": max_cte_bucket,
        "context_encoding_buckets": cte_buckets,
        "token_generation_buckets": [args.seq_len],
        "enable_bucketing": len(cte_buckets) > 1,
        "logical_nc_config": args.logical_nc_config,
        "torch_dtype": "bfloat16",
        "save_sharded_checkpoint": True,
        "pa_block_size": args.block_size,
    }
    if (
        args.enable_prefix_caching
        or args.enable_hybrid_apc
        or args.enable_vllm_chunked_prefill
    ):
        neuron_config["is_block_kv_layout"] = True
    if args.enable_prefix_caching or args.enable_hybrid_apc:
        neuron_config["is_prefix_caching"] = True
    if args.enable_vllm_chunked_prefill:
        neuron_config.update(
            {
                "chunked_prefill_config": {
                    "max_num_seqs": args.max_num_seqs,
                    "tkg_model_enabled": True,
                    "kernel_q_tile_size": args.kernel_q_tile_size,
                    "kernel_kv_tile_size": args.kernel_kv_tile_size,
                },
            }
        )
    return {
        "max_prompt_length": max_cte_bucket,
        "use_hybrid_apc_manager": args.enable_hybrid_apc,
        "use_text_only_cte_inputs": args.text_only_cte,
        "use_compact_cte_attention_mask": args.compact_cte_attention_mask,
        "use_cold_zero_conv_fast_path": args.cold_zero_conv_fast_path,
        "gdn_checkpoint_interval": args.gdn_checkpoint_interval,
        "max_gdn_checkpoint_slots": args.max_gdn_checkpoint_slots,
        "gdn_recurrent_cache_dtype": recurrent_cache_dtype,
        "gdn_conv_cache_dtype": conv_cache_dtype,
        "hybrid_recurrent_cache_dtype": recurrent_cache_dtype,
        "hybrid_conv_cache_dtype": conv_cache_dtype,
        "hybrid_cache_mode": args.hybrid_cache_mode,
        "hybrid_cache_prefix_boundary_only": args.hybrid_cache_prefix_boundary_only,
        "hybrid_cache_block_boundary_only": args.hybrid_cache_prefix_boundary_only,
        "hybrid_cache_validate_exact": args.hybrid_cache_validate_exact,
        "hybrid_apc_require_vllm_metadata": args.hybrid_apc_require_vllm_metadata,
        "hybrid_apc_allow_local_hash_fallback": not args.hybrid_apc_require_vllm_metadata,
        "hybrid_apc_require_attention_block_refs": args.hybrid_apc_require_vllm_metadata,
        "override_neuron_config": neuron_config,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", default=None)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--compiled-artifacts", default=None)
    parser.add_argument("--prompt", default="What is 17 * 23? Answer with the number only.")
    parser.add_argument("--prompt-file", type=Path, default=None)
    parser.add_argument("--chat", action="store_true")
    parser.add_argument("--enable-vllm-chunked-prefill", action="store_true")
    parser.add_argument("--enable-prefix-caching", action="store_true")
    parser.add_argument("--enable-hybrid-apc", action="store_true")
    parser.add_argument("--mamba-cache-mode", default=None)
    parser.add_argument("--mamba-cache-dtype", default=None)
    parser.add_argument("--mamba-ssm-cache-dtype", default=None)
    parser.add_argument("--gdn-checkpoint-interval", type=int, default=256)
    parser.add_argument("--max-gdn-checkpoint-slots", type=int, default=8)
    parser.add_argument("--gdn-recurrent-cache-dtype", default="float32")
    parser.add_argument("--gdn-conv-cache-dtype", default="bfloat16")
    parser.add_argument("--hybrid-gdn-recurrent-cache-dtype", default=None)
    parser.add_argument("--hybrid-gdn-conv-cache-dtype", default=None)
    parser.add_argument("--hybrid-cache-mode", default="all")
    parser.add_argument(
        "--hybrid-cache-prefix-boundary-only",
        "--hybrid-cache-block-boundary-only",
        dest="hybrid_cache_prefix_boundary_only",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--hybrid-cache-validate-exact", action="store_true")
    parser.add_argument(
        "--hybrid-apc-require-vllm-metadata",
        action="store_true",
        help=(
            "Require serving-provided vLLM cumulative prefix hashes and attention "
            "block refs instead of the local token-hash validation fallback."
        ),
    )
    parser.add_argument("--num-gpu-blocks-override", type=int, default=None)
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-k", type=int, default=1)
    parser.add_argument("--tensor-parallel-size", type=int, default=4)
    parser.add_argument("--logical-nc-config", type=int, default=2)
    parser.add_argument("--max-num-seqs", type=int, default=1)
    parser.add_argument("--ctx-batch-size", type=int, default=1)
    parser.add_argument("--max-model-len", type=int, default=512)
    parser.add_argument("--seq-len", type=int, default=512)
    parser.add_argument("--cte-bucket", type=int, default=512)
    parser.add_argument("--cte-buckets", nargs="+", default=None)
    parser.add_argument(
        "--cte-bucket-profile",
        choices=("single", "short", "general", "long", "262k"),
        default="single",
    )
    parser.add_argument("--block-size", type=int, default=128)
    parser.add_argument("--kernel-q-tile-size", type=int, default=128)
    parser.add_argument("--kernel-kv-tile-size", type=int, default=1024)
    parser.add_argument(
        "--text-only-cte",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--compact-cte-attention-mask",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--cold-zero-conv-fast-path",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    args = parser.parse_args()

    contrib_root = _contrib_root(args.repo_root)
    script_dir = Path(__file__).resolve().parent
    sys.path.insert(0, str(script_dir))
    sys.path.insert(0, str(contrib_root))
    os.environ["PYTHONPATH"] = (
        f"{script_dir}:{contrib_root}:{os.environ.get('PYTHONPATH', '')}"
    )
    os.environ.setdefault("VLLM_NEURON_FRAMEWORK", "neuronx-distributed-inference")
    os.environ.setdefault("VLLM_PLUGINS", "neuron")
    os.environ.setdefault("USE_NKI_FUSED", "1")
    os.environ.setdefault("USE_NKI_CHUNKED", "0")
    os.environ.setdefault("USE_PYTORCH_CHUNK", "0")
    if args.enable_vllm_chunked_prefill:
        os.environ["DISABLE_NEURON_CUSTOM_SCHEDULER"] = "1"
    if args.compiled_artifacts:
        os.environ["NEURON_COMPILED_ARTIFACTS"] = str(
            Path(args.compiled_artifacts).expanduser().resolve()
        )

    from hf_qwen35_config import register_qwen35_config  # noqa: WPS433

    register_qwen35_config()

    from vllm import LLM, SamplingParams  # noqa: WPS433

    prompt = (
        args.prompt_file.expanduser().read_text()
        if args.prompt_file is not None
        else args.prompt
    )
    tokenizer = None
    if args.chat:
        from transformers import AutoTokenizer  # noqa: WPS433

        tokenizer = AutoTokenizer.from_pretrained(
            args.model_path,
            trust_remote_code=True,
        )
        prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )

    additional_config = _override_config(args)
    print("VLLM_QWEN36_CONFIG", json.dumps(additional_config, sort_keys=True), flush=True)
    print("GDN_CTE_KERNEL", _gdn_cte_kernel_from_env(), flush=True)
    max_cte_bucket = max(_cte_buckets(args))
    actual_prompt_len = _prompt_token_count(
        args.model_path,
        prompt,
        tokenizer=tokenizer,
    )

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
    recurrent_cache_dtype = (
        args.hybrid_gdn_recurrent_cache_dtype or args.gdn_recurrent_cache_dtype
    )
    if args.enable_prefix_caching or args.enable_hybrid_apc:
        llm_kwargs["mamba_cache_mode"] = args.mamba_cache_mode or "all"
        llm_kwargs["mamba_ssm_cache_dtype"] = (
            args.mamba_ssm_cache_dtype or recurrent_cache_dtype
        )
    elif args.mamba_cache_mode is not None:
        llm_kwargs["mamba_cache_mode"] = args.mamba_cache_mode
    if args.mamba_cache_dtype is not None:
        llm_kwargs["mamba_cache_dtype"] = args.mamba_cache_dtype
    if (
        args.mamba_ssm_cache_dtype is not None
        and "mamba_ssm_cache_dtype" not in llm_kwargs
    ):
        llm_kwargs["mamba_ssm_cache_dtype"] = args.mamba_ssm_cache_dtype
    if args.num_gpu_blocks_override is not None:
        llm_kwargs["num_gpu_blocks_override"] = args.num_gpu_blocks_override
    if (
        args.enable_prefix_caching
        or args.enable_hybrid_apc
        or args.enable_vllm_chunked_prefill
    ):
        llm_kwargs["block_size"] = args.block_size
    if args.enable_vllm_chunked_prefill:
        llm_kwargs["max_num_batched_tokens"] = max_cte_bucket
    llm = LLM(**llm_kwargs)

    sampling = SamplingParams(
        temperature=args.temperature,
        top_k=args.top_k,
        max_tokens=args.max_tokens,
    )
    start = time.perf_counter()
    outputs = llm.generate([prompt], sampling)
    elapsed = time.perf_counter() - start
    text = outputs[0].outputs[0].text
    token_ids = outputs[0].outputs[0].token_ids
    generation_metrics = _generation_metrics_from_vllm_output(
        outputs[0],
        request_start_time=start,
        elapsed_seconds=elapsed,
        generated_token_count=len(token_ids),
        max_tokens=args.max_tokens,
    )
    metrics = _cold_prefill_metrics(
        args,
        actual_prompt_len=actual_prompt_len,
        elapsed_seconds=elapsed,
        generated_token_count=len(token_ids),
        hbm_usage=_hbm_usage_if_available(),
    )
    for key, value in generation_metrics.items():
        if metrics.get(key) is None:
            metrics[key] = value

    print("PROMPT", prompt)
    print("OUTPUT", text)
    print("TOKENS", list(token_ids))
    print("ELAPSED_SECONDS", f"{elapsed:.3f}")
    print("GENERATION_METRICS", json.dumps(generation_metrics, sort_keys=True), flush=True)
    print("COLD_PREFILL_METRICS", json.dumps(metrics, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
