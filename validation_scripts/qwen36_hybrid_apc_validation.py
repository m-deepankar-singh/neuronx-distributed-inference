#!/usr/bin/env python3
"""Trainium validation harness for Qwen3.6 hybrid APC.

This script is intentionally separate from unit tests because it expects a
Neuron/vLLM runtime and compiled artifacts. It covers two gates:

* token exactness for cold vs warm full-prefix and partial-prefix reuse;
* HBM planning for GDN checkpoint slot budgets.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace


REPO_ROOT = Path(__file__).resolve().parents[1]
QWEN_ROOT = REPO_ROOT / "contrib" / "models" / "Qwen3.6-27B"
RUNNER_PATH = QWEN_ROOT / "vllm" / "run_offline_inference.py"
HYBRID_APC_PATH = QWEN_ROOT / "src" / "hybrid_apc.py"


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _runner_args(args, *, enable_hybrid_apc: bool):
    return SimpleNamespace(
        cte_bucket=args.cte_bucket,
        cte_buckets=args.cte_buckets,
        cte_bucket_profile=args.cte_bucket_profile,
        seq_len=args.seq_len,
        tensor_parallel_size=args.tensor_parallel_size,
        max_num_seqs=1,
        ctx_batch_size=args.ctx_batch_size,
        logical_nc_config=args.logical_nc_config,
        block_size=args.block_size,
        enable_prefix_caching=enable_hybrid_apc,
        enable_hybrid_apc=enable_hybrid_apc,
        enable_vllm_chunked_prefill=args.enable_vllm_chunked_prefill,
        kernel_q_tile_size=args.kernel_q_tile_size,
        kernel_kv_tile_size=args.kernel_kv_tile_size,
        hybrid_gdn_recurrent_cache_dtype=None,
        gdn_recurrent_cache_dtype=args.gdn_recurrent_cache_dtype,
        hybrid_gdn_conv_cache_dtype=None,
        gdn_conv_cache_dtype=args.gdn_conv_cache_dtype,
        gdn_checkpoint_interval=args.gdn_checkpoint_interval,
        max_gdn_checkpoint_slots=args.max_gdn_checkpoint_slots,
        hybrid_cache_mode="all",
        hybrid_cache_prefix_boundary_only=True,
        hybrid_cache_validate_exact=True,
        text_only_cte=True,
        compact_cte_attention_mask=True,
        cold_zero_conv_fast_path=False,
    )


def _build_llm(args, *, enable_hybrid_apc: bool):
    sys.path.insert(0, str(QWEN_ROOT / "vllm"))
    sys.path.insert(0, str(QWEN_ROOT))
    os.environ["PYTHONPATH"] = (
        f"{QWEN_ROOT / 'vllm'}:{QWEN_ROOT}:{os.environ.get('PYTHONPATH', '')}"
    )
    os.environ.setdefault("VLLM_NEURON_FRAMEWORK", "neuronx-distributed-inference")
    os.environ.setdefault("VLLM_PLUGINS", "neuron")
    if args.enable_vllm_chunked_prefill:
        os.environ["DISABLE_NEURON_CUSTOM_SCHEDULER"] = "1"
    if args.compiled_artifacts:
        os.environ["NEURON_COMPILED_ARTIFACTS"] = str(
            Path(args.compiled_artifacts).expanduser().resolve()
        )

    runner = _load_module("qwen36_run_offline_inference_validation", RUNNER_PATH)
    from hf_qwen35_config import register_qwen35_config  # noqa: WPS433

    register_qwen35_config()
    from vllm import LLM, SamplingParams  # noqa: WPS433

    runner_args = _runner_args(args, enable_hybrid_apc=enable_hybrid_apc)
    additional_config = runner._override_config(runner_args)
    llm_kwargs = {
        "model": str(Path(args.model_path).expanduser().resolve()),
        "trust_remote_code": True,
        "dtype": "bfloat16",
        "tensor_parallel_size": args.tensor_parallel_size,
        "max_num_seqs": 1,
        "max_model_len": args.max_model_len,
        "enable_prefix_caching": enable_hybrid_apc,
        "enable_chunked_prefill": args.enable_vllm_chunked_prefill,
        "additional_config": additional_config,
    }
    if enable_hybrid_apc or args.enable_vllm_chunked_prefill:
        llm_kwargs["block_size"] = args.block_size
    if enable_hybrid_apc:
        llm_kwargs["mamba_cache_mode"] = "all"
        llm_kwargs["mamba_ssm_cache_dtype"] = args.gdn_recurrent_cache_dtype
    if args.enable_vllm_chunked_prefill:
        llm_kwargs["max_num_batched_tokens"] = max(runner._cte_buckets(runner_args))
    if args.num_gpu_blocks_override is not None:
        llm_kwargs["num_gpu_blocks_override"] = args.num_gpu_blocks_override
    sampling = SamplingParams(temperature=0.0, top_k=1, max_tokens=args.max_tokens)
    return LLM(**llm_kwargs), sampling


def _generate(llm, sampling, prompt: str):
    start = time.perf_counter()
    outputs = llm.generate([prompt], sampling)
    elapsed = time.perf_counter() - start
    token_ids = list(outputs[0].outputs[0].token_ids)
    return {"tokens": token_ids, "elapsed_seconds": elapsed}


def run_exactness(args) -> int:
    shared = args.shared_prefix
    prompt_a = shared + args.suffix_a
    prompt_b = shared + args.suffix_b

    cold_llm, sampling = _build_llm(args, enable_hybrid_apc=False)
    cold_full = _generate(cold_llm, sampling, prompt_a)
    cold_partial = _generate(cold_llm, sampling, prompt_b)

    warm_llm, warm_sampling = _build_llm(args, enable_hybrid_apc=True)
    warmup_full = _generate(warm_llm, warm_sampling, prompt_a)
    warm_full = _generate(warm_llm, warm_sampling, prompt_a)
    _warmup_partial = _generate(warm_llm, warm_sampling, prompt_a)
    warm_partial = _generate(warm_llm, warm_sampling, prompt_b)

    report = {
        "full_prefix_exact": cold_full["tokens"] == warm_full["tokens"],
        "partial_prefix_exact": cold_partial["tokens"] == warm_partial["tokens"],
        "cold_full": cold_full,
        "warmup_full": warmup_full,
        "warm_full": warm_full,
        "cold_partial": cold_partial,
        "warm_partial": warm_partial,
        "negative_tests": {
            "missing_gdn_state_fallback": "requires scheduler fault injection",
            "zeroed_conv_state": "requires model debug hook",
        },
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["full_prefix_exact"] and report["partial_prefix_exact"] else 1


def run_hbm(args) -> int:
    hybrid_apc = _load_module("qwen36_hybrid_apc_validation", HYBRID_APC_PATH)
    rows = []
    for context_len in args.context_lens:
        for interval in args.checkpoint_intervals:
            estimate = hybrid_apc.estimate_qwen_hybrid_cache_bytes_per_rank(
                max_context_len=context_len,
                checkpoint_interval=interval,
                recurrent_dtype=args.gdn_recurrent_cache_dtype,
                conv_dtype=args.gdn_conv_cache_dtype,
            )
            rows.append(
                {
                    "context_len": context_len,
                    "checkpoint_interval": interval,
                    "num_gdn_checkpoints": estimate["num_gdn_checkpoints"],
                    "attention_kv_gib": estimate["attention_kv_bytes"] / 2**30,
                    "gdn_checkpoint_gib": estimate["gdn_checkpoint_bytes"] / 2**30,
                    "total_gib": estimate["total_bytes"] / 2**30,
                    "bytes_per_gdn_checkpoint": estimate["gdn_bytes_per_checkpoint"],
                }
            )
    print(json.dumps(rows, indent=2, sort_keys=True))
    return 0


def parse_args():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    exact = subparsers.add_parser("exactness")
    exact.add_argument("--model-path", required=True)
    exact.add_argument("--compiled-artifacts")
    exact.add_argument("--max-model-len", type=int, default=2048)
    exact.add_argument("--seq-len", type=int, default=2048)
    exact.add_argument("--cte-bucket", type=int, default=512)
    exact.add_argument("--cte-buckets", nargs="+", default=["256,512"])
    exact.add_argument("--cte-bucket-profile", default="single")
    exact.add_argument("--tensor-parallel-size", type=int, default=4)
    exact.add_argument("--logical-nc-config", type=int, default=2)
    exact.add_argument("--ctx-batch-size", type=int, default=1)
    exact.add_argument("--block-size", type=int, default=256)
    exact.add_argument("--gdn-checkpoint-interval", type=int, default=256)
    exact.add_argument("--max-gdn-checkpoint-slots", type=int, default=8)
    exact.add_argument("--gdn-recurrent-cache-dtype", default="float32")
    exact.add_argument("--gdn-conv-cache-dtype", default="bfloat16")
    exact.add_argument("--enable-vllm-chunked-prefill", action="store_true")
    exact.add_argument("--kernel-q-tile-size", type=int, default=128)
    exact.add_argument("--kernel-kv-tile-size", type=int, default=1024)
    exact.add_argument("--num-gpu-blocks-override", type=int)
    exact.add_argument("--max-tokens", type=int, default=32)
    exact.add_argument("--shared-prefix", default="System: answer deterministically.\n" * 64)
    exact.add_argument("--suffix-a", default="\nUser: What is 17 * 23?\nAssistant:")
    exact.add_argument("--suffix-b", default="\nUser: What is 19 * 29?\nAssistant:")
    exact.set_defaults(func=run_exactness)

    hbm = subparsers.add_parser("hbm")
    hbm.add_argument("--context-lens", nargs="+", type=int, default=[131072, 262144])
    hbm.add_argument("--checkpoint-intervals", nargs="+", type=int, default=[128, 256, 512])
    hbm.add_argument("--gdn-recurrent-cache-dtype", default="float32")
    hbm.add_argument("--gdn-conv-cache-dtype", default="bfloat16")
    hbm.set_defaults(func=run_hbm)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
