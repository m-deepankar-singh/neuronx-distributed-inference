#!/usr/bin/env python3
"""Trainium validation harness for Qwen3.6 hybrid APC.

This script is intentionally separate from unit tests because it expects a
Neuron/vLLM runtime and compiled artifacts. It covers two gates:

* token exactness for cold vs warm full-prefix and partial-prefix reuse;
* HBM planning for GDN checkpoint slot budgets.
"""

from __future__ import annotations

import argparse
import gc
import importlib.util
import json
import multiprocessing
import os
import queue
import sys
import time
import traceback
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
        num_gpu_blocks_override=args.num_gpu_blocks_override,
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
        hybrid_apc_require_vllm_metadata=getattr(
            args, "hybrid_apc_require_vllm_metadata", False
        ),
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
    if (
        runner_args.enable_prefix_caching
        or runner_args.enable_hybrid_apc
        or runner_args.enable_vllm_chunked_prefill
    ):
        llm_kwargs["num_gpu_blocks_override"] = runner._pa_num_blocks(runner_args)
    sampling = SamplingParams(temperature=0.0, top_k=1, max_tokens=args.max_tokens)
    return LLM(**llm_kwargs), sampling


def _generate(llm, sampling, prompt: str):
    start = time.perf_counter()
    outputs = llm.generate([prompt], sampling)
    elapsed = time.perf_counter() - start
    token_ids = list(outputs[0].outputs[0].token_ids)
    return {"tokens": token_ids, "elapsed_seconds": elapsed}


def _shutdown_llm(llm) -> None:
    if llm is None:
        return
    for target in (
        llm,
        getattr(llm, "llm_engine", None),
        getattr(getattr(llm, "llm_engine", None), "engine_core", None),
        getattr(getattr(llm, "llm_engine", None), "engine_core_client", None),
    ):
        shutdown = getattr(target, "shutdown", None)
        if shutdown is None:
            continue
        try:
            shutdown()
        except Exception:
            pass
    del llm
    gc.collect()


def _generate_batch_worker(args_dict, enable_hybrid_apc: bool, labeled_prompts, result_queue):
    llm = None
    try:
        args = argparse.Namespace(**args_dict)
        llm, sampling = _build_llm(args, enable_hybrid_apc=enable_hybrid_apc)
        results = {}
        for label, prompt in labeled_prompts:
            if os.environ.get("QWEN36_HYBRID_APC_DEBUG") == "1":
                print(
                    "[hybrid_apc_debug] generate "
                    f"label={label} enable_hybrid_apc={enable_hybrid_apc} "
                    f"prompt_chars={len(prompt)}",
                    flush=True,
                )
            results[label] = _generate(llm, sampling, prompt)
        result_queue.put({"ok": True, "results": results})
    except BaseException:
        result_queue.put({"ok": False, "traceback": traceback.format_exc()})
    finally:
        _shutdown_llm(llm)


def _generate_batch(args, *, enable_hybrid_apc: bool, labeled_prompts):
    ctx = multiprocessing.get_context("spawn")
    result_queue = ctx.Queue()
    proc = ctx.Process(
        target=_generate_batch_worker,
        args=(vars(args), enable_hybrid_apc, labeled_prompts, result_queue),
    )
    proc.start()
    proc.join()

    try:
        message = result_queue.get(timeout=1.0)
    except queue.Empty as exc:
        raise RuntimeError(
            f"generation worker exited with code {proc.exitcode} without a report"
        ) from exc
    if not message["ok"]:
        raise RuntimeError(message["traceback"])
    if proc.exitcode not in (0, None):
        raise RuntimeError(f"generation worker exited with code {proc.exitcode}")
    return message["results"]


def run_exactness(args) -> int:
    shared = args.shared_prefix
    prompt_a = shared + args.suffix_a
    prompt_b = shared + args.suffix_b

    # This validation uses the v3 vLLM APC artifact, which is compiled for
    # prefix/block KV layout. Keep prefix metadata enabled even for cold
    # references, and isolate each cold prompt in a fresh process so it cannot
    # observe cache state from the other reference prompt.
    cold_full = _generate_batch(
        args,
        enable_hybrid_apc=True,
        labeled_prompts=[
            ("cold_full", prompt_a),
        ],
    )["cold_full"]
    cold_partial = _generate_batch(
        args,
        enable_hybrid_apc=True,
        labeled_prompts=[
            ("cold_partial", prompt_b),
        ],
    )["cold_partial"]
    warm_results = _generate_batch(
        args,
        enable_hybrid_apc=True,
        labeled_prompts=[
            ("warmup_full", prompt_a),
            ("warm_full", prompt_a),
            ("warmup_partial", prompt_a),
            ("warm_partial", prompt_b),
        ],
    )

    warmup_full = warm_results["warmup_full"]
    warm_full = warm_results["warm_full"]
    warm_partial = warm_results["warm_partial"]

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
    exact.add_argument("--hybrid-apc-require-vllm-metadata", action="store_true")
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
