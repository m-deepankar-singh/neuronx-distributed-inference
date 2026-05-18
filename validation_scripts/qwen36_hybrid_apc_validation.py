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
FP8_ENV_DEFAULTS = {
    "XLA_HANDLE_SPECIAL_SCALAR": "1",
    "UNSAFE_FP8FNCAST": "1",
}


def _ensure_fp8_environment() -> None:
    for name, value in FP8_ENV_DEFAULTS.items():
        os.environ.setdefault(name, value)


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _compiled_neuron_config(args) -> dict:
    if not args.compiled_artifacts:
        return {}
    config_path = Path(args.compiled_artifacts).expanduser() / "neuron_config.json"
    if not config_path.exists():
        return {}
    with config_path.open(encoding="utf-8") as handle:
        config = json.load(handle)
    nested = config.get("neuron_config")
    return nested if isinstance(nested, dict) else config


def _validate_generation_batch_support(args) -> None:
    if args.max_tokens <= 0 or args.max_num_seqs <= 1:
        return
    neuron_config = _compiled_neuron_config(args)
    if not neuron_config:
        return
    tkg_batch_size = int(
        neuron_config.get("tkg_batch_size")
        or neuron_config.get("batch_size")
        or neuron_config.get("max_batch_size")
        or 1
    )
    if args.max_num_seqs > tkg_batch_size:
        raise ValueError(
            "batched generation requires a compiled artifact with "
            f"tkg_batch_size >= --max-num-seqs; got tkg_batch_size={tkg_batch_size} "
            f"and max_num_seqs={args.max_num_seqs}"
        )
    ctx_batch_size = int(
        neuron_config.get("ctx_batch_size")
        or neuron_config.get("batch_size")
        or neuron_config.get("max_batch_size")
        or 1
    )
    if args.max_num_seqs > ctx_batch_size:
        raise ValueError(
            "batched generation requires a compiled artifact with "
            f"ctx_batch_size >= --max-num-seqs for grouped prefill host logits; "
            f"got ctx_batch_size={ctx_batch_size} and max_num_seqs={args.max_num_seqs}"
        )


def _parse_bucket_values(values) -> list[int]:
    buckets = []
    for value in values:
        for part in str(value).split(","):
            part = part.strip()
            if part:
                buckets.append(int(part))
    return sorted(set(buckets))


def _next_bucket(token_count: int, buckets: list[int]) -> int:
    for bucket in buckets:
        if token_count <= bucket:
            return bucket
    raise ValueError(
        f"prompt token length {token_count} exceeds compiled CTE buckets {buckets}"
    )


def _padding_token_id(tokenizer) -> int:
    for token_id in (tokenizer.pad_token_id, tokenizer.eos_token_id):
        if token_id is not None:
            return int(token_id)
    raise ValueError("tokenizer must define a pad_token_id or eos_token_id")


def _maybe_bucket_align_labeled_prompts(args, labeled_prompts):
    if not getattr(args, "align_prompts_to_cte_buckets", False):
        return labeled_prompts

    from transformers import AutoTokenizer  # noqa: WPS433

    tokenizer = AutoTokenizer.from_pretrained(
        str(Path(args.model_path).expanduser().resolve()),
        trust_remote_code=True,
    )
    buckets = _parse_bucket_values(args.cte_buckets)
    pad_token_id = _padding_token_id(tokenizer)
    aligned = []
    for label, prompt in labeled_prompts:
        prompt_token_ids = tokenizer.encode(prompt, add_special_tokens=False)
        bucket = _next_bucket(len(prompt_token_ids), buckets)
        aligned.append(
            (
                label,
                {
                    "prompt_token_ids": prompt_token_ids
                    + [pad_token_id] * (bucket - len(prompt_token_ids)),
                },
            )
        )
    return aligned


def _runner_args(args, *, enable_hybrid_apc: bool):
    return SimpleNamespace(
        cte_bucket=args.cte_bucket,
        cte_buckets=args.cte_buckets,
        cte_bucket_profile=args.cte_bucket_profile,
        seq_len=args.seq_len,
        tensor_parallel_size=args.tensor_parallel_size,
        max_num_seqs=args.max_num_seqs,
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
        hybrid_apc_reject_unbacked_attention_hits=getattr(
            args,
            "hybrid_apc_reject_unbacked_attention_hits",
            True,
        ),
        hybrid_apc_disable_unbacked_prefix_reads=getattr(
            args,
            "hybrid_apc_disable_unbacked_prefix_reads",
            False,
        ),
        hybrid_apc_enable_backed_prefix_reads=getattr(
            args,
            "hybrid_apc_enable_backed_prefix_reads",
            False,
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
    if args.hybrid_apc_disable_unbacked_prefix_reads:
        os.environ["QWEN36_HYBRID_APC_DISABLE_UNBACKED_PREFIX_READS"] = "1"
    if args.compiled_artifacts:
        os.environ["NEURON_COMPILED_ARTIFACTS"] = str(
            Path(args.compiled_artifacts).expanduser().resolve()
        )
        if not args.skip_fp8_env:
            _ensure_fp8_environment()

    runner = _load_module("qwen36_run_offline_inference_validation", RUNNER_PATH)
    from hf_qwen35_config import register_qwen35_config  # noqa: WPS433
    from qwen36_hybrid_apc_scheduler_patch import (  # noqa: WPS433
        install_import_hook as install_hybrid_apc_scheduler_patch,
    )

    register_qwen35_config()
    install_hybrid_apc_scheduler_patch()
    from vllm import LLM, SamplingParams  # noqa: WPS433

    runner_args = _runner_args(args, enable_hybrid_apc=enable_hybrid_apc)
    additional_config = runner._override_config(runner_args)
    llm_kwargs = {
        "model": str(Path(args.model_path).expanduser().resolve()),
        "trust_remote_code": True,
        "dtype": "bfloat16",
        "tensor_parallel_size": args.tensor_parallel_size,
        "max_num_seqs": args.max_num_seqs,
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


def _generate_many(llm, sampling, labeled_prompts):
    start = time.perf_counter()
    outputs = llm.generate([prompt for _label, prompt in labeled_prompts], sampling)
    elapsed = time.perf_counter() - start
    return {
        label: {
            "tokens": list(output.outputs[0].token_ids),
            "elapsed_seconds": elapsed,
        }
        for (label, _prompt), output in zip(labeled_prompts, outputs)
    }


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
        labeled_prompts = _maybe_bucket_align_labeled_prompts(args, labeled_prompts)
        results = {}
        for label, prompt in labeled_prompts:
            if os.environ.get("QWEN36_HYBRID_APC_DEBUG") == "1":
                prompt_len = (
                    len(prompt.get("prompt_token_ids", []))
                    if isinstance(prompt, dict)
                    else len(prompt)
                )
                print(
                    "[hybrid_apc_debug] generate "
                    f"label={label} enable_hybrid_apc={enable_hybrid_apc} "
                    f"prompt_len={prompt_len}",
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


def _generate_grouped_batch_worker(
    args_dict,
    enable_hybrid_apc: bool,
    labeled_prompt_groups,
    result_queue,
):
    llm = None
    try:
        args = argparse.Namespace(**args_dict)
        llm, sampling = _build_llm(args, enable_hybrid_apc=enable_hybrid_apc)
        results = {}
        for group in labeled_prompt_groups:
            group = _maybe_bucket_align_labeled_prompts(args, group)
            if os.environ.get("QWEN36_HYBRID_APC_DEBUG") == "1":
                print(
                    "[hybrid_apc_debug] generate-group "
                    f"labels={[label for label, _prompt in group]} "
                    f"enable_hybrid_apc={enable_hybrid_apc}",
                    flush=True,
                )
            if len(group) == 1:
                label, prompt = group[0]
                results[label] = _generate(llm, sampling, prompt)
            else:
                results.update(_generate_many(llm, sampling, group))
        result_queue.put({"ok": True, "results": results})
    except BaseException:
        result_queue.put({"ok": False, "traceback": traceback.format_exc()})
    finally:
        _shutdown_llm(llm)


def _generate_grouped_batch(args, *, enable_hybrid_apc: bool, labeled_prompt_groups):
    ctx = multiprocessing.get_context("spawn")
    result_queue = ctx.Queue()
    proc = ctx.Process(
        target=_generate_grouped_batch_worker,
        args=(vars(args), enable_hybrid_apc, labeled_prompt_groups, result_queue),
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


def _token_check(label: str, result: dict, dummy_token_ids: set[int]) -> dict:
    tokens = [int(token) for token in result.get("tokens", [])]
    unique_token_ids = sorted(set(tokens))
    non_dummy_tokens = [token for token in tokens if token not in dummy_token_ids]
    passed = bool(non_dummy_tokens)
    check = {
        "label": label,
        "generated_token_count": len(tokens),
        "unique_token_ids": unique_token_ids,
        "dummy_token_ids": sorted(dummy_token_ids),
        "non_dummy_token_count": len(non_dummy_tokens),
        "passed": passed,
    }
    if not passed:
        check["failure"] = "generated tokens are empty or all configured dummy tokens"
    return check


def _real_token_checks(results_by_label: dict[str, dict], dummy_token_ids: set[int]) -> dict:
    checks = {
        label: _token_check(label, result, dummy_token_ids)
        for label, result in sorted(results_by_label.items())
    }
    return {
        "passed": bool(checks) and all(check["passed"] for check in checks.values()),
        "checks": checks,
    }


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
    all_results = {
        "cold_full": cold_full,
        "warmup_full": warmup_full,
        "warm_full": warm_full,
        "cold_partial": cold_partial,
        "warm_partial": warm_partial,
    }
    real_token_checks = _real_token_checks(
        all_results,
        {int(token_id) for token_id in args.dummy_token_ids},
    )

    report = {
        "full_prefix_exact": cold_full["tokens"] == warm_full["tokens"],
        "partial_prefix_exact": cold_partial["tokens"] == warm_partial["tokens"],
        "cold_full": cold_full,
        "warmup_full": warmup_full,
        "warm_full": warm_full,
        "cold_partial": cold_partial,
        "warm_partial": warm_partial,
        "real_generated_tokens_required": args.require_real_tokens,
        "real_generated_tokens_passed": real_token_checks["passed"],
        "real_generated_token_checks": real_token_checks["checks"],
        "negative_tests": {
            "missing_gdn_state_fallback": "requires scheduler fault injection",
            "zeroed_conv_state": "requires model debug hook",
        },
    }
    if args.output_json:
        args.output_json.expanduser().write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    print(json.dumps(report, indent=2, sort_keys=True))
    passed = report["full_prefix_exact"] and report["partial_prefix_exact"]
    if args.require_real_tokens:
        passed = passed and real_token_checks["passed"]
    return 0 if passed else 1


def run_batched_exactness(args) -> int:
    if not args.shared_prefix_2:
        raise ValueError("--shared-prefix-2 is required for batched-exactness")
    _validate_generation_batch_support(args)

    prompt_full_a = args.shared_prefix + args.suffix_a
    prompt_partial_a = args.shared_prefix + args.suffix_b
    prompt_full_b = args.shared_prefix_2 + args.suffix_c
    prompt_partial_b = args.shared_prefix_2 + args.suffix_d

    cold_partial_a = _generate_batch(
        args,
        enable_hybrid_apc=True,
        labeled_prompts=[
            ("cold_partial_a", prompt_partial_a),
        ],
    )["cold_partial_a"]
    cold_partial_b = _generate_batch(
        args,
        enable_hybrid_apc=True,
        labeled_prompts=[
            ("cold_partial_b", prompt_partial_b),
        ],
    )["cold_partial_b"]
    warm_results = _generate_grouped_batch(
        args,
        enable_hybrid_apc=True,
        labeled_prompt_groups=[
            [("warmup_full_a", prompt_full_a)],
            [("warmup_full_b", prompt_full_b)],
            [
                ("warm_partial_a", prompt_partial_a),
                ("warm_partial_b", prompt_partial_b),
            ],
        ],
    )

    all_results = {
        "cold_partial_a": cold_partial_a,
        "cold_partial_b": cold_partial_b,
        **warm_results,
    }
    real_token_checks = _real_token_checks(
        all_results,
        {int(token_id) for token_id in args.dummy_token_ids},
    )
    report = {
        "batched_partial_a_exact": (
            cold_partial_a["tokens"] == warm_results["warm_partial_a"]["tokens"]
        ),
        "batched_partial_b_exact": (
            cold_partial_b["tokens"] == warm_results["warm_partial_b"]["tokens"]
        ),
        "max_num_seqs": args.max_num_seqs,
        "cold_partial_a": cold_partial_a,
        "cold_partial_b": cold_partial_b,
        "warmup_full_a": warm_results["warmup_full_a"],
        "warmup_full_b": warm_results["warmup_full_b"],
        "warm_partial_a": warm_results["warm_partial_a"],
        "warm_partial_b": warm_results["warm_partial_b"],
        "real_generated_tokens_required": args.require_real_tokens,
        "real_generated_tokens_passed": real_token_checks["passed"],
        "real_generated_token_checks": real_token_checks["checks"],
    }
    if args.output_json:
        args.output_json.expanduser().write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    print(json.dumps(report, indent=2, sort_keys=True))
    passed = report["batched_partial_a_exact"] and report["batched_partial_b_exact"]
    if args.require_real_tokens:
        passed = passed and real_token_checks["passed"]
    return 0 if passed else 1


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

    def add_common_exact_args(exact):
        exact.add_argument("--model-path", required=True)
        exact.add_argument("--compiled-artifacts")
        exact.add_argument(
            "--skip-fp8-env",
            action="store_true",
            help="Do not set FP8 runtime environment defaults for BF16 control artifacts.",
        )
        exact.add_argument("--max-model-len", type=int, default=2048)
        exact.add_argument("--seq-len", type=int, default=2048)
        exact.add_argument("--cte-bucket", type=int, default=512)
        exact.add_argument("--cte-buckets", nargs="+", default=["256,512"])
        exact.add_argument(
            "--align-prompts-to-cte-buckets",
            action="store_true",
            help=(
                "Tokenize prompts and pad token ids to the next compiled CTE bucket "
                "before calling vLLM. This is useful for static Neuron artifacts "
                "that reject non-bucket prompt shapes."
            ),
        )
        exact.add_argument("--cte-bucket-profile", default="single")
        exact.add_argument("--tensor-parallel-size", type=int, default=4)
        exact.add_argument("--max-num-seqs", type=int, default=1)
        exact.add_argument("--logical-nc-config", type=int, default=2)
        exact.add_argument("--ctx-batch-size", type=int, default=1)
        exact.add_argument("--block-size", type=int, default=256)
        exact.add_argument("--gdn-checkpoint-interval", type=int, default=256)
        exact.add_argument("--max-gdn-checkpoint-slots", type=int, default=8)
        exact.add_argument("--gdn-recurrent-cache-dtype", default="float32")
        exact.add_argument("--gdn-conv-cache-dtype", default="bfloat16")
        exact.add_argument("--hybrid-apc-require-vllm-metadata", action="store_true")
        exact.add_argument(
            "--hybrid-apc-reject-unbacked-attention-hits",
            action=argparse.BooleanOptionalAction,
            default=True,
        )
        exact.add_argument(
            "--hybrid-apc-disable-unbacked-prefix-reads",
            action=argparse.BooleanOptionalAction,
            default=False,
        )
        exact.add_argument(
            "--hybrid-apc-enable-backed-prefix-reads",
            action=argparse.BooleanOptionalAction,
            default=False,
        )
        exact.add_argument("--enable-vllm-chunked-prefill", action="store_true")
        exact.add_argument("--kernel-q-tile-size", type=int, default=128)
        exact.add_argument("--kernel-kv-tile-size", type=int, default=1024)
        exact.add_argument("--num-gpu-blocks-override", type=int)
        exact.add_argument("--max-tokens", type=int, default=32)
        exact.add_argument(
            "--shared-prefix",
            default="System: answer deterministically.\n" * 64,
        )
        exact.add_argument("--suffix-a", default="\nUser: What is 17 * 23?\nAssistant:")
        exact.add_argument("--suffix-b", default="\nUser: What is 19 * 29?\nAssistant:")
        exact.add_argument(
            "--require-real-tokens",
            action="store_true",
            help=(
                "Fail exactness if every generated token for any checked request is a "
                "configured dummy token."
            ),
        )
        exact.add_argument(
            "--dummy-token-ids",
            nargs="+",
            type=int,
            default=[0],
            help="Token ids treated as dummy generated output when --require-real-tokens is set.",
        )
        exact.add_argument("--output-json", type=Path)

    exact = subparsers.add_parser("exactness")
    add_common_exact_args(exact)
    exact.set_defaults(func=run_exactness)

    batched = subparsers.add_parser("batched-exactness")
    add_common_exact_args(batched)
    batched.add_argument("--shared-prefix-2", required=True)
    batched.add_argument("--suffix-c", default="")
    batched.add_argument("--suffix-d", default="\nUser: What is 23 * 31?\nAssistant:")
    batched.set_defaults(func=run_batched_exactness)

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
