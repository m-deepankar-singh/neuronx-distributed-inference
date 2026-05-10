#!/usr/bin/env python3
"""Offline token-exact vLLM prefix-cache validation for Qwen3.6 Neuron.

Run this on the Trainium box. It constructs one vLLM engine with native prefix
caching enabled, sends the same long prompt twice, and compares generated token
IDs exactly. This is the first gate before implementing any Neuron-specific
GDN-state APC path.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass
class RunResult:
    name: str
    wall_s: float
    output_text: str
    token_ids: list[int]
    metrics: dict[str, Any]


def contrib_root(repo_root: str | None) -> Path:
    if repo_root:
        return Path(repo_root).expanduser().resolve() / "contrib" / "models" / "Qwen3.6-27B"
    return Path(__file__).resolve().parents[1] / "contrib" / "models" / "Qwen3.6-27B"


def override_config(args: argparse.Namespace) -> dict[str, Any]:
    neuron_config: dict[str, Any] = {
        "tp_degree": args.tensor_parallel_size,
        "batch_size": args.max_num_seqs,
        "ctx_batch_size": 1,
        "tkg_batch_size": args.max_num_seqs,
        "seq_len": args.seq_len,
        "max_length": args.seq_len,
        "max_context_length": args.cte_bucket,
        "context_encoding_buckets": [args.cte_bucket],
        "token_generation_buckets": [args.seq_len],
        "enable_bucketing": False,
        "logical_nc_config": args.logical_nc_config,
        "torch_dtype": "bfloat16",
        "save_sharded_checkpoint": True,
    }
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


def long_prompt(repeats: int) -> str:
    prefix = (
        "Reference dossier: Tokyo hosted the postponed 2020 Summer Olympics in 2021. "
        "The Games were held under COVID-19 restrictions, with limited spectators. "
        "This sentence is repeated to create a deterministic shared prefix. "
    )
    return (
        prefix * repeats
        + "\nAnswer in one concise sentence: where were the 2020 Olympics held?"
    )


def collect_metrics(llm: Any) -> dict[str, Any]:
    try:
        metrics = llm.llm_engine.get_metrics()
    except Exception as exc:  # pragma: no cover - version-dependent vLLM API
        return {"error": repr(exc)}

    records: dict[str, Any] = {}
    for metric in metrics:
        name = getattr(metric, "name", "")
        if "prefix_cache" in name:
            records[name] = getattr(metric, "value", None)
    return records


def run_once(llm: Any, prompt: str, sampling: Any, name: str) -> RunResult:
    start = time.perf_counter()
    outputs = llm.generate([prompt], sampling)
    wall_s = time.perf_counter() - start
    output = outputs[0].outputs[0]
    token_ids = list(output.token_ids)
    result = RunResult(
        name=name,
        wall_s=wall_s,
        output_text=output.text,
        token_ids=token_ids,
        metrics=collect_metrics(llm),
    )
    print("PREFIX_CACHE_OFFLINE_CASE " + json.dumps(asdict(result), sort_keys=True), flush=True)
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", default=None)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--compiled-artifacts", required=True)
    parser.add_argument("--prefix-repeats", type=int, default=220)
    parser.add_argument("--max-tokens", type=int, default=32)
    parser.add_argument("--min-speedup", type=float, default=1.25)
    parser.add_argument("--no-fail-on-speed", action="store_true")
    parser.add_argument("--mamba-cache-mode", default=None)
    parser.add_argument("--mamba-cache-dtype", default=None)
    parser.add_argument("--mamba-ssm-cache-dtype", default=None)
    parser.add_argument("--enable-vllm-chunked-prefill", action="store_true")
    parser.add_argument("--tensor-parallel-size", type=int, default=4)
    parser.add_argument("--logical-nc-config", type=int, default=2)
    parser.add_argument("--max-num-seqs", type=int, default=1)
    parser.add_argument("--max-model-len", type=int, default=131072)
    parser.add_argument("--seq-len", type=int, default=131072)
    parser.add_argument("--cte-bucket", type=int, default=512)
    parser.add_argument("--block-size", type=int, default=256)
    args = parser.parse_args()

    root = contrib_root(args.repo_root)
    vllm_dir = root / "vllm"
    sys.path.insert(0, str(vllm_dir))
    sys.path.insert(0, str(root))
    os.environ["PYTHONPATH"] = f"{vllm_dir}:{root}:{os.environ.get('PYTHONPATH', '')}"
    os.environ.setdefault("VLLM_NEURON_FRAMEWORK", "neuronx-distributed-inference")
    os.environ.setdefault("VLLM_PLUGINS", "neuron")
    if args.enable_vllm_chunked_prefill:
        os.environ["DISABLE_NEURON_CUSTOM_SCHEDULER"] = "1"
    os.environ["NEURON_COMPILED_ARTIFACTS"] = str(
        Path(args.compiled_artifacts).expanduser().resolve()
    )

    from hf_qwen35_config import register_qwen35_config  # noqa: WPS433

    register_qwen35_config()

    from transformers import AutoTokenizer  # noqa: WPS433
    from vllm import LLM, SamplingParams  # noqa: WPS433

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": long_prompt(args.prefix_repeats)}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )

    additional_config = override_config(args)
    llm_kwargs: dict[str, Any] = {
        "model": str(Path(args.model_path).expanduser().resolve()),
        "trust_remote_code": True,
        "dtype": "bfloat16",
        "tensor_parallel_size": args.tensor_parallel_size,
        "max_num_seqs": args.max_num_seqs,
        "max_model_len": args.max_model_len,
        "enable_prefix_caching": True,
        "enable_chunked_prefill": args.enable_vllm_chunked_prefill,
        "additional_config": additional_config,
    }
    if args.enable_vllm_chunked_prefill:
        llm_kwargs["max_num_batched_tokens"] = args.cte_bucket
        llm_kwargs["block_size"] = args.block_size
    if args.mamba_cache_mode is not None:
        llm_kwargs["mamba_cache_mode"] = args.mamba_cache_mode
    if args.mamba_cache_dtype is not None:
        llm_kwargs["mamba_cache_dtype"] = args.mamba_cache_dtype
    if args.mamba_ssm_cache_dtype is not None:
        llm_kwargs["mamba_ssm_cache_dtype"] = args.mamba_ssm_cache_dtype

    print("PREFIX_CACHE_OFFLINE_CONFIG " + json.dumps(llm_kwargs, default=str, sort_keys=True), flush=True)
    llm = LLM(**llm_kwargs)
    sampling = SamplingParams(temperature=0.0, top_k=1, max_tokens=args.max_tokens)

    cold = run_once(llm, prompt, sampling, "cold_fill")
    warm = run_once(llm, prompt, sampling, "warm_hit")

    token_match = cold.token_ids == warm.token_ids
    speedup = cold.wall_s / warm.wall_s if warm.wall_s > 0 else 0.0
    summary = {
        "token_match": token_match,
        "text_match": cold.output_text == warm.output_text,
        "cold_wall_s": cold.wall_s,
        "warm_wall_s": warm.wall_s,
        "speedup": speedup,
        "cold_metrics": cold.metrics,
        "warm_metrics": warm.metrics,
    }
    print("PREFIX_CACHE_OFFLINE_SUMMARY " + json.dumps(summary, sort_keys=True), flush=True)

    if not token_match:
        raise SystemExit("prefix cache changed greedy token IDs")
    if not args.no_fail_on_speed and speedup < args.min_speedup:
        raise SystemExit(f"prefix cache speedup {speedup:.2f}x below {args.min_speedup:.2f}x")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
