#!/usr/bin/env python3
"""Offline partial-prefix APC validation for Qwen3.6 Neuron vLLM.

The exact-repeat APC check proves a full prompt hit. This harness checks the
more useful production case: request B should match a no-cache baseline after a
different request A has populated only the shared prefix blocks.

It uses subprocess workers so the no-cache and APC engines do not share Neuron
runtime state in one Python process.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
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


def shared_prefix(repeats: int) -> str:
    block = (
        "Reference dossier: Tokyo hosted the postponed 2020 Summer Olympics in 2021. "
        "The Games were held under COVID-19 restrictions, with limited spectators. "
        "This repeated paragraph is the shared prefix for prefix-cache validation. "
    )
    return block * repeats


def apply_chat_template(tokenizer: Any, prompt: str) -> str:
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )


def build_prompt(tokenizer: Any, args: argparse.Namespace, suffix: str) -> str:
    return apply_chat_template(tokenizer, shared_prefix(args.prefix_repeats) + suffix)


def configure_env(args: argparse.Namespace) -> Path:
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
    return root


def make_llm(args: argparse.Namespace, enable_prefix_caching: bool) -> Any:
    from vllm import LLM  # noqa: WPS433

    llm_kwargs: dict[str, Any] = {
        "model": str(Path(args.model_path).expanduser().resolve()),
        "trust_remote_code": True,
        "dtype": "bfloat16",
        "tensor_parallel_size": args.tensor_parallel_size,
        "max_num_seqs": args.max_num_seqs,
        "max_model_len": args.max_model_len,
        "enable_prefix_caching": enable_prefix_caching,
        "enable_chunked_prefill": args.enable_vllm_chunked_prefill,
        "additional_config": override_config(args),
    }
    if args.enable_vllm_chunked_prefill:
        llm_kwargs["max_num_batched_tokens"] = args.cte_bucket
        llm_kwargs["block_size"] = args.block_size
    if enable_prefix_caching and args.mamba_cache_mode is not None:
        llm_kwargs["mamba_cache_mode"] = args.mamba_cache_mode
    if enable_prefix_caching and args.mamba_cache_dtype is not None:
        llm_kwargs["mamba_cache_dtype"] = args.mamba_cache_dtype
    if enable_prefix_caching and args.mamba_ssm_cache_dtype is not None:
        llm_kwargs["mamba_ssm_cache_dtype"] = args.mamba_ssm_cache_dtype
    return LLM(**llm_kwargs)


def run_once(llm: Any, prompt: str, sampling: Any, name: str) -> RunResult:
    start = time.perf_counter()
    outputs = llm.generate([prompt], sampling)
    wall_s = time.perf_counter() - start
    output = outputs[0].outputs[0]
    result = RunResult(
        name=name,
        wall_s=wall_s,
        output_text=output.text,
        token_ids=list(output.token_ids),
    )
    print("PREFIX_CACHE_PARTIAL_CASE " + json.dumps(asdict(result), sort_keys=True), flush=True)
    return result


def worker_main(args: argparse.Namespace) -> int:
    configure_env(args)

    from hf_qwen35_config import register_qwen35_config  # noqa: WPS433

    register_qwen35_config()

    from transformers import AutoTokenizer  # noqa: WPS433
    from vllm import SamplingParams  # noqa: WPS433

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    sampling = SamplingParams(temperature=0.0, top_k=1, max_tokens=args.max_tokens)

    target_suffix = (
        "\nAnswer in one concise sentence: where were the 2020 Olympics held?"
    )
    warmup_suffix = (
        "\nAnswer in one concise sentence: what restriction shaped those Games?"
    )

    enable_prefix_caching = args.worker_mode == "apc"
    llm = make_llm(args, enable_prefix_caching=enable_prefix_caching)

    results: list[RunResult] = []
    if args.worker_mode == "apc":
        warmup_prompt = build_prompt(tokenizer, args, warmup_suffix)
        results.append(run_once(llm, warmup_prompt, sampling, "apc_prefix_warmup"))

    target_prompt = build_prompt(tokenizer, args, target_suffix)
    name = "apc_partial_hit" if args.worker_mode == "apc" else "baseline_no_cache"
    results.append(run_once(llm, target_prompt, sampling, name))

    Path(args.output_json).write_text(
        json.dumps([asdict(result) for result in results], indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return 0


def worker_command(args: argparse.Namespace, mode: str, output_json: Path) -> list[str]:
    cmd = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--worker-mode",
        mode,
        "--output-json",
        str(output_json),
        "--model-path",
        args.model_path,
        "--compiled-artifacts",
        args.compiled_artifacts,
        "--prefix-repeats",
        str(args.prefix_repeats),
        "--max-tokens",
        str(args.max_tokens),
        "--tensor-parallel-size",
        str(args.tensor_parallel_size),
        "--logical-nc-config",
        str(args.logical_nc_config),
        "--max-num-seqs",
        str(args.max_num_seqs),
        "--max-model-len",
        str(args.max_model_len),
        "--seq-len",
        str(args.seq_len),
        "--cte-bucket",
        str(args.cte_bucket),
        "--block-size",
        str(args.block_size),
    ]
    if args.repo_root is not None:
        cmd.extend(["--repo-root", args.repo_root])
    if args.enable_vllm_chunked_prefill:
        cmd.append("--enable-vllm-chunked-prefill")
    if args.mamba_cache_mode is not None:
        cmd.extend(["--mamba-cache-mode", args.mamba_cache_mode])
    if args.mamba_cache_dtype is not None:
        cmd.extend(["--mamba-cache-dtype", args.mamba_cache_dtype])
    if args.mamba_ssm_cache_dtype is not None:
        cmd.extend(["--mamba-ssm-cache-dtype", args.mamba_ssm_cache_dtype])
    return cmd


def run_worker(args: argparse.Namespace, mode: str, output_json: Path) -> list[dict[str, Any]]:
    cmd = worker_command(args, mode, output_json)
    print("PREFIX_CACHE_PARTIAL_WORKER " + json.dumps({"mode": mode, "cmd": cmd}, sort_keys=True), flush=True)
    subprocess.run(cmd, check=True)
    return json.loads(output_json.read_text(encoding="utf-8"))


def orchestrator_main(args: argparse.Namespace) -> int:
    with tempfile.TemporaryDirectory(prefix="qwen36_apc_partial_") as tmp:
        tmpdir = Path(tmp)
        baseline_rows = run_worker(args, "baseline", tmpdir / "baseline.json")
        apc_rows = run_worker(args, "apc", tmpdir / "apc.json")

    baseline = baseline_rows[-1]
    apc = apc_rows[-1]
    token_match = baseline["token_ids"] == apc["token_ids"]
    text_match = baseline["output_text"] == apc["output_text"]
    speedup = baseline["wall_s"] / apc["wall_s"] if apc["wall_s"] > 0 else 0.0
    summary = {
        "token_match": token_match,
        "text_match": text_match,
        "baseline_wall_s": baseline["wall_s"],
        "apc_partial_hit_wall_s": apc["wall_s"],
        "speedup": speedup,
        "baseline_tokens": baseline["token_ids"],
        "apc_tokens": apc["token_ids"],
    }
    print("PREFIX_CACHE_PARTIAL_SUMMARY " + json.dumps(summary, sort_keys=True), flush=True)

    if not token_match:
        raise SystemExit("partial-prefix cache changed greedy token IDs")
    if not args.no_fail_on_speed and speedup < args.min_speedup:
        raise SystemExit(f"partial-prefix speedup {speedup:.2f}x below {args.min_speedup:.2f}x")
    return 0


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
    parser.add_argument("--worker-mode", choices=["baseline", "apc"], default=None)
    parser.add_argument("--output-json", default=None)
    args = parser.parse_args()

    if args.worker_mode is not None:
        if args.output_json is None:
            raise SystemExit("--output-json is required in worker mode")
        return worker_main(args)
    return orchestrator_main(args)


if __name__ == "__main__":
    raise SystemExit(main())
