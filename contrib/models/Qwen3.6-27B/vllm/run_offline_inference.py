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


def _override_config(args: argparse.Namespace) -> dict:
    return {
        "override_neuron_config": {
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
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", default=None)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--compiled-artifacts", default=None)
    parser.add_argument("--prompt", default="What is 17 * 23? Answer with the number only.")
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-k", type=int, default=1)
    parser.add_argument("--tensor-parallel-size", type=int, default=4)
    parser.add_argument("--logical-nc-config", type=int, default=2)
    parser.add_argument("--max-num-seqs", type=int, default=1)
    parser.add_argument("--max-model-len", type=int, default=512)
    parser.add_argument("--seq-len", type=int, default=512)
    parser.add_argument("--cte-bucket", type=int, default=512)
    args = parser.parse_args()

    contrib_root = _contrib_root(args.repo_root)
    sys.path.insert(0, str(contrib_root))
    os.environ["PYTHONPATH"] = f"{contrib_root}:{os.environ.get('PYTHONPATH', '')}"
    os.environ.setdefault("VLLM_NEURON_FRAMEWORK", "neuronx-distributed-inference")
    if args.compiled_artifacts:
        os.environ["NEURON_COMPILED_ARTIFACTS"] = str(
            Path(args.compiled_artifacts).expanduser().resolve()
        )

    from vllm import LLM, SamplingParams  # noqa: WPS433

    additional_config = _override_config(args)
    print("VLLM_QWEN36_CONFIG", json.dumps(additional_config, sort_keys=True), flush=True)

    llm = LLM(
        model=str(Path(args.model_path).expanduser().resolve()),
        trust_remote_code=True,
        dtype="bfloat16",
        tensor_parallel_size=args.tensor_parallel_size,
        max_num_seqs=args.max_num_seqs,
        max_model_len=args.max_model_len,
        enable_prefix_caching=False,
        enable_chunked_prefill=False,
        additional_config=additional_config,
    )

    sampling = SamplingParams(
        temperature=args.temperature,
        top_k=args.top_k,
        max_tokens=args.max_tokens,
    )
    start = time.perf_counter()
    outputs = llm.generate([args.prompt], sampling)
    elapsed = time.perf_counter() - start
    text = outputs[0].outputs[0].text
    token_ids = outputs[0].outputs[0].token_ids

    print("PROMPT", args.prompt)
    print("OUTPUT", text)
    print("TOKENS", list(token_ids))
    print("ELAPSED_SECONDS", f"{elapsed:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
