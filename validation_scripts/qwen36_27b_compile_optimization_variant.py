#!/usr/bin/env python3
"""Compile Qwen3.6-27B 64K optimization variants.

This is intended to run on a Trn2 instance from the repository root. It keeps
the proven 64K hybrid/chunked-prefill settings fixed and varies only the
optimization knobs we want to measure.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
from pathlib import Path

import torch


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _load_text_config(model_path: Path) -> dict:
    with (model_path / "config.json").open() as f:
        full_config = json.load(f)
    text_config = full_config.get("text_config", full_config)
    config_dict = dict(text_config)
    config_dict["pad_token_id"] = text_config.get("eos_token_id", 248044)
    if "rope_parameters" in text_config:
        config_dict["rope_theta"] = text_config["rope_parameters"].get(
            "rope_theta", 10000000
        )
    config_dict.setdefault("tie_word_embeddings", False)
    return config_dict


def _variant_flags(name: str) -> dict[str, bool]:
    variants = {
        "baseline": {},
        "config": {
            "async_mode": True,
            "flash_decoding_enabled": True,
            "fused_qkv": True,
            "qkv_kernel_enabled": True,
            "qkv_nki_kernel_enabled": True,
        },
        "mlp": {
            "mlp_kernel_enabled": True,
        },
        "combined": {
            "async_mode": True,
            "flash_decoding_enabled": True,
            "fused_qkv": True,
            "qkv_kernel_enabled": True,
            "qkv_nki_kernel_enabled": True,
            "mlp_kernel_enabled": True,
        },
    }
    return variants[name]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--compiled-path", required=True)
    parser.add_argument(
        "--variant",
        choices=["baseline", "config", "mlp", "combined"],
        default="combined",
    )
    parser.add_argument("--seq-len", type=int, default=65536)
    parser.add_argument("--cte-bucket", type=int, default=512)
    parser.add_argument("--tp-degree", type=int, default=4)
    parser.add_argument("--logical-nc-config", type=int, default=2)
    parser.add_argument("--load-after-compile", action="store_true")
    args = parser.parse_args()

    repo = _repo_root()
    contrib_model_dir = repo / "contrib" / "models" / "Qwen3.6-27B"
    sys.path.insert(0, str(repo))
    sys.path.insert(0, str(contrib_model_dir))

    from neuronx_distributed_inference.models.config import (  # noqa: WPS433
        NeuronConfig,
        OnDeviceSamplingConfig,
    )
    from src.modeling_qwen35 import (  # noqa: WPS433
        NeuronQwen35ForCausalLM,
        Qwen35InferenceConfig,
    )

    model_path = Path(args.model_path)
    compiled_path = Path(args.compiled_path)

    flags = _variant_flags(args.variant)
    neuron_kwargs = {
        "tp_degree": args.tp_degree,
        "batch_size": 1,
        "ctx_batch_size": 1,
        "tkg_batch_size": 1,
        "seq_len": args.seq_len,
        "max_context_length": args.seq_len,
        "max_length": args.seq_len,
        "context_encoding_buckets": [args.cte_bucket],
        "torch_dtype": torch.bfloat16,
        "on_device_sampling_config": OnDeviceSamplingConfig(
            do_sample=False,
            top_k=1,
            top_p=1.0,
            temperature=1.0,
        ),
        "enable_bucketing": False,
        "logical_nc_config": args.logical_nc_config,
        "save_sharded_checkpoint": True,
        **flags,
    }
    neuron_config = NeuronConfig(**neuron_kwargs)

    config_dict = _load_text_config(model_path)
    # These are consumed by the 64K internal Qwen3.6 branch. They are harmless
    # on branches that do not implement the custom hybrid/chunked path.
    config_dict.setdefault("use_hybrid_cache_manager", True)
    config_dict.setdefault("use_qwen_hybrid_chunked_prefill", True)
    config_dict.setdefault("use_qwen_hybrid_chunked_prefill_nki", True)

    print("COMPILE_VARIANT", args.variant, flush=True)
    print("COMPILE_FLAGS", json.dumps(flags, sort_keys=True), flush=True)
    print("COMPILE_PATH", str(compiled_path), flush=True)

    inf_config = Qwen35InferenceConfig(neuron_config=neuron_config, **config_dict)
    model = NeuronQwen35ForCausalLM(str(model_path), inf_config)
    model.compile(str(compiled_path))
    del model
    gc.collect()

    if args.load_after_compile:
        model = NeuronQwen35ForCausalLM(str(compiled_path))
        model.load(str(compiled_path))
        print("LOAD_AFTER_COMPILE_OK", flush=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
