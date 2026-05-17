#!/usr/bin/env python3
"""Compile Qwen3.6-27B 64K with scoped weight-mode ablations.

This script intentionally starts from the validated 64K hybrid/chunked-prefill
baseline and changes only weight quantization. Supported modes:

* ``fp8_mlp_only``: MLP linear weights are converted to FP8 while attention,
  DeltaNet, normalization, embeddings, lm_head, KV cache, and recurrent state
  remain BF16.
* ``bf16_control``: no FP8 conversion; this is the real-token host-logits
  control for separating FP8 conversion failures from serving/logits failures.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
from pathlib import Path

import torch


_FP8_ENV_DEFAULTS = {
    "XLA_HANDLE_SPECIAL_SCALAR": "1",
    "UNSAFE_FP8FNCAST": "1",
}

_WEIGHT_DTYPE_FP8_MLP_ONLY = "fp8_mlp_only"
_WEIGHT_DTYPE_BF16_CONTROL = "bf16_control"


def _ensure_fp8_environment() -> None:
    for name, value in _FP8_ENV_DEFAULTS.items():
        os.environ.setdefault(name, value)


def _repo_root(path: str | None) -> Path:
    if path:
        return Path(path).expanduser().resolve()
    return Path(__file__).resolve().parents[5]


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


def _parse_int_list(values: list[str] | None) -> list[int] | None:
    if values is None:
        return None
    tokens: list[str] = []
    for value in values:
        tokens.extend(value.replace(",", " ").split())
    return [int(token) for token in tokens]


def _cte_buckets(args: argparse.Namespace) -> list[int]:
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


def _prefix_buckets(args: argparse.Namespace, cte_buckets: list[int]) -> list[int]:
    buckets = _parse_int_list(args.prefix_buckets) or cte_buckets
    buckets = sorted(set(buckets))
    if not buckets:
        raise ValueError("At least one prefix bucket is required")
    for bucket in buckets:
        if bucket <= 0:
            raise ValueError(f"Prefix buckets must be positive, got {bucket}")
        if bucket % args.block_size != 0:
            raise ValueError(
                f"Prefix bucket {bucket} must be divisible by block size {args.block_size}"
            )
    if buckets[-1] > args.seq_len:
        raise ValueError(
            f"Largest prefix bucket {buckets[-1]} exceeds --seq-len {args.seq_len}"
        )
    return buckets


def _pa_num_blocks(args: argparse.Namespace) -> int:
    min_blocks = max(1, (args.seq_len + args.block_size - 1) // args.block_size)
    if args.pa_num_blocks is None:
        requested_blocks = min_blocks
    else:
        requested_blocks = args.pa_num_blocks
    if requested_blocks < min_blocks:
        raise ValueError(
            f"--pa-num-blocks {requested_blocks} is too small for seq_len="
            f"{args.seq_len} and block_size={args.block_size}; need at least {min_blocks}"
        )
    # vLLM Neuron reserves one additional null block at runtime. Compile the
    # physical PA table with the same extra block so block ids stay in-bounds.
    return requested_blocks + 1


def _mlp_only_modules_to_not_convert(num_layers: int) -> list[str]:
    """Exclude numerically sensitive or unsupported modules from FP8 conversion."""
    modules = [
        "embed_tokens",
        "model.embed_tokens",
        "lm_head",
        "norm",
        "model.norm",
        "rotary_emb",
        "model.rotary_emb",
    ]
    for layer_idx in range(num_layers):
        for prefix in ("layers", "model.layers"):
            modules.extend(
                [
                    f"{prefix}.{layer_idx}.self_attn",
                    f"{prefix}.{layer_idx}.linear_attn",
                    f"{prefix}.{layer_idx}.input_layernorm",
                    f"{prefix}.{layer_idx}.post_attention_layernorm",
                ]
            )
    return modules


def _quantized_checkpoint_ready(path: Path) -> bool:
    if path.is_file():
        return True
    if path.is_dir():
        return any(path.iterdir())
    return False


def _is_mlp_weight(name: str) -> bool:
    parts = name.split(".")
    return (
        len(parts) >= 4
        and parts[-3] == "mlp"
        and parts[-2] in {"gate_proj", "up_proj", "down_proj"}
        and parts[-1] == "weight"
    )


def _scale_name(weight_name: str) -> str:
    return weight_name[: -len(".weight")] + ".weight_scale"


def _clear_quantized_checkpoint_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    for child in path.iterdir():
        if child.name.endswith(".safetensors") or child.name.endswith(".json"):
            child.unlink()


def _save_mlp_only_fp8_state_dict(model_path: Path, output_path: Path) -> None:
    """Create a sharded FP8 checkpoint directly from HF safetensors.

    Loading the HF architecture requires a newer Transformers than the Neuron
    venv uses internally. For this MLP-only ablation, we do not need model
    execution: the checkpoint transform is a direct tensor rewrite.
    """
    from safetensors.torch import load_file, save_file  # noqa: WPS433
    from neuronx_distributed.quantization.quantization_utils import (  # noqa: WPS433
        quantize_fp8_per_channel,
    )

    index_path = model_path / "model.safetensors.index.json"
    if index_path.exists():
        with index_path.open() as f:
            source_index = json.load(f)
        source_weight_map = source_index["weight_map"]
        filenames = sorted(set(source_weight_map.values()))
    elif (model_path / "model.safetensors").exists():
        source_weight_map = None
        filenames = ["model.safetensors"]
    else:
        raise FileNotFoundError(f"No safetensors checkpoint found in {model_path}")

    _clear_quantized_checkpoint_dir(output_path)
    output_weight_map: dict[str, str] = {}
    total_size = 0
    quantized_count = 0

    for filename in filenames:
        shard = load_file(str(model_path / filename))
        output_shard = {}
        for name, tensor in shard.items():
            if _is_mlp_weight(name):
                weight, scale = quantize_fp8_per_channel(
                    tensor,
                    torch.float8_e4m3fn,
                    channel_axis=0,
                )
                output_shard[name] = weight
                output_shard[_scale_name(name)] = scale
                output_weight_map[_scale_name(name)] = filename
                total_size += weight.numel() * weight.element_size()
                total_size += scale.numel() * scale.element_size()
                quantized_count += 1
            else:
                output_shard[name] = tensor
                total_size += tensor.numel() * tensor.element_size()
            output_weight_map[name] = filename

        save_file(output_shard, str(output_path / filename), metadata={"format": "pt"})
        del shard
        del output_shard
        gc.collect()

    if source_weight_map is not None:
        with (output_path / "model.safetensors.index.json").open("w") as f:
            json.dump(
                {
                    "metadata": {"total_size": total_size},
                    "weight_map": output_weight_map,
                },
                f,
                indent=2,
                sort_keys=True,
            )

    print("MANUAL_FP8_MLP_WEIGHT_COUNT", quantized_count, flush=True)


def _build_config(args: argparse.Namespace):
    from neuronx_distributed_inference.models.config import (  # noqa: WPS433
        ChunkedPrefillConfig,
        NeuronConfig,
        OnDeviceSamplingConfig,
    )
    from src.modeling_qwen35 import Qwen35InferenceConfig  # noqa: WPS433

    model_path = Path(args.model_path).expanduser().resolve()
    config_dict = _load_text_config(model_path)
    num_layers = int(config_dict["num_hidden_layers"])
    modules_to_not_convert = _mlp_only_modules_to_not_convert(num_layers)
    cte_buckets = _cte_buckets(args)
    max_cte_bucket = cte_buckets[-1]
    prefix_buckets = _prefix_buckets(args, cte_buckets)

    neuron_config_kwargs = {
        "tp_degree": args.tp_degree,
        "batch_size": 1,
        "ctx_batch_size": 1,
        "tkg_batch_size": 1,
        "seq_len": args.seq_len,
        "max_context_length": max_cte_bucket,
        "max_length": args.seq_len,
        "context_encoding_buckets": cte_buckets,
        "token_generation_buckets": [args.seq_len],
        "torch_dtype": torch.bfloat16,
        "enable_bucketing": len(cte_buckets) > 1,
        "logical_nc_config": args.logical_nc_config,
        "save_sharded_checkpoint": True,
    }
    if args.weight_dtype == _WEIGHT_DTYPE_FP8_MLP_ONLY:
        neuron_config_kwargs.update(
            {
                "quantized": True,
                "quantized_checkpoints_path": str(
                    Path(args.quantized_checkpoints_path).expanduser().resolve()
                ),
                "quantization_type": "per_channel_symmetric",
                "quantization_dtype": "f8e4m3",
                "modules_to_not_convert": modules_to_not_convert,
                "kv_cache_quant": False,
                "quantized_mlp_kernel_enabled": False,
                "activation_quantization_type": None,
            }
        )
    else:
        neuron_config_kwargs["quantized"] = False
    if args.disable_on_device_sampling:
        # vLLM/host-side sampling consumes logits from the Neuron trace. Without
        # logits, the serving path can only surface placeholder token ids.
        neuron_config_kwargs["output_logits"] = True
    else:
        neuron_config_kwargs["on_device_sampling_config"] = OnDeviceSamplingConfig(
            do_sample=False,
            top_k=1,
            top_p=1.0,
            temperature=1.0,
        )
    if args.enable_prefix_caching or args.enable_hybrid_apc or args.enable_vllm_chunked_prefill:
        neuron_config_kwargs["is_block_kv_layout"] = True
        neuron_config_kwargs["pa_block_size"] = args.block_size
        neuron_config_kwargs["pa_num_blocks"] = _pa_num_blocks(args)
    if args.enable_prefix_caching or args.enable_hybrid_apc:
        neuron_config_kwargs["is_prefix_caching"] = True
        neuron_config_kwargs["prefix_buckets"] = prefix_buckets
    if args.enable_vllm_chunked_prefill:
        neuron_config_kwargs["chunked_prefill_config"] = ChunkedPrefillConfig(
            max_num_seqs=1,
            tkg_model_enabled=True,
            kernel_q_tile_size=args.kernel_q_tile_size,
            kernel_kv_tile_size=args.kernel_kv_tile_size,
        )

    neuron_config = NeuronConfig(**neuron_config_kwargs)

    if args.disable_static_hybrid_cache or args.enable_prefix_caching or args.enable_hybrid_apc:
        config_dict["use_hybrid_cache_manager"] = False
    else:
        config_dict.setdefault("use_hybrid_cache_manager", True)
    config_dict["use_hybrid_apc_manager"] = args.enable_hybrid_apc
    config_dict["gdn_checkpoint_interval"] = args.gdn_checkpoint_interval
    config_dict["max_gdn_checkpoint_slots"] = args.max_gdn_checkpoint_slots
    config_dict["gdn_recurrent_cache_dtype"] = args.gdn_recurrent_cache_dtype
    config_dict["gdn_conv_cache_dtype"] = args.gdn_conv_cache_dtype
    config_dict["hybrid_recurrent_cache_dtype"] = args.gdn_recurrent_cache_dtype
    config_dict["hybrid_conv_cache_dtype"] = args.gdn_conv_cache_dtype
    config_dict["hybrid_cache_mode"] = args.hybrid_cache_mode
    config_dict["hybrid_apc_require_vllm_metadata"] = args.hybrid_apc_require_vllm_metadata
    config_dict["hybrid_apc_allow_local_hash_fallback"] = (
        not args.hybrid_apc_require_vllm_metadata
    )
    config_dict["hybrid_apc_require_attention_block_refs"] = (
        args.hybrid_apc_require_vllm_metadata
    )
    config_dict["use_qwen_hybrid_chunked_prefill"] = args.enable_vllm_chunked_prefill
    config_dict["use_qwen_hybrid_chunked_prefill_nki"] = args.enable_vllm_chunked_prefill

    inf_config = Qwen35InferenceConfig(neuron_config=neuron_config, **config_dict)
    return inf_config, modules_to_not_convert


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", default=None)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--compiled-path", required=True)
    parser.add_argument("--quantized-checkpoints-path")
    parser.add_argument(
        "--weight-dtype",
        choices=[_WEIGHT_DTYPE_FP8_MLP_ONLY, _WEIGHT_DTYPE_BF16_CONTROL],
        default=_WEIGHT_DTYPE_FP8_MLP_ONLY,
        help=(
            "Weight mode to compile. Use bf16_control for the non-FP8 "
            "host-logits real-token control."
        ),
    )
    parser.add_argument("--seq-len", type=int, default=65536)
    parser.add_argument("--cte-bucket", type=int, default=512)
    parser.add_argument("--cte-buckets", nargs="+", default=None)
    parser.add_argument("--prefix-buckets", nargs="+", default=None)
    parser.add_argument("--block-size", type=int, default=256)
    parser.add_argument("--pa-num-blocks", type=int, default=None)
    parser.add_argument("--tp-degree", type=int, default=4)
    parser.add_argument("--logical-nc-config", type=int, default=2)
    parser.add_argument("--enable-prefix-caching", action="store_true")
    parser.add_argument("--enable-hybrid-apc", action="store_true")
    parser.add_argument("--enable-vllm-chunked-prefill", action="store_true")
    parser.add_argument("--disable-on-device-sampling", action="store_true")
    parser.add_argument("--kernel-q-tile-size", type=int, default=128)
    parser.add_argument("--kernel-kv-tile-size", type=int, default=1024)
    parser.add_argument("--disable-static-hybrid-cache", action="store_true")
    parser.add_argument("--gdn-checkpoint-interval", type=int, default=256)
    parser.add_argument("--max-gdn-checkpoint-slots", type=int, default=8)
    parser.add_argument("--gdn-recurrent-cache-dtype", default="float32")
    parser.add_argument("--gdn-conv-cache-dtype", default="bfloat16")
    parser.add_argument("--hybrid-cache-mode", default="all")
    parser.add_argument("--hybrid-apc-require-vllm-metadata", action="store_true")
    parser.add_argument("--force-quantize", action="store_true")
    parser.add_argument("--quantize-only", action="store_true")
    parser.add_argument("--load-after-compile", action="store_true")
    args = parser.parse_args()
    if (
        args.weight_dtype == _WEIGHT_DTYPE_FP8_MLP_ONLY
        and not args.quantized_checkpoints_path
    ):
        parser.error("--quantized-checkpoints-path is required for fp8_mlp_only")

    repo = _repo_root(args.repo_root)
    contrib_model_dir = repo / "contrib" / "models" / "Qwen3.6-27B"
    sys.path.insert(0, str(repo))
    sys.path.insert(0, str(contrib_model_dir))
    if args.weight_dtype == _WEIGHT_DTYPE_FP8_MLP_ONLY:
        _ensure_fp8_environment()

    from src.modeling_qwen35 import NeuronQwen35ForCausalLM  # noqa: WPS433

    model_path = Path(args.model_path).expanduser().resolve()
    compiled_path = Path(args.compiled_path).expanduser().resolve()
    quantized_path = (
        Path(args.quantized_checkpoints_path).expanduser().resolve()
        if args.quantized_checkpoints_path
        else None
    )

    inf_config, modules_to_not_convert = _build_config(args)

    print("WEIGHT_DTYPE_MODE", args.weight_dtype, flush=True)
    if args.weight_dtype == _WEIGHT_DTYPE_FP8_MLP_ONLY:
        print("FP8_MODE mlp_only", flush=True)
    else:
        print("FP8_MODE disabled_bf16_control", flush=True)
    print("MODEL_PATH", str(model_path), flush=True)
    print("COMPILED_PATH", str(compiled_path), flush=True)
    if quantized_path is not None:
        print("QUANTIZED_CHECKPOINTS_PATH", str(quantized_path), flush=True)
    for env_name in _FP8_ENV_DEFAULTS:
        print(env_name, os.environ.get(env_name), flush=True)
    print("MODULES_TO_NOT_CONVERT_COUNT", len(modules_to_not_convert), flush=True)
    print(
        "CONTEXT_TRACE_SHAPE",
        json.dumps(
            {
                "seq_len": args.seq_len,
                "max_context_length": max(_cte_buckets(args)),
                "context_encoding_buckets": _cte_buckets(args),
                "prefix_buckets": _prefix_buckets(args, _cte_buckets(args)),
                "enable_prefix_caching": args.enable_prefix_caching,
                "enable_hybrid_apc": args.enable_hybrid_apc,
                "enable_vllm_chunked_prefill": args.enable_vllm_chunked_prefill,
                "block_size": args.block_size,
                "pa_num_blocks": _pa_num_blocks(args),
                "gdn_checkpoint_interval": args.gdn_checkpoint_interval,
                "max_gdn_checkpoint_slots": args.max_gdn_checkpoint_slots,
            },
            sort_keys=True,
        ),
        flush=True,
    )

    if args.weight_dtype == _WEIGHT_DTYPE_BF16_CONTROL:
        print("QUANTIZE_SKIP bf16_control", flush=True)
    elif args.force_quantize or not _quantized_checkpoint_ready(quantized_path):
        print("QUANTIZE_START manual_mlp_only", flush=True)
        _save_mlp_only_fp8_state_dict(model_path, quantized_path)
        print("QUANTIZE_DONE", flush=True)
    else:
        print("QUANTIZE_SKIP existing checkpoint found", flush=True)

    if args.quantize_only:
        return 0

    print("COMPILE_START", flush=True)
    model = NeuronQwen35ForCausalLM(str(model_path), inf_config)
    model.compile(str(compiled_path))
    del model
    gc.collect()
    print("COMPILE_DONE", flush=True)

    if args.load_after_compile:
        model = NeuronQwen35ForCausalLM(str(compiled_path))
        model.load(str(compiled_path))
        print("LOAD_AFTER_COMPILE_OK", flush=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
