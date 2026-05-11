#!/usr/bin/env python3
"""Compile Qwen3.6-27B with native MTP speculative decoding enabled."""

from __future__ import annotations

import argparse
import copy
import gc
import json
import sys
from pathlib import Path

import torch


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


def _build_config(args: argparse.Namespace):
    from neuronx_distributed_inference.models.config import (  # noqa: WPS433
        FusedSpecNeuronConfig,
        NeuronConfig,
        OnDeviceSamplingConfig,
    )
    from src.modeling_qwen35 import (  # noqa: WPS433
        NeuronQwen35ForCausalLM,
        NeuronQwen35MTPDraftForCausalLM,
        Qwen35InferenceConfig,
    )

    model_path = Path(args.model_path).expanduser().resolve()
    config_dict = _load_text_config(model_path)

    target_neuron_config = NeuronConfig(
        tp_degree=args.tp_degree,
        batch_size=1,
        ctx_batch_size=1,
        tkg_batch_size=1,
        seq_len=args.seq_len,
        max_context_length=args.cte_bucket,
        max_length=args.seq_len,
        context_encoding_buckets=[args.cte_bucket],
        torch_dtype=torch.bfloat16,
        on_device_sampling_config=OnDeviceSamplingConfig(
            do_sample=False,
            top_k=1,
            top_p=1.0,
            temperature=1.0,
        ),
        enable_bucketing=False,
        logical_nc_config=args.logical_nc_config,
        save_sharded_checkpoint=True,
        enable_fused_speculation=True,
        enable_eagle_speculation=True,
        speculation_length=args.speculation_length,
        disable_kv_cache_tiling=True,
    )

    target_config_dict = dict(config_dict)
    target_config_dict.update(
        {
            "use_hybrid_cache_manager": True,
            "use_qwen_hybrid_chunked_prefill": True,
            "use_qwen_hybrid_chunked_prefill_nki": True,
            "enable_mtp_speculation": True,
            "enable_mtp_step_state_output": True,
        }
    )

    draft_neuron_config = copy.deepcopy(target_neuron_config)
    draft_neuron_config.enable_fused_speculation = False
    draft_neuron_config.is_eagle_draft = True
    draft_neuron_config.n_active_tokens = 1

    draft_config_dict = dict(config_dict)
    draft_config_dict.update(
        {
            "use_hybrid_cache_manager": False,
            "enable_mtp_weight_loading": True,
            "enable_mtp_speculation": True,
            "is_mtp_draft_model": True,
        }
    )

    draft_config = Qwen35InferenceConfig(
        neuron_config=draft_neuron_config,
        **draft_config_dict,
    )
    fused_spec_config = FusedSpecNeuronConfig(
        NeuronQwen35ForCausalLM._model_cls,
        draft_config=draft_config,
        draft_model_path=str(model_path),
        draft_model_cls=NeuronQwen35MTPDraftForCausalLM,
    )

    return Qwen35InferenceConfig(
        neuron_config=target_neuron_config,
        fused_spec_config=fused_spec_config,
        **target_config_dict,
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", default=None)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--compiled-path", required=True)
    parser.add_argument("--seq-len", type=int, default=65536)
    parser.add_argument("--cte-bucket", type=int, default=512)
    parser.add_argument("--tp-degree", type=int, default=4)
    parser.add_argument("--logical-nc-config", type=int, default=2)
    parser.add_argument("--speculation-length", type=int, default=2)
    parser.add_argument("--load-after-compile", action="store_true")
    args = parser.parse_args()

    repo = _repo_root(args.repo_root)
    contrib_model_dir = repo / "contrib" / "models" / "Qwen3.6-27B"
    sys.path.insert(0, str(repo))
    sys.path.insert(0, str(contrib_model_dir))

    from src.modeling_qwen35 import NeuronQwen35ForCausalLM  # noqa: WPS433

    model_path = Path(args.model_path).expanduser().resolve()
    compiled_path = Path(args.compiled_path).expanduser().resolve()
    inf_config = _build_config(args)

    print("MTP_SPECULATION native_qwen", flush=True)
    print("MODEL_PATH", str(model_path), flush=True)
    print("COMPILED_PATH", str(compiled_path), flush=True)
    print(
        "CONTEXT_TRACE_SHAPE",
        json.dumps(
            {
                "seq_len": args.seq_len,
                "max_context_length": args.cte_bucket,
                "context_encoding_buckets": [args.cte_bucket],
                "speculation_length": args.speculation_length,
            },
            sort_keys=True,
        ),
        flush=True,
    )

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
