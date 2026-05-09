import argparse
import gc
import importlib.util
import json
import os
import sys
import time

import torch
from neuronx_distributed_inference.models.config import (
    NeuronConfig,
    OnDeviceSamplingConfig,
)


def load_model_module(contrib_root):
    sys.path.insert(0, contrib_root)
    spec = importlib.util.spec_from_file_location(
        "qwen35_modeling",
        os.path.join(contrib_root, "src", "modeling_qwen35.py"),
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def read_text_config(model_path):
    with open(os.path.join(model_path, "config.json")) as f:
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


def build_config(module, model_path, seq_len, tp_degree, save_sharded):
    neuron_config = NeuronConfig(
        tp_degree=tp_degree,
        batch_size=1,
        ctx_batch_size=1,
        tkg_batch_size=1,
        seq_len=seq_len,
        torch_dtype=torch.bfloat16,
        on_device_sampling_config=OnDeviceSamplingConfig(top_k=1),
        enable_bucketing=False,
        flash_decoding_enabled=False,
        logical_nc_config=2,
        save_sharded_checkpoint=save_sharded,
    )
    return module.Qwen35InferenceConfig(
        neuron_config=neuron_config,
        **read_text_config(model_path),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--compiled-path", required=True)
    ap.add_argument("--contrib-root", required=True)
    ap.add_argument("--seq-len", type=int, default=65536)
    ap.add_argument("--tp-degree", type=int, default=4)
    ap.add_argument("--save-sharded", action="store_true")
    ap.add_argument("--use-hybrid-cache", action="store_true")
    ap.add_argument("--use-qwen-hybrid-chunked-prefill", action="store_true")
    ap.add_argument("--use-qwen-hybrid-chunked-prefill-nki", action="store_true")
    ap.add_argument("--context-bucket", type=int, default=None)
    ap.add_argument("--token-bucket", type=int, default=None)
    args = ap.parse_args()

    module = load_model_module(args.contrib_root)
    os.makedirs(args.compiled_path, exist_ok=True)
    config = build_config(
        module,
        args.model_path,
        args.seq_len,
        args.tp_degree,
        args.save_sharded,
    )
    if args.context_bucket or args.token_bucket:
        config.neuron_config.enable_bucketing = True
        if args.context_bucket:
            config.neuron_config.context_encoding_buckets = [args.context_bucket]
        if args.token_bucket:
            config.neuron_config.token_generation_buckets = [args.token_bucket]
    config.use_hybrid_cache_manager = args.use_hybrid_cache
    config.use_qwen_hybrid_chunked_prefill = args.use_qwen_hybrid_chunked_prefill
    config.use_qwen_hybrid_chunked_prefill_nki = (
        args.use_qwen_hybrid_chunked_prefill_nki
    )

    print(f"HYBRID_CACHE use_hybrid_cache_manager={config.use_hybrid_cache_manager}")
    print(
        "QWEN_CHUNKED_PREFILL "
        f"use_qwen_hybrid_chunked_prefill={config.use_qwen_hybrid_chunked_prefill}"
    )
    print(
        "QWEN_CHUNKED_PREFILL_NKI "
        f"use_qwen_hybrid_chunked_prefill_nki={config.use_qwen_hybrid_chunked_prefill_nki}"
    )
    print(
        "COMPILE_CONFIG "
        f"seq_len={args.seq_len} tp={args.tp_degree} save_sharded={args.save_sharded} "
        f"context_bucket={args.context_bucket} token_bucket={args.token_bucket}"
    )
    start = time.time()
    model = module.NeuronQwen35ForCausalLM(args.model_path, config)
    model.compile(args.compiled_path)
    del model
    gc.collect()
    elapsed = time.time() - start
    print(f"SAMPLE_COMPILE_DONE seconds={elapsed:.2f} minutes={elapsed / 60:.2f}")


if __name__ == "__main__":
    main()
