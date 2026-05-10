#!/usr/bin/env python3
"""Compare direct NxDI/HF generation quality for the Qwen3.6 artifact."""

from __future__ import annotations

import argparse
import sys

import torch


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--compiled-path", required=True)
    parser.add_argument("--contrib-root", required=True)
    parser.add_argument("--max-new-tokens", type=int, default=48)
    args = parser.parse_args()

    if args.contrib_root not in sys.path:
        sys.path.insert(0, args.contrib_root)

    import transformers
    from transformers import AutoTokenizer, GenerationConfig

    from neuronx_distributed_inference.utils.hf_adapter import (
        HuggingFaceGenerationAdapter,
    )
    from src.modeling_qwen35 import NeuronQwen35ForCausalLM

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, padding_side="right")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    prompts = {
        "raw_math": "Question: What is 137 * 23? Answer with only the integer.\nAnswer:",
        "chat_math_thinking": tokenizer.apply_chat_template(
            [
                {
                    "role": "user",
                    "content": "What is 137 * 23? Answer with only the integer.",
                }
            ],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=True,
        ),
        "chat_math_no_thinking": tokenizer.apply_chat_template(
            [
                {
                    "role": "user",
                    "content": "What is 137 * 23? Answer with only the integer.",
                }
            ],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        ),
    }

    print("loading model", flush=True)
    model = NeuronQwen35ForCausalLM(args.compiled_path)
    model.load(args.compiled_path)
    model.reset()
    gen_model = HuggingFaceGenerationAdapter(model)

    generation_config = GenerationConfig(
        do_sample=True,
        top_k=1,
        temperature=1.0,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    gen_model.generation_config.transformers_version = transformers.__version__
    generation_config.transformers_version = transformers.__version__

    for name, prompt in prompts.items():
        print(f"\n=== {name} prompt ===")
        print(prompt[-300:])
        inputs = tokenizer(prompt, padding=True, return_tensors="pt", add_special_tokens=False)
        with torch.no_grad():
            outputs = gen_model.generate(
                inputs.input_ids,
                generation_config=generation_config,
                attention_mask=inputs.attention_mask,
                max_new_tokens=args.max_new_tokens,
            )
        full_ids = outputs[0].tolist()
        prompt_len = int(inputs.input_ids.shape[1])
        new_ids = full_ids[prompt_len:]
        print("new_ids:", new_ids)
        print("decoded_new:", tokenizer.decode(new_ids, skip_special_tokens=True))
        model.reset()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
