import argparse
import os
import sys
import time

import torch
from transformers import AutoTokenizer


def build_prompt_ids(tokenizer, target_tokens):
    facts = (
        "Tokyo 2020 Olympics were held in 2021 because of the COVID-19 pandemic. "
        "The Games included athletes from around the world, new sports such as skateboarding, "
        "sport climbing, surfing, and karate, and major performances in athletics, swimming, "
        "gymnastics, and team sports. "
    )
    question = (
        "\n\nQuestion: Give me a concise summary of the Tokyo 2020 Olympics, "
        "including why they were delayed and what made the Games notable.\nAnswer:"
    )
    q_ids = tokenizer(question, add_special_tokens=False).input_ids
    keep = max(1, target_tokens - len(q_ids))
    base_filler_ids = tokenizer(facts, add_special_tokens=False).input_ids
    repeat_count = max(1, (keep + len(base_filler_ids) - 1) // len(base_filler_ids))
    filler_ids = (base_filler_ids * repeat_count)[:keep]
    ids = (filler_ids[:keep] + q_ids)[:target_tokens]
    return torch.tensor([ids], dtype=torch.long)


def build_custom_prompt_ids(tokenizer, prompt):
    ids = tokenizer(prompt, add_special_tokens=False).input_ids
    if not ids:
        raise ValueError("custom prompt produced zero tokens")
    return torch.tensor([ids], dtype=torch.long)


def token_scalar(tokens):
    if hasattr(tokens, "detach"):
        tokens = tokens.detach().cpu()
    if tokens.ndim == 0:
        return int(tokens.item())
    return int(tokens.reshape(-1)[0].item())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--compiled-path", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--contrib-root", required=True)
    parser.add_argument("--prompt-tokens", type=int, default=762)
    parser.add_argument("--chunk-size", type=int, default=128)
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--seq-len", type=int, default=65536)
    parser.add_argument("--prompt", default=None)
    parser.add_argument("--prompt-file", default=None)
    args = parser.parse_args()

    sys.path.insert(0, args.contrib_root)
    from src.modeling_qwen35 import NeuronQwen35ForCausalLM

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, padding_side="right")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    pad_id = tokenizer.pad_token_id

    print("LOAD_START", flush=True)
    t_load0 = time.perf_counter()
    model = NeuronQwen35ForCausalLM(args.compiled_path)
    model.load(args.compiled_path)
    model.reset()
    print(f"LOAD_DONE seconds={time.perf_counter() - t_load0:.3f}", flush=True)
    print(
        "CONFIG "
        f"hybrid={getattr(model.config, 'use_hybrid_cache_manager', None)} "
        f"chunked={getattr(model.config, 'use_qwen_hybrid_chunked_prefill', None)} "
        f"nki={getattr(model.config, 'use_qwen_hybrid_chunked_prefill_nki', None)} "
        f"seq_len={model.config.neuron_config.seq_len} "
        f"ctx_buckets={model.config.neuron_config.context_encoding_buckets}",
        flush=True,
    )

    if args.prompt is not None and args.prompt_file is not None:
        raise ValueError("use only one of --prompt or --prompt-file")
    if args.prompt_file is not None:
        with open(args.prompt_file, "r", encoding="utf-8") as f:
            prompt_text = f.read()
        input_ids = build_custom_prompt_ids(tokenizer, prompt_text)
    elif args.prompt is not None:
        input_ids = build_custom_prompt_ids(tokenizer, args.prompt)
    else:
        input_ids = build_prompt_ids(tokenizer, args.prompt_tokens)
    prompt_len = input_ids.shape[1]
    if prompt_len + args.max_new_tokens > args.seq_len:
        raise ValueError(
            f"prompt_len + max_new_tokens must fit inside seq_len: "
            f"{prompt_len} + {args.max_new_tokens} > {args.seq_len}"
        )
    seq_ids = torch.tensor([0], dtype=torch.int32)
    generated = []
    cte_times = []

    print(f"PREFILL_START prompt_tokens={prompt_len} chunk_size={args.chunk_size}", flush=True)
    first_token = None
    for start in range(0, prompt_len, args.chunk_size):
        end = min(start + args.chunk_size, prompt_len)
        valid = end - start
        chunk_ids = input_ids[:, start:end]
        if valid < args.chunk_size:
            pad = torch.full(
                (1, args.chunk_size - valid),
                pad_id,
                dtype=chunk_ids.dtype,
            )
            chunk_ids = torch.cat([chunk_ids, pad], dim=1)
        attn_mask = torch.zeros((1, args.chunk_size), dtype=torch.long)
        attn_mask[:, :valid] = 1
        pos = torch.arange(start, start + args.chunk_size, dtype=torch.long).unsqueeze(0)

        t0 = time.perf_counter()
        with torch.no_grad():
            out = model(
                input_ids=chunk_ids,
                attention_mask=attn_mask,
                position_ids=pos,
                seq_ids=seq_ids,
                return_dict=True,
            )
        dt = time.perf_counter() - t0
        cte_times.append(dt)
        tok = token_scalar(out.tokens)
        print(
            f"PREFILL_CHUNK index={start // args.chunk_size} "
            f"start={start} valid={valid} seconds={dt:.3f} token={tok}",
            flush=True,
        )
        if end == prompt_len:
            first_token = tok

    assert first_token is not None
    generated.append(first_token)
    invalid = [
        tok for tok in generated if tok < 0 or tok >= len(tokenizer)
    ]
    if invalid:
        raise RuntimeError(f"invalid token after prefill: {invalid}")

    decode_times = []
    current_token = first_token
    for step in range(1, args.max_new_tokens):
        pos_value = prompt_len + step - 1
        ids = torch.tensor([[current_token]], dtype=torch.long)
        pos = torch.tensor([[pos_value]], dtype=torch.long)
        attn_mask = torch.zeros((1, args.seq_len), dtype=torch.long)
        attn_mask[:, : pos_value + 1] = 1

        t0 = time.perf_counter()
        with torch.no_grad():
            out = model(
                input_ids=ids,
                attention_mask=attn_mask,
                position_ids=pos,
                seq_ids=seq_ids,
                return_dict=True,
            )
        dt = time.perf_counter() - t0
        decode_times.append(dt)
        current_token = token_scalar(out.tokens)
        generated.append(current_token)
        if current_token < 0 or current_token >= len(tokenizer):
            raise RuntimeError(
                f"invalid decode token step={step} token={current_token}"
            )
        print(
            f"DECODE_STEP step={step} pos={pos_value} seconds={dt:.3f} token={current_token}",
            flush=True,
        )

    total_cte = sum(cte_times)
    ingest_tps = prompt_len / total_cte if total_cte else 0.0
    decode_tps = len(decode_times) / sum(decode_times) if decode_times else 0.0
    text = tokenizer.decode(generated, skip_special_tokens=True)
    print(
        "SUMMARY "
        f"prompt_tokens={prompt_len} chunks={len(cte_times)} "
        f"cte_seconds={total_cte:.3f} ingest_tok_s={ingest_tps:.2f} "
        f"first_token={first_token} decode_steps={len(decode_times)} "
        f"decode_tok_s={decode_tps:.2f}",
        flush=True,
    )
    print("GENERATED_TOKENS", generated, flush=True)
    print("GENERATED_TEXT_START")
    print(text)
    print("GENERATED_TEXT_END")


if __name__ == "__main__":
    main()
