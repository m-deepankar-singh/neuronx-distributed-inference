import argparse
import json
import sys
import time

import torch
from transformers import AutoTokenizer


FILLER = (
    "Archive note: The Tokyo 2020 Olympics were held in 2021 after a global delay. "
    "Athletes competed in athletics, swimming, gymnastics, team sports, and new events. "
    "This sentence is background context and does not contain the answer. "
)


def repeat_to_length(piece, length):
    if length <= 0:
        return []
    reps = (length + len(piece) - 1) // len(piece)
    return (piece * reps)[:length]


def build_needle_prompt(tokenizer, target_tokens, needle_ratio, code):
    intro = (
        "You are reading a long archive. Exactly one note contains a secret code phrase. "
        "Remember it and answer the final question using only that phrase.\n\n"
    )
    needle = f"Important note: the secret code phrase is {code}.\n\n"
    question = "\n\nQuestion: What is the secret code phrase? Answer with only the exact phrase.\nAnswer:"

    intro_ids = tokenizer(intro, add_special_tokens=False).input_ids
    needle_ids = tokenizer(needle, add_special_tokens=False).input_ids
    question_ids = tokenizer(question, add_special_tokens=False).input_ids
    filler_ids = tokenizer(FILLER, add_special_tokens=False).input_ids

    body_budget = target_tokens - len(intro_ids) - len(question_ids)
    if body_budget <= len(needle_ids):
        raise ValueError("target_tokens too small for needle prompt")

    needle_start = int((body_budget - len(needle_ids)) * needle_ratio)
    before = repeat_to_length(filler_ids, needle_start)
    after = repeat_to_length(filler_ids, body_budget - len(before) - len(needle_ids))
    ids = intro_ids + before + needle_ids + after + question_ids
    ids = ids[:target_tokens]
    if len(ids) != target_tokens:
        raise AssertionError((len(ids), target_tokens))
    return torch.tensor([ids], dtype=torch.long)


def token_scalar(tokens):
    if hasattr(tokens, "detach"):
        tokens = tokens.detach().cpu()
    if tokens.ndim == 0:
        return int(tokens.item())
    return int(tokens.reshape(-1)[0].item())


def run_generation(model, tokenizer, input_ids, chunk_size, max_new_tokens, seq_len):
    prompt_len = input_ids.shape[1]
    if prompt_len + max_new_tokens > seq_len:
        raise ValueError(f"prompt_len + max_new_tokens exceeds seq_len: {prompt_len} + {max_new_tokens} > {seq_len}")

    pad_id = tokenizer.pad_token_id
    seq_ids = torch.tensor([0], dtype=torch.int32)
    cte_times = []
    first_token = None

    for start in range(0, prompt_len, chunk_size):
        end = min(start + chunk_size, prompt_len)
        valid = end - start
        chunk_ids = input_ids[:, start:end]
        if valid < chunk_size:
            pad = torch.full((1, chunk_size - valid), pad_id, dtype=chunk_ids.dtype)
            chunk_ids = torch.cat([chunk_ids, pad], dim=1)
        attn_mask = torch.zeros((1, chunk_size), dtype=torch.long)
        attn_mask[:, :valid] = 1
        pos = torch.arange(start, start + chunk_size, dtype=torch.long).unsqueeze(0)

        t0 = time.perf_counter()
        with torch.no_grad():
            out = model(
                input_ids=chunk_ids,
                attention_mask=attn_mask,
                position_ids=pos,
                seq_ids=seq_ids,
                return_dict=True,
            )
        cte_times.append(time.perf_counter() - t0)
        if end == prompt_len:
            first_token = token_scalar(out.tokens)

    generated = [first_token]
    current_token = first_token
    decode_times = []
    for step in range(1, max_new_tokens):
        pos_value = prompt_len + step - 1
        ids = torch.tensor([[current_token]], dtype=torch.long)
        pos = torch.tensor([[pos_value]], dtype=torch.long)
        attn_mask = torch.zeros((1, seq_len), dtype=torch.long)
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
        decode_times.append(time.perf_counter() - t0)
        current_token = token_scalar(out.tokens)
        generated.append(current_token)
        if current_token < 0 or current_token >= len(tokenizer):
            raise RuntimeError(f"invalid token step={step} token={current_token}")

    text = tokenizer.decode(generated, skip_special_tokens=True)
    return {
        "prompt_tokens": prompt_len,
        "chunks": len(cte_times),
        "cte_seconds": sum(cte_times),
        "ingest_tok_s": prompt_len / sum(cte_times),
        "decode_steps": len(decode_times),
        "decode_tok_s": len(decode_times) / sum(decode_times) if decode_times else 0.0,
        "generated_tokens": generated,
        "generated_text": text,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--compiled-path", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--contrib-root", required=True)
    parser.add_argument("--chunk-size", type=int, default=128)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--seq-len", type=int, default=65536)
    args = parser.parse_args()

    sys.path.insert(0, args.contrib_root)
    from src.modeling_qwen35 import NeuronQwen35ForCausalLM

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, padding_side="right")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    cases = [
        ("needle_4k_middle", 4096, 0.50, "CRIMSON LANTERN"),
        ("needle_16k_middle", 16384, 0.50, "ORANGE FALCON"),
        ("needle_32k_late", 32768, 0.85, "SILVER HARBOR"),
        ("needle_64k_late", 65472, 0.90, "VIOLET SIGNAL"),
    ]

    print("LOAD_START", flush=True)
    t0 = time.perf_counter()
    model = NeuronQwen35ForCausalLM(args.compiled_path)
    model.load(args.compiled_path)
    print(f"LOAD_DONE seconds={time.perf_counter() - t0:.3f}", flush=True)
    print(
        "CONFIG "
        f"hybrid={getattr(model.config, 'use_hybrid_cache_manager', None)} "
        f"chunked={getattr(model.config, 'use_qwen_hybrid_chunked_prefill', None)} "
        f"nki={getattr(model.config, 'use_qwen_hybrid_chunked_prefill_nki', None)} "
        f"seq_len={model.config.neuron_config.seq_len} "
        f"ctx_buckets={model.config.neuron_config.context_encoding_buckets}",
        flush=True,
    )

    results = []
    for name, target_tokens, ratio, code in cases:
        print(f"CASE_START name={name} target_tokens={target_tokens} ratio={ratio} expected={code}", flush=True)
        model.reset()
        input_ids = build_needle_prompt(tokenizer, target_tokens, ratio, code)
        result = run_generation(model, tokenizer, input_ids, args.chunk_size, args.max_new_tokens, args.seq_len)
        passed = code.lower() in result["generated_text"].lower()
        row = {
            "name": name,
            "expected": code,
            "passed": passed,
            **result,
        }
        results.append(row)
        print("CASE_RESULT " + json.dumps(row, ensure_ascii=False), flush=True)

    passed_count = sum(1 for row in results if row["passed"])
    print("EVAL_SUMMARY " + json.dumps({"passed": passed_count, "total": len(results), "results": results}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
