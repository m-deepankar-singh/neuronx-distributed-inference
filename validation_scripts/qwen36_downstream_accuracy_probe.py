#!/usr/bin/env python3
"""Small downstream accuracy probe for Qwen3.6-27B NxDI artifacts.

This validates a compiled artifact directly instead of going through vLLM or
the OpenAI proxy. It is intentionally small: enough to catch drift between two
artifacts without turning every kernel experiment into a full benchmark run.

The preferred path uses public Hugging Face datasets:

* `cais/mmlu` for MMLU-style multiple choice.
* `openai/gsm8k` for GSM8K short-answer math.

If those datasets cannot be read on the instance, the script falls back to a
small self-contained smoke set and labels the output accordingly.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path
from typing import Any

import torch
from transformers import AutoTokenizer


MMLU_SUBJECTS = [
    "elementary_mathematics",
    "high_school_world_history",
    "college_computer_science",
    "high_school_physics",
    "professional_medicine",
]

FALLBACK_MMLU = [
    {
        "subject": "fallback_math",
        "question": "What is the value of 12 * 8?",
        "choices": ["20", "80", "96", "108"],
        "answer": 2,
    },
    {
        "subject": "fallback_history",
        "question": "The Magna Carta was first issued in which century?",
        "choices": ["11th", "12th", "13th", "15th"],
        "answer": 2,
    },
    {
        "subject": "fallback_computer_science",
        "question": "Which data structure is normally used for FIFO ordering?",
        "choices": ["Stack", "Queue", "Heap", "Trie"],
        "answer": 1,
    },
    {
        "subject": "fallback_physics",
        "question": "What quantity is measured in newtons?",
        "choices": ["Force", "Energy", "Power", "Charge"],
        "answer": 0,
    },
]

FALLBACK_GSM = [
    {
        "question": "A box has 7 red balls and 5 blue balls. How many balls are in 3 boxes?",
        "answer": "36",
    },
    {
        "question": "Mia buys 4 packs of pencils with 6 pencils each and gives away 5. How many remain?",
        "answer": "19",
    },
    {
        "question": "A train travels 45 miles per hour for 3 hours. How many miles does it travel?",
        "answer": "135",
    },
    {
        "question": "There are 18 cookies. Sam eats 3 and shares the rest equally among 5 friends. How many does each friend get?",
        "answer": "3",
    },
]


def repo_root(path: str | None) -> Path:
    if path:
        return Path(path).expanduser().resolve()
    return Path(__file__).resolve().parents[1]


def load_model(args: argparse.Namespace):
    root = repo_root(args.repo_root)
    contrib_root = root / "contrib" / "models" / "Qwen3.6-27B"
    sys.path.insert(0, str(root))
    sys.path.insert(0, str(contrib_root))

    from neuronx_distributed_inference.modules.generation.sampling import (  # noqa: WPS433
        prepare_sampling_params,
    )
    from src.modeling_qwen35 import NeuronQwen35ForCausalLM  # noqa: WPS433

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, padding_side="right")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("LOAD_START", args.compiled_path, flush=True)
    t0 = time.perf_counter()
    model = NeuronQwen35ForCausalLM(args.compiled_path)
    model.load(args.compiled_path)
    print(f"LOAD_DONE seconds={time.perf_counter() - t0:.3f}", flush=True)

    sampling_params = prepare_sampling_params(
        batch_size=1,
        top_k=[1],
        top_p=[1.0],
        temperature=[1.0],
    )
    return model, tokenizer, sampling_params


def render_prompt(tokenizer: AutoTokenizer, user_content: str) -> str:
    messages = [{"role": "user", "content": user_content}]
    try:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
    except TypeError:
        rendered = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        if rendered.endswith("<think>\n"):
            rendered = rendered[: -len("<think>\n")] + "<think>\n\n</think>\n\n"
        return rendered


def token_scalar(tokens: Any) -> int:
    if hasattr(tokens, "detach"):
        tokens = tokens.detach().cpu()
    if tokens.ndim == 0:
        return int(tokens.item())
    return int(tokens.reshape(-1)[0].item())


def prefill(
    model,
    tokenizer,
    sampling_params,
    prompt: str,
    chunk_size: int,
) -> tuple[torch.Tensor | None, int, int, float]:
    input_ids = torch.tensor(
        [tokenizer(prompt, add_special_tokens=False).input_ids],
        dtype=torch.long,
    )
    prompt_len = int(input_ids.shape[1])
    seq_len = int(model.config.neuron_config.seq_len)
    if prompt_len > seq_len:
        raise ValueError(f"prompt_len={prompt_len} exceeds seq_len={seq_len}")

    final_logits: torch.Tensor | None = None
    final_token: int | None = None
    seq_ids = torch.tensor([0], dtype=torch.int32)
    total_s = 0.0
    for start in range(0, prompt_len, chunk_size):
        end = min(start + chunk_size, prompt_len)
        chunk_ids = input_ids[:, start:end]
        attn_mask = torch.ones((1, end - start), dtype=torch.long)
        position_ids = torch.arange(start, end, dtype=torch.long).unsqueeze(0)
        t0 = time.perf_counter()
        with torch.no_grad():
            out = model(
                input_ids=chunk_ids,
                attention_mask=attn_mask,
                position_ids=position_ids,
                seq_ids=seq_ids,
                sampling_params=sampling_params,
                return_dict=True,
            )
        total_s += time.perf_counter() - t0
        final_token = token_scalar(out.tokens)
        if hasattr(out, "logits") and out.logits is not None:
            logits = out.logits.detach().cpu().float()
            if logits.ndim == 3:
                final_logits = logits[0, -1, :].contiguous()
            elif logits.ndim == 2:
                final_logits = logits[0, :].contiguous()
            else:
                raise RuntimeError(f"unexpected logits shape: {tuple(logits.shape)}")

    assert final_token is not None
    return final_logits, final_token, prompt_len, total_s


def generate(
    model,
    tokenizer,
    sampling_params,
    prompt: str,
    chunk_size: int,
    max_new_tokens: int,
) -> dict[str, Any]:
    model.reset()
    _, first_token, prompt_len, prefill_s = prefill(
        model,
        tokenizer,
        sampling_params,
        prompt,
        chunk_size,
    )

    seq_ids = torch.tensor([0], dtype=torch.int32)
    seq_len = int(model.config.neuron_config.seq_len)
    vocab_size = len(tokenizer)
    eos_id = tokenizer.eos_token_id
    eos_ids = set(eos_id) if isinstance(eos_id, (list, tuple, set)) else {eos_id}

    generated: list[int] = []
    current_token = first_token
    decode_s = 0.0
    with torch.no_grad():
        for step in range(max_new_tokens):
            if current_token in eos_ids:
                break
            if current_token < 0 or current_token >= vocab_size:
                raise RuntimeError(f"invalid token id generated: {current_token}")
            generated.append(current_token)
            if step == max_new_tokens - 1:
                break

            pos_value = prompt_len + step
            input_ids = torch.tensor([[current_token]], dtype=torch.long)
            position_ids = torch.tensor([[pos_value]], dtype=torch.long)
            attention_mask = torch.ones((1, min(seq_len, pos_value + 1)), dtype=torch.long)
            t0 = time.perf_counter()
            out = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                seq_ids=seq_ids,
                sampling_params=sampling_params,
                return_dict=True,
            )
            decode_s += time.perf_counter() - t0
            current_token = token_scalar(out.tokens)

    text = tokenizer.decode(generated, skip_special_tokens=True)
    return {
        "prompt_tokens": prompt_len,
        "tokens": generated,
        "text": text,
        "prefill_seconds": prefill_s,
        "decode_seconds": decode_s,
    }


def load_hf_mmlu(limit: int) -> tuple[list[dict[str, Any]], str]:
    from huggingface_hub import hf_hub_download
    import pandas as pd

    cases: list[dict[str, Any]] = []
    per_subject = max(1, (limit + len(MMLU_SUBJECTS) - 1) // len(MMLU_SUBJECTS))
    for subject in MMLU_SUBJECTS:
        path = hf_hub_download(
            repo_id="cais/mmlu",
            repo_type="dataset",
            filename=f"{subject}/test-00000-of-00001.parquet",
        )
        frame = pd.read_parquet(path).head(per_subject)
        for _, row in frame.iterrows():
            choices = row["choices"]
            if not isinstance(choices, list):
                choices = list(choices)
            cases.append(
                {
                    "subject": subject,
                    "question": str(row["question"]),
                    "choices": [str(x) for x in choices],
                    "answer": int(row["answer"]),
                }
            )
            if len(cases) >= limit:
                return cases, "hf:cais/mmlu"
    return cases, "hf:cais/mmlu"


def load_hf_gsm(limit: int) -> tuple[list[dict[str, str]], str]:
    from huggingface_hub import hf_hub_download
    import pandas as pd

    path = hf_hub_download(
        repo_id="openai/gsm8k",
        repo_type="dataset",
        filename="main/test-00000-of-00001.parquet",
    )
    frame = pd.read_parquet(path).head(limit)
    cases: list[dict[str, str]] = []
    for _, row in frame.iterrows():
        answer = str(row["answer"])
        match = re.search(r"####\s*([-+]?\d+(?:\.\d+)?)", answer.replace(",", ""))
        cases.append(
            {
                "question": str(row["question"]),
                "answer": match.group(1) if match else answer,
            }
        )
    return cases, "hf:openai/gsm8k"


def get_mmlu_cases(limit: int, use_fallback: bool) -> tuple[list[dict[str, Any]], str]:
    if not use_fallback:
        try:
            return load_hf_mmlu(limit)
        except Exception as exc:
            print(f"MMLU_HF_LOAD_FAILED {type(exc).__name__}: {exc}", flush=True)
    return FALLBACK_MMLU[:limit], "fallback:self-contained"


def get_gsm_cases(limit: int, use_fallback: bool) -> tuple[list[dict[str, str]], str]:
    if not use_fallback:
        try:
            return load_hf_gsm(limit)
        except Exception as exc:
            print(f"GSM_HF_LOAD_FAILED {type(exc).__name__}: {exc}", flush=True)
    return FALLBACK_GSM[:limit], "fallback:self-contained"


def letter_token_ids(tokenizer, label: str) -> list[int]:
    ids: list[int] = []
    for text in (label, f" {label}"):
        encoded = tokenizer(text, add_special_tokens=False).input_ids
        if len(encoded) == 1:
            ids.append(int(encoded[0]))
    return sorted(set(ids))


def run_mmlu(
    model,
    tokenizer,
    sampling_params,
    cases: list[dict[str, Any]],
    chunk_size: int,
) -> dict[str, Any]:
    labels = ["A", "B", "C", "D"]
    label_ids = {label: letter_token_ids(tokenizer, label) for label in labels}
    rows = []
    correct = 0
    for index, case in enumerate(cases):
        model.reset()
        choices = case["choices"]
        body = [case["question"].strip(), ""]
        for label, choice in zip(labels, choices):
            body.append(f"{label}. {choice}")
        body.append("")
        body.append("Answer with only the letter A, B, C, or D.")
        body.append("Answer:")
        prompt = render_prompt(tokenizer, "\n".join(body))
        logits, next_token, prompt_len, prefill_s = prefill(
            model,
            tokenizer,
            sampling_params,
            prompt,
            chunk_size,
        )
        if logits is None:
            raise RuntimeError("MMLU requires an output-logits artifact")

        scores = {}
        for label in labels:
            ids = label_ids[label]
            if not ids:
                raise RuntimeError(f"could not find single-token ids for {label}")
            scores[label] = max(float(logits[token_id].item()) for token_id in ids)
        pred_label = max(scores, key=scores.get)
        expected_label = labels[int(case["answer"])]
        is_correct = pred_label == expected_label
        correct += int(is_correct)
        row = {
            "index": index,
            "subject": case.get("subject"),
            "prompt_tokens": prompt_len,
            "prefill_seconds": prefill_s,
            "prediction": pred_label,
            "expected": expected_label,
            "correct": is_correct,
            "next_token": next_token,
            "scores": scores,
        }
        print("MMLU_CASE " + json.dumps(row, sort_keys=True), flush=True)
        rows.append(row)

    return {
        "count": len(rows),
        "correct": correct,
        "accuracy": correct / len(rows) if rows else 0.0,
        "cases": rows,
    }


def normalize_number(text: str) -> str | None:
    cleaned = text.replace(",", "")
    matches = re.findall(r"[-+]?\d+(?:\.\d+)?", cleaned)
    if not matches:
        return None
    value = matches[-1]
    if value.endswith(".0"):
        value = value[:-2]
    return value


def run_gsm(
    model,
    tokenizer,
    sampling_params,
    cases: list[dict[str, str]],
    chunk_size: int,
    max_new_tokens: int,
) -> dict[str, Any]:
    rows = []
    correct = 0
    for index, case in enumerate(cases):
        prompt = render_prompt(
            tokenizer,
            "Solve the math problem. Answer with only the final number.\n\n"
            f"Problem: {case['question']}\nAnswer:",
        )
        result = generate(
            model,
            tokenizer,
            sampling_params,
            prompt,
            chunk_size,
            max_new_tokens=max_new_tokens,
        )
        pred = normalize_number(result["text"])
        expected = normalize_number(case["answer"])
        is_correct = pred == expected
        correct += int(is_correct)
        row = {
            "index": index,
            "prompt_tokens": result["prompt_tokens"],
            "prefill_seconds": result["prefill_seconds"],
            "decode_seconds": result["decode_seconds"],
            "prediction": pred,
            "expected": expected,
            "correct": is_correct,
            "text": result["text"],
            "tokens": result["tokens"],
        }
        print("GSM_CASE " + json.dumps(row, sort_keys=True), flush=True)
        rows.append(row)

    return {
        "count": len(rows),
        "correct": correct,
        "accuracy": correct / len(rows) if rows else 0.0,
        "cases": rows,
    }


def run_artifact(args: argparse.Namespace) -> None:
    model, tokenizer, sampling_params = load_model(args)
    chunk_size = int(args.chunk_size or model.config.neuron_config.max_context_length)

    mmlu_cases, mmlu_source = get_mmlu_cases(args.mmlu_limit, args.fallback_only)
    gsm_cases, gsm_source = get_gsm_cases(args.gsm_limit, args.fallback_only)

    report = {
        "model_path": args.model_path,
        "compiled_path": args.compiled_path,
        "chunk_size": chunk_size,
        "mmlu_source": mmlu_source,
        "gsm_source": gsm_source,
        "mmlu": run_mmlu(model, tokenizer, sampling_params, mmlu_cases, chunk_size),
        "gsm8k": run_gsm(
            model,
            tokenizer,
            sampling_params,
            gsm_cases,
            chunk_size,
            args.gsm_max_new_tokens,
        ),
    }

    out_path = Path(args.out).expanduser()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print("RUN_WROTE", out_path, flush=True)


def compare_reports(args: argparse.Namespace) -> None:
    baseline = json.loads(Path(args.baseline).expanduser().read_text())
    candidate = json.loads(Path(args.candidate).expanduser().read_text())

    def summarize(task: str) -> dict[str, Any]:
        b = baseline[task]
        c = candidate[task]
        delta = float(c["accuracy"] - b["accuracy"])
        if b["count"] != c["count"]:
            raise ValueError(f"{task} count mismatch: {b['count']} vs {c['count']}")
        case_matches = 0
        for brow, crow in zip(b["cases"], c["cases"]):
            if brow.get("prediction") == crow.get("prediction"):
                case_matches += 1
        return {
            "baseline_accuracy": b["accuracy"],
            "candidate_accuracy": c["accuracy"],
            "delta": delta,
            "abs_delta": abs(delta),
            "count": b["count"],
            "baseline_correct": b["correct"],
            "candidate_correct": c["correct"],
            "prediction_match_count": case_matches,
            "prediction_match_rate": case_matches / b["count"] if b["count"] else 0.0,
            "passed_delta_gate": abs(delta) <= args.max_delta,
        }

    report = {
        "baseline": baseline["compiled_path"],
        "candidate": candidate["compiled_path"],
        "max_delta": args.max_delta,
        "mmlu": summarize("mmlu"),
        "gsm8k": summarize("gsm8k"),
    }
    report["passed"] = (
        report["mmlu"]["passed_delta_gate"]
        and report["gsm8k"]["passed_delta_gate"]
    )

    if args.out:
        out_path = Path(args.out).expanduser()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        print("COMPARE_WROTE", out_path, flush=True)
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)
    if not report["passed"]:
        raise SystemExit(3)


def main() -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)

    run = sub.add_parser("run")
    run.add_argument("--repo-root", default=None)
    run.add_argument("--model-path", required=True)
    run.add_argument("--compiled-path", required=True)
    run.add_argument("--out", required=True)
    run.add_argument("--chunk-size", type=int, default=None)
    run.add_argument("--mmlu-limit", type=int, default=40)
    run.add_argument("--gsm-limit", type=int, default=20)
    run.add_argument("--gsm-max-new-tokens", type=int, default=32)
    run.add_argument("--fallback-only", action="store_true")

    compare = sub.add_parser("compare")
    compare.add_argument("--baseline", required=True)
    compare.add_argument("--candidate", required=True)
    compare.add_argument("--out", default=None)
    compare.add_argument("--max-delta", type=float, default=0.01)

    args = parser.parse_args()
    if args.cmd == "run":
        run_artifact(args)
    elif args.cmd == "compare":
        compare_reports(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
