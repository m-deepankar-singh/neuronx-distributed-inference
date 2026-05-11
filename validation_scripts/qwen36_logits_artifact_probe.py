#!/usr/bin/env python3
"""Direct logits probe for Qwen3.6-27B NxDI artifacts.

The vLLM/OpenAI logprobs path currently fails for this model with
``list index out of range``. This script bypasses that server layer and loads a
compiled NxDI artifact directly. It requires artifacts compiled with
``neuron_config.output_logits=True``.

Typical use:

  python validation_scripts/qwen36_logits_artifact_probe.py dump \
    --model-path /opt/dlami/nvme/models/Qwen3.6-27B \
    --compiled-path /opt/dlami/nvme/qwen_artifacts/baseline_logits \
    --out /tmp/baseline_logits.pt

  python validation_scripts/qwen36_logits_artifact_probe.py compare \
    --baseline /tmp/baseline_logits.pt \
    --candidate /tmp/candidate_logits.pt \
    --out /tmp/logits_compare.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import torch
from transformers import AutoTokenizer


DEFAULT_CASES: list[dict[str, Any]] = [
    {
        "name": "short",
        "messages": [
            {"role": "user", "content": "What is 17 * 23? Answer with only the number."}
        ],
    },
    {
        "name": "medium",
        "messages": [
            {
                "role": "user",
                "content": (
                    "Summarize the main causes of the 2020 Tokyo Olympics delay in "
                    "two concise sentences."
                ),
            }
        ],
    },
    {
        "name": "long",
        "messages": [
            {
                "role": "user",
                "content": (
                    "Read this repeated context and answer the final question. "
                    + ("Metal Gear Solid involves stealth, espionage, nuclear deterrence, "
                       "identity, soldiers, and political deception. " * 80)
                    + "Question: name two recurring themes."
                ),
            }
        ],
    },
    {
        "name": "code",
        "messages": [
            {
                "role": "user",
                "content": (
                    "Write a Python expression that returns the squares of even numbers "
                    "from 0 through 10."
                ),
            }
        ],
    },
    {
        "name": "multilingual",
        "messages": [
            {
                "role": "user",
                "content": "Responde en espanol: cual es la capital de Francia?",
            }
        ],
    },
]


def repo_root(path: str | None) -> Path:
    if path:
        return Path(path).expanduser().resolve()
    return Path(__file__).resolve().parents[1]


def load_cases(path: str | None) -> list[dict[str, Any]]:
    if path is None:
        return DEFAULT_CASES
    with Path(path).expanduser().open() as handle:
        data = json.load(handle)
    if not isinstance(data, list):
        raise ValueError("cases file must contain a JSON list")
    return data


def render_prompt(tokenizer: AutoTokenizer, messages: list[dict[str, str]], enable_thinking: bool) -> str:
    try:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=enable_thinking,
        )
    except TypeError:
        rendered = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        if not enable_thinking and rendered.endswith("<think>\n"):
            rendered = rendered[: -len("<think>\n")] + "<think>\n\n</think>\n\n"
        return rendered


def token_scalar(tokens: Any) -> int:
    if hasattr(tokens, "detach"):
        tokens = tokens.detach().cpu()
    if tokens.ndim == 0:
        return int(tokens.item())
    return int(tokens.reshape(-1)[0].item())


def topk_payload(logits: torch.Tensor, k: int) -> dict[str, list[float | int]]:
    values, indices = torch.topk(logits.float(), k=min(k, logits.numel()))
    return {
        "ids": [int(x) for x in indices.cpu().tolist()],
        "values": [float(x) for x in values.cpu().tolist()],
    }


def dump_logits(args: argparse.Namespace) -> None:
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

    cases = load_cases(args.cases)

    print("LOAD_START", args.compiled_path, flush=True)
    t0 = time.perf_counter()
    model = NeuronQwen35ForCausalLM(args.compiled_path)
    model.load(args.compiled_path)
    print(f"LOAD_DONE seconds={time.perf_counter() - t0:.3f}", flush=True)

    if not getattr(model.config.neuron_config, "output_logits", False):
        raise RuntimeError(
            "compiled artifact has output_logits=False; recompile with --output-logits"
        )

    chunk_size = int(args.chunk_size or model.config.neuron_config.max_context_length)
    seq_len = int(model.config.neuron_config.seq_len)
    sampling_params = prepare_sampling_params(
        batch_size=1,
        top_k=[1],
        top_p=[1.0],
        temperature=[1.0],
    )
    seq_ids = torch.tensor([0], dtype=torch.int32)

    records: list[dict[str, Any]] = []
    for case in cases:
        model.reset()
        prompt = render_prompt(
            tokenizer,
            case["messages"],
            enable_thinking=bool(case.get("enable_thinking", False)),
        )
        input_ids = torch.tensor(
            [tokenizer(prompt, add_special_tokens=False).input_ids],
            dtype=torch.long,
        )
        prompt_len = int(input_ids.shape[1])
        if prompt_len > seq_len:
            raise ValueError(f"{case['name']} prompt_len={prompt_len} exceeds seq_len={seq_len}")

        final_logits: torch.Tensor | None = None
        final_token: int | None = None
        prefill_s = 0.0
        for start in range(0, prompt_len, chunk_size):
            end = min(start + chunk_size, prompt_len)
            chunk_ids = input_ids[:, start:end]
            attn_mask = torch.ones((1, end - start), dtype=torch.long)
            pos = torch.arange(start, end, dtype=torch.long).unsqueeze(0)
            t1 = time.perf_counter()
            with torch.no_grad():
                out = model(
                    input_ids=chunk_ids,
                    attention_mask=attn_mask,
                    position_ids=pos,
                    seq_ids=seq_ids,
                    sampling_params=sampling_params,
                    return_dict=True,
                )
            prefill_s += time.perf_counter() - t1
            if not hasattr(out, "logits") or out.logits is None:
                raise RuntimeError("model output did not include logits")
            logits = out.logits.detach().cpu().float()
            if logits.ndim == 3:
                final_logits = logits[0, -1, :].contiguous()
            elif logits.ndim == 2:
                final_logits = logits[0, :].contiguous()
            else:
                raise RuntimeError(f"unexpected logits shape: {tuple(logits.shape)}")
            final_token = token_scalar(out.tokens)

        assert final_logits is not None and final_token is not None
        record = {
            "name": case["name"],
            "messages": case["messages"],
            "prompt_tokens": prompt_len,
            "prefill_seconds": prefill_s,
            "next_token": final_token,
            "next_text": tokenizer.decode([final_token], skip_special_tokens=True),
            "topk": topk_payload(final_logits, args.top_k),
            "logits": final_logits,
        }
        print(
            "CASE",
            case["name"],
            "prompt_tokens",
            prompt_len,
            "next_token",
            final_token,
            "prefill_s",
            f"{prefill_s:.3f}",
            flush=True,
        )
        records.append(record)

    payload = {
        "model_path": args.model_path,
        "compiled_path": args.compiled_path,
        "chunk_size": chunk_size,
        "seq_len": seq_len,
        "records": records,
    }
    out_path = Path(args.out).expanduser()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, out_path)
    print("DUMP_WROTE", out_path, flush=True)


def compare_logits(args: argparse.Namespace) -> None:
    baseline = torch.load(args.baseline, map_location="cpu", weights_only=False)
    candidate = torch.load(args.candidate, map_location="cpu", weights_only=False)
    base_records = {row["name"]: row for row in baseline["records"]}
    cand_records = {row["name"]: row for row in candidate["records"]}
    names = [row["name"] for row in baseline["records"]]

    results: list[dict[str, Any]] = []
    for name in names:
        if name not in cand_records:
            raise ValueError(f"candidate missing case {name}")
        b = base_records[name]["logits"].float()
        c = cand_records[name]["logits"].float()
        if b.shape != c.shape:
            raise ValueError(f"{name} shape mismatch: {tuple(b.shape)} vs {tuple(c.shape)}")
        cosine = float(torch.nn.functional.cosine_similarity(b, c, dim=0).item())
        max_abs = float((b - c).abs().max().item())
        base_top = set(int(x) for x in base_records[name]["topk"]["ids"])
        cand_top = set(int(x) for x in cand_records[name]["topk"]["ids"])
        overlap = len(base_top & cand_top)
        result = {
            "name": name,
            "cosine": cosine,
            "max_abs": max_abs,
            "baseline_next_token": int(base_records[name]["next_token"]),
            "candidate_next_token": int(cand_records[name]["next_token"]),
            "top1_match": int(base_records[name]["next_token"]) == int(cand_records[name]["next_token"]),
            "topk_overlap": overlap,
            "topk_size": len(base_top),
            "baseline_prompt_tokens": int(base_records[name]["prompt_tokens"]),
            "candidate_prompt_tokens": int(cand_records[name]["prompt_tokens"]),
        }
        results.append(result)

    avg_cosine = sum(row["cosine"] for row in results) / len(results)
    report = {
        "baseline": baseline.get("compiled_path"),
        "candidate": candidate.get("compiled_path"),
        "average_cosine": avg_cosine,
        "threshold": args.threshold,
        "passed": avg_cosine >= args.threshold,
        "results": results,
    }
    if args.out:
        out_path = Path(args.out).expanduser()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        print("COMPARE_WROTE", out_path, flush=True)
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)
    if avg_cosine < args.threshold:
        raise SystemExit(3)


def main() -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)

    dump = sub.add_parser("dump")
    dump.add_argument("--repo-root", default=None)
    dump.add_argument("--model-path", required=True)
    dump.add_argument("--compiled-path", required=True)
    dump.add_argument("--out", required=True)
    dump.add_argument("--cases", default=None)
    dump.add_argument("--chunk-size", type=int, default=None)
    dump.add_argument("--top-k", type=int, default=20)

    compare = sub.add_parser("compare")
    compare.add_argument("--baseline", required=True)
    compare.add_argument("--candidate", required=True)
    compare.add_argument("--out", default=None)
    compare.add_argument("--threshold", type=float, default=0.999)

    args = parser.parse_args()
    if args.cmd == "dump":
        dump_logits(args)
    elif args.cmd == "compare":
        compare_logits(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
