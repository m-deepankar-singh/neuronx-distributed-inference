from __future__ import annotations

import json
import math
import os
import re
import time
from typing import Any, Dict, Iterable, List

import requests
import torch
import torch.nn.functional as F


def _accuracy_config(lm_eval_config: Dict[str, Any]) -> Dict[str, Any]:
    return lm_eval_config.get("accuracy") or lm_eval_config


def _tasks(lm_eval_config: Dict[str, Any]) -> List[str]:
    tasks = _accuracy_config(lm_eval_config).get("tasks") or ["hellaswag"]
    return [str(task).lower() for task in tasks]


def _limit(lm_eval_config: Dict[str, Any]) -> int:
    value = _accuracy_config(lm_eval_config).get("limit", 5)
    return max(1, int(value))


def _method(lm_eval_config: Dict[str, Any]) -> str:
    return str(_accuracy_config(lm_eval_config).get("method", "loglikelihood")).lower()


def _fetch_hellaswag_rows(limit: int) -> List[Dict[str, Any]]:
    response = requests.get(
        "https://datasets-server.huggingface.co/rows",
        params={
            "dataset": "Rowan/hellaswag",
            "config": "default",
            "split": "validation",
            "offset": 0,
            "length": limit,
        },
        timeout=60,
    )
    response.raise_for_status()
    rows = response.json()["rows"]
    return [row["row"] for row in rows[:limit]]


def _bench_model_config(output_dir: str) -> Dict[str, Any]:
    exp_dir = os.path.dirname(os.path.dirname(output_dir))
    config_path = os.path.join(exp_dir, "bench_config.yaml")
    import yaml

    with open(config_path) as f:
        raw = yaml.safe_load(f)
    model_config = raw["model"]
    return {
        "model_name": os.path.basename(model_config["model_path"]),
        "model_path": model_config["model_path"],
        "compiled_model_path": model_config["compiled_model_path"],
        "model_class": model_config["model_class"],
        "config_class": model_config["config_class"],
    }


def _choice_ids(tokenizer: Any, ending: str) -> torch.Tensor:
    # HellaSwag endings are continuations of ctx; lm-eval scores the continuation.
    # A leading space keeps the tokenizer boundary consistent with ctx + ending.
    ids = tokenizer(" " + ending, add_special_tokens=False, return_tensors="pt").input_ids
    if ids.numel() == 0:
        raise ValueError("empty choice tokenization")
    return ids.to(torch.long)


def _prompt_ids(tokenizer: Any, prompt: str) -> torch.Tensor:
    return tokenizer(prompt, return_tensors="pt", add_special_tokens=True).input_ids.to(torch.long)


def _chat_prompt_ids(tokenizer: Any, prompt: str) -> torch.Tensor:
    content = prompt + "\n/no_think"
    messages = [{"role": "user", "content": content}]
    if hasattr(tokenizer, "apply_chat_template"):
        try:
            encoded = tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                enable_thinking=False,
                return_tensors="pt",
            )
            if hasattr(encoded, "input_ids"):
                encoded = encoded.input_ids
            elif isinstance(encoded, dict):
                encoded = encoded["input_ids"]
            return encoded.to(torch.long)
        except TypeError:
            try:
                text = tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=False,
                )
                return tokenizer(text, return_tensors="pt", add_special_tokens=False).input_ids.to(torch.long)
            except TypeError:
                pass
    return tokenizer(content, return_tensors="pt", add_special_tokens=True).input_ids.to(torch.long)


def _score_from_logits(logits: torch.Tensor, token_ids: torch.Tensor) -> Dict[str, float]:
    logits = torch.nan_to_num(logits.float(), nan=0.0, posinf=1e6, neginf=-1e6)
    vocab = logits.shape[-1]
    token_ids = token_ids.reshape(-1).to(torch.long)
    valid_len = min(logits.shape[0], token_ids.numel())
    if valid_len == 0:
        return {"score": -math.inf, "score_norm": -math.inf, "tokens": 0}
    token_ids = token_ids[:valid_len]
    if int(token_ids.max().item()) >= vocab:
        return {"score": -math.inf, "score_norm": -math.inf, "tokens": int(valid_len)}
    selected = F.log_softmax(logits[:valid_len, :], dim=-1)[
        torch.arange(valid_len), token_ids
    ]
    score = float(selected.sum().item())
    return {
        "score": score,
        "score_norm": score / float(valid_len),
        "tokens": int(valid_len),
    }


def _score_hf_choice(model: Any, tokenizer: Any, ctx: str, ending: str) -> Dict[str, float]:
    prompt = _prompt_ids(tokenizer, ctx)
    choice = _choice_ids(tokenizer, ending)
    full = torch.cat([prompt, choice], dim=1)
    with torch.no_grad():
        output = model(full)
    logits = output.logits[0].float()
    start = prompt.shape[1] - 1
    choice_logits = logits[start : start + choice.shape[1], :]
    return _score_from_logits(choice_logits, choice)


def _stage7_logits_from_outputs(outputs: Any) -> torch.Tensor:
    logits = outputs.logits if hasattr(outputs, "logits") else outputs
    if isinstance(logits, (list, tuple)):
        candidates = [
            tensor
            for tensor in logits
            if hasattr(tensor, "ndim")
            and tensor.ndim >= 2
            and torch.is_floating_point(tensor)
            and tensor.shape[-1] > 1000
        ]
        if not candidates:
            raise RuntimeError("no floating logits tensor found in target outputs")
        max_last_dim = max(tensor.shape[-1] for tensor in candidates)
        shard_candidates = [tensor for tensor in candidates if tensor.shape[-1] == max_last_dim]
        if len(shard_candidates) > 1:
            base_shape = shard_candidates[0].shape[:-1]
            if all(tensor.shape[:-1] == base_shape for tensor in shard_candidates):
                logits = torch.cat(shard_candidates, dim=-1)
            else:
                logits = max(candidates, key=lambda tensor: tensor.shape[-1])
        else:
            logits = max(candidates, key=lambda tensor: tensor.shape[-1])
    if logits.ndim == 2:
        return logits.float()
    return logits[:, -1, :].float()


def _score_neuron_choice(model: Any, tokenizer: Any, ctx: str, ending: str) -> Dict[str, float]:
    import run_teacher_forced_comparison_qwen_bucket as qwen_tf

    qwen_tf._logits_from_outputs = _stage7_logits_from_outputs
    prompt = _prompt_ids(tokenizer, ctx)
    choice = _choice_ids(tokenizer, ending)
    with torch.no_grad():
        logits = qwen_tf.get_target_logits_teacher_forced(
            model, tokenizer, prompt, choice, choice.shape[1]
        )
    if logits.ndim == 3:
        logits = logits[:, 0, :]
    return _score_from_logits(logits, choice)


def _evaluate_hellaswag(
    *,
    model: Any,
    tokenizer: Any,
    rows: Iterable[Dict[str, Any]],
    output_dir: str,
    model_name: str,
    scorer,
) -> Dict[str, Any]:
    details = []
    raw_correct = 0
    norm_correct = 0
    total = 0

    for index, row in enumerate(rows):
        ctx = row["ctx"]
        endings = row["endings"]
        label = int(row["label"])
        scores = []
        for ending in endings:
            scores.append(scorer(model, tokenizer, ctx, ending))

        raw_pred = max(range(len(scores)), key=lambda i: scores[i]["score"])
        norm_pred = max(range(len(scores)), key=lambda i: scores[i]["score_norm"])
        raw_correct += int(raw_pred == label)
        norm_correct += int(norm_pred == label)
        total += 1
        details.append({
            "index": index,
            "dataset_index": row.get("ind"),
            "ctx": ctx,
            "label": label,
            "raw_pred": raw_pred,
            "norm_pred": norm_pred,
            "scores": scores,
            "endings": endings,
        })
        print(
            f"  {model_name} hellaswag[{index}] label={label} "
            f"raw_pred={raw_pred} norm_pred={norm_pred}"
        )

    result = {
        "hellaswag": {
            "acc": raw_correct / total if total else 0.0,
            "acc_norm": norm_correct / total if total else 0.0,
            "correct": raw_correct,
            "correct_norm": norm_correct,
            "total": total,
            "details": details,
        }
    }
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, f"{model_name}_hellaswag_details.json"), "w") as f:
        json.dump(result, f, indent=2)
    return result


def _choice_prompt(row: Dict[str, Any]) -> str:
    choices = "\n".join(
        f"{chr(ord('A') + idx)}. {ending}" for idx, ending in enumerate(row["endings"])
    )
    return (
        "Choose the most plausible continuation for the context. "
        "Respond with exactly one letter: A, B, C, or D. Do not explain.\n\n"
        f"Context: {row['ctx']}\n"
        f"{choices}\n"
        "Answer:"
    )


def _parse_choice(text: str) -> int:
    match = re.search(r"\b([ABCD])\b", text.upper())
    if not match:
        match = re.search(r"([ABCD])", text.upper())
    if not match:
        return -1
    return ord(match.group(1)) - ord("A")


def _generate_hf_choice(model: Any, tokenizer: Any, prompt: str, max_new_tokens: int) -> str:
    input_ids = _chat_prompt_ids(tokenizer, prompt)
    with torch.no_grad():
        output = model.generate(
            input_ids=input_ids,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )
    return tokenizer.decode(output[0, input_ids.shape[1]:], skip_special_tokens=True)


def _generate_neuron_choice(model: Any, tokenizer: Any, prompt: str, max_new_tokens: int) -> str:
    from validator.accuracy import generate_with_neuron_model

    input_ids = _chat_prompt_ids(tokenizer, prompt)
    with torch.no_grad():
        output = generate_with_neuron_model(model, input_ids, max_new_tokens=max_new_tokens)
    return tokenizer.decode(output[0, input_ids.shape[1]:], skip_special_tokens=True)


def _evaluate_hellaswag_generate(
    *,
    model: Any,
    tokenizer: Any,
    rows: Iterable[Dict[str, Any]],
    output_dir: str,
    model_name: str,
    generator,
    max_new_tokens: int = 4,
) -> Dict[str, Any]:
    details = []
    correct = 0
    total = 0

    for index, row in enumerate(rows):
        prompt = _choice_prompt(row)
        label = int(row["label"])
        text = generator(model, tokenizer, prompt, max_new_tokens)
        pred = _parse_choice(text)
        correct += int(pred == label)
        total += 1
        details.append({
            "index": index,
            "dataset_index": row.get("ind"),
            "ctx": row["ctx"],
            "label": label,
            "pred": pred,
            "generated": text,
            "endings": row["endings"],
        })
        print(
            f"  {model_name} hellaswag[{index}] label={label} pred={pred} "
            f"generated={text!r}"
        )

    result = {
        "hellaswag": {
            "acc": correct / total if total else 0.0,
            "correct": correct,
            "total": total,
            "method": "generate_choice",
            "details": details,
        }
    }
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, f"{model_name}_hellaswag_details.json"), "w") as f:
        json.dump(result, f, indent=2)
    return result


def run_hf_lm_eval_scenarios(model_path: str, lm_eval_config: Dict[str, Any], output_dir: str) -> Dict[str, Any]:
    from transformers import AutoTokenizer
    from validator.main import load_hf_golden_model

    if any(task not in {"hellaswag"} for task in _tasks(lm_eval_config)):
        raise NotImplementedError("local Stage 7 compatibility layer currently supports hellaswag")

    limit = _limit(lm_eval_config)
    rows = _fetch_hellaswag_rows(limit)
    print(f"Loaded {len(rows)} exact HellaSwag validation rows from Hugging Face dataset-server")
    print("Loading HF baseline model in FP32...")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token or tokenizer.bos_token
    model = load_hf_golden_model(
        model_path,
        config=_bench_model_config(output_dir),
        hf_dtype_override="float32",
    )
    start = time.perf_counter()
    if _method(lm_eval_config) == "generate_choice":
        result = _evaluate_hellaswag_generate(
            model=model,
            tokenizer=tokenizer,
            rows=rows,
            output_dir=output_dir,
            model_name="hf",
            generator=_generate_hf_choice,
            max_new_tokens=int(_accuracy_config(lm_eval_config).get("max_new_tokens", 4)),
        )
    else:
        result = _evaluate_hellaswag(
            model=model,
            tokenizer=tokenizer,
            rows=rows,
            output_dir=output_dir,
            model_name="hf",
            scorer=_score_hf_choice,
        )
    result["hellaswag"]["elapsed_s"] = time.perf_counter() - start
    return result


def run_lm_eval_scenarios(
    model: Any,
    tokenizer: Any,
    generation_config: Any,
    lm_eval_config: Dict[str, Any],
    output_dir: str,
) -> Dict[str, Any]:
    del generation_config
    if any(task not in {"hellaswag"} for task in _tasks(lm_eval_config)):
        raise NotImplementedError("local Stage 7 compatibility layer currently supports hellaswag")

    limit = _limit(lm_eval_config)
    rows = _fetch_hellaswag_rows(limit)
    print(f"Loaded {len(rows)} exact HellaSwag validation rows from Hugging Face dataset-server")
    start = time.perf_counter()
    if _method(lm_eval_config) == "generate_choice":
        result = _evaluate_hellaswag_generate(
            model=model,
            tokenizer=tokenizer,
            rows=rows,
            output_dir=output_dir,
            model_name="neuron",
            generator=_generate_neuron_choice,
            max_new_tokens=int(_accuracy_config(lm_eval_config).get("max_new_tokens", 4)),
        )
    else:
        result = _evaluate_hellaswag(
            model=model,
            tokenizer=tokenizer,
            rows=rows,
            output_dir=output_dir,
            model_name="neuron",
            scorer=_score_neuron_choice,
        )
    result["hellaswag"]["elapsed_s"] = time.perf_counter() - start
    if hasattr(model, "reset"):
        model.reset()
    return result


def run_longbench_scenarios(*args, **kwargs):
    raise NotImplementedError("LongBench scenarios are not configured for this Stage 7 run")
