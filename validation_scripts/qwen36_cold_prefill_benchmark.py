#!/usr/bin/env python3
"""Cold-prefill benchmark matrix for Qwen3.6-27B vLLM on Neuron."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
RUNNER = (
    REPO_ROOT
    / "contrib"
    / "models"
    / "Qwen3.6-27B"
    / "vllm"
    / "run_offline_inference.py"
)
GDN_RECURRENT_DIFF_KEYS = (
    "gdn_recurrent_state_max_abs_diff",
    "recurrent_state_max_abs_diff",
    "recurrent_max_abs_diff",
    "max_recurrent_diff",
)
GDN_CONV_DIFF_KEYS = (
    "gdn_conv_state_max_abs_diff",
    "conv_state_max_abs_diff",
    "conv_max_abs_diff",
    "max_conv_diff",
)
BENCHMARK_SCHEMA_VERSION = 1


def _load_tokenizer(model_path: str):
    model_path_obj = Path(model_path).expanduser()
    if model_path_obj.is_absolute() and not model_path_obj.exists():
        return None
    try:
        from transformers import AutoTokenizer  # noqa: WPS433

        return AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    except Exception:
        return None


def _token_count(prompt: str, tokenizer) -> int | None:
    if tokenizer is None:
        return None
    try:
        return len(tokenizer.encode(prompt, add_special_tokens=False))
    except TypeError:
        encoded = tokenizer(prompt, add_special_tokens=False)
        return len(encoded["input_ids"])


def _prompt_from_filler_count(filler_count: int) -> str:
    header = "Answer the final arithmetic question with one number.\n"
    filler = " ".join(f"fact{i % 997}" for i in range(max(filler_count, 1)))
    return f"{header}{filler}\nQuestion: What is 17 * 23?"


def _prompt_for_target_tokens(target_tokens: int, tokenizer=None) -> tuple[str, int | None]:
    if tokenizer is None:
        prompt = _prompt_from_filler_count(max(target_tokens - 16, 1))
        return prompt, None

    low = 0
    high = max(target_tokens, 1)
    while True:
        count = _token_count(_prompt_from_filler_count(high), tokenizer)
        if count is None or count >= target_tokens or high > target_tokens * 4:
            break
        low = high + 1
        high *= 2

    best_prompt = _prompt_from_filler_count(high)
    best_count = _token_count(best_prompt, tokenizer)
    while low <= high:
        mid = (low + high) // 2
        prompt = _prompt_from_filler_count(mid)
        count = _token_count(prompt, tokenizer)
        if count is None:
            return prompt, None
        if best_count is None or abs(count - target_tokens) < abs(best_count - target_tokens):
            best_prompt = prompt
            best_count = count
        if count < target_tokens:
            low = mid + 1
        elif count > target_tokens:
            high = mid - 1
        else:
            return prompt, count
    return best_prompt, best_count


def _prompt_sha256(prompt: str) -> str:
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()


def _prompt_file(args, spec_name: str, prompt_len: int, prompt: str) -> Path:
    prompt_dir = args.prompt_dir
    prompt_dir.mkdir(parents=True, exist_ok=True)
    digest = _prompt_sha256(prompt)[:12]
    path = prompt_dir / f"{spec_name}_{prompt_len}_{digest}.txt"
    if not path.exists():
        path.write_text(prompt, encoding="utf-8")
    return path


def _compiled_artifacts_by_len(args) -> dict[int, str]:
    cached = getattr(args, "_compiled_artifacts_by_len_cache", None)
    if cached is not None:
        return cached

    mapping: dict[int, str] = {}
    for raw in getattr(args, "compiled_artifacts_by_len", None) or []:
        if "=" not in raw:
            raise ValueError(
                "--compiled-artifacts-by-len entries must be shaped LEN=PATH"
            )
        seq_len_raw, artifact_path = raw.split("=", 1)
        seq_len = int(seq_len_raw)
        if seq_len <= 0:
            raise ValueError(f"artifact sequence length must be positive: {raw}")
        if not artifact_path:
            raise ValueError(f"artifact path cannot be empty: {raw}")
        mapping[seq_len] = artifact_path
    setattr(args, "_compiled_artifacts_by_len_cache", mapping)
    return mapping


def _compiled_artifacts_for_case(args, seq_len: int) -> str | None:
    return _compiled_artifacts_by_len(args).get(seq_len) or args.compiled_artifacts


def _compiled_artifact_path_evidence(compiled_artifacts: str | None) -> dict:
    if not compiled_artifacts:
        return {
            "compiled_artifacts_resolved": None,
            "compiled_artifacts_path_exists": None,
            "compiled_artifacts_path_nonempty": None,
        }
    artifact_path = Path(compiled_artifacts).expanduser()
    exists = artifact_path.exists()
    nonempty = None
    if artifact_path.is_dir():
        nonempty = any(artifact_path.iterdir())
    elif artifact_path.is_file():
        nonempty = artifact_path.stat().st_size > 0
    return {
        "compiled_artifacts_resolved": str(artifact_path.resolve()) if exists else None,
        "compiled_artifacts_path_exists": exists,
        "compiled_artifacts_path_nonempty": nonempty,
    }


def _base_command(
    args,
    prompt_file: Path,
    seq_len: int,
    max_tokens: int,
    compiled_artifacts: str | None,
    enable_chunked_prefill: bool = True,
) -> list[str]:
    command = [
        sys.executable,
        str(RUNNER),
        "--model-path",
        args.model_path,
        "--prompt-file",
        str(prompt_file),
        "--max-model-len",
        str(seq_len),
        "--seq-len",
        str(seq_len),
        "--tensor-parallel-size",
        str(args.tensor_parallel_size),
        "--logical-nc-config",
        str(args.logical_nc_config),
        "--max-num-seqs",
        str(args.max_num_seqs),
        "--ctx-batch-size",
        str(args.ctx_batch_size),
        "--temperature",
        "0",
        "--top-k",
        "1",
        "--max-tokens",
        str(max_tokens),
    ]
    if enable_chunked_prefill:
        command.append("--enable-vllm-chunked-prefill")
    if compiled_artifacts:
        command.extend(["--compiled-artifacts", compiled_artifacts])
    if args.num_gpu_blocks_override is not None:
        command.extend(
            ["--num-gpu-blocks-override", str(args.num_gpu_blocks_override)]
        )
    return command


def _variant_specs():
    old_chunked_env = {
        "USE_NKI_FUSED": "0",
        "USE_NKI_CHUNKED": "1",
        "USE_PYTORCH_CHUNK": "0",
    }
    fused_env = {
        "USE_NKI_FUSED": "1",
        "USE_NKI_CHUNKED": "0",
        "USE_PYTORCH_CHUNK": "0",
    }
    pytorch_chunk_env = {
        "USE_NKI_FUSED": "0",
        "USE_NKI_CHUNKED": "0",
        "USE_PYTORCH_CHUNK": "1",
    }
    return [
        {
            "name": "A_single512_old_chunked",
            "cte": ["--cte-bucket", "512"],
            "flags": ["--no-text-only-cte", "--no-compact-cte-attention-mask"],
            "env": old_chunked_env,
            "tile_sweep": False,
        },
        {
            "name": "B_short_buckets_old_chunked",
            "cte": ["--cte-bucket-profile", "short"],
            "flags": ["--no-text-only-cte", "--no-compact-cte-attention-mask"],
            "env": old_chunked_env,
            "tile_sweep": False,
        },
        {
            "name": "C_short_text_only_old_chunked",
            "cte": ["--cte-bucket-profile", "short"],
            "flags": ["--text-only-cte", "--no-compact-cte-attention-mask"],
            "env": old_chunked_env,
            "tile_sweep": False,
        },
        {
            "name": "D_short_text_compact_old_chunked",
            "cte": ["--cte-bucket-profile", "short"],
            "flags": ["--text-only-cte", "--compact-cte-attention-mask"],
            "env": old_chunked_env,
            "tile_sweep": False,
        },
        {
            "name": "E_short_text_compact_fused",
            "cte": ["--cte-bucket-profile", "short"],
            "flags": ["--text-only-cte", "--compact-cte-attention-mask"],
            "env": fused_env,
            "tile_sweep": False,
        },
        {
            "name": "K_short_text_compact_pytorch_chunk",
            "cte": ["--cte-bucket-profile", "short"],
            "flags": ["--text-only-cte", "--compact-cte-attention-mask"],
            "env": pytorch_chunk_env,
            "tile_sweep": False,
            "optional": True,
        },
        {
            "name": "L_small_dense_mask_fallback",
            "cte": ["--cte-bucket", "512"],
            "flags": ["--text-only-cte", "--no-compact-cte-attention-mask"],
            "env": old_chunked_env,
            "tile_sweep": False,
            "enable_chunked_prefill": False,
            "only_prompt_lengths": [256, 512],
            "optional": True,
        },
        {
            "name": "F_short_text_compact_fused_cold_zero",
            "cte": ["--cte-bucket-profile", "short"],
            "flags": [
                "--text-only-cte",
                "--compact-cte-attention-mask",
            ],
            "env": fused_env,
            "tile_sweep": False,
        },
        {
            "name": "G_tile_block_sweep",
            "cte": ["--cte-bucket-profile", "short"],
            "flags": [
                "--text-only-cte",
                "--compact-cte-attention-mask",
            ],
            "env": fused_env,
            "tile_sweep": True,
        },
        {
            "name": "M_short_text_compact_fused_cold_zero_ablation",
            "cte": ["--cte-bucket-profile", "short"],
            "flags": [
                "--text-only-cte",
                "--compact-cte-attention-mask",
                "--cold-zero-conv-fast-path",
            ],
            "env": fused_env,
            "tile_sweep": False,
            "optional": True,
        },
        {
            "name": "H_128k_candidate",
            "cte": ["--cte-buckets", "256,512,1024,2048"],
            "flags": ["--text-only-cte", "--compact-cte-attention-mask"],
            "env": fused_env,
            "tile_sweep": False,
            "only_prompt_lengths": [131072],
        },
        {
            "name": "I_262k_recovery_block256",
            "cte": ["--cte-bucket-profile", "262k"],
            "flags": ["--text-only-cte", "--compact-cte-attention-mask"],
            "env": fused_env,
            "tile_cases": [
                {
                    "kernel_q_tile_size": 128,
                    "kernel_kv_tile_size": 1024,
                    "block_size": 256,
                }
            ],
            "only_prompt_lengths": [262144],
        },
        {
            "name": "J_262k_recovery_block128",
            "cte": ["--cte-bucket-profile", "262k"],
            "flags": ["--text-only-cte", "--compact-cte-attention-mask"],
            "env": fused_env,
            "tile_cases": [
                {
                    "kernel_q_tile_size": 128,
                    "kernel_kv_tile_size": 1024,
                    "block_size": 128,
                }
            ],
            "only_prompt_lengths": [262144],
        },
    ]


def _tile_cases(spec, args):
    if "tile_cases" in spec:
        return spec["tile_cases"]
    if not spec["tile_sweep"]:
        return [
            {
                "kernel_q_tile_size": args.kernel_q_tile_size,
                "kernel_kv_tile_size": args.kernel_kv_tile_size,
                "block_size": args.block_size,
            }
        ]
    return [
        {"kernel_q_tile_size": 128, "kernel_kv_tile_size": 512, "block_size": 128},
        {"kernel_q_tile_size": 128, "kernel_kv_tile_size": 1024, "block_size": 128},
        {"kernel_q_tile_size": 128, "kernel_kv_tile_size": 2048, "block_size": 128},
        {"kernel_q_tile_size": 256, "kernel_kv_tile_size": 1024, "block_size": 128},
        {"kernel_q_tile_size": 128, "kernel_kv_tile_size": 1024, "block_size": 256},
    ]


def _spec_supports_prompt_len(spec, prompt_len: int) -> bool:
    only_prompt_lengths = spec.get("only_prompt_lengths")
    return only_prompt_lengths is None or prompt_len in only_prompt_lengths


def _extract_prefixed_json(stdout: str, prefix: str) -> dict | None:
    payload = None
    for line in stdout.splitlines():
        if line.startswith(f"{prefix} "):
            payload = json.loads(line.split(" ", 1)[1])
    return payload


def _extract_metrics(stdout: str) -> dict | None:
    return _extract_prefixed_json(stdout, "COLD_PREFILL_METRICS")


def _extract_generation_metrics(stdout: str) -> dict | None:
    return _extract_prefixed_json(stdout, "GENERATION_METRICS")


def _first_float(payload: dict, names: tuple[str, ...]) -> float | None:
    for name in names:
        value = payload.get(name)
        if value is not None:
            return float(value)
    return None


def _normalise_gdn_state_diff(payload: dict | None) -> dict | None:
    if payload is None:
        return None
    recurrent_diff = _first_float(payload, GDN_RECURRENT_DIFF_KEYS)
    conv_diff = _first_float(payload, GDN_CONV_DIFF_KEYS)
    normalised = dict(payload)
    normalised.setdefault("source", "GDN_STATE_DIFF")
    normalised["raw"] = dict(payload)
    if recurrent_diff is not None:
        normalised["recurrent_state_max_abs_diff"] = recurrent_diff
        normalised["recurrent_max_abs_diff"] = recurrent_diff
    if conv_diff is not None:
        normalised["conv_state_max_abs_diff"] = conv_diff
        normalised["conv_max_abs_diff"] = conv_diff
    return normalised


def _extract_gdn_state_diff(stdout: str) -> dict | None:
    return _normalise_gdn_state_diff(_extract_prefixed_json(stdout, "GDN_STATE_DIFF"))


def _state_diff_key(
    variant: str,
    prompt_len: int,
    max_tokens: int,
    repetition: int,
    tile_case: dict | None = None,
) -> str:
    parts = [variant, str(prompt_len), str(max_tokens), str(repetition)]
    if tile_case is not None:
        parts.extend(
            [
                str(tile_case.get("kernel_q_tile_size")),
                str(tile_case.get("kernel_kv_tile_size")),
                str(tile_case.get("block_size")),
            ]
        )
    return "|".join(parts)


def _state_diff_keys(
    variant: str,
    prompt_len: int,
    max_tokens: int,
    repetition: int,
    tile_case: dict,
) -> tuple[str, ...]:
    return (
        _state_diff_key(variant, prompt_len, max_tokens, repetition, tile_case),
        _state_diff_key(variant, prompt_len, max_tokens, repetition, None),
        f"{variant}|{prompt_len}|{max_tokens}",
        f"{variant}|{prompt_len}",
    )


def _load_gdn_state_diff_sidecar(path: Path | None) -> dict[str, dict]:
    if path is None:
        return {}
    payload = json.loads(path.expanduser().read_text())
    if isinstance(payload, dict):
        records = payload.get("rows", payload)
        if isinstance(records, list):
            return _state_diff_sidecar_from_records(records)
        if isinstance(records, dict):
            return {
                str(key): value
                for key, value in records.items()
                if isinstance(value, dict)
            }
    if isinstance(payload, list):
        return _state_diff_sidecar_from_records(payload)
    raise ValueError("--gdn-state-diff-json must contain a JSON object or list")


def _state_diff_sidecar_from_records(records: list[dict]) -> dict[str, dict]:
    sidecar: dict[str, dict] = {}
    for record in records:
        if not isinstance(record, dict):
            continue
        variant = record.get("variant")
        prompt_len = record.get("target_prompt_tokens", record.get("prompt_len"))
        if variant is None or prompt_len is None:
            continue
        max_tokens = int(record.get("max_tokens", 1))
        repetition = int(record.get("repetition", 0))
        tile_case = {
            name: record.get(name)
            for name in ("kernel_q_tile_size", "kernel_kv_tile_size", "block_size")
            if record.get(name) is not None
        }
        diff_payload = record.get("gdn_state_diff", record)
        key = _state_diff_key(
            str(variant),
            int(prompt_len),
            max_tokens,
            repetition,
            tile_case if len(tile_case) == 3 else None,
        )
        sidecar[key] = diff_payload
    return sidecar


def _gdn_state_diff_from_sidecar(
    sidecar: dict[str, dict],
    *,
    variant: str,
    prompt_len: int,
    max_tokens: int,
    repetition: int,
    tile_case: dict,
) -> dict | None:
    for key in _state_diff_keys(variant, prompt_len, max_tokens, repetition, tile_case):
        if key in sidecar:
            normalised = _normalise_gdn_state_diff(sidecar[key])
            if normalised is not None:
                normalised["source"] = sidecar[key].get("source", "gdn_state_diff_json")
            return normalised
    return None


def _extract_tokens(stdout: str) -> list[int] | None:
    for line in stdout.splitlines():
        if not line.startswith("TOKENS "):
            continue
        payload = line.split(" ", 1)[1]
        try:
            return list(json.loads(payload))
        except json.JSONDecodeError:
            return None
    return None


def _run_case(
    args,
    spec,
    prompt_len: int,
    tile_case: dict,
    max_tokens: int,
    repetition: int = 0,
) -> dict:
    seq_len = max(args.seq_len, prompt_len, 1024)
    prompt_cache = getattr(args, "_prompt_cache", None)
    if prompt_cache is None:
        prompt_cache = {}
        setattr(args, "_prompt_cache", prompt_cache)
    tokenizer = getattr(args, "_tokenizer", None)
    if prompt_len not in prompt_cache:
        prompt_cache[prompt_len] = _prompt_for_target_tokens(prompt_len, tokenizer)
    prompt, prompt_token_count = prompt_cache[prompt_len]
    prompt_file = _prompt_file(args, spec["name"], prompt_len, prompt)
    compiled_artifacts = _compiled_artifacts_for_case(args, seq_len)
    enable_chunked_prefill = bool(spec.get("enable_chunked_prefill", True))
    command = _base_command(
        args,
        prompt_file,
        seq_len,
        max_tokens,
        compiled_artifacts,
        enable_chunked_prefill=enable_chunked_prefill,
    )
    command.extend(spec["cte"])
    command.extend(spec["flags"])
    command.extend(
        [
            "--block-size",
            str(tile_case["block_size"]),
            "--kernel-q-tile-size",
            str(tile_case["kernel_q_tile_size"]),
            "--kernel-kv-tile-size",
            str(tile_case["kernel_kv_tile_size"]),
        ]
    )
    env = os.environ.copy()
    env.update(spec["env"])
    row = {
        "benchmark_schema_version": BENCHMARK_SCHEMA_VERSION,
        "benchmark_script": str(Path(__file__).resolve()),
        "runner": str(RUNNER),
        "variant": spec["name"],
        "target_prompt_tokens": prompt_len,
        "prompt_token_count": prompt_token_count,
        "prompt_sha256": _prompt_sha256(prompt),
        "prompt_bytes": len(prompt.encode("utf-8")),
        "model_path": args.model_path,
        "compiled_artifacts": compiled_artifacts,
        **_compiled_artifact_path_evidence(compiled_artifacts),
        "max_model_len": seq_len,
        "seq_len": seq_len,
        "tensor_parallel_size": args.tensor_parallel_size,
        "logical_nc_config": args.logical_nc_config,
        "max_num_seqs": args.max_num_seqs,
        "ctx_batch_size": args.ctx_batch_size,
        "max_tokens": max_tokens,
        "temperature": 0,
        "top_k": 1,
        "repetition": repetition,
        **tile_case,
        "prompt_file": str(prompt_file),
        "command": command,
        "cte_args": list(spec["cte"]),
        "flags": list(spec["flags"]),
        "env": spec["env"],
        "enable_vllm_chunked_prefill": enable_chunked_prefill,
    }
    if args.dry_run:
        row["dry_run"] = True
        return row

    start = time.perf_counter()
    proc = subprocess.run(
        command,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    row["elapsed_seconds"] = time.perf_counter() - start
    row["returncode"] = proc.returncode
    row["metrics"] = _extract_metrics(proc.stdout)
    row["generation_metrics"] = _extract_generation_metrics(proc.stdout)
    if row["metrics"] is not None and row["generation_metrics"] is not None:
        for key, value in row["generation_metrics"].items():
            if key == "prefill_latency_ms" and value is not None:
                row["metrics"][key] = value
            elif row["metrics"].get(key) is None:
                row["metrics"][key] = value
    row["gdn_state_diff"] = _extract_gdn_state_diff(proc.stdout)
    if row["gdn_state_diff"] is None:
        row["gdn_state_diff"] = _gdn_state_diff_from_sidecar(
            getattr(args, "_gdn_state_diff_sidecar", {}),
            variant=spec["name"],
            prompt_len=prompt_len,
            max_tokens=max_tokens,
            repetition=repetition,
            tile_case=tile_case,
        )
    if row["metrics"] is not None and row["gdn_state_diff"] is not None:
        row["metrics"].setdefault("gdn_state_diff", row["gdn_state_diff"])
    artifact_path_ok = (
        compiled_artifacts is None
        or (
            row.get("compiled_artifacts_path_exists") is True
            and row.get("compiled_artifacts_path_nonempty") is True
        )
    )
    row["artifact_load_success"] = (
        proc.returncode == 0 and row["metrics"] is not None and artifact_path_ok
    )
    row["token_ids"] = _extract_tokens(proc.stdout)
    row["output_tail"] = proc.stdout.splitlines()[-40:]
    return row


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", default="/path/to/Qwen3.6-27B")
    parser.add_argument("--compiled-artifacts")
    parser.add_argument(
        "--compiled-artifacts-by-len",
        nargs="+",
        default=None,
        help=(
            "Override artifacts by selected seq_len, e.g. "
            "2048=/artifacts/2k 131072=/artifacts/128k 262144=/artifacts/262k. "
            "--compiled-artifacts remains the fallback."
        ),
    )
    parser.add_argument(
        "--prompt-lengths",
        nargs="+",
        type=int,
        default=[128, 256, 384, 512, 1024, 2048, 8192, 32768],
    )
    parser.add_argument("--seq-len", type=int, default=2048)
    parser.add_argument(
        "--max-tokens-values",
        nargs="+",
        type=int,
        default=[1, 32],
        help="Run pure-prefill and end-to-end timing variants; default: 1 32.",
    )
    parser.add_argument(
        "--repetitions",
        type=int,
        default=1,
        help="Repeat each matrix row so acceptance can report p50/p95 latency.",
    )
    parser.add_argument("--tensor-parallel-size", type=int, default=4)
    parser.add_argument("--logical-nc-config", type=int, default=2)
    parser.add_argument("--max-num-seqs", type=int, default=1)
    parser.add_argument("--ctx-batch-size", type=int, default=1)
    parser.add_argument("--block-size", type=int, default=128)
    parser.add_argument("--kernel-q-tile-size", type=int, default=128)
    parser.add_argument("--kernel-kv-tile-size", type=int, default=1024)
    parser.add_argument("--num-gpu-blocks-override", type=int)
    parser.add_argument("--prompt-dir", type=Path)
    parser.add_argument(
        "--include-128k",
        action="store_true",
        help="Append the 128K prompt-length gate when a matching artifact is available.",
    )
    parser.add_argument(
        "--include-262k",
        action="store_true",
        help="Append the 262K prompt-length gate after the recovery artifact loads.",
    )
    parser.add_argument(
        "--include-dense-fallback",
        action="store_true",
        help="Append small-S dense-mask fallback rows for token exactness checks.",
    )
    parser.add_argument("--variants", nargs="+", default=None)
    parser.add_argument("--output-json", type=Path)
    parser.add_argument(
        "--gdn-state-diff-json",
        type=Path,
        help=(
            "Optional sidecar with GDN recurrent/conv state diffs keyed by "
            "variant|prompt_len|max_tokens|repetition or as records with row fields."
        ),
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="Stop on the first failing runtime row instead of writing all evidence.",
    )
    return parser.parse_args()


def _selected_specs(args):
    specs = _variant_specs()
    if args.variants:
        wanted = set(args.variants)
        specs = [spec for spec in specs if spec["name"] in wanted]
        missing = wanted - {spec["name"] for spec in specs}
        if missing:
            raise SystemExit(f"unknown variants: {sorted(missing)}")
    else:
        include_optional = set()
        if args.include_dense_fallback:
            include_optional.add("L_small_dense_mask_fallback")
        specs = [
            spec
            for spec in specs
            if not spec.get("optional") or spec["name"] in include_optional
        ]
    return specs


def _run_matrix(args) -> tuple[list[dict], int]:
    prompt_lengths = list(args.prompt_lengths)
    if args.include_128k and 131072 not in prompt_lengths:
        prompt_lengths.append(131072)
    if args.include_262k and 262144 not in prompt_lengths:
        prompt_lengths.append(262144)
    specs = _selected_specs(args)

    rows = []
    first_failure_returncode = 0
    for spec in specs:
        for prompt_len in sorted(set(prompt_lengths)):
            if not _spec_supports_prompt_len(spec, prompt_len):
                continue
            for tile_case in _tile_cases(spec, args):
                for max_tokens in args.max_tokens_values:
                    for repetition in range(args.repetitions):
                        row = _run_case(
                            args,
                            spec,
                            prompt_len,
                            tile_case,
                            max_tokens,
                            repetition=repetition,
                        )
                        rows.append(row)
                        print(json.dumps(row, sort_keys=True), flush=True)
                        if not args.dry_run and row.get("returncode") != 0:
                            if first_failure_returncode == 0:
                                first_failure_returncode = int(row["returncode"])
                            if args.fail_fast:
                                return rows, first_failure_returncode
    return rows, first_failure_returncode


def main() -> int:
    args = parse_args()
    if args.prompt_dir is None:
        args.prompt_dir = Path(tempfile.mkdtemp(prefix="qwen36_cold_prefill_prompts_"))
    args._tokenizer = _load_tokenizer(args.model_path)
    args._prompt_cache = {}
    args._gdn_state_diff_sidecar = _load_gdn_state_diff_sidecar(args.gdn_state_diff_json)
    rows, first_failure_returncode = _run_matrix(args)

    if args.output_json:
        args.output_json.write_text(json.dumps(rows, indent=2, sort_keys=True) + "\n")
    return first_failure_returncode


if __name__ == "__main__":
    raise SystemExit(main())
