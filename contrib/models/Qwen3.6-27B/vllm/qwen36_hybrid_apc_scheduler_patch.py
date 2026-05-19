"""vLLM scheduler patch for Qwen Hybrid APC fallback.

This module is intentionally opt-in. The safe fallback for current Hybrid APC
validation is to make vLLM skip attention-prefix reads before slot allocation
when the GDN checkpoint side is not integrated with the scheduler yet.
"""

from __future__ import annotations

import hashlib
import importlib.abc
import importlib.machinery
import logging
import os
import struct
import sys
from typing import Any, Hashable, NamedTuple


logger = logging.getLogger(__name__)
_SCHEDULER_MODULE = "vllm.v1.core.sched.scheduler"
_VLLM_NEURON_RUNNER_MODULE = "vllm_neuron.worker.neuronx_distributed_model_runner"
_PATCHED_MODULES = {_SCHEDULER_MODULE, _VLLM_NEURON_RUNNER_MODULE}


class HybridGDNPrefixKey(NamedTuple):
    cumulative_prefix_hash: Hashable
    prefix_len: int
    block_size: int
    cache_salt: Hashable | None
    model_revision: str
    layout_version: int
    tp_rank: int
    recurrent_dtype: str
    conv_dtype: str


_GDN_PREFIX_KEYS: set[HybridGDNPrefixKey] = set()
_AUTHORIZED_PREFIX_READS: dict[int, list[HybridGDNPrefixKey]] = {}
_AUTHORIZED_PREFIX_READS_BY_REQUEST: dict[Hashable, list[HybridGDNPrefixKey]] = {}
_SCHEDULER_OUTPUT_METADATA_ATTR = "_qwen36_hybrid_apc_metadata_by_request_id"
_HYBRID_APC_RUNTIME_CONFIG_KEYS = (
    "use_hybrid_apc_manager",
    "use_qwen_hybrid_chunked_prefill",
    "use_qwen_hybrid_chunked_prefill_nki",
    "gdn_checkpoint_interval",
    "max_gdn_checkpoint_slots",
    "gdn_recurrent_cache_dtype",
    "gdn_conv_cache_dtype",
    "hybrid_recurrent_cache_dtype",
    "hybrid_conv_cache_dtype",
    "hybrid_cache_mode",
    "hybrid_cache_prefix_boundary_only",
    "hybrid_cache_block_boundary_only",
    "hybrid_cache_validate_exact",
    "hybrid_apc_layout_version",
    "hybrid_apc_allow_residual_replay",
    "hybrid_apc_cache_salt",
    "hybrid_apc_model_revision",
    "hybrid_apc_require_vllm_metadata",
    "hybrid_apc_allow_local_hash_fallback",
    "hybrid_apc_require_attention_block_refs",
    "hybrid_apc_reject_unbacked_attention_hits",
    "hybrid_apc_disable_unbacked_prefix_reads",
    "hybrid_apc_enable_backed_prefix_reads",
    "hybrid_apc_max_backed_prefix_read_len",
)
_HYBRID_APC_BRIDGE_CONFIG_ATTRS = {
    "hybrid_apc_allow_local_hash_fallback": "allow_local_hash_fallback",
    "hybrid_apc_require_attention_block_refs": "require_attention_block_refs",
    "hybrid_apc_reject_unbacked_attention_hits": "reject_unbacked_attention_hits",
    "hybrid_apc_cache_salt": "cache_salt",
    "hybrid_apc_model_revision": "model_revision",
    "hybrid_apc_layout_version": "layout_version",
    "hybrid_recurrent_cache_dtype": "recurrent_dtype",
    "hybrid_conv_cache_dtype": "conv_dtype",
}


def _env_flag(name: str) -> bool:
    value = os.environ.get(name)
    return value is not None and value.strip().lower() not in {
        "",
        "0",
        "false",
        "no",
        "off",
    }


def _env_int(name: str) -> int | None:
    value = os.environ.get(name)
    if value is None or value.strip() == "":
        return None
    return int(value)


def _get_hf_config(vllm_config: Any) -> Any:
    model_config = getattr(vllm_config, "model_config", None)
    return getattr(model_config, "hf_config", None)


def _get_additional_config(vllm_config: Any) -> dict[str, Any]:
    additional_config = getattr(vllm_config, "additional_config", None)
    return additional_config if isinstance(additional_config, dict) else {}


def _config_flag(config: Any, name: str, default: bool = False) -> bool:
    return bool(getattr(config, name, default))


def _config_value(config: Any, name: str, default: Any) -> Any:
    return getattr(config, name, default)


def _scheduler_config_flag(
    scheduler: Any,
    name: str,
    default: bool = False,
) -> bool:
    vllm_config = getattr(scheduler, "vllm_config", None)
    additional_config = _get_additional_config(vllm_config)
    if name in additional_config:
        return bool(additional_config[name])
    return _config_flag(_get_hf_config(vllm_config), name, default)


def _scheduler_config_value(
    scheduler: Any,
    name: str,
    default: Any,
) -> Any:
    vllm_config = getattr(scheduler, "vllm_config", None)
    additional_config = _get_additional_config(vllm_config)
    if name in additional_config:
        return additional_config[name]
    return _config_value(_get_hf_config(vllm_config), name, default)


def _max_num_seqs_for_scheduler(scheduler: Any) -> int:
    scheduler_config = getattr(scheduler, "scheduler_config", None)
    max_num_seqs = getattr(scheduler_config, "max_num_seqs", 1)
    return int(max_num_seqs or 1)


def _normalize_dtype(value: Any, default: str) -> str:
    if value is None:
        value = default
    normalized = str(value).lower()
    aliases = {
        "fp32": "float32",
        "float32": "float32",
        "torch.float32": "float32",
        "bf16": "bfloat16",
        "bfloat16": "bfloat16",
        "torch.bfloat16": "bfloat16",
    }
    return aliases.get(normalized, normalized)


def _normalize_request_id(request_id: Any) -> Hashable | None:
    if request_id is None:
        return None
    if isinstance(request_id, list):
        return tuple(request_id)
    try:
        hash(request_id)
    except TypeError:
        return repr(request_id)
    return request_id


def _to_registry_key(key: Any) -> HybridGDNPrefixKey:
    return HybridGDNPrefixKey(
        cumulative_prefix_hash=getattr(key, "cumulative_prefix_hash"),
        prefix_len=int(getattr(key, "prefix_len")),
        block_size=int(getattr(key, "block_size")),
        cache_salt=getattr(key, "cache_salt", None),
        model_revision=str(getattr(key, "model_revision", "unknown")),
        layout_version=int(getattr(key, "layout_version", 1)),
        tp_rank=int(getattr(key, "tp_rank", 0)),
        recurrent_dtype=_normalize_dtype(
            getattr(key, "recurrent_dtype", None),
            "float32",
        ),
        conv_dtype=_normalize_dtype(getattr(key, "conv_dtype", None), "bfloat16"),
    )


def register_hybrid_apc_gdn_checkpoint(key: Any) -> HybridGDNPrefixKey:
    """Publish a committed GDN checkpoint boundary to the scheduler process."""

    registry_key = _to_registry_key(key)
    _GDN_PREFIX_KEYS.add(registry_key)
    if _env_flag("QWEN36_HYBRID_APC_DEBUG"):
        print(
            "[hybrid_apc_debug] scheduler-register "
            f"prefix_len={registry_key.prefix_len} "
            f"model_revision={registry_key.model_revision} "
            f"registry_size={len(_GDN_PREFIX_KEYS)}",
            flush=True,
        )
    return registry_key


def unregister_hybrid_apc_gdn_checkpoint(key: Any) -> bool:
    registry_key = _to_registry_key(key)
    if registry_key not in _GDN_PREFIX_KEYS:
        return False
    _GDN_PREFIX_KEYS.remove(registry_key)
    if _env_flag("QWEN36_HYBRID_APC_DEBUG"):
        print(
            "[hybrid_apc_debug] scheduler-unregister "
            f"prefix_len={registry_key.prefix_len} "
            f"model_revision={registry_key.model_revision} "
            f"registry_size={len(_GDN_PREFIX_KEYS)}",
            flush=True,
        )
    return True


def clear_hybrid_apc_gdn_checkpoint_registry() -> None:
    _GDN_PREFIX_KEYS.clear()
    _AUTHORIZED_PREFIX_READS.clear()
    _AUTHORIZED_PREFIX_READS_BY_REQUEST.clear()


def authorize_hybrid_apc_prefix_read(
    key: Any,
    *,
    request_id: Hashable | None = None,
) -> HybridGDNPrefixKey:
    """Publish a scheduler-approved prefix read for suffix-only model prep."""

    registry_key = _to_registry_key(key)
    normalized_request_id = _normalize_request_id(request_id)
    if normalized_request_id is None:
        _AUTHORIZED_PREFIX_READS.setdefault(registry_key.prefix_len, []).append(
            registry_key
        )
    else:
        _AUTHORIZED_PREFIX_READS_BY_REQUEST.setdefault(
            normalized_request_id,
            [],
        ).append(registry_key)
    return registry_key


def _pop_matching_authorized_key(
    candidates: list[HybridGDNPrefixKey],
    *,
    prefix_len: int,
    cache_salt: Hashable | None,
    model_revision: str,
    layout_version: int,
    tp_rank: int,
    recurrent_dtype: str,
    conv_dtype: str,
) -> HybridGDNPrefixKey | None:
    for idx, key in enumerate(candidates):
        if key.prefix_len != prefix_len:
            continue
        if key.cache_salt != cache_salt:
            continue
        if key.model_revision != str(model_revision):
            continue
        if key.layout_version != int(layout_version):
            continue
        if key.tp_rank != int(tp_rank):
            continue
        if key.recurrent_dtype != recurrent_dtype or key.conv_dtype != conv_dtype:
            continue
        return candidates.pop(idx)
    return None


def pop_hybrid_apc_authorized_prefix_key(
    *,
    prefix_len: int,
    request_id: Hashable | None = None,
    cache_salt: Hashable | None = None,
    model_revision: str = "unknown",
    layout_version: int = 1,
    tp_rank: int = 0,
    recurrent_dtype: str = "float32",
    conv_dtype: str = "bfloat16",
) -> HybridGDNPrefixKey | None:
    """Consume the exact key for a prefix read allowed by the scheduler."""

    prefix_len = int(prefix_len)
    recurrent_dtype = _normalize_dtype(recurrent_dtype, "float32")
    conv_dtype = _normalize_dtype(conv_dtype, "bfloat16")
    normalized_request_id = _normalize_request_id(request_id)
    if normalized_request_id is not None:
        candidates = _AUTHORIZED_PREFIX_READS_BY_REQUEST.get(normalized_request_id)
        if candidates:
            matched = _pop_matching_authorized_key(
                candidates,
                prefix_len=prefix_len,
                cache_salt=cache_salt,
                model_revision=model_revision,
                layout_version=layout_version,
                tp_rank=tp_rank,
                recurrent_dtype=recurrent_dtype,
                conv_dtype=conv_dtype,
            )
            if matched is not None:
                if not candidates:
                    _AUTHORIZED_PREFIX_READS_BY_REQUEST.pop(
                        normalized_request_id,
                        None,
                    )
                return matched

    candidates = _AUTHORIZED_PREFIX_READS.get(prefix_len)
    if not candidates:
        return None
    matched = _pop_matching_authorized_key(
        candidates,
        prefix_len=prefix_len,
        cache_salt=cache_salt,
        model_revision=model_revision,
        layout_version=layout_version,
        tp_rank=tp_rank,
        recurrent_dtype=recurrent_dtype,
        conv_dtype=conv_dtype,
    )
    if matched is not None and not candidates:
        _AUTHORIZED_PREFIX_READS.pop(prefix_len, None)
    if matched is not None:
        return matched
    return None


def _block_size_for_scheduler(scheduler: Any) -> int:
    cache_config = getattr(scheduler, "cache_config", None)
    block_size = getattr(cache_config, "block_size", None)
    if block_size is None:
        hf_config = _get_hf_config(getattr(scheduler, "vllm_config", None))
        block_size = _config_value(
            hf_config,
            "gdn_checkpoint_interval",
            0,
        )
    return int(block_size or 0)


def _local_cumulative_prefix_hashes(
    token_ids: list[int] | tuple[int, ...],
    *,
    block_size: int,
    max_prefix_len: int,
) -> dict[int, str]:
    max_prefix_len = max(0, int(max_prefix_len))
    max_prefix_len = max_prefix_len // block_size * block_size
    parent_digest = b""
    hashes: dict[int, str] = {}
    for block_start in range(0, max_prefix_len, block_size):
        block_end = block_start + block_size
        block = [int(token) for token in token_ids[block_start:block_end]]
        digest = hashlib.blake2b(digest_size=16)
        digest.update(parent_digest)
        digest.update(struct.pack("<QQ", block_size, block_end))
        digest.update(struct.pack("<" + "q" * len(block), *block))
        parent_digest = digest.digest()
        hashes[block_end] = parent_digest.hex()
    return hashes


def _vllm_cumulative_prefix_hashes(
    request: Any,
    *,
    block_size: int,
    max_prefix_len: int | None = None,
) -> dict[int, Hashable]:
    block_hashes = list(getattr(request, "block_hashes", ()) or ())
    if not block_hashes:
        return {}
    if max_prefix_len is None:
        max_prefix_len = len(block_hashes) * block_size
    max_prefix_len = max(0, int(max_prefix_len))
    max_prefix_len = max_prefix_len // block_size * block_size
    hashes: dict[int, Hashable] = {}
    for index, block_hash in enumerate(block_hashes):
        prefix_len = (index + 1) * block_size
        if prefix_len > max_prefix_len:
            break
        hashes[prefix_len] = block_hash
    return hashes


def _candidate_cumulative_prefix_hashes(
    scheduler: Any,
    request: Any,
    *,
    max_prefix_len: int,
) -> list[dict[int, Hashable]]:
    block_size = _block_size_for_scheduler(scheduler)
    if block_size <= 0:
        return []
    candidates = []
    vllm_hashes = _vllm_cumulative_prefix_hashes(
        request,
        block_size=block_size,
        max_prefix_len=max_prefix_len,
    )
    if vllm_hashes:
        candidates.append(vllm_hashes)
    token_ids = getattr(request, "prompt_token_ids", None)
    if token_ids:
        local_hashes = _local_cumulative_prefix_hashes(
            token_ids,
            block_size=block_size,
            max_prefix_len=max_prefix_len,
        )
        if local_hashes:
            candidates.append(local_hashes)
    return candidates


def _request_registry_key(
    *,
    scheduler: Any,
    request: Any,
    cumulative_prefix_hash: Hashable,
    prefix_len: int,
    block_size: int,
) -> HybridGDNPrefixKey:
    return HybridGDNPrefixKey(
        cumulative_prefix_hash=cumulative_prefix_hash,
        prefix_len=int(prefix_len),
        block_size=int(block_size),
        cache_salt=getattr(request, "cache_salt", None),
        model_revision=str(
            _scheduler_config_value(
                scheduler,
                "hybrid_apc_model_revision",
                "unknown",
            )
        ),
        layout_version=int(
            _scheduler_config_value(scheduler, "hybrid_apc_layout_version", 1)
        ),
        tp_rank=int(_scheduler_config_value(scheduler, "tp_rank", 0)),
        recurrent_dtype=_normalize_dtype(
            _scheduler_config_value(
                scheduler,
                "hybrid_recurrent_cache_dtype",
                _scheduler_config_value(
                    scheduler,
                    "gdn_recurrent_cache_dtype",
                    "float32",
                ),
            ),
            "float32",
        ),
        conv_dtype=_normalize_dtype(
            _scheduler_config_value(
                scheduler,
                "hybrid_conv_cache_dtype",
                _scheduler_config_value(
                    scheduler,
                    "gdn_conv_cache_dtype",
                    "bfloat16",
                ),
            ),
            "bfloat16",
        ),
    )


def _request_max_cache_hit_len(scheduler: Any, request: Any) -> int:
    if request is None:
        return 0
    block_size = _block_size_for_scheduler(scheduler)
    if block_size <= 0:
        return 0
    token_ids = getattr(request, "prompt_token_ids", None)
    token_count = int(getattr(request, "num_tokens", len(token_ids or ())))
    max_cache_hit_len = max(0, token_count - 1)
    if token_ids:
        max_cache_hit_len = min(max_cache_hit_len, len(token_ids))
    return max_cache_hit_len


def backed_gdn_prefix_hits(scheduler: Any, request: Any) -> dict[int, HybridGDNPrefixKey]:
    """Return request prefix lengths with registered GDN checkpoints."""

    if request is None:
        return {}
    block_size = _block_size_for_scheduler(scheduler)
    max_cache_hit_len = _request_max_cache_hit_len(scheduler, request)
    if block_size <= 0 or max_cache_hit_len <= 0:
        return {}
    hits: dict[int, HybridGDNPrefixKey] = {}
    for hashes in _candidate_cumulative_prefix_hashes(
        scheduler,
        request,
        max_prefix_len=max_cache_hit_len,
    ):
        for prefix_len in sorted(hashes, reverse=True):
            if prefix_len in hits:
                continue
            key = _request_registry_key(
                scheduler=scheduler,
                request=request,
                cumulative_prefix_hash=hashes[prefix_len],
                prefix_len=prefix_len,
                block_size=block_size,
            )
            if key in _GDN_PREFIX_KEYS:
                hits[prefix_len] = key
    return hits


def backed_gdn_prefix_hit(scheduler: Any, request: Any) -> HybridGDNPrefixKey | None:
    """Return the largest request prefix with a registered GDN checkpoint."""

    hits = backed_gdn_prefix_hits(scheduler, request)
    if not hits:
        return None
    return hits[max(hits)]


def _required_backed_prefix_lens(scheduler: Any, request: Any) -> tuple[int, ...]:
    if request is None:
        return ()
    block_size = _block_size_for_scheduler(scheduler)
    max_cache_hit_len = _request_max_cache_hit_len(scheduler, request)
    if block_size <= 0 or max_cache_hit_len <= 0:
        return ()
    required: set[int] = set()
    for hashes in _candidate_cumulative_prefix_hashes(
        scheduler,
        request,
        max_prefix_len=max_cache_hit_len,
    ):
        required.update(int(prefix_len) for prefix_len in hashes)
    return tuple(sorted(required))


def _block_id_groups(block_ids: Any) -> list[list[int]]:
    if block_ids is None:
        return []
    if isinstance(block_ids, tuple):
        groups = block_ids
    elif (
        isinstance(block_ids, list)
        and block_ids
        and all(isinstance(item, (list, tuple)) for item in block_ids)
    ):
        groups = tuple(block_ids)
    else:
        groups = (block_ids,)
    normalized = []
    for group in groups:
        try:
            normalized.append([int(block_id) for block_id in group])
        except TypeError:
            continue
    return normalized


def _attention_block_refs_by_prefix_len(
    block_ids: Any,
    *,
    block_size: int,
) -> dict[int, tuple[int, ...]]:
    groups = _block_id_groups(block_ids)
    if not groups:
        return {}
    max_blocks = max(len(group) for group in groups)
    refs_by_prefix_len: dict[int, tuple[int, ...]] = {}
    for block_count in range(1, max_blocks + 1):
        refs: list[int] = []
        for group in groups:
            refs.extend(group[:block_count])
        if refs:
            refs_by_prefix_len[block_count * block_size] = tuple(refs)
    return refs_by_prefix_len


def _scheduler_request_metadata(
    scheduler: Any,
    request: Any,
    *,
    block_ids: Any = None,
    num_computed_tokens: int | None = None,
) -> dict[str, Any]:
    block_size = _block_size_for_scheduler(scheduler)
    if request is None or block_size <= 0:
        return {}
    token_ids = getattr(request, "prompt_token_ids", None)
    request_prefix_len = int(getattr(request, "num_tokens", len(token_ids or ())))
    cumulative_hashes = _vllm_cumulative_prefix_hashes(
        request,
        block_size=block_size,
        max_prefix_len=request_prefix_len,
    )
    metadata: dict[str, Any] = {}
    if cumulative_hashes:
        metadata["cumulative_hashes_by_prefix_len"] = cumulative_hashes
    refs_by_prefix_len = _attention_block_refs_by_prefix_len(
        block_ids,
        block_size=block_size,
    )
    if refs_by_prefix_len:
        metadata["attention_block_refs_by_prefix_len"] = refs_by_prefix_len
    metadata["request_prefix_len"] = request_prefix_len
    if num_computed_tokens is not None:
        metadata["vllm_attention_hit_len"] = int(num_computed_tokens)
    return metadata


def _request_from_scheduler(scheduler: Any, req_id: Any) -> Any:
    requests = getattr(scheduler, "requests", None)
    if isinstance(requests, dict):
        return requests.get(req_id)
    return None


def _attach_scheduler_output_metadata(scheduler: Any, scheduler_output: Any) -> None:
    metadata_by_request_id: dict[Hashable, dict[str, Any]] = {}
    for req_data in getattr(scheduler_output, "scheduled_new_reqs", ()) or ():
        req_id = getattr(req_data, "req_id", None)
        request = _request_from_scheduler(scheduler, req_id)
        metadata = _scheduler_request_metadata(
            scheduler,
            request,
            block_ids=getattr(req_data, "block_ids", None),
            num_computed_tokens=getattr(req_data, "num_computed_tokens", None),
        )
        if metadata:
            metadata_by_request_id[_normalize_request_id(req_id)] = metadata

    cached_reqs = getattr(scheduler_output, "scheduled_cached_reqs", None)
    req_ids = list(getattr(cached_reqs, "req_ids", ()) or ())
    new_block_ids = list(getattr(cached_reqs, "new_block_ids", ()) or ())
    num_computed_tokens = list(
        getattr(cached_reqs, "num_computed_tokens", ()) or ()
    )
    for index, req_id in enumerate(req_ids):
        request = _request_from_scheduler(scheduler, req_id)
        block_ids = new_block_ids[index] if index < len(new_block_ids) else None
        computed = (
            int(num_computed_tokens[index])
            if index < len(num_computed_tokens)
            else None
        )
        metadata = _scheduler_request_metadata(
            scheduler,
            request,
            block_ids=block_ids,
            num_computed_tokens=computed,
        )
        if metadata:
            metadata_by_request_id[_normalize_request_id(req_id)] = metadata

    if metadata_by_request_id:
        setattr(
            scheduler_output,
            _SCHEDULER_OUTPUT_METADATA_ATTR,
            metadata_by_request_id,
        )


def backed_gdn_prefix_hit_len(scheduler: Any, request: Any) -> int:
    hit = backed_gdn_prefix_hit(scheduler, request)
    if hit is None:
        return 0
    return hit.prefix_len


def _request_id_for_scheduler_request(request: Any) -> Hashable | None:
    if request is None:
        return None
    for attr in ("request_id", "req_id", "id"):
        request_id = getattr(request, attr, None)
        if request_id is not None:
            return _normalize_request_id(request_id)
    return None


def _supports_backed_prefix_reads(scheduler: Any) -> bool:
    """Return whether this artifact can consume a backed Hybrid APC prefix."""

    if _env_flag("QWEN36_HYBRID_APC_ENABLE_BACKED_PREFIX_READS"):
        return True

    if not _scheduler_config_flag(scheduler, "hybrid_apc_enable_backed_prefix_reads"):
        return False

    # A backed GDN checkpoint is not enough on its own. The CTE graph must also
    # consume attention KV prefix state; otherwise warm requests restore GDN
    # state but full-attention layers still see only the suffix.
    return _scheduler_config_flag(scheduler, "use_qwen_hybrid_chunked_prefill")


def _max_backed_prefix_read_len(scheduler: Any) -> int:
    env_value = _env_int("QWEN36_HYBRID_APC_MAX_BACKED_PREFIX_READ_LEN")
    if env_value is not None:
        return max(0, env_value)
    return max(
        0,
        int(
            _scheduler_config_value(
                scheduler,
                "hybrid_apc_max_backed_prefix_read_len",
                0,
            )
            or 0
        ),
    )


def should_disable_unbacked_prefix_reads(scheduler: Any, request: Any = None) -> bool:
    """Return whether this scheduler should avoid vLLM APC reads.

    The current Qwen Hybrid APC control plane can prove an attention hit is
    invalid only inside model request prep. That is too late for allocation.
    This opt-in fallback makes vLLM allocate the request as no-prefix unless
    the scheduler process has a registered matching GDN checkpoint boundary and
    the compiled artifact can consume the matching attention KV prefix in CTE.
    """

    if _env_flag("QWEN36_HYBRID_APC_ENABLE_PREFIX_READS"):
        return False

    disable_requested = _env_flag("QWEN36_HYBRID_APC_DISABLE_UNBACKED_PREFIX_READS")
    if not disable_requested:
        if not _scheduler_config_flag(scheduler, "use_hybrid_apc_manager"):
            return False
        disable_requested = _scheduler_config_flag(
            scheduler,
            "hybrid_apc_disable_unbacked_prefix_reads",
        )
    if not disable_requested:
        return False
    backed_hits = backed_gdn_prefix_hits(scheduler, request)
    required_prefix_lens = _required_backed_prefix_lens(scheduler, request)
    backed_hit_len = max(backed_hits) if backed_hits else 0
    missing_backed_lens = [
        prefix_len for prefix_len in required_prefix_lens if prefix_len not in backed_hits
    ]
    supports_backed = _supports_backed_prefix_reads(scheduler)
    max_backed_prefix_read_len = _max_backed_prefix_read_len(scheduler)
    exceeds_backed_prefix_cap = (
        max_backed_prefix_read_len > 0
        and bool(required_prefix_lens)
        and max(required_prefix_lens) > max_backed_prefix_read_len
    )
    if _env_flag("QWEN36_HYBRID_APC_DEBUG"):
        prompt_len = len(getattr(request, "prompt_token_ids", ()) or ())
        print(
            "[hybrid_apc_debug] scheduler-decision "
            f"disable_requested={disable_requested} "
            f"backed_hit_len={backed_hit_len} "
            f"supports_backed={supports_backed} "
            f"max_num_seqs={_max_num_seqs_for_scheduler(scheduler)} "
            f"prompt_len={prompt_len} "
            f"required_backed_lens={required_prefix_lens} "
            f"missing_backed_lens={tuple(missing_backed_lens)} "
            f"max_backed_prefix_read_len={max_backed_prefix_read_len} "
            f"exceeds_backed_prefix_cap={exceeds_backed_prefix_cap} "
            f"registry_size={len(_GDN_PREFIX_KEYS)}",
            flush=True,
        )
    if (
        required_prefix_lens
        and not missing_backed_lens
        and not exceeds_backed_prefix_cap
        and supports_backed
    ):
        request_id = _request_id_for_scheduler_request(request)
        for prefix_len in required_prefix_lens:
            authorize_hybrid_apc_prefix_read(
                backed_hits[prefix_len],
                request_id=request_id,
            )
        return False
    return True


def patch_scheduler_class(scheduler_cls: type) -> bool:
    """Patch a vLLM Scheduler class in-place.

    Returns True if this call installed the patch, False if the class was
    already patched.
    """

    original_add_request = getattr(scheduler_cls, "add_request", None)
    if original_add_request is None:
        raise AttributeError(f"{scheduler_cls!r} has no add_request method")
    installed = False

    if not getattr(original_add_request, "_qwen36_hybrid_apc_patched", False):

        def add_request_with_hybrid_apc_fallback(self, request):
            if should_disable_unbacked_prefix_reads(self, request):
                request.skip_reading_prefix_cache = True
            return original_add_request(self, request)

        add_request_with_hybrid_apc_fallback._qwen36_hybrid_apc_patched = True
        add_request_with_hybrid_apc_fallback._qwen36_original_add_request = (
            original_add_request
        )
        scheduler_cls.add_request = add_request_with_hybrid_apc_fallback
        installed = True

    original_schedule = getattr(scheduler_cls, "schedule", None)
    if original_schedule is not None and not getattr(
        original_schedule,
        "_qwen36_hybrid_apc_metadata_patched",
        False,
    ):

        def schedule_with_hybrid_apc_metadata(self, *args, **kwargs):
            scheduler_output = original_schedule(self, *args, **kwargs)
            _attach_scheduler_output_metadata(self, scheduler_output)
            return scheduler_output

        schedule_with_hybrid_apc_metadata._qwen36_hybrid_apc_metadata_patched = True
        schedule_with_hybrid_apc_metadata._qwen36_original_schedule = (
            original_schedule
        )
        scheduler_cls.schedule = schedule_with_hybrid_apc_metadata
        installed = True

    return installed


def _patch_scheduler_module(module: Any) -> bool:
    scheduler_cls = getattr(module, "Scheduler", None)
    if scheduler_cls is None:
        return False
    installed = patch_scheduler_class(scheduler_cls)
    if installed:
        logger.info("Installed Qwen Hybrid APC scheduler fallback patch")
    return installed


def _request_ids_from_model_input(model_input: Any) -> tuple[Hashable, ...] | None:
    request_ids = getattr(model_input, "request_ids", None)
    return _as_request_id_tuple(request_ids)


def _as_request_id_tuple(request_ids: Any) -> tuple[Hashable, ...] | None:
    if request_ids is None:
        return None
    if isinstance(request_ids, tuple):
        return request_ids
    if isinstance(request_ids, list):
        return tuple(request_ids)
    if isinstance(request_ids, (str, bytes)):
        return (request_ids,)
    try:
        return tuple(request_ids)
    except TypeError:
        return (request_ids,)


def _request_ids_from_scheduler_output(
    scheduler_output: Any,
    *,
    kind: str,
) -> tuple[Hashable, ...] | None:
    if kind == "cached":
        cached_reqs = getattr(scheduler_output, "scheduled_cached_reqs", None)
        return _as_request_id_tuple(getattr(cached_reqs, "req_ids", None))
    if kind == "new":
        new_reqs = getattr(scheduler_output, "scheduled_new_reqs", None)
        if new_reqs is None:
            return None
        return tuple(getattr(req, "req_id") for req in new_reqs)
    raise ValueError(f"unknown scheduler request kind: {kind}")


def _request_id_target_models(model: Any) -> list[Any]:
    targets = []
    seen = set()
    current = model
    for _ in range(4):
        if current is None:
            break
        current_id = id(current)
        if current_id in seen:
            break
        seen.add(current_id)
        targets.append(current)
        current = getattr(current, "model", None)
    return targets


def _runner_hybrid_apc_runtime_config(runner: Any) -> dict[str, Any]:
    additional_config = _get_additional_config(getattr(runner, "vllm_config", None))
    runtime_config = {
        key: additional_config[key]
        for key in _HYBRID_APC_RUNTIME_CONFIG_KEYS
        if key in additional_config
    }
    if runtime_config.get("hybrid_apc_require_vllm_metadata"):
        runtime_config["hybrid_apc_allow_local_hash_fallback"] = False
        runtime_config["hybrid_apc_require_attention_block_refs"] = True
        runtime_config["hybrid_apc_reject_unbacked_attention_hits"] = True
    return runtime_config


def _config_targets_for_model(model: Any) -> list[Any]:
    targets = []
    seen = set()
    for target in _request_id_target_models(model):
        config = getattr(target, "config", None)
        if config is not None:
            config_id = id(config)
            if config_id not in seen:
                seen.add(config_id)
                targets.append(config)
        if any(hasattr(target, key) for key in _HYBRID_APC_RUNTIME_CONFIG_KEYS):
            target_id = id(target)
            if target_id not in seen:
                seen.add(target_id)
                targets.append(target)
    return targets


def _apply_runtime_config_values(
    *,
    target: Any,
    values: dict[str, Any],
    previous_values: list[tuple[Any, str, Any]],
    missing: Any,
) -> None:
    for attr, value in values.items():
        previous_values.append((target, attr, getattr(target, attr, missing)))
        setattr(target, attr, value)


def _apply_hybrid_apc_runtime_config(
    model: Any,
    values: dict[str, Any],
    *,
    previous_values: list[tuple[Any, str, Any]],
    missing: Any,
) -> None:
    if not values:
        return
    for target in _config_targets_for_model(model):
        _apply_runtime_config_values(
            target=target,
            values=values,
            previous_values=previous_values,
            missing=missing,
        )
    bridge_values = {
        bridge_attr: values[config_attr]
        for config_attr, bridge_attr in _HYBRID_APC_BRIDGE_CONFIG_ATTRS.items()
        if config_attr in values
    }
    if not bridge_values:
        return
    for target in _request_id_target_models(model):
        bridge = getattr(target, "hybrid_apc_bridge", None)
        if bridge is None:
            continue
        _apply_runtime_config_values(
            target=bridge,
            values=bridge_values,
            previous_values=previous_values,
            missing=missing,
        )


def _debug_logits_tensor(stage: str, tensor: Any) -> None:
    if not _env_flag("QWEN36_VLLM_LOGITS_DEBUG"):
        return
    if tensor is None or not hasattr(tensor, "numel"):
        print(f"[qwen36_vllm_logits_debug] stage={stage} tensor=none", flush=True)
        return
    try:
        import torch  # noqa: WPS433

        if tensor.numel() == 0:
            print(
                "[qwen36_vllm_logits_debug] "
                f"stage={stage} shape={tuple(tensor.shape)} dtype={tensor.dtype} empty",
                flush=True,
            )
            return
        flat = tensor.detach().reshape(-1)
        if torch.is_floating_point(flat):
            finite_mask = torch.isfinite(flat)
            finite_count = int(finite_mask.sum().item())
            nan_count = int(torch.isnan(flat).sum().item())
            posinf_count = int(
                torch.logical_and(torch.isinf(flat), flat > 0).sum().item()
            )
            neginf_count = int(
                torch.logical_and(torch.isinf(flat), flat < 0).sum().item()
            )
            if finite_count:
                finite_flat = flat[finite_mask].float()
                finite_min = float(finite_flat.min().item())
                finite_max = float(finite_flat.max().item())
            else:
                finite_min = "none"
                finite_max = "none"
            row_argmax = []
            row_argmax_values = []
            if tensor.ndim >= 2:
                rows = tensor.detach().float().reshape(tensor.shape[0], -1)
                argmax = rows.argmax(dim=-1)
                row_argmax = [int(item) for item in argmax[:8].cpu().tolist()]
                row_argmax_values = [
                    float(rows[row, argmax[row]].item())
                    for row in range(min(rows.shape[0], 8))
                ]
            print(
                "[qwen36_vllm_logits_debug] "
                f"stage={stage} shape={tuple(tensor.shape)} dtype={tensor.dtype} "
                f"finite={finite_count} nan={nan_count} posinf={posinf_count} "
                f"neginf={neginf_count} finite_min={finite_min} "
                f"finite_max={finite_max} row_argmax={row_argmax} "
                f"row_argmax_values={row_argmax_values}",
                flush=True,
            )
        else:
            flat_i64 = flat.to(torch.int64)
            print(
                "[qwen36_vllm_logits_debug] "
                f"stage={stage} shape={tuple(tensor.shape)} dtype={tensor.dtype} "
                f"min={int(flat_i64.min().item())} max={int(flat_i64.max().item())}",
                flush=True,
            )
    except Exception as exc:  # pragma: no cover - diagnostic only
        print(
            "[qwen36_vllm_logits_debug] "
            f"stage={stage} summary_error={type(exc).__name__}: {exc}",
            flush=True,
        )


def _expand_completed_prefill_logits(hidden_states: Any, model_input: Any) -> Any:
    """Restore completed-only CTE logits to vLLM's scheduled request rows."""
    prefill_state = getattr(model_input, "prefill_completion_state", None)
    if prefill_state is None or not hasattr(hidden_states, "shape"):
        return hidden_states
    if len(getattr(hidden_states, "shape", ())) == 0:
        return hidden_states

    try:
        import torch  # noqa: WPS433

        if hasattr(prefill_state, "detach"):
            state_values = [
                bool(item)
                for item in prefill_state.detach().cpu().reshape(-1).tolist()
            ]
        else:
            state_values = [bool(item) for item in prefill_state]
        scheduled_rows = len(state_values)
        output_rows = int(hidden_states.shape[0])
        if output_rows == scheduled_rows:
            return hidden_states

        completed_rows = [idx for idx, is_done in enumerate(state_values) if is_done]
        if output_rows != len(completed_rows):
            return hidden_states
        if not torch.is_floating_point(hidden_states):
            return hidden_states

        expanded = hidden_states.new_full(
            (scheduled_rows, *tuple(hidden_states.shape[1:])),
            float("-inf"),
        )
        for src_row, dst_row in enumerate(completed_rows):
            expanded[dst_row] = hidden_states[src_row]
        if _env_flag("QWEN36_VLLM_LOGITS_DEBUG"):
            print(
                "[qwen36_vllm_logits_debug] "
                f"expanded_completed_prefill_logits output_rows={output_rows} "
                f"scheduled_rows={scheduled_rows} completed_rows={completed_rows}",
                flush=True,
            )
        return expanded
    except Exception as exc:  # pragma: no cover - defensive shim only
        if _env_flag("QWEN36_VLLM_LOGITS_DEBUG"):
            print(
                "[qwen36_vllm_logits_debug] "
                f"expand_completed_prefill_logits_error={type(exc).__name__}: {exc}",
                flush=True,
            )
        return hidden_states


def patch_neuron_model_runner_class(runner_cls: type) -> bool:
    """Patch vLLM-Neuron runner to expose scheduler row metadata."""

    original_execute = getattr(runner_cls, "_execute_model_for_text", None)
    if original_execute is None:
        raise AttributeError(
            f"{runner_cls!r} has no _execute_model_for_text method"
        )
    original_prepare = getattr(runner_cls, "_prepare_model_input", None)
    original_prepare_logits = getattr(
        runner_cls,
        "_prepare_logits_for_sampling",
        None,
    )

    missing = object()
    installed = False

    if original_prepare is not None and not getattr(
        original_prepare,
        "_qwen36_hybrid_apc_model_input_patched",
        False,
    ):

        def prepare_model_input_with_hybrid_apc_metadata(
            self,
            scheduler_output,
            *args,
            **kwargs,
        ):
            model_input = original_prepare(self, scheduler_output, *args, **kwargs)
            object.__setattr__(
                model_input,
                "_qwen36_cached_request_ids",
                _request_ids_from_scheduler_output(
                    scheduler_output,
                    kind="cached",
                ),
            )
            object.__setattr__(
                model_input,
                "_qwen36_new_request_ids",
                _request_ids_from_scheduler_output(
                    scheduler_output,
                    kind="new",
                ),
            )
            metadata_by_request_id = getattr(
                scheduler_output,
                _SCHEDULER_OUTPUT_METADATA_ATTR,
                None,
            )
            if metadata_by_request_id is not None:
                object.__setattr__(
                    model_input,
                    _SCHEDULER_OUTPUT_METADATA_ATTR,
                    metadata_by_request_id,
                )
            return model_input

        prepare_model_input_with_hybrid_apc_metadata._qwen36_hybrid_apc_model_input_patched = (
            True
        )
        prepare_model_input_with_hybrid_apc_metadata._qwen36_original_prepare_model_input = (
            original_prepare
        )
        runner_cls._prepare_model_input = prepare_model_input_with_hybrid_apc_metadata
        installed = True

    if original_prepare_logits is not None and not getattr(
        original_prepare_logits,
        "_qwen36_vllm_logits_debug_patched",
        False,
    ):

        def prepare_logits_for_sampling_with_debug(
            self,
            hidden_states,
            model_input,
            *args,
            **kwargs,
        ):
            if _env_flag("QWEN36_VLLM_LOGITS_DEBUG"):
                _debug_logits_tensor("runner_hidden_states_before_prepare", hidden_states)
                prefill_state = getattr(model_input, "prefill_completion_state", None)
                request_ids = getattr(model_input, "request_ids", None)
                print(
                    "[qwen36_vllm_logits_debug] "
                    f"request_ids={request_ids} prefill_completion_state={prefill_state}",
                    flush=True,
                )
            hidden_states = _expand_completed_prefill_logits(hidden_states, model_input)
            _debug_logits_tensor(
                "runner_hidden_states_after_prefill_expand",
                hidden_states,
            )
            logits = original_prepare_logits(
                self,
                hidden_states,
                model_input,
                *args,
                **kwargs,
            )
            _debug_logits_tensor("runner_logits_after_prepare", logits)
            return logits

        prepare_logits_for_sampling_with_debug._qwen36_vllm_logits_debug_patched = (
            True
        )
        prepare_logits_for_sampling_with_debug._qwen36_original_prepare_logits = (
            original_prepare_logits
        )
        runner_cls._prepare_logits_for_sampling = prepare_logits_for_sampling_with_debug
        installed = True

    if getattr(original_execute, "_qwen36_hybrid_apc_request_ids_patched", False):
        return installed

    def execute_model_for_text_with_request_ids(self, model_input, *args, **kwargs):
        model = getattr(self, "model", None)
        runtime_config = _runner_hybrid_apc_runtime_config(self)
        metadata = {
            "_qwen36_vllm_request_ids": _request_ids_from_model_input(model_input),
            "_qwen36_vllm_cached_request_ids": getattr(
                model_input,
                "_qwen36_cached_request_ids",
                None,
            ),
            "_qwen36_vllm_new_request_ids": getattr(
                model_input,
                "_qwen36_new_request_ids",
                None,
            ),
            "_qwen36_vllm_prefill_completion_state": getattr(
                model_input,
                "prefill_completion_state",
                None,
            ),
            "_qwen36_vllm_hybrid_apc_metadata_by_request_id": getattr(
                model_input,
                _SCHEDULER_OUTPUT_METADATA_ATTR,
                None,
            ),
        }
        previous_values = []
        _apply_hybrid_apc_runtime_config(
            model,
            runtime_config,
            previous_values=previous_values,
            missing=missing,
        )
        if any(value is not None for value in metadata.values()):
            for target in _request_id_target_models(model):
                for attr, value in metadata.items():
                    if value is None:
                        continue
                    previous_values.append(
                        (
                            target,
                            attr,
                            getattr(target, attr, missing),
                        )
                    )
                    setattr(target, attr, value)
        try:
            return original_execute(self, model_input, *args, **kwargs)
        finally:
            for target, attr, previous_value in reversed(previous_values):
                if previous_value is missing:
                    try:
                        delattr(target, attr)
                    except AttributeError:
                        pass
                else:
                    setattr(target, attr, previous_value)

    execute_model_for_text_with_request_ids._qwen36_hybrid_apc_request_ids_patched = (
        True
    )
    execute_model_for_text_with_request_ids._qwen36_original_execute_model_for_text = (
        original_execute
    )
    runner_cls._execute_model_for_text = execute_model_for_text_with_request_ids
    return True


def _patch_neuron_runner_module(module: Any) -> bool:
    runner_cls = getattr(module, "NeuronxDistributedModelRunner", None)
    if runner_cls is None:
        return False
    installed = patch_neuron_model_runner_class(runner_cls)
    if installed:
        logger.info("Installed Qwen Hybrid APC vLLM-Neuron runner patch")
    return installed


def _patch_module(module_name: str, module: Any) -> bool:
    if module_name == _SCHEDULER_MODULE:
        return _patch_scheduler_module(module)
    if module_name == _VLLM_NEURON_RUNNER_MODULE:
        return _patch_neuron_runner_module(module)
    return False


class _HybridAPCSchedulerPatchLoader(importlib.abc.Loader):
    _qwen36_hybrid_apc_loader = True

    def __init__(self, wrapped_loader: importlib.abc.Loader):
        self.wrapped_loader = wrapped_loader

    def create_module(self, spec):
        create_module = getattr(self.wrapped_loader, "create_module", None)
        if create_module is None:
            return None
        return create_module(spec)

    def exec_module(self, module):
        self.wrapped_loader.exec_module(module)
        _patch_module(module.__name__, module)


class _HybridAPCSchedulerPatchFinder(importlib.abc.MetaPathFinder):
    _qwen36_hybrid_apc_import_hook = True

    def find_spec(self, fullname, path, target=None):
        if fullname not in _PATCHED_MODULES:
            return None
        spec = importlib.machinery.PathFinder.find_spec(fullname, path)
        if spec is None or spec.loader is None:
            return spec
        if getattr(spec.loader, "_qwen36_hybrid_apc_loader", False):
            return spec
        spec.loader = _HybridAPCSchedulerPatchLoader(spec.loader)
        return spec


def install_import_hook() -> bool:
    """Patch vLLM components lazily, without importing vLLM at Python startup."""

    installed = False
    for module_name in _PATCHED_MODULES:
        module = sys.modules.get(module_name)
        if module is not None:
            installed = _patch_module(module_name, module) or installed
    for finder in sys.meta_path:
        if getattr(finder, "_qwen36_hybrid_apc_import_hook", False):
            return installed
    sys.meta_path.insert(0, _HybridAPCSchedulerPatchFinder())
    return installed


def install() -> bool:
    """Install the vLLM scheduler patch when vLLM is available."""

    from vllm.v1.core.sched.scheduler import Scheduler  # noqa: WPS433

    installed = False
    module = sys.modules.get(_SCHEDULER_MODULE)
    if module is not None:
        installed = _patch_scheduler_module(module)
    else:
        installed = patch_scheduler_class(Scheduler)
    runner_module = sys.modules.get(_VLLM_NEURON_RUNNER_MODULE)
    if runner_module is not None:
        installed = _patch_neuron_runner_module(runner_module) or installed
    if installed:
        logger.info("Installed Qwen Hybrid APC scheduler fallback patch")
    return installed
