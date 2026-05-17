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


def _env_flag(name: str) -> bool:
    value = os.environ.get(name)
    return value is not None and value.strip().lower() not in {
        "",
        "0",
        "false",
        "no",
        "off",
    }


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


def authorize_hybrid_apc_prefix_read(key: Any) -> HybridGDNPrefixKey:
    """Publish a scheduler-approved prefix read for suffix-only model prep."""

    registry_key = _to_registry_key(key)
    _AUTHORIZED_PREFIX_READS.setdefault(registry_key.prefix_len, []).append(
        registry_key
    )
    return registry_key


def pop_hybrid_apc_authorized_prefix_key(
    *,
    prefix_len: int,
    cache_salt: Hashable | None = None,
    model_revision: str = "unknown",
    layout_version: int = 1,
    tp_rank: int = 0,
    recurrent_dtype: str = "float32",
    conv_dtype: str = "bfloat16",
) -> HybridGDNPrefixKey | None:
    """Consume the exact key for a prefix read allowed by the scheduler."""

    prefix_len = int(prefix_len)
    candidates = _AUTHORIZED_PREFIX_READS.get(prefix_len)
    if not candidates:
        return None
    recurrent_dtype = _normalize_dtype(recurrent_dtype, "float32")
    conv_dtype = _normalize_dtype(conv_dtype, "bfloat16")
    for idx, key in enumerate(candidates):
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
        matched = candidates.pop(idx)
        if not candidates:
            _AUTHORIZED_PREFIX_READS.pop(prefix_len, None)
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


def backed_gdn_prefix_hit(scheduler: Any, request: Any) -> HybridGDNPrefixKey | None:
    """Return the largest request prefix with a registered GDN checkpoint."""

    if request is None:
        return None
    token_ids = getattr(request, "prompt_token_ids", None)
    if not token_ids:
        return None
    block_size = _block_size_for_scheduler(scheduler)
    if block_size <= 0:
        return None
    max_cache_hit_len = max(0, int(getattr(request, "num_tokens", len(token_ids))) - 1)
    max_cache_hit_len = min(max_cache_hit_len, len(token_ids))
    hashes = _local_cumulative_prefix_hashes(
        token_ids,
        block_size=block_size,
        max_prefix_len=max_cache_hit_len,
    )
    for prefix_len in sorted(hashes, reverse=True):
        key = _request_registry_key(
            scheduler=scheduler,
            request=request,
            cumulative_prefix_hash=hashes[prefix_len],
            prefix_len=prefix_len,
            block_size=block_size,
        )
        if key in _GDN_PREFIX_KEYS:
            return key
    return None


def backed_gdn_prefix_hit_len(scheduler: Any, request: Any) -> int:
    hit = backed_gdn_prefix_hit(scheduler, request)
    if hit is None:
        return 0
    return hit.prefix_len


def _supports_backed_prefix_reads(scheduler: Any) -> bool:
    """Return whether this artifact can consume a backed Hybrid APC prefix."""

    if _env_flag("QWEN36_HYBRID_APC_ENABLE_BACKED_PREFIX_READS"):
        return True

    if not _scheduler_config_flag(scheduler, "hybrid_apc_enable_backed_prefix_reads"):
        return False
    if _max_num_seqs_for_scheduler(scheduler) != 1:
        return False

    # A backed GDN checkpoint is not enough on its own. The CTE graph must also
    # consume attention KV prefix state; otherwise warm requests restore GDN
    # state but full-attention layers still see only the suffix.
    return _scheduler_config_flag(scheduler, "use_qwen_hybrid_chunked_prefill")


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
    backed_hit = backed_gdn_prefix_hit(scheduler, request)
    backed_hit_len = 0 if backed_hit is None else backed_hit.prefix_len
    supports_backed = _supports_backed_prefix_reads(scheduler)
    if _env_flag("QWEN36_HYBRID_APC_DEBUG"):
        prompt_len = len(getattr(request, "prompt_token_ids", ()) or ())
        print(
            "[hybrid_apc_debug] scheduler-decision "
            f"disable_requested={disable_requested} "
            f"backed_hit_len={backed_hit_len} "
            f"supports_backed={supports_backed} "
            f"prompt_len={prompt_len} "
            f"registry_size={len(_GDN_PREFIX_KEYS)}",
            flush=True,
        )
    if backed_hit is not None and supports_backed:
        authorize_hybrid_apc_prefix_read(backed_hit)
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
    if getattr(original_add_request, "_qwen36_hybrid_apc_patched", False):
        return False

    def add_request_with_hybrid_apc_fallback(self, request):
        if should_disable_unbacked_prefix_reads(self, request):
            request.skip_reading_prefix_cache = True
        return original_add_request(self, request)

    add_request_with_hybrid_apc_fallback._qwen36_hybrid_apc_patched = True
    add_request_with_hybrid_apc_fallback._qwen36_original_add_request = (
        original_add_request
    )
    scheduler_cls.add_request = add_request_with_hybrid_apc_fallback
    return True


def _patch_scheduler_module(module: Any) -> bool:
    scheduler_cls = getattr(module, "Scheduler", None)
    if scheduler_cls is None:
        return False
    installed = patch_scheduler_class(scheduler_cls)
    if installed:
        logger.info("Installed Qwen Hybrid APC scheduler fallback patch")
    return installed


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
        _patch_scheduler_module(module)


class _HybridAPCSchedulerPatchFinder(importlib.abc.MetaPathFinder):
    _qwen36_hybrid_apc_import_hook = True

    def find_spec(self, fullname, path, target=None):
        if fullname != _SCHEDULER_MODULE:
            return None
        spec = importlib.machinery.PathFinder.find_spec(fullname, path)
        if spec is None or spec.loader is None:
            return spec
        if getattr(spec.loader, "_qwen36_hybrid_apc_loader", False):
            return spec
        spec.loader = _HybridAPCSchedulerPatchLoader(spec.loader)
        return spec


def install_import_hook() -> bool:
    """Patch Scheduler lazily, without importing vLLM at Python startup."""

    module = sys.modules.get(_SCHEDULER_MODULE)
    if module is not None:
        return _patch_scheduler_module(module)
    for finder in sys.meta_path:
        if getattr(finder, "_qwen36_hybrid_apc_import_hook", False):
            return False
    sys.meta_path.insert(0, _HybridAPCSchedulerPatchFinder())
    return False


def install() -> bool:
    """Install the vLLM scheduler patch when vLLM is available."""

    from vllm.v1.core.sched.scheduler import Scheduler  # noqa: WPS433

    module = sys.modules.get(_SCHEDULER_MODULE)
    if module is not None:
        return _patch_scheduler_module(module)
    installed = patch_scheduler_class(Scheduler)
    if installed:
        logger.info("Installed Qwen Hybrid APC scheduler fallback patch")
    return installed
