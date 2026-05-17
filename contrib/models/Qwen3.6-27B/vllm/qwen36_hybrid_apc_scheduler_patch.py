"""vLLM scheduler patch for Qwen Hybrid APC fallback.

This module is intentionally opt-in. The safe fallback for current Hybrid APC
validation is to make vLLM skip attention-prefix reads before slot allocation
when the GDN checkpoint side is not integrated with the scheduler yet.
"""

from __future__ import annotations

import logging
import os
from typing import Any


logger = logging.getLogger(__name__)


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


def _config_flag(config: Any, name: str, default: bool = False) -> bool:
    return bool(getattr(config, name, default))


def should_disable_unbacked_prefix_reads(scheduler: Any) -> bool:
    """Return whether this scheduler should avoid vLLM APC reads.

    The current Qwen Hybrid APC control plane can prove an attention hit is
    invalid only inside model request prep. That is too late for allocation.
    Until the scheduler has first-class GDN checkpoint metadata, this opt-in
    fallback makes vLLM allocate the request as a normal no-prefix prefill.
    """

    hf_config = _get_hf_config(getattr(scheduler, "vllm_config", None))
    if not _config_flag(hf_config, "use_hybrid_apc_manager"):
        return False
    if _env_flag("QWEN36_HYBRID_APC_ENABLE_PREFIX_READS"):
        return False
    return (
        _env_flag("QWEN36_HYBRID_APC_DISABLE_UNBACKED_PREFIX_READS")
        or _config_flag(hf_config, "hybrid_apc_disable_unbacked_prefix_reads")
    )


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
        if should_disable_unbacked_prefix_reads(self):
            request.skip_reading_prefix_cache = True
        return original_add_request(self, request)

    add_request_with_hybrid_apc_fallback._qwen36_hybrid_apc_patched = True
    add_request_with_hybrid_apc_fallback._qwen36_original_add_request = (
        original_add_request
    )
    scheduler_cls.add_request = add_request_with_hybrid_apc_fallback
    return True


def install() -> bool:
    """Install the vLLM scheduler patch when vLLM is available."""

    from vllm.v1.core.sched.scheduler import Scheduler  # noqa: WPS433

    installed = patch_scheduler_class(Scheduler)
    if installed:
        logger.info("Installed Qwen Hybrid APC scheduler fallback patch")
    return installed
