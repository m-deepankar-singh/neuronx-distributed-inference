# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Qwen hybrid APC metadata lifecycle.

This module intentionally stores only control-plane metadata. GDN recurrent and
conv checkpoint tensors live in the model-side checkpoint bank; the metadata
store owns prefix identity, validity, refcounts, LRU state, and memory
accounting.
"""

from __future__ import annotations

import hashlib
import struct
from collections import OrderedDict, deque
from dataclasses import dataclass
from typing import Hashable, Iterable, NamedTuple

import torch


class HybridPrefixKey(NamedTuple):
    cumulative_prefix_hash: Hashable
    prefix_len: int
    block_size: int
    cache_salt: Hashable | None
    model_revision: str
    layout_version: int
    tp_rank: int
    recurrent_dtype: str
    conv_dtype: str


class HybridAPCHitPlan(NamedTuple):
    attention_hit_len: int
    recurrent_hit_len: int
    conv_hit_len: int
    usable_hit_len: int
    restore_checkpoint_prefix_len: int
    residual_replay_len: int
    suffix_len: int
    checkpoint_slot: int | None
    checkpoint_key: HybridPrefixKey | None


class HybridAPCPreparedRequest(NamedTuple):
    request_id: Hashable
    input_dict: dict[str, torch.Tensor]
    plan: HybridAPCHitPlan
    commit_prefix_len: int
    commit_key: HybridPrefixKey | None
    commit_slot: int | None
    attention_block_refs: tuple[int, ...]


@dataclass
class HybridAPCStats:
    checkpoints: int = 0
    bytes_used: int = 0
    evictions: int = 0
    hits: int = 0
    misses: int = 0


@dataclass
class HybridAPCRequestRecord:
    request_id: Hashable
    state: str
    restored_key: HybridPrefixKey | None = None
    committed_keys: list[HybridPrefixKey] | None = None
    reserved_slots: list[int] | None = None

    def __post_init__(self):
        if self.committed_keys is None:
            self.committed_keys = []
        if self.reserved_slots is None:
            self.reserved_slots = []


@dataclass
class HybridPrefixCheckpoint:
    key: HybridPrefixKey
    prefix_len: int
    attention_block_refs: tuple[int, ...]
    gdn_checkpoint_slot: int
    valid_recurrent_layers: torch.Tensor
    valid_conv_layers: torch.Tensor
    refcount: int = 0
    last_access_step: int = 0
    bytes_used: int = 0
    evictable: bool = True
    attention_valid: bool = True

    def has_valid_recurrent(self, required_layers: tuple[int, ...]) -> bool:
        return _mask_has_layers(self.valid_recurrent_layers, required_layers)

    def has_valid_conv(self, required_layers: tuple[int, ...]) -> bool:
        return _mask_has_layers(self.valid_conv_layers, required_layers)

    def has_valid_gdn(self, required_layers: tuple[int, ...]) -> bool:
        return self.has_valid_recurrent(required_layers) and self.has_valid_conv(
            required_layers
        )

    def has_valid_hybrid_state(self, required_layers: tuple[int, ...]) -> bool:
        return self.attention_valid and self.has_valid_gdn(required_layers)


def _normalize_dtype(dtype: str | torch.dtype) -> str:
    if dtype == torch.float32:
        return "float32"
    if dtype == torch.bfloat16:
        return "bfloat16"
    normalized = str(dtype).lower()
    aliases = {
        "fp32": "float32",
        "float32": "float32",
        "torch.float32": "float32",
        "bf16": "bfloat16",
        "bfloat16": "bfloat16",
        "torch.bfloat16": "bfloat16",
    }
    if normalized not in aliases:
        raise ValueError(f"unsupported hybrid APC dtype: {dtype}")
    return aliases[normalized]


def _mask_has_layers(mask: torch.Tensor, required_layers: tuple[int, ...]) -> bool:
    if mask.numel() == 0:
        return False
    for layer in required_layers:
        if layer >= mask.numel() or not bool(mask[layer].item()):
            return False
    return True


def estimate_qwen_gdn_checkpoint_bytes_per_rank(
    *,
    num_gdn_layers: int = 48,
    local_value_heads: int = 12,
    local_key_heads: int = 4,
    key_dim: int = 128,
    value_dim: int = 128,
    conv_kernel_size: int = 4,
    recurrent_dtype: str | torch.dtype = "float32",
    conv_dtype: str | torch.dtype = "bfloat16",
) -> int:
    recurrent_dtype = _normalize_dtype(recurrent_dtype)
    conv_dtype = _normalize_dtype(conv_dtype)
    recurrent_bytes = 4 if recurrent_dtype == "float32" else 2
    conv_bytes = 4 if conv_dtype == "float32" else 2
    recurrent_numel = num_gdn_layers * local_value_heads * key_dim * value_dim
    conv_dim = 2 * local_key_heads * key_dim + local_value_heads * value_dim
    conv_numel = num_gdn_layers * conv_dim * (conv_kernel_size - 1)
    return recurrent_numel * recurrent_bytes + conv_numel * conv_bytes


def estimate_qwen_hybrid_cache_bytes_per_rank(
    *,
    max_context_len: int,
    checkpoint_interval: int,
    num_attention_layers: int = 16,
    local_kv_heads: int = 1,
    attention_head_dim: int = 256,
    attention_kv_dtype: str | torch.dtype = "bfloat16",
    **gdn_kwargs,
) -> dict[str, int]:
    attention_dtype = _normalize_dtype(attention_kv_dtype)
    attention_bytes = 4 if attention_dtype == "float32" else 2
    attention_kv = (
        int(max_context_len)
        * num_attention_layers
        * 2
        * local_kv_heads
        * attention_head_dim
        * attention_bytes
    )
    checkpoints = max(0, int(max_context_len)) // int(checkpoint_interval)
    gdn_per_checkpoint = estimate_qwen_gdn_checkpoint_bytes_per_rank(**gdn_kwargs)
    gdn_total = checkpoints * gdn_per_checkpoint
    return {
        "attention_kv_bytes": attention_kv,
        "gdn_checkpoint_bytes": gdn_total,
        "gdn_bytes_per_checkpoint": gdn_per_checkpoint,
        "num_gdn_checkpoints": checkpoints,
        "total_bytes": attention_kv + gdn_total,
    }


def _flatten_single_request_tokens(token_ids: torch.Tensor | Iterable[int]) -> torch.Tensor:
    if isinstance(token_ids, torch.Tensor):
        tokens = token_ids.detach().cpu()
    else:
        tokens = torch.tensor(list(token_ids), dtype=torch.int64)
    if tokens.ndim == 2 and tokens.shape[0] == 1:
        tokens = tokens.reshape(-1)
    elif tokens.ndim != 1:
        raise ValueError(
            "token_ids must be a single request tensor with shape [seq] or [1, seq], "
            f"got {tuple(tokens.shape)}"
        )
    return tokens.to(torch.int64).contiguous()


def build_cumulative_prefix_hashes(
    token_ids: torch.Tensor | Iterable[int],
    *,
    block_size: int,
    prefix_lens: Iterable[int] | None = None,
) -> dict[int, str]:
    """Build deterministic cumulative prefix hashes at block boundaries.

    This is a local scheduler bridge helper, not a replacement for vLLM's
    production block hash. It deliberately hashes the parent digest plus the
    next block's token bytes so a reused final block with a different parent
    prefix produces a different cumulative hash.
    """

    block_size = int(block_size)
    if block_size <= 0:
        raise ValueError(f"block_size must be positive, got {block_size}")

    tokens = _flatten_single_request_tokens(token_ids)
    seq_len = int(tokens.numel())
    if prefix_lens is None:
        requested_lens = set(range(block_size, seq_len + 1, block_size))
    else:
        requested_lens = {int(prefix_len) for prefix_len in prefix_lens}
    requested_lens = {prefix_len for prefix_len in requested_lens if prefix_len > 0}
    for prefix_len in requested_lens:
        if prefix_len > seq_len:
            raise ValueError(f"prefix_len {prefix_len} exceeds token length {seq_len}")
        if prefix_len % block_size != 0:
            raise ValueError(
                f"prefix_len {prefix_len} must be a multiple of block_size {block_size}"
            )

    if not requested_lens:
        return {}

    max_prefix_len = max(requested_lens)
    parent_digest = b""
    hashes: dict[int, str] = {}
    for block_start in range(0, max_prefix_len, block_size):
        block_end = block_start + block_size
        block = tokens[block_start:block_end]
        digest = hashlib.blake2b(digest_size=16)
        digest.update(parent_digest)
        digest.update(struct.pack("<QQ", block_size, block_end))
        digest.update(block.numpy().tobytes())
        parent_digest = digest.digest()
        if block_end in requested_lens:
            hashes[block_end] = parent_digest.hex()
    return hashes


def floor_to_checkpoint_boundary(prefix_len: int, checkpoint_interval: int) -> int:
    checkpoint_interval = int(checkpoint_interval)
    if checkpoint_interval <= 0:
        raise ValueError(
            f"checkpoint_interval must be positive, got {checkpoint_interval}"
        )
    return max(0, int(prefix_len)) // checkpoint_interval * checkpoint_interval


def apply_hybrid_apc_prefill_plan(
    input_dict: dict[str, torch.Tensor],
    *,
    plan: HybridAPCHitPlan,
    commit_slot: int | None = None,
    request_prefix_len: int | None = None,
) -> dict[str, torch.Tensor]:
    """Materialize model inputs for a scheduler-selected hybrid APC hit plan.

    The serving scheduler owns prefix hashing, attention block-table selection,
    checkpoint lookup, and checkpoint-slot reservation. This helper only applies
    the chosen restore boundary to the token tensors and emits explicit
    restore/commit control tensors. GDN state is restored only when the plan has
    a checkpoint slot; slot ID presence alone is never treated as a cache hit.
    """

    if "input_ids" not in input_dict:
        raise KeyError("input_ids is required to apply a hybrid APC prefill plan")

    input_ids = input_dict["input_ids"]
    if input_ids.ndim != 2:
        raise ValueError(f"input_ids must be [batch, seq], got {tuple(input_ids.shape)}")

    batch_size, available_len = input_ids.shape
    prompt_len = available_len if request_prefix_len is None else int(request_prefix_len)
    restore_len = int(plan.restore_checkpoint_prefix_len)
    if prompt_len < 0:
        raise ValueError(f"request_prefix_len must be non-negative, got {prompt_len}")
    if restore_len < 0 or restore_len > prompt_len:
        raise ValueError(
            "restore_checkpoint_prefix_len must be in [0, request_prefix_len], "
            f"got {restore_len} and {prompt_len}"
        )
    if prompt_len > available_len:
        raise ValueError(
            f"request_prefix_len {prompt_len} exceeds input_ids length {available_len}"
        )
    if plan.checkpoint_slot is None and restore_len != 0:
        raise ValueError("restore checkpoint prefix length requires a checkpoint slot")
    if plan.checkpoint_slot is not None and restore_len == 0:
        raise ValueError("checkpoint slot restore requires a positive prefix length")

    output = dict(input_dict)
    suffix_len = prompt_len - restore_len
    device = input_ids.device

    output["input_ids"] = input_ids[:, restore_len:prompt_len]

    attention_mask = input_dict.get("attention_mask")
    if (
        isinstance(attention_mask, torch.Tensor)
        and attention_mask.ndim >= 2
        and attention_mask.shape[0] == batch_size
        and attention_mask.shape[1] >= prompt_len
    ):
        output["attention_mask"] = attention_mask[:, restore_len:prompt_len]

    inputs_embeds = input_dict.get("inputs_embeds")
    if (
        isinstance(inputs_embeds, torch.Tensor)
        and inputs_embeds.ndim >= 3
        and inputs_embeds.shape[0] == batch_size
        and inputs_embeds.shape[1] >= prompt_len
    ):
        output["inputs_embeds"] = inputs_embeds[:, restore_len:prompt_len]

    slot_mapping = input_dict.get("slot_mapping")
    if (
        isinstance(slot_mapping, torch.Tensor)
        and slot_mapping.ndim >= 2
        and slot_mapping.shape[0] == batch_size
        and slot_mapping.shape[1] >= prompt_len
    ):
        output["slot_mapping"] = slot_mapping[:, restore_len:prompt_len]

    position_template = input_dict.get("position_ids")
    position_dtype = (
        position_template.dtype
        if isinstance(position_template, torch.Tensor)
        else torch.int64
    )
    position_ids = torch.arange(
        restore_len,
        prompt_len,
        dtype=position_dtype,
        device=device,
    ).unsqueeze(0)
    output["position_ids"] = position_ids.expand(batch_size, suffix_len).contiguous()

    for key in ("rotary_position_id", "rotary_position_ids"):
        value = input_dict.get(key)
        if not isinstance(value, torch.Tensor):
            continue
        if (
            value.ndim == 2
            and value.shape[0] == batch_size
            and value.shape[1] >= prompt_len
        ):
            output[key] = value[:, restore_len:prompt_len]
        elif (
            value.ndim == 3
            and value.shape[1] == batch_size
            and value.shape[2] >= prompt_len
        ):
            output[key] = value[:, :, restore_len:prompt_len]

    def _batch_i32(value: int) -> torch.Tensor:
        return torch.full((batch_size,), int(value), dtype=torch.int32, device=device)

    def _batch_i32_col(value: int) -> torch.Tensor:
        return torch.full((batch_size, 1), int(value), dtype=torch.int32, device=device)

    restore_enabled = plan.checkpoint_slot is not None
    output["computed_context_lens"] = _batch_i32_col(restore_len)
    output["full_context_lens"] = _batch_i32_col(prompt_len)
    output["num_queries"] = _batch_i32_col(suffix_len)
    output["hybrid_restore_slot_ids"] = _batch_i32(
        0 if plan.checkpoint_slot is None else int(plan.checkpoint_slot)
    )
    output["hybrid_restore_mask"] = _batch_i32(1 if restore_enabled else 0)
    output["hybrid_restore_prefix_lens"] = _batch_i32(restore_len)
    output["hybrid_commit_slot_ids"] = _batch_i32(0 if commit_slot is None else commit_slot)
    output["hybrid_commit_mask"] = _batch_i32(0 if commit_slot is None else 1)

    return output


class HybridAPCSlotAllocator:
    """Small checkpoint-slot allocator for local scheduler integration tests."""

    def __init__(self, num_slots: int):
        num_slots = int(num_slots)
        if num_slots <= 0:
            raise ValueError(f"num_slots must be positive, got {num_slots}")
        self.num_slots = num_slots
        self._free = deque(range(num_slots))
        self._reserved: set[int] = set()
        self._committed: set[int] = set()

    @property
    def free_slots(self) -> tuple[int, ...]:
        return tuple(self._free)

    @property
    def reserved_slots(self) -> tuple[int, ...]:
        return tuple(sorted(self._reserved))

    @property
    def committed_slots(self) -> tuple[int, ...]:
        return tuple(sorted(self._committed))

    def reserve(self) -> int:
        if not self._free:
            raise RuntimeError("no hybrid APC checkpoint slots available")
        slot = int(self._free.popleft())
        self._reserved.add(slot)
        return slot

    def mark_committed(self, slot: int):
        slot = int(slot)
        self.validate_slot_range(slot)
        if slot not in self._reserved and slot not in self._committed:
            raise ValueError(f"hybrid APC checkpoint slot {slot} is not reserved")
        self._reserved.discard(slot)
        self._committed.add(slot)

    def release(self, slot: int):
        slot = int(slot)
        self.validate_slot_range(slot)
        was_known = slot in self._reserved or slot in self._committed
        self._reserved.discard(slot)
        self._committed.discard(slot)
        if was_known and slot not in self._free:
            self._free.append(slot)

    def validate_slot_range(self, slot: int):
        slot = int(slot)
        if slot < 0 or slot >= self.num_slots:
            raise ValueError(
                f"hybrid APC checkpoint slot {slot} is outside "
                f"[0, {self.num_slots})"
            )


class HybridAPCSchedulerBridge:
    """Local request-prep bridge for production hybrid APC scheduler wiring.

    The real vLLM/NxDI scheduler must supply the attention APC hit length,
    active attention block refs, and tenant/cache metadata. This bridge performs
    the Qwen hybrid-specific part: intersect attention hits with GDN checkpoint
    metadata, materialize suffix model inputs, reserve a GDN checkpoint slot,
    and commit checkpoint metadata after a successful prefill.
    """

    def __init__(
        self,
        *,
        store: "HybridAPCMetadataStore",
        slot_allocator: HybridAPCSlotAllocator,
        cache_salt: Hashable | None = None,
        model_revision: str | None = None,
        layout_version: int | None = None,
        tp_rank: int | None = None,
        recurrent_dtype: str | torch.dtype | None = None,
        conv_dtype: str | torch.dtype | None = None,
        allow_local_hash_fallback: bool = True,
        require_attention_block_refs: bool = False,
    ):
        self.store = store
        self.slot_allocator = slot_allocator
        self.cache_salt = cache_salt
        self.model_revision = model_revision
        self.layout_version = layout_version
        self.tp_rank = tp_rank
        self.recurrent_dtype = recurrent_dtype
        self.conv_dtype = conv_dtype
        self.allow_local_hash_fallback = bool(allow_local_hash_fallback)
        self.require_attention_block_refs = bool(require_attention_block_refs)

    @property
    def requires_external_metadata(self) -> bool:
        return (
            not self.allow_local_hash_fallback
            or self.require_attention_block_refs
        )

    def prepare_request(
        self,
        *,
        request_id: Hashable,
        input_dict: dict[str, torch.Tensor],
        attention_hit_len: int,
        request_prefix_len: int | None = None,
        cumulative_hashes_by_prefix_len: dict[int, Hashable] | None = None,
        attention_block_refs_by_prefix_len: dict[int, Iterable[int]] | None = None,
    ) -> HybridAPCPreparedRequest:
        if "input_ids" not in input_dict:
            raise KeyError("input_ids is required for hybrid APC request prep")
        input_ids = input_dict["input_ids"]
        prompt_len = (
            int(input_ids.shape[1])
            if request_prefix_len is None
            else int(request_prefix_len)
        )
        commit_prefix_len = floor_to_checkpoint_boundary(
            prompt_len,
            self.store.checkpoint_interval,
        )

        if cumulative_hashes_by_prefix_len is None:
            if not self.allow_local_hash_fallback:
                raise ValueError(
                    "hybrid APC production mode requires vLLM cumulative prefix "
                    "hashes; set hybrid_apc_allow_local_hash_fallback=True only "
                    "for controlled local validation"
                )
            cumulative_hashes_by_prefix_len = build_cumulative_prefix_hashes(
                input_ids,
                block_size=self.store.block_size,
            )

        plan = self.store.compute_hit_plan(
            cumulative_hashes_by_prefix_len=cumulative_hashes_by_prefix_len,
            attention_hit_len=attention_hit_len,
            request_prefix_len=prompt_len,
            cache_salt=self.cache_salt,
            model_revision=self.model_revision,
            layout_version=self.layout_version,
            tp_rank=self.tp_rank,
            recurrent_dtype=self.recurrent_dtype,
            conv_dtype=self.conv_dtype,
        )
        if plan.checkpoint_slot is not None:
            self.slot_allocator.validate_slot_range(plan.checkpoint_slot)

        commit_key = None
        commit_slot = None
        attention_block_refs: tuple[int, ...] = ()
        if commit_prefix_len > 0:
            if commit_prefix_len not in cumulative_hashes_by_prefix_len:
                raise ValueError(
                    f"missing cumulative prefix hash for commit boundary {commit_prefix_len}"
                )
            commit_key = self.store.make_key(
                cumulative_prefix_hash=cumulative_hashes_by_prefix_len[commit_prefix_len],
                prefix_len=commit_prefix_len,
                cache_salt=self.cache_salt,
                model_revision=self.model_revision,
                layout_version=self.layout_version,
                tp_rank=self.tp_rank,
                recurrent_dtype=self.recurrent_dtype,
                conv_dtype=self.conv_dtype,
            )
            if attention_block_refs_by_prefix_len is not None:
                attention_block_refs = tuple(
                    int(ref)
                    for ref in attention_block_refs_by_prefix_len.get(
                        commit_prefix_len,
                        (),
                    )
                )
            if not attention_block_refs and not self.require_attention_block_refs:
                attention_block_refs = tuple(
                    range(commit_prefix_len // self.store.block_size)
                )
            if self.store.lookup(commit_key) is None:
                commit_slot = self.slot_allocator.reserve()

        model_inputs = apply_hybrid_apc_prefill_plan(
            input_dict,
            plan=plan,
            commit_slot=commit_slot,
            request_prefix_len=prompt_len,
        )
        record = self.store.on_request_restore(
            request_id=request_id,
            checkpoint_key=plan.checkpoint_key,
        )
        if commit_slot is not None:
            record.reserved_slots.append(commit_slot)
        self.store.on_prefill_running(request_id)

        return HybridAPCPreparedRequest(
            request_id=request_id,
            input_dict=model_inputs,
            plan=plan,
            commit_prefix_len=commit_prefix_len,
            commit_key=commit_key,
            commit_slot=commit_slot,
            attention_block_refs=attention_block_refs,
        )

    def commit_prefill(
        self,
        prepared: HybridAPCPreparedRequest,
        *,
        attention_block_refs: Iterable[int] | None = None,
        bytes_used: int = 0,
    ) -> HybridPrefixCheckpoint | None:
        if prepared.commit_key is None or prepared.commit_slot is None:
            return None
        refs = (
            tuple(int(ref) for ref in attention_block_refs)
            if attention_block_refs is not None
            else prepared.attention_block_refs
        )
        if self.require_attention_block_refs and not refs:
            raise ValueError(
                "hybrid APC checkpoint commit requires real attention block refs "
                "from the vLLM/NxDI APC allocator"
            )
        checkpoint = self.store.insert(
            key=prepared.commit_key,
            attention_block_refs=refs,
            gdn_checkpoint_slot=prepared.commit_slot,
            bytes_used=bytes_used,
        )
        self.slot_allocator.mark_committed(prepared.commit_slot)
        record = self.store.on_checkpoint_committed(
            request_id=prepared.request_id,
            checkpoint_key=prepared.commit_key,
        )
        if prepared.commit_slot in record.reserved_slots:
            record.reserved_slots.remove(prepared.commit_slot)
        return checkpoint

    def finish_request(self, request_id: Hashable) -> HybridAPCRequestRecord | None:
        record = self.store.on_request_finish(request_id)
        if record is not None:
            for slot in record.reserved_slots:
                self.slot_allocator.release(slot)
        return record

    def cancel_request(
        self,
        prepared: HybridAPCPreparedRequest,
    ) -> HybridAPCRequestRecord | None:
        record = self.store._requests.get(prepared.request_id)
        if (
            prepared.commit_slot is not None
            and record is not None
            and prepared.commit_slot in record.reserved_slots
        ):
            self.slot_allocator.release(prepared.commit_slot)
        return self.store.on_request_cancel(prepared.request_id)


class HybridAPCMetadataStore:
    """CPU-side lifecycle store for hybrid prefix-boundary checkpoints."""

    def __init__(
        self,
        *,
        required_gdn_layers: Iterable[int],
        block_size: int,
        checkpoint_interval: int | None = None,
        max_checkpoints: int | None = None,
        max_bytes: int | None = None,
        layout_version: int = 1,
        model_revision: str = "unknown",
        tp_rank: int = 0,
        recurrent_dtype: str | torch.dtype = "float32",
        conv_dtype: str | torch.dtype = "bfloat16",
        allow_residual_replay: bool = False,
    ):
        self.required_gdn_layers = tuple(sorted({int(x) for x in required_gdn_layers}))
        if not self.required_gdn_layers:
            raise ValueError("required_gdn_layers must not be empty")
        self.num_layer_mask_bits = max(self.required_gdn_layers) + 1
        self.block_size = int(block_size)
        if self.block_size <= 0:
            raise ValueError(f"block_size must be positive, got {block_size}")
        self.checkpoint_interval = (
            self.block_size
            if checkpoint_interval is None
            else int(checkpoint_interval)
        )
        if self.checkpoint_interval <= 0:
            raise ValueError(
                f"checkpoint_interval must be positive, got {checkpoint_interval}"
            )
        if self.checkpoint_interval % self.block_size != 0:
            raise ValueError(
                "checkpoint_interval must be a multiple of block_size for v0 "
                f"hybrid APC, got {self.checkpoint_interval} and {self.block_size}"
            )
        self.max_checkpoints = max_checkpoints
        if self.max_checkpoints is not None and self.max_checkpoints <= 0:
            raise ValueError(f"max_checkpoints must be positive, got {max_checkpoints}")
        self.max_bytes = max_bytes
        if self.max_bytes is not None and self.max_bytes <= 0:
            raise ValueError(f"max_bytes must be positive, got {max_bytes}")
        self.layout_version = int(layout_version)
        self.model_revision = str(model_revision)
        self.tp_rank = int(tp_rank)
        self.recurrent_dtype = _normalize_dtype(recurrent_dtype)
        self.conv_dtype = _normalize_dtype(conv_dtype)
        self.allow_residual_replay = bool(allow_residual_replay)

        self._by_key: OrderedDict[HybridPrefixKey, HybridPrefixCheckpoint] = (
            OrderedDict()
        )
        self._slot_to_key: dict[int, HybridPrefixKey] = {}
        self._requests: dict[Hashable, HybridAPCRequestRecord] = {}
        self._step = 0
        self.stats = HybridAPCStats()

    def __len__(self) -> int:
        return len(self._by_key)

    @property
    def bytes_used(self) -> int:
        return sum(checkpoint.bytes_used for checkpoint in self._by_key.values())

    def _next_step(self) -> int:
        self._step += 1
        return self._step

    def make_key(
        self,
        *,
        cumulative_prefix_hash: Hashable,
        prefix_len: int,
        cache_salt: Hashable | None = None,
        model_revision: str | None = None,
        layout_version: int | None = None,
        tp_rank: int | None = None,
        recurrent_dtype: str | torch.dtype | None = None,
        conv_dtype: str | torch.dtype | None = None,
    ) -> HybridPrefixKey:
        prefix_len = int(prefix_len)
        if prefix_len < 0:
            raise ValueError(f"prefix_len must be non-negative, got {prefix_len}")
        if prefix_len % self.checkpoint_interval != 0:
            raise ValueError(
                "prefix_len must align to checkpoint_interval "
                f"{self.checkpoint_interval}, got {prefix_len}"
            )
        return HybridPrefixKey(
            cumulative_prefix_hash=cumulative_prefix_hash,
            prefix_len=prefix_len,
            block_size=self.block_size,
            cache_salt=cache_salt,
            model_revision=self.model_revision
            if model_revision is None
            else str(model_revision),
            layout_version=self.layout_version
            if layout_version is None
            else int(layout_version),
            tp_rank=self.tp_rank if tp_rank is None else int(tp_rank),
            recurrent_dtype=self.recurrent_dtype
            if recurrent_dtype is None
            else _normalize_dtype(recurrent_dtype),
            conv_dtype=self.conv_dtype if conv_dtype is None else _normalize_dtype(conv_dtype),
        )

    def _make_mask(self, valid_layers: torch.Tensor | int | Iterable[int] | None):
        if valid_layers is None:
            layers = self.required_gdn_layers
            mask = torch.zeros(self.num_layer_mask_bits, dtype=torch.bool)
            mask[list(layers)] = True
            return mask
        if isinstance(valid_layers, torch.Tensor):
            mask = valid_layers.detach().cpu().to(torch.bool).flatten().clone()
            if mask.numel() < self.num_layer_mask_bits:
                padded = torch.zeros(self.num_layer_mask_bits, dtype=torch.bool)
                padded[: mask.numel()] = mask
                mask = padded
            return mask
        mask = torch.zeros(self.num_layer_mask_bits, dtype=torch.bool)
        if isinstance(valid_layers, int):
            bitmask = int(valid_layers)
            for layer in range(self.num_layer_mask_bits):
                mask[layer] = bool(bitmask & (1 << layer))
            return mask
        for layer in valid_layers:
            layer = int(layer)
            if layer >= mask.numel():
                padded = torch.zeros(layer + 1, dtype=torch.bool)
                padded[: mask.numel()] = mask
                mask = padded
            mask[layer] = True
        return mask

    def insert(
        self,
        *,
        key: HybridPrefixKey,
        attention_block_refs: Iterable[int],
        gdn_checkpoint_slot: int,
        valid_recurrent_layers: torch.Tensor | int | Iterable[int] | None = None,
        valid_conv_layers: torch.Tensor | int | Iterable[int] | None = None,
        bytes_used: int = 0,
        evictable: bool = True,
    ) -> HybridPrefixCheckpoint:
        if key.block_size != self.block_size:
            raise ValueError(
                f"key block_size {key.block_size} does not match store block_size {self.block_size}"
            )
        if key.layout_version != self.layout_version:
            raise ValueError(
                f"key layout_version {key.layout_version} does not match store layout_version {self.layout_version}"
            )
        recurrent_mask = self._make_mask(valid_recurrent_layers)
        conv_mask = self._make_mask(valid_conv_layers)
        checkpoint = HybridPrefixCheckpoint(
            key=key,
            prefix_len=key.prefix_len,
            attention_block_refs=tuple(int(ref) for ref in attention_block_refs),
            gdn_checkpoint_slot=int(gdn_checkpoint_slot),
            valid_recurrent_layers=recurrent_mask,
            valid_conv_layers=conv_mask,
            last_access_step=self._next_step(),
            bytes_used=int(bytes_used),
            evictable=bool(evictable),
        )
        if not checkpoint.has_valid_gdn(self.required_gdn_layers):
            raise ValueError("checkpoint is missing recurrent or conv state")

        old = self._slot_to_key.get(checkpoint.gdn_checkpoint_slot)
        if old is not None and old != key:
            self.mark_invalid(old)
        if key in self._by_key:
            old_checkpoint = self._by_key[key]
            self._slot_to_key.pop(old_checkpoint.gdn_checkpoint_slot, None)
        self._by_key[key] = checkpoint
        self._by_key.move_to_end(key)
        self._slot_to_key[checkpoint.gdn_checkpoint_slot] = key
        self._evict_over_budget()
        self._refresh_stats()
        return checkpoint

    def lookup(
        self,
        key: HybridPrefixKey,
        *,
        require_attention: bool = True,
        require_gdn: bool = True,
    ) -> HybridPrefixCheckpoint | None:
        checkpoint = self._by_key.get(key)
        if checkpoint is None:
            self.stats.misses += 1
            return None
        if require_attention and not checkpoint.attention_valid:
            self.stats.misses += 1
            return None
        if require_gdn and not checkpoint.has_valid_gdn(self.required_gdn_layers):
            self.stats.misses += 1
            return None
        checkpoint.last_access_step = self._next_step()
        self._by_key.move_to_end(key)
        self.stats.hits += 1
        return checkpoint

    def mark_invalid(
        self,
        key: HybridPrefixKey | None = None,
        *,
        checkpoint_slot: int | None = None,
        state_kind: str | None = None,
        layer_id: int | None = None,
    ) -> bool:
        if key is None:
            if checkpoint_slot is None:
                raise ValueError("key or checkpoint_slot is required")
            key = self._slot_to_key.get(int(checkpoint_slot))
            if key is None:
                return False
        checkpoint = self._by_key.get(key)
        if checkpoint is None:
            return False

        if state_kind is None:
            self._delete_checkpoint(key)
            self._refresh_stats()
            return True
        if state_kind == "attention":
            checkpoint.attention_valid = False
        elif state_kind == "recurrent":
            if layer_id is None:
                checkpoint.valid_recurrent_layers.zero_()
            elif int(layer_id) < checkpoint.valid_recurrent_layers.numel():
                checkpoint.valid_recurrent_layers[int(layer_id)] = False
        elif state_kind == "conv":
            if layer_id is None:
                checkpoint.valid_conv_layers.zero_()
            elif int(layer_id) < checkpoint.valid_conv_layers.numel():
                checkpoint.valid_conv_layers[int(layer_id)] = False
        else:
            raise ValueError(f"unknown state_kind: {state_kind}")
        return True

    def inc_ref(self, key: HybridPrefixKey) -> int:
        checkpoint = self.lookup(key, require_attention=False, require_gdn=False)
        if checkpoint is None:
            raise KeyError(key)
        checkpoint.refcount += 1
        return checkpoint.refcount

    def dec_ref(self, key: HybridPrefixKey) -> int:
        checkpoint = self.lookup(key, require_attention=False, require_gdn=False)
        if checkpoint is None:
            raise KeyError(key)
        checkpoint.refcount = max(0, checkpoint.refcount - 1)
        return checkpoint.refcount

    def on_request_restore(
        self,
        *,
        request_id: Hashable,
        checkpoint_key: HybridPrefixKey | None,
    ) -> HybridAPCRequestRecord:
        record = HybridAPCRequestRecord(
            request_id=request_id,
            state="NEW",
            restored_key=checkpoint_key,
        )
        if checkpoint_key is not None:
            self.inc_ref(checkpoint_key)
            record.state = "RESTORED_FROM_HYBRID_APC"
        self._requests[request_id] = record
        return record

    def on_prefill_running(self, request_id: Hashable) -> HybridAPCRequestRecord:
        record = self._requests[request_id]
        record.state = "PREFILL_RUNNING"
        return record

    def on_checkpoint_committed(
        self,
        *,
        request_id: Hashable,
        checkpoint_key: HybridPrefixKey,
    ) -> HybridAPCRequestRecord:
        record = self._requests.setdefault(
            request_id,
            HybridAPCRequestRecord(request_id=request_id, state="PREFILL_RUNNING"),
        )
        record.state = "PREFILL_COMMIT_PENDING"
        record.committed_keys.append(checkpoint_key)
        return record

    def on_decode_running(self, request_id: Hashable) -> HybridAPCRequestRecord:
        record = self._requests[request_id]
        record.state = "DECODE_RUNNING"
        return record

    def on_request_finish(self, request_id: Hashable) -> HybridAPCRequestRecord | None:
        record = self._requests.pop(request_id, None)
        if record is None:
            return None
        if record.restored_key is not None and record.restored_key in self._by_key:
            self.dec_ref(record.restored_key)
        record.state = "FINISHED"
        return record

    def on_request_cancel(self, request_id: Hashable) -> HybridAPCRequestRecord | None:
        record = self._requests.pop(request_id, None)
        if record is None:
            return None
        if record.restored_key is not None and record.restored_key in self._by_key:
            self.dec_ref(record.restored_key)
        for key in record.committed_keys:
            if key in self._by_key:
                self.mark_invalid(key)
        record.state = "CANCELLED"
        return record

    def evict_lru(self, *, target_checkpoints: int | None = None) -> list[HybridPrefixKey]:
        target = self.max_checkpoints if target_checkpoints is None else target_checkpoints
        if target is None:
            return []
        evicted: list[HybridPrefixKey] = []
        for key, checkpoint in list(self._by_key.items()):
            if len(self._by_key) <= target:
                break
            if checkpoint.refcount > 0 or not checkpoint.evictable:
                continue
            self._delete_checkpoint(key)
            evicted.append(key)
        self.stats.evictions += len(evicted)
        self._refresh_stats()
        return evicted

    def on_attention_block_evicted(self, block_ref: int) -> list[HybridPrefixKey]:
        invalidated: list[HybridPrefixKey] = []
        for key, checkpoint in self._by_key.items():
            if int(block_ref) in checkpoint.attention_block_refs:
                checkpoint.attention_valid = False
                invalidated.append(key)
        return invalidated

    def on_gdn_checkpoint_evicted(self, checkpoint_slot: int) -> bool:
        return self.mark_invalid(checkpoint_slot=int(checkpoint_slot))

    def compute_hit_plan(
        self,
        *,
        cumulative_hashes_by_prefix_len: dict[int, Hashable],
        attention_hit_len: int,
        request_prefix_len: int,
        cache_salt: Hashable | None = None,
        model_revision: str | None = None,
        layout_version: int | None = None,
        tp_rank: int | None = None,
        recurrent_dtype: str | torch.dtype | None = None,
        conv_dtype: str | torch.dtype | None = None,
    ) -> HybridAPCHitPlan:
        attention_hit_len = max(0, int(attention_hit_len))
        request_prefix_len = max(0, int(request_prefix_len))
        target_hit_len = min(attention_hit_len, request_prefix_len)
        candidate_lens = sorted(
            (
                int(prefix_len)
                for prefix_len in cumulative_hashes_by_prefix_len
                if int(prefix_len) <= target_hit_len
                and int(prefix_len) % self.checkpoint_interval == 0
            ),
            reverse=True,
        )

        for prefix_len in candidate_lens:
            key = self.make_key(
                cumulative_prefix_hash=cumulative_hashes_by_prefix_len[prefix_len],
                prefix_len=prefix_len,
                cache_salt=cache_salt,
                model_revision=model_revision,
                layout_version=layout_version,
                tp_rank=tp_rank,
                recurrent_dtype=recurrent_dtype,
                conv_dtype=conv_dtype,
            )
            checkpoint = self.lookup(key)
            if checkpoint is None:
                continue

            if self.allow_residual_replay:
                usable_hit_len = target_hit_len
                residual_replay_len = target_hit_len - prefix_len
                suffix_len = request_prefix_len - target_hit_len
            else:
                usable_hit_len = prefix_len
                residual_replay_len = 0
                suffix_len = request_prefix_len - prefix_len
            return HybridAPCHitPlan(
                attention_hit_len=attention_hit_len,
                recurrent_hit_len=prefix_len,
                conv_hit_len=prefix_len,
                usable_hit_len=usable_hit_len,
                restore_checkpoint_prefix_len=prefix_len,
                residual_replay_len=residual_replay_len,
                suffix_len=suffix_len,
                checkpoint_slot=checkpoint.gdn_checkpoint_slot,
                checkpoint_key=key,
            )

        return HybridAPCHitPlan(
            attention_hit_len=attention_hit_len,
            recurrent_hit_len=0,
            conv_hit_len=0,
            usable_hit_len=0,
            restore_checkpoint_prefix_len=0,
            residual_replay_len=0,
            suffix_len=request_prefix_len,
            checkpoint_slot=None,
            checkpoint_key=None,
        )

    def _evict_over_budget(self):
        if self.max_checkpoints is not None:
            self.evict_lru(target_checkpoints=self.max_checkpoints)
        if self.max_bytes is None:
            return
        evicted = 0
        for key, checkpoint in list(self._by_key.items()):
            if self.bytes_used <= self.max_bytes:
                break
            if checkpoint.refcount > 0 or not checkpoint.evictable:
                continue
            self._delete_checkpoint(key)
            evicted += 1
        self.stats.evictions += evicted

    def _delete_checkpoint(self, key: HybridPrefixKey):
        checkpoint = self._by_key.pop(key, None)
        if checkpoint is not None:
            self._slot_to_key.pop(checkpoint.gdn_checkpoint_slot, None)

    def _refresh_stats(self):
        self.stats.checkpoints = len(self._by_key)
        self.stats.bytes_used = self.bytes_used
