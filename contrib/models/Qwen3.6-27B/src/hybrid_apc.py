# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Qwen hybrid APC metadata lifecycle.

This module intentionally stores only control-plane metadata. GDN recurrent and
conv checkpoint tensors live in the model-side checkpoint bank; the metadata
store owns prefix identity, validity, refcounts, LRU state, and memory
accounting.
"""

from __future__ import annotations

from collections import OrderedDict
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


@dataclass
class HybridAPCStats:
    checkpoints: int = 0
    bytes_used: int = 0
    evictions: int = 0
    hits: int = 0
    misses: int = 0


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
