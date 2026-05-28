# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU-only contract tests for Qwen3.6 native MTP heads.

These tests intentionally avoid NxDI/Neuron imports. They validate the tensor
contract we need before wiring the checkpoint's ``mtp.*`` weights into the
Trainium speculative decode path:

* the Qwen3.6 checkpoint exposes the expected MTP key family
* vLLM-style weight remapping is deterministic
* an executable tiny MTP reference produces finite logits
"""

import json
import math
import os
import unittest
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping

import torch
from torch import nn
from torch.nn import functional as F


BASE_MTP_KEYS = {
    "mtp.fc.weight",
    "mtp.norm.weight",
    "mtp.pre_fc_norm_embedding.weight",
    "mtp.pre_fc_norm_hidden.weight",
}

PER_LAYER_MTP_KEYS = {
    "input_layernorm.weight",
    "mlp.down_proj.weight",
    "mlp.gate_proj.weight",
    "mlp.up_proj.weight",
    "post_attention_layernorm.weight",
    "self_attn.k_norm.weight",
    "self_attn.k_proj.weight",
    "self_attn.o_proj.weight",
    "self_attn.q_norm.weight",
    "self_attn.q_proj.weight",
    "self_attn.v_proj.weight",
}


@dataclass(frozen=True)
class TinyMTPConfig:
    vocab_size: int = 32
    hidden_size: int = 16
    intermediate_size: int = 32
    num_attention_heads: int = 4
    num_key_value_heads: int = 2
    num_nextn_predict_layers: int = 1
    rms_norm_eps: float = 1e-6

    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.num_attention_heads


def mtp_required_keys(num_mtp_layers: int) -> set[str]:
    keys = set(BASE_MTP_KEYS)
    for layer_idx in range(num_mtp_layers):
        for suffix in PER_LAYER_MTP_KEYS:
            keys.add(f"mtp.layers.{layer_idx}.{suffix}")
    return keys


def missing_mtp_keys(keys: Iterable[str], num_mtp_layers: int) -> set[str]:
    return mtp_required_keys(num_mtp_layers).difference(set(keys))


def read_qwen_text_config(model_path: str | os.PathLike[str]) -> Mapping[str, object]:
    with open(Path(model_path) / "config.json", encoding="utf-8") as f:
        config = json.load(f)
    return config.get("text_config", config)


def checkpoint_index_keys(model_path: str | os.PathLike[str]) -> set[str]:
    index_path = Path(model_path) / "model.safetensors.index.json"
    with open(index_path, encoding="utf-8") as f:
        index = json.load(f)
    return set(index["weight_map"])


def remap_mtp_weight_name(weight_name: str) -> str | None:
    """Mirror vLLM's MTP loader naming contract without importing vLLM.

    Native MTP weights are loaded under ``model.*`` in the predictor. Shared
    embedding/lm-head weights stay shared with the target model.
    """

    if weight_name.startswith("mtp."):
        return "model." + weight_name[len("mtp.") :]
    if weight_name in {"embed_tokens.weight", "lm_head.weight"}:
        return weight_name
    return None


class RMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        variance = hidden_states.float().pow(2).mean(dim=-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.eps)
        return hidden_states * self.weight


class TinyQwenAttention(nn.Module):
    def __init__(self, config: TinyMTPConfig):
        super().__init__()
        self.config = config
        h = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        self.q_proj = nn.Linear(h, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(h, self.num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(h, self.num_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, h, bias=False)
        self.q_norm = RMSNorm(self.head_dim, config.rms_norm_eps)
        self.k_norm = RMSNorm(self.head_dim, config.rms_norm_eps)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch, seq_len, _ = hidden_states.shape
        q = self.q_proj(hidden_states).view(batch, seq_len, self.num_heads, self.head_dim)
        k = self.k_proj(hidden_states).view(batch, seq_len, self.num_kv_heads, self.head_dim)
        v = self.v_proj(hidden_states).view(batch, seq_len, self.num_kv_heads, self.head_dim)
        q = self.q_norm(q)
        k = self.k_norm(k)

        repeat_factor = self.num_heads // self.num_kv_heads
        k = k.repeat_interleave(repeat_factor, dim=2)
        v = v.repeat_interleave(repeat_factor, dim=2)

        scores = torch.einsum("bshd,bthd->bhst", q, k) / math.sqrt(self.head_dim)
        causal_mask = torch.triu(
            torch.ones(seq_len, seq_len, dtype=torch.bool, device=hidden_states.device),
            diagonal=1,
        )
        scores = scores.masked_fill(causal_mask, torch.finfo(scores.dtype).min)
        probs = F.softmax(scores.float(), dim=-1).to(v.dtype)
        out = torch.einsum("bhst,bthd->bshd", probs, v).reshape(batch, seq_len, -1)
        return self.o_proj(out)


class TinyQwenMLP(nn.Module):
    def __init__(self, config: TinyMTPConfig):
        super().__init__()
        h = config.hidden_size
        i = config.intermediate_size
        self.gate_proj = nn.Linear(h, i, bias=False)
        self.up_proj = nn.Linear(h, i, bias=False)
        self.down_proj = nn.Linear(i, h, bias=False)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(hidden_states)) * self.up_proj(hidden_states))


class TinyQwenDecoderLayer(nn.Module):
    def __init__(self, config: TinyMTPConfig):
        super().__init__()
        self.input_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.self_attn = TinyQwenAttention(config)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.mlp = TinyQwenMLP(config)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = hidden_states + self.self_attn(self.input_layernorm(hidden_states))
        hidden_states = hidden_states + self.mlp(self.post_attention_layernorm(hidden_states))
        return hidden_states


class QwenMTPReference(nn.Module):
    """Executable CPU reference for the MTP predictor tensor contract."""

    def __init__(self, config: TinyMTPConfig):
        super().__init__()
        self.config = config
        h = config.hidden_size
        self.embed_tokens = nn.Embedding(config.vocab_size, h)
        self.fc = nn.Linear(h * 2, h, bias=False)
        self.layers = nn.ModuleList(
            TinyQwenDecoderLayer(config)
            for _ in range(config.num_nextn_predict_layers)
        )
        self.norm = RMSNorm(h, config.rms_norm_eps)
        self.pre_fc_norm_embedding = RMSNorm(h, config.rms_norm_eps)
        self.pre_fc_norm_hidden = RMSNorm(h, config.rms_norm_eps)
        self.lm_head = nn.Linear(h, config.vocab_size, bias=False)

    def forward(
        self,
        input_ids: torch.Tensor,
        target_hidden_states: torch.Tensor,
        spec_step_idx: int = 0,
    ) -> torch.Tensor:
        inputs_embeds = self.embed_tokens(input_ids)
        inputs_embeds = self.pre_fc_norm_embedding(inputs_embeds)
        target_hidden_states = self.pre_fc_norm_hidden(target_hidden_states)
        hidden_states = self.fc(torch.cat([inputs_embeds, target_hidden_states], dim=-1))
        layer_idx = spec_step_idx % len(self.layers)
        hidden_states = self.layers[layer_idx](hidden_states)
        hidden_states = self.norm(hidden_states)
        return self.lm_head(hidden_states)


class TestMTPCheckpointContract(unittest.TestCase):
    def test_required_key_list_matches_qwen36_family(self):
        keys = mtp_required_keys(num_mtp_layers=1)
        self.assertIn("mtp.fc.weight", keys)
        self.assertIn("mtp.layers.0.self_attn.q_proj.weight", keys)
        self.assertIn("mtp.layers.0.mlp.down_proj.weight", keys)
        self.assertEqual(len(keys), 15)

    def test_missing_key_detector(self):
        keys = mtp_required_keys(num_mtp_layers=1)
        keys.remove("mtp.fc.weight")
        self.assertEqual(missing_mtp_keys(keys, 1), {"mtp.fc.weight"})

    def test_vllm_style_weight_name_remap(self):
        self.assertEqual(remap_mtp_weight_name("mtp.fc.weight"), "model.fc.weight")
        self.assertEqual(
            remap_mtp_weight_name("mtp.layers.0.self_attn.q_proj.weight"),
            "model.layers.0.self_attn.q_proj.weight",
        )
        self.assertEqual(remap_mtp_weight_name("embed_tokens.weight"), "embed_tokens.weight")
        self.assertEqual(remap_mtp_weight_name("lm_head.weight"), "lm_head.weight")
        self.assertIsNone(remap_mtp_weight_name("layers.0.mlp.up_proj.weight"))

    def test_real_checkpoint_index_when_available(self):
        model_path = os.environ.get("QWEN36_MODEL_PATH")
        if not model_path:
            self.skipTest("QWEN36_MODEL_PATH not set")

        text_config = read_qwen_text_config(model_path)
        self.assertEqual(text_config.get("model_type"), "qwen3_5_text")
        num_mtp_layers = int(text_config.get("mtp_num_hidden_layers", 1))
        keys = checkpoint_index_keys(model_path)
        self.assertEqual(missing_mtp_keys(keys, num_mtp_layers), set())


class TestMTPReferenceForward(unittest.TestCase):
    def test_forward_produces_finite_logits(self):
        torch.manual_seed(0)
        config = TinyMTPConfig()
        model = QwenMTPReference(config).eval()
        input_ids = torch.tensor([[1, 2, 3, 4]], dtype=torch.long)
        target_hidden = torch.randn(1, 4, config.hidden_size)

        with torch.no_grad():
            logits = model(input_ids, target_hidden, spec_step_idx=0)

        self.assertEqual(logits.shape, (1, 4, config.vocab_size))
        self.assertTrue(torch.isfinite(logits).all())

    def test_step_index_wraps_mtp_layers(self):
        torch.manual_seed(1)
        config = TinyMTPConfig(num_nextn_predict_layers=2)
        model = QwenMTPReference(config).eval()
        input_ids = torch.tensor([[5, 6]], dtype=torch.long)
        target_hidden = torch.randn(1, 2, config.hidden_size)

        with torch.no_grad():
            logits_0 = model(input_ids, target_hidden, spec_step_idx=0)
            logits_2 = model(input_ids, target_hidden, spec_step_idx=2)

        torch.testing.assert_close(logits_0, logits_2)


if __name__ == "__main__":
    unittest.main()
