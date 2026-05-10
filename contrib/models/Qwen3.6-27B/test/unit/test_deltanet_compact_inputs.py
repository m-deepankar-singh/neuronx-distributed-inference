# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU checks for compact DeltaNet chunk gate/decay inputs."""

import unittest

import torch


class TestDeltaNetCompactInputs(unittest.TestCase):
    def test_compact_gate_inputs_match_expanded_columns(self):
        batch_size, num_heads, num_chunks, chunk_size, dim = 1, 3, 4, 128, 128
        total_seq_len = num_chunks * chunk_size

        beta = torch.randn(batch_size, num_heads, total_seq_len, dtype=torch.float32)
        g = torch.randn(batch_size, num_heads, total_seq_len, dtype=torch.float32) * 0.01

        g_reshaped = g.reshape(batch_size, num_heads, num_chunks, chunk_size)
        g_cs = g_reshaped.cumsum(dim=-1)
        g_last = g_cs[:, :, :, -1:].expand(-1, -1, -1, chunk_size)

        old_beta = (
            beta.reshape(batch_size, num_heads, num_chunks, chunk_size)
            .unsqueeze(-1)
            .expand(-1, -1, -1, -1, dim)
            .reshape(batch_size * num_heads, num_chunks, chunk_size, dim)
        )
        old_gc = (
            g_cs.unsqueeze(-1)
            .expand(-1, -1, -1, -1, dim)
            .reshape(batch_size * num_heads, num_chunks, chunk_size, dim)
        )
        old_gl = (
            g_last.unsqueeze(-1)
            .expand(-1, -1, -1, -1, dim)
            .reshape(batch_size * num_heads, num_chunks, chunk_size, dim)
        )

        compact_beta = (
            beta.reshape(batch_size, num_heads, num_chunks, chunk_size)
            .unsqueeze(-1)
            .reshape(batch_size * num_heads, num_chunks, chunk_size, 1)
        )
        compact_gc = g_cs.unsqueeze(-1).reshape(
            batch_size * num_heads, num_chunks, chunk_size, 1
        )
        compact_gl = g_last.unsqueeze(-1).reshape(
            batch_size * num_heads, num_chunks, chunk_size, 1
        )

        torch.testing.assert_close(compact_beta[..., 0], old_beta[..., 0])
        torch.testing.assert_close(compact_gc[..., 0], old_gc[..., 0])
        torch.testing.assert_close(compact_gl[..., 0], old_gl[..., 0])

        self.assertEqual(compact_beta.numel() * dim, old_beta.numel())
        self.assertEqual(compact_gc.numel() * dim, old_gc.numel())
        self.assertEqual(compact_gl.numel() * dim, old_gl.numel())


if __name__ == "__main__":
    unittest.main()
