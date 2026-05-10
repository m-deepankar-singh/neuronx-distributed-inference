# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU checks for materialized-N DeltaNet kernel cleanups."""

import unittest

import torch


class TestDeltaNetMaterializedNCleanup(unittest.TestCase):
    def test_strict_lower_decay_makes_second_mask_redundant(self):
        torch.manual_seed(0)

        chunk_size = 128
        qk = torch.randn(chunk_size, chunk_size, dtype=torch.float32)
        lower_mask = torch.tril(torch.ones(chunk_size, chunk_size, dtype=torch.float32), diagonal=-1)
        g = torch.randn(chunk_size, dtype=torch.float32).cumsum(0) * 0.01
        decay = torch.exp(g[:, None] - g[None, :]) * lower_mask

        old_a = -(qk * decay) * lower_mask
        new_a = -(qk * decay)

        torch.testing.assert_close(new_a, old_a, rtol=0.0, atol=0.0)


if __name__ == "__main__":
    unittest.main()
