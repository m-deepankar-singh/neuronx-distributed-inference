# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU checks for the direct DeltaNet RHS triangular solve."""

import unittest

import torch


class TestDeltaNetDirectRhsSolve(unittest.TestCase):
    def test_direct_rhs_solve_matches_materialized_inverse_path(self):
        torch.manual_seed(0)

        chunk_size = 128
        dim = 128
        dtype = torch.float32

        qk_decay = torch.randn(chunk_size, chunk_size, dtype=dtype) * 0.01
        lower_mask = torch.tril(torch.ones(chunk_size, chunk_size, dtype=dtype), diagonal=-1)
        a_mat = -qk_decay * lower_mask
        system = torch.eye(chunk_size, dtype=dtype) - a_mat

        v_beta = torch.randn(chunk_size, dim, dtype=dtype)
        kb_exp_gc = torch.randn(chunk_size, dim, dtype=dtype)
        state = torch.randn(dim, dim, dtype=dtype) * 0.01

        identity = torch.eye(chunk_size, dtype=dtype)
        n_mat = torch.linalg.solve_triangular(
            system,
            identity,
            upper=False,
            unitriangular=True,
        )

        materialized_value_corr = n_mat @ v_beta
        materialized_k_cumdecay = n_mat @ kb_exp_gc
        materialized_v_new = materialized_value_corr - materialized_k_cumdecay @ state

        rhs = v_beta - kb_exp_gc @ state
        direct_v_new = torch.linalg.solve_triangular(
            system,
            rhs,
            upper=False,
            unitriangular=True,
        )

        torch.testing.assert_close(
            direct_v_new,
            materialized_v_new,
            rtol=2e-5,
            atol=2e-5,
        )


if __name__ == "__main__":
    unittest.main()
