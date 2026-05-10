# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU reference tests for the blocked DeltaNet triangular solve.

The NKI chunked DeltaNet kernel solves N = inv(I - A) for a strictly lower
triangular A. These tests validate the block algorithm before it is encoded in
NKI, so compile cycles are not spent on basic math mistakes.
"""

import unittest

import torch


def _make_strict_lower(size=128, dtype=torch.float32):
    torch.manual_seed(0)
    # Keep values small enough that the inverse is well conditioned, but not
    # so tiny that a broken block merge can pass by accident.
    raw = torch.randn(size, size, dtype=dtype) * 0.03
    return torch.tril(raw, diagonal=-1)


def _rowwise_inverse(a):
    """Reference forward substitution for N = inv(I - A)."""
    size = a.shape[0]
    n = torch.zeros_like(a)
    eye = torch.eye(size, dtype=a.dtype)
    for row in range(size):
        if row:
            n[row] = a[row, :row] @ n[:row]
        n[row] += eye[row]
    return n


def _blocked_inverse(a, block=16):
    """Blocked forward substitution used by the planned NKI implementation."""
    size = a.shape[0]
    n = torch.zeros_like(a)
    eye = torch.eye(size, dtype=a.dtype)
    for base in range(0, size, block):
        end = min(base + block, size)
        if base:
            external = a[base:end, :base] @ n[:base]
        else:
            external = torch.zeros(end - base, size, dtype=a.dtype)

        for inner, row in enumerate(range(base, end)):
            update = external[inner].clone()
            if inner:
                update += a[row, base:row] @ n[base:row]
            update += eye[row]
            n[row] = update
    return n


class TestBlockedDeltaNetSolve(unittest.TestCase):
    def test_blocked_inverse_matches_rowwise_reference(self):
        a = _make_strict_lower()

        expected = _rowwise_inverse(a)
        actual = _blocked_inverse(a, block=16)

        torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-6)

    def test_blocked_inverse_solves_identity_system(self):
        a = _make_strict_lower()
        n = _blocked_inverse(a, block=16)

        residual = n @ (torch.eye(a.shape[0], dtype=a.dtype) - a)

        torch.testing.assert_close(
            residual,
            torch.eye(a.shape[0], dtype=a.dtype),
            atol=1e-5,
            rtol=1e-5,
        )

    def test_blocked_rhs_matches_materialized_inverse(self):
        a = _make_strict_lower()
        rhs = torch.randn(128, 32, dtype=torch.float32)

        n = _blocked_inverse(a, block=16)
        materialized = n @ rhs

        direct = torch.zeros_like(rhs)
        for base in range(0, 128, 16):
            end = base + 16
            if base:
                external = a[base:end, :base] @ direct[:base]
            else:
                external = torch.zeros(16, rhs.shape[1], dtype=rhs.dtype)
            for inner, row in enumerate(range(base, end)):
                update = external[inner].clone()
                if inner:
                    update += a[row, base:row] @ direct[base:row]
                direct[row] = rhs[row] + update

        torch.testing.assert_close(direct, materialized, atol=1e-5, rtol=1e-5)


if __name__ == "__main__":
    unittest.main()
