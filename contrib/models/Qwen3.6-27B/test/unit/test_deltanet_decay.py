# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU-only regressions for the fused DeltaNet decay reference math."""

import importlib.util
import os
import types
import unittest

import torch


_CONTRIB_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
_VALIDATOR_PATH = os.path.join(
    _CONTRIB_ROOT,
    "scripts",
    "validate_deltanet_fused_nki.py",
)


def _load_validator():
    spec = importlib.util.spec_from_file_location(
        "qwen36_validate_deltanet_fused_nki",
        _VALIDATOR_PATH,
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class TestFusedDeltaNetDecayMath(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.validator = _load_validator()

    def test_stable_causal_decay_masks_before_exp(self):
        gc = torch.linspace(0.0, -300.0, 128, dtype=torch.float32).reshape(128, 1)
        lower = torch.tril(torch.ones((128, 128), dtype=torch.float32), diagonal=-1)
        lower_diag = torch.tril(torch.ones((128, 128), dtype=torch.float32))

        strict_decay = self.validator.stable_causal_decay(torch, gc, lower)
        diag_decay = self.validator.stable_causal_decay(torch, gc, lower_diag)

        self.assertTrue(torch.isfinite(strict_decay).all())
        self.assertTrue(torch.isfinite(diag_decay).all())
        self.assertTrue(torch.equal(strict_decay.triu(), torch.zeros_like(strict_decay.triu())))
        torch.testing.assert_close(torch.diagonal(diag_decay), torch.ones(128))

    def test_reference_math_is_finite_for_realistic_gate_scale(self):
        args = types.SimpleNamespace(
            seed=1234,
            seq_len=256,
            value_scale=0.05,
            state_scale=0.01,
            gate_scale=1.0,
        )

        inputs = self.validator.make_inputs(torch, args)
        output, state = self.validator.reference_math(torch, inputs)

        self.assertTrue(torch.isfinite(output).all())
        self.assertTrue(torch.isfinite(state).all())


if __name__ == "__main__":
    unittest.main()
