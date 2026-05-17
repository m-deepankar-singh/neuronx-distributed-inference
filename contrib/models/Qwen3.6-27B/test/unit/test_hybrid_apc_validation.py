# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


_REPO_ROOT = Path(__file__).resolve().parents[5]
_VALIDATION_PATH = _REPO_ROOT / "validation_scripts" / "qwen36_hybrid_apc_validation.py"
_SPEC = importlib.util.spec_from_file_location(
    "qwen36_hybrid_apc_validation_under_test",
    _VALIDATION_PATH,
)
_VALIDATION = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _VALIDATION
_SPEC.loader.exec_module(_VALIDATION)


def _args(**overrides):
    defaults = {
        "shared_prefix": "shared",
        "suffix_a": " suffix a",
        "suffix_b": " suffix b",
        "require_real_tokens": True,
        "dummy_token_ids": [0],
        "output_json": None,
    }
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


def _fake_generate_batch(tokens_by_label):
    def fake_generate_batch(_args, *, enable_hybrid_apc, labeled_prompts):
        return {
            label: {
                "tokens": list(tokens_by_label[label]),
                "elapsed_seconds": 0.01,
            }
            for label, _prompt in labeled_prompts
        }

    return fake_generate_batch


class TestHybridAPCValidationRealTokens(unittest.TestCase):
    def test_real_token_checks_fail_all_dummy_tokens(self):
        checks = _VALIDATION._real_token_checks(
            {
                "cold_full": {"tokens": [0, 0, 0]},
                "warm_full": {"tokens": [0, 0, 0]},
            },
            {0},
        )

        self.assertFalse(checks["passed"])
        self.assertEqual(
            checks["checks"]["cold_full"]["failure"],
            "generated tokens are empty or all configured dummy tokens",
        )

    def test_exactness_can_require_non_dummy_generated_tokens(self):
        tokens_by_label = {
            "cold_full": [0, 0],
            "cold_partial": [0, 0],
            "warmup_full": [0, 0],
            "warm_full": [0, 0],
            "warmup_partial": [0, 0],
            "warm_partial": [0, 0],
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            output_json = Path(tmpdir) / "report.json"
            with patch.object(
                _VALIDATION,
                "_generate_batch",
                side_effect=_fake_generate_batch(tokens_by_label),
            ):
                rc = _VALIDATION.run_exactness(
                    _args(output_json=output_json),
                )

            self.assertEqual(rc, 1)
            report = output_json.read_text(encoding="utf-8")
            self.assertIn('"full_prefix_exact": true', report)
            self.assertIn('"partial_prefix_exact": true', report)
            self.assertIn('"real_generated_tokens_passed": false', report)

    def test_exactness_passes_when_real_tokens_are_present(self):
        tokens_by_label = {
            "cold_full": [42, 0],
            "cold_partial": [43, 0],
            "warmup_full": [42, 0],
            "warm_full": [42, 0],
            "warmup_partial": [42, 0],
            "warm_partial": [43, 0],
        }
        with patch.object(
            _VALIDATION,
            "_generate_batch",
            side_effect=_fake_generate_batch(tokens_by_label),
        ):
            rc = _VALIDATION.run_exactness(_args())

        self.assertEqual(rc, 0)


if __name__ == "__main__":
    unittest.main()
