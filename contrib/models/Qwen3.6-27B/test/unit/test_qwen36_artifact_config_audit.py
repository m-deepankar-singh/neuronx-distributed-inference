# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path


_REPO_ROOT = Path(__file__).resolve().parents[5]
_AUDIT_PATH = _REPO_ROOT / "validation_scripts" / "qwen36_artifact_config_audit.py"
_SPEC = importlib.util.spec_from_file_location(
    "qwen36_artifact_config_audit_under_test",
    _AUDIT_PATH,
)
_AUDIT = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _AUDIT
_SPEC.loader.exec_module(_AUDIT)


class TestQwen36ArtifactConfigAudit(unittest.TestCase):
    def test_audit_flags_current_low_headroom_nki_chunked_shape(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            artifact = Path(tmpdir) / "qwen36_nki_chunked_artifact"
            artifact.mkdir()
            (artifact / "neuron_config.json").write_text(
                json.dumps(
                    {
                        "seq_len": 4096,
                        "batch_size": 2,
                        "ctx_batch_size": 2,
                        "pa_block_size": 256,
                        "pa_num_blocks": 33,
                        "max_gdn_checkpoint_slots": 8,
                        "context_encoding_buckets": [256, 512, 1024, 2048, 4096],
                        "prefix_buckets": [4096],
                        "is_prefix_caching": True,
                        "use_hybrid_apc_manager": True,
                    }
                )
            )

            summary = _AUDIT.audit(
                artifact=artifact,
                compile_log=None,
                recommended_block_size=32,
                min_usable_headroom_blocks=8,
                strict_hybrid_gate=True,
            )

        warning_codes = {warning["code"] for warning in summary["warnings"]}
        self.assertEqual(summary["pa_min_blocks"], 32)
        self.assertEqual(summary["pa_usable_headroom_blocks_excluding_null"], 0)
        self.assertIn("non_recommended_block_size", warning_codes)
        self.assertIn("low_pa_headroom", warning_codes)
        self.assertIn("strict_gate_boundary_slots_exceed_gdn_slots", warning_codes)
        self.assertIn("nki_chunked_deltanet_cte", warning_codes)


if __name__ == "__main__":
    unittest.main()
