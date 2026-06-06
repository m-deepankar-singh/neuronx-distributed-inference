# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import importlib.util
import json
import subprocess
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
    def _sampletok_config(self):
        return {
            "seq_len": 32768,
            "max_context_length": 32768,
            "batch_size": 1,
            "ctx_batch_size": 1,
            "pa_block_size": 256,
            "pa_num_blocks": 128,
            "max_gdn_checkpoint_slots": 64,
            "context_encoding_buckets": [2048],
            "token_generation_buckets": [512, 16384, 16640, 32768],
            "prefix_buckets": [256, 512, 1024, 2048, 4096, 8192, 16384, 32768],
            "context_encoding_bucket_pairs": [
                [2048, 256],
                [2048, 512],
                [2048, 1024],
                [2048, 2048],
                [2048, 4096],
                [2048, 8192],
                [2048, 16384],
                [2048, 32768],
            ],
            "gdn_recurrent_cache_dtype": "bfloat16",
            "gdn_conv_cache_dtype": "bfloat16",
            "qkv_nki_kernel_enabled": True,
            "qkv_cte_nki_kernel_fuse_rope": False,
            "qkv_cte_nki_kernel_fuse_qk_norm": True,
            "out_proj_kernel_enabled": False,
            "kv_cache_quant": False,
            "prefix_cte_attention_backend": "attention_cte",
            "prefix_cte_attention_segment_size": 512,
            "output_logits": False,
            "on_device_sampling_config": {"do_sample": False, "top_k": 1},
            "vocab_parallel": True,
            "modules_to_not_convert": ["model.embed_tokens", "model.norm"],
            "is_prefix_caching": True,
            "use_hybrid_apc_manager": True,
            "async_mode": True,
        }

    def _sampletok_env(self):
        return {
            "ARTIFACT": "/artifact",
            "SEQ_LEN": "32768",
            "MAX_CONTEXT_LENGTH": "32768",
            "CTE_BUCKETS": "2048",
            "TOKEN_GENERATION_BUCKETS": "512 16384 16640 32768",
            "PREFIX_BUCKETS": "256 512 1024 2048 4096 8192 16384 32768",
            "CONTEXT_ENCODING_BUCKET_PAIRS": (
                "2048:256 2048:512 2048:1024 2048:2048 "
                "2048:4096 2048:8192 2048:16384 2048:32768"
            ),
            "MAX_GDN_CHECKPOINT_SLOTS": "64",
            "GDN_RECURRENT_CACHE_DTYPE": "bfloat16",
            "GDN_CONV_CACHE_DTYPE": "bfloat16",
            "ENABLE_QKV_NKI_KERNELS": "1",
            "ENABLE_QKV_CTE_NKI_KERNEL_FUSE_ROPE": "0",
            "ENABLE_QKV_CTE_NKI_KERNEL_FUSE_QK_NORM": "1",
            "ENABLE_OUT_PROJ_NKI_KERNEL": "0",
            "ENABLE_KV_CACHE_QUANT": "0",
            "PREFIX_CTE_ATTENTION_BACKEND": "attention_cte",
            "PREFIX_CTE_ATTENTION_SEGMENT_SIZE": "512",
            "DISABLE_ON_DEVICE_SAMPLING": "0",
            "OUTPUT_LOGITS_WITH_ON_DEVICE_SAMPLING": "0",
            "QUANTIZE_LM_HEAD": "1",
        }

    def _write_env(self, path, values):
        path.write_text("".join(f"{key}={value}\n" for key, value in values.items()))

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
                env_log=None,
                recommended_block_size=32,
                min_usable_headroom_blocks=8,
                strict_hybrid_gate=True,
            )

        warning_codes = {warning["code"] for warning in summary["warnings"]}
        self.assertEqual(summary["pa_min_blocks"], 32)
        self.assertEqual(summary["pa_usable_headroom_blocks"], 1)
        self.assertIn("non_recommended_block_size", warning_codes)
        self.assertIn("low_pa_headroom", warning_codes)
        self.assertIn("strict_gate_boundary_slots_exceed_gdn_slots", warning_codes)
        self.assertIn("nki_chunked_deltanet_cte", warning_codes)

    def test_audit_reads_nested_neuron_config(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            artifact = Path(tmpdir) / "qwen36_128k_fp8_artifact"
            artifact.mkdir()
            (artifact / "neuron_config.json").write_text(
                json.dumps(
                    {
                        "ctx_batch_size": 1,
                        "max_gdn_checkpoint_slots": 64,
                        "use_hybrid_apc_manager": True,
                        "neuron_config": {
                            "seq_len": 131072,
                            "batch_size": 1,
                            "pa_block_size": 256,
                            "pa_num_blocks": 512,
                            "context_encoding_buckets": [256, 512],
                            "prefix_buckets": [256, 512, 1024, 2048, 4096, 8192, 16384],
                            "is_prefix_caching": True,
                        },
                    }
                )
            )

            summary = _AUDIT.audit(
                artifact=artifact,
                compile_log=None,
                env_log=None,
                recommended_block_size=256,
                min_usable_headroom_blocks=0,
                strict_hybrid_gate=False,
            )

        self.assertEqual(summary["seq_len"], 131072)
        self.assertEqual(summary["pa_block_size"], 256)
        self.assertEqual(summary["pa_num_blocks"], 512)
        self.assertEqual(summary["pa_min_blocks"], 512)
        self.assertEqual(summary["context_encoding_buckets"], [256, 512])
        self.assertEqual(summary["prefix_buckets"][-1], 16384)
        self.assertTrue(summary["is_prefix_caching"])

    def test_policy_passes_for_sample_token_only_speed_slice(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            artifact = root / "artifact"
            artifact.mkdir()
            env_log = root / "compile_env.txt"
            (artifact / "neuron_config.json").write_text(
                json.dumps(self._sampletok_config())
            )
            self._write_env(env_log, self._sampletok_env())

            summary = _AUDIT.audit(
                artifact=artifact,
                compile_log=None,
                env_log=env_log,
                recommended_block_size=256,
                min_usable_headroom_blocks=0,
                strict_hybrid_gate=False,
            )

        self.assertTrue(summary["policy_passed"])
        self.assertEqual(summary["policy_errors"], [])

    def test_policy_fails_when_sample_token_artifact_outputs_logits(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            artifact = root / "artifact"
            artifact.mkdir()
            env_log = root / "compile_env.txt"
            config = self._sampletok_config()
            config["output_logits"] = True
            config["on_device_sampling_config"] = None
            (artifact / "neuron_config.json").write_text(json.dumps(config))
            self._write_env(env_log, self._sampletok_env())

            summary = _AUDIT.audit(
                artifact=artifact,
                compile_log=None,
                env_log=env_log,
                recommended_block_size=256,
                min_usable_headroom_blocks=0,
                strict_hybrid_gate=False,
            )

        error_codes = {error["code"] for error in summary["policy_errors"]}
        self.assertFalse(summary["policy_passed"])
        self.assertIn("on_device_sampling_mismatch", error_codes)
        self.assertIn("output_logits_mismatch", error_codes)

    def test_cli_returns_nonzero_for_policy_error(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            artifact = root / "artifact"
            artifact.mkdir()
            env_log = root / "compile_env.txt"
            output_json = root / "audit.json"
            config = self._sampletok_config()
            config["qkv_cte_nki_kernel_fuse_qk_norm"] = False
            (artifact / "neuron_config.json").write_text(json.dumps(config))
            self._write_env(env_log, self._sampletok_env())

            completed = subprocess.run(
                [
                    sys.executable,
                    str(_AUDIT_PATH),
                    str(artifact),
                    "--env-log",
                    str(env_log),
                    "--output-json",
                    str(output_json),
                ],
                check=False,
                capture_output=True,
                text=True,
            )
            payload = json.loads(output_json.read_text())

        self.assertEqual(completed.returncode, 1)
        self.assertFalse(payload["policy_passed"])
        self.assertIn("qkv_cte_nki_kernel_fuse_qk_norm_mismatch", completed.stdout)


if __name__ == "__main__":
    unittest.main()
