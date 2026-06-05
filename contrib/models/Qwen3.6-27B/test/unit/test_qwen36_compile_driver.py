# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import subprocess
import tempfile
import unittest
from pathlib import Path


_REPO_ROOT = Path(__file__).resolve().parents[5]
_DRIVER = _REPO_ROOT / "tmp_compile_qwen32k_segcte2048_gdnseg512.sh"


def _parse_key_values(text):
    parsed = {}
    for line in text.splitlines():
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        parsed[key] = value
    return parsed


class TestQwen36CompileDriver(unittest.TestCase):
    def _speed_anchor_overrides(self):
        return {
            "ENABLE_QKV_NKI_KERNELS": "1",
            "ENABLE_QKV_CTE_NKI_KERNEL_FUSE_QK_NORM": "1",
            "ENABLE_QKV_CTE_NKI_KERNEL_FUSE_ROPE": "0",
            "ENABLE_OUT_PROJ_NKI_KERNEL": "0",
            "ENABLE_KV_CACHE_QUANT": "0",
            "PREFIX_CTE_ATTENTION_BACKEND": "attention_cte",
            "QWEN36_DELTANET_FUSED_SEGMENT_TOKENS": "0",
            "QWEN36_DELTANET_MULTIHEAD_CTE": "0",
            "QWEN36_DELTANET_SOLVE_MODE": "direct",
            "QWEN36_DELTANET_SOLVE_SCAN_STEPS": "0",
            "GDN_RECURRENT_CACHE_DTYPE": "bfloat16",
            "GDN_CONV_CACHE_DTYPE": "bfloat16",
            "CTE_BUCKETS_RAW": "2048",
        }

    def _run_dry_driver(self, check=True, **overrides):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            repo = root / "repo"
            model = root / "model"
            art_root = root / "artifacts"
            logdir = root / "logs"
            repo.mkdir()
            model.mkdir()

            env = os.environ.copy()
            env.update(
                {
                    "REPO": str(repo),
                    "MODEL": str(model),
                    "ART_ROOT": str(art_root),
                    "LOGDIR": str(logdir),
                    "TS": "unittest",
                    "COMPILE_DRY_RUN": "1",
                }
            )
            env.update({key: str(value) for key, value in overrides.items()})

            completed = subprocess.run(
                ["bash", str(_DRIVER)],
                check=check,
                capture_output=True,
                env=env,
                text=True,
            )
            if not check:
                return completed
            stdout = _parse_key_values(completed.stdout)
            envlog = Path(stdout["ENVLOG"])
            return stdout, _parse_key_values(envlog.read_text()), Path(stdout["PIDFILE"])

    def test_sample_token_only_mode_is_tagged_and_does_not_start_compile(self):
        stdout, envlog, pidfile = self._run_dry_driver(
            DISABLE_ON_DEVICE_SAMPLING=0,
            OUTPUT_LOGITS_WITH_ON_DEVICE_SAMPLING=0,
        )

        self.assertIn("sampletokonly", stdout["BASE"])
        self.assertEqual(envlog["SAMPLING"], "on_device_greedy_sampletokonly")
        self.assertEqual(envlog["DISABLE_ON_DEVICE_SAMPLING"], "0")
        self.assertEqual(envlog["OUTPUT_LOGITS_WITH_ON_DEVICE_SAMPLING"], "0")
        self.assertEqual(envlog["COMPILE_DRY_RUN"], "1")
        self.assertEqual(envlog["ENVLOG"], stdout["ENVLOG"])
        self.assertFalse(pidfile.exists())

    def test_host_logits_mode_is_tagged_without_on_device_label(self):
        stdout, envlog, pidfile = self._run_dry_driver(
            DISABLE_ON_DEVICE_SAMPLING=1,
            OUTPUT_LOGITS_WITH_ON_DEVICE_SAMPLING=0,
        )

        self.assertIn("hostlogits", stdout["BASE"])
        self.assertEqual(envlog["SAMPLING"], "host_logits")
        self.assertEqual(envlog["DISABLE_ON_DEVICE_SAMPLING"], "1")
        self.assertEqual(envlog["OUTPUT_LOGITS_WITH_ON_DEVICE_SAMPLING"], "0")
        self.assertEqual(envlog["ENVLOG"], stdout["ENVLOG"])
        self.assertFalse(pidfile.exists())

    def test_speed_slice_hostlogits_keeps_lm_head_fp8_anchor(self):
        stdout, envlog, pidfile = self._run_dry_driver(
            **self._speed_anchor_overrides(),
            SPEED_SLICE="hostlogits",
            DISABLE_ON_DEVICE_SAMPLING=1,
            OUTPUT_LOGITS_WITH_ON_DEVICE_SAMPLING=0,
            QUANTIZE_LM_HEAD=1,
        )

        self.assertIn("hostlogits", stdout["BASE"])
        self.assertIn("lmheadfp8", stdout["BASE"])
        self.assertIn("attention_cte512", stdout["BASE"])
        self.assertIn("gdnseg0", stdout["BASE"])
        self.assertEqual(envlog["SPEED_SLICE"], "hostlogits")
        self.assertEqual(envlog["SAMPLING"], "host_logits")
        self.assertEqual(envlog["QUANTIZE_LM_HEAD"], "1")
        self.assertEqual(envlog["ENABLE_OUT_PROJ_NKI_KERNEL"], "0")
        self.assertEqual(envlog["PREFIX_CTE_ATTENTION_BACKEND"], "attention_cte")
        self.assertFalse(pidfile.exists())

    def test_speed_slice_hostlogits_rejects_confounded_lm_head_bf16(self):
        completed = self._run_dry_driver(
            False,
            **self._speed_anchor_overrides(),
            SPEED_SLICE="hostlogits",
            DISABLE_ON_DEVICE_SAMPLING=1,
            OUTPUT_LOGITS_WITH_ON_DEVICE_SAMPLING=0,
            QUANTIZE_LM_HEAD=0,
        )

        self.assertEqual(completed.returncode, 2)
        self.assertIn(
            "SPEED_SLICE=hostlogits requires QUANTIZE_LM_HEAD=1",
            completed.stderr,
        )

    def test_speed_slice_hostlogits_lmheadbf16_is_explicit_second_flip(self):
        stdout, envlog, pidfile = self._run_dry_driver(
            **self._speed_anchor_overrides(),
            SPEED_SLICE="hostlogits_lmheadbf16",
            DISABLE_ON_DEVICE_SAMPLING=1,
            OUTPUT_LOGITS_WITH_ON_DEVICE_SAMPLING=0,
            QUANTIZE_LM_HEAD=0,
        )

        self.assertIn("hostlogits", stdout["BASE"])
        self.assertIn("lmheadbf16", stdout["BASE"])
        self.assertEqual(envlog["SPEED_SLICE"], "hostlogits_lmheadbf16")
        self.assertEqual(envlog["SAMPLING"], "host_logits")
        self.assertEqual(envlog["QUANTIZE_LM_HEAD"], "0")
        self.assertFalse(pidfile.exists())

    def test_speed_slice_rejects_non_anchor_attention_backend(self):
        overrides = self._speed_anchor_overrides()
        overrides["PREFIX_CTE_ATTENTION_BACKEND"] = "segmented_cte"
        completed = self._run_dry_driver(
            False,
            **overrides,
            SPEED_SLICE="hostlogits",
            DISABLE_ON_DEVICE_SAMPLING=1,
            OUTPUT_LOGITS_WITH_ON_DEVICE_SAMPLING=0,
            QUANTIZE_LM_HEAD=1,
        )

        self.assertEqual(completed.returncode, 2)
        self.assertIn(
            "SPEED_SLICE=hostlogits requires PREFIX_CTE_ATTENTION_BACKEND=attention_cte",
            completed.stderr,
        )


if __name__ == "__main__":
    unittest.main()
