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
    def _run_dry_driver(self, **overrides):
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
                check=True,
                capture_output=True,
                env=env,
                text=True,
            )
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
        self.assertFalse(pidfile.exists())


if __name__ == "__main__":
    unittest.main()
