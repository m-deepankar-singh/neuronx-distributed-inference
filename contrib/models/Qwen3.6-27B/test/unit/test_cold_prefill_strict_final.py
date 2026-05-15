# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU-only tests for the strict-final cold-prefill orchestrator."""

import argparse
import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import unittest
from io import StringIO
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


_REPO_ROOT = Path(__file__).resolve().parents[5]
_STRICT_FINAL_PATH = (
    _REPO_ROOT / "validation_scripts" / "qwen36_cold_prefill_strict_final.py"
)
_WRAPPER_PATH = _REPO_ROOT / "validation_scripts" / "qwen36_trainium_strict_final.sh"


def _load_strict_final():
    spec = importlib.util.spec_from_file_location(
        "qwen36_cold_prefill_strict_final",
        _STRICT_FINAL_PATH,
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _args(output_dir: Path):
    return argparse.Namespace(
        model_path="/tmp/model",
        artifacts_2k="/tmp/artifacts-2k",
        artifacts_8k="/tmp/artifacts-8k",
        artifacts_32k="/tmp/artifacts-32k",
        artifacts_128k="/tmp/artifacts-128k",
        artifacts_262k="/tmp/artifacts-262k",
        output_dir=output_dir,
        output_prefix="strict",
        repetitions=3,
        strict_min_samples=3,
        tensor_parallel_size=4,
        logical_nc_config=2,
        max_num_seqs=1,
        ctx_batch_size=1,
        num_gpu_blocks_override=None,
        gdn_state_diff_json=None,
        short_latency_speedup=1.5,
        baseline_cold_tok_per_s_target=420.0,
        baseline_cold_tok_per_s_tolerance=0.15,
        bucket_tok_regression_tolerance=0.20,
        prompt_token_tolerance=0.05,
        python="python3",
        expected_branch="qwen36-cold-prefill-perf",
        fail_fast=False,
        dry_run=True,
        preflight=False,
        preflight_json=None,
        evidence_check=False,
        evidence_check_json=None,
    )


def _write_preflight_fixture(strict_final, args, *, passed=True, checks=None):
    payload = {
        "passed": passed,
        "checks": checks or [{"label": "git_branch", "ok": passed}],
        "expected_outputs": strict_final._expected_output_paths(args),
        "expected_logs": strict_final._phase_log_paths(args),
        "run_manifest": str(strict_final._run_manifest_path(args)),
        "output_dir": str(args.output_dir.expanduser()),
        "output_prefix": args.output_prefix,
    }
    strict_final._preflight_json_path(args).write_text(
        json.dumps(payload) + "\n",
        encoding="utf-8",
    )


def _strict_acceptance_payload(*, passed=True, strict_final=True, audit_status="pass"):
    return {
        "passed": passed,
        "failures": [] if passed else ["token mismatch"],
        "strict_final": strict_final,
        "audit_checklist": [{"name": "strict-final", "status": audit_status}],
    }


class TestColdPrefillStrictFinal(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.strict_final = _load_strict_final()

    def test_builds_strict_final_matrix_long_context_and_acceptance_commands(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            commands = self.strict_final._build_commands(_args(Path(tmpdir)))

        matrix = commands["matrix"]
        long_context = commands["long_context"]
        hybrid_apc = commands["hybrid_apc"]
        acceptance = commands["acceptance"]

        self.assertIn("--compiled-artifacts-by-len", matrix)
        self.assertIn("8192=/tmp/artifacts-8k", matrix)
        self.assertIn("32768=/tmp/artifacts-32k", matrix)
        self.assertIn("--prompt-lengths", matrix)
        self.assertIn("32768", matrix)
        self.assertIn("--include-dense-fallback", matrix)
        self.assertIn("--max-num-seqs", matrix)
        self.assertIn("1", matrix)
        matrix_max_tokens_index = matrix.index("--max-tokens-values")
        self.assertEqual(
            matrix[matrix_max_tokens_index + 1 : matrix_max_tokens_index + 3],
            ["1", "32"],
        )
        self.assertIn("--include-128k", long_context)
        self.assertIn("--include-262k", long_context)
        self.assertIn("131072=/tmp/artifacts-128k", long_context)
        self.assertIn("262144=/tmp/artifacts-262k", long_context)
        self.assertIn("J_262k_recovery_block128", long_context)
        long_context_max_tokens_index = long_context.index("--max-tokens-values")
        self.assertEqual(
            long_context[
                long_context_max_tokens_index + 1 : long_context_max_tokens_index + 3
            ],
            ["1", "32"],
        )
        self.assertIn("exactness", hybrid_apc)
        self.assertIn("--compiled-artifacts", hybrid_apc)
        self.assertIn("/tmp/artifacts-2k", hybrid_apc)
        self.assertNotIn("--cold-zero-conv-fast-path", hybrid_apc)
        self.assertIn("--max-num-seqs", hybrid_apc)
        self.assertIn("--hybrid-apc-require-vllm-metadata", hybrid_apc)
        self.assertIn("--output-json", hybrid_apc)
        self.assertIn("--strict-final", acceptance)
        self.assertIn("--baseline-cold-tok-per-s-target", acceptance)
        self.assertIn("--hybrid-apc-report", acceptance)
        self.assertIn("420.0", acceptance)
        self.assertIn("--strict-min-samples", acceptance)
        self.assertIn("3", acceptance)

    def test_build_commands_threads_gdn_state_diff_sidecar_to_benchmarks(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            args = _args(Path(tmpdir))
            args.gdn_state_diff_json = Path("/tmp/gdn_state_diff.json")
            commands = self.strict_final._build_commands(args)

        self.assertIn("--gdn-state-diff-json", commands["matrix"])
        self.assertIn("/tmp/gdn_state_diff.json", commands["matrix"])
        self.assertIn("--gdn-state-diff-json", commands["long_context"])
        self.assertIn("/tmp/gdn_state_diff.json", commands["long_context"])
        self.assertNotIn("--gdn-state-diff-json", commands["acceptance"])

    def test_parse_args_rejects_too_few_repetitions(self):
        argv = [
            "strict_final",
            "--model-path",
            "/tmp/model",
            "--artifacts-2k",
            "/tmp/2k",
            "--artifacts-8k",
            "/tmp/8k",
            "--artifacts-32k",
            "/tmp/32k",
            "--artifacts-128k",
            "/tmp/128k",
            "--artifacts-262k",
            "/tmp/262k",
            "--repetitions",
            "2",
            "--strict-min-samples",
            "3",
        ]

        with patch("sys.argv", argv), self.assertRaises(SystemExit):
            self.strict_final.parse_args()

    def test_parse_args_allows_evidence_check_without_runtime_paths(self):
        argv = [
            "strict_final",
            "--output-dir",
            "/tmp/strict-final",
            "--evidence-check",
        ]

        with patch("sys.argv", argv):
            args = self.strict_final.parse_args()

        self.assertTrue(args.evidence_check)
        self.assertIsNone(args.model_path)

    def test_preflight_passes_for_existing_paths(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            args = _args(root / "out")
            args.python = str(root / "python")
            Path(args.python).write_text("#!/usr/bin/env python3\n", encoding="utf-8")
            args.model_path = str(root / "model")
            args.artifacts_2k = str(root / "artifacts-2k")
            args.artifacts_8k = str(root / "artifacts-8k")
            args.artifacts_32k = str(root / "artifacts-32k")
            args.artifacts_128k = str(root / "artifacts-128k")
            args.artifacts_262k = str(root / "artifacts-262k")
            for path in (
                args.model_path,
                args.artifacts_2k,
                args.artifacts_8k,
                args.artifacts_32k,
                args.artifacts_128k,
                args.artifacts_262k,
            ):
                path_obj = Path(path)
                path_obj.mkdir()
                (path_obj / "marker").write_text("x", encoding="utf-8")

            checks = self.strict_final._preflight_checks(args)

        self.assertTrue(all(check["ok"] for check in checks), checks)

    def test_git_branch_check_accepts_expected_branch(self):
        with patch.object(
            self.strict_final.subprocess,
            "run",
            return_value=SimpleNamespace(
                returncode=0,
                stdout="qwen36-cold-prefill-perf\n",
                stderr="",
            ),
        ):
            check = self.strict_final._git_branch_check("qwen36-cold-prefill-perf")

        self.assertTrue(check["ok"])
        self.assertEqual(check["actual_branch"], "qwen36-cold-prefill-perf")

    def test_git_branch_check_rejects_wrong_branch(self):
        with patch.object(
            self.strict_final.subprocess,
            "run",
            return_value=SimpleNamespace(
                returncode=0,
                stdout="experimental\n",
                stderr="",
            ),
        ):
            check = self.strict_final._git_branch_check("qwen36-cold-prefill-perf")

        self.assertFalse(check["ok"])
        self.assertEqual(check["actual_branch"], "experimental")

    def test_print_preflight_handles_git_branch_check(self):
        checks = [
            {
                "label": "git_branch",
                "kind": "git_branch",
                "path": "/tmp/repo",
                "expected_branch": "qwen36-cold-prefill-perf",
                "actual_branch": "qwen36-cold-prefill-perf",
                "ok": True,
            }
        ]
        output = StringIO()

        with patch("sys.stdout", output):
            self.strict_final._print_preflight(checks)

        self.assertIn("actual=qwen36-cold-prefill-perf", output.getvalue())
        self.assertIn("expected=qwen36-cold-prefill-perf", output.getvalue())

    def test_preflight_reports_missing_artifact_paths(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            args = _args(root / "out")
            args.model_path = str(root / "model")
            args.artifacts_2k = str(root / "artifacts-2k")
            Path(args.model_path).mkdir()
            (Path(args.model_path) / "config.json").write_text("{}", encoding="utf-8")
            Path(args.artifacts_2k).mkdir()
            (Path(args.artifacts_2k) / "manifest.json").write_text(
                "{}",
                encoding="utf-8",
            )

            checks = self.strict_final._preflight_checks(args)

        by_label = {check["label"]: check for check in checks}
        self.assertTrue(by_label["model_path"]["ok"])
        self.assertTrue(by_label["artifacts_2k"]["ok"])
        self.assertFalse(by_label["artifacts_8k"]["ok"])
        self.assertFalse(by_label["artifacts_32k"]["ok"])
        self.assertFalse(by_label["artifacts_128k"]["ok"])
        self.assertFalse(by_label["artifacts_262k"]["ok"])

    def test_preflight_rejects_empty_artifact_directory(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            args = _args(root / "out")
            args.model_path = str(root / "model")
            args.artifacts_2k = str(root / "artifacts-2k")
            Path(args.model_path).mkdir()
            (Path(args.model_path) / "config.json").write_text("{}", encoding="utf-8")
            Path(args.artifacts_2k).mkdir()

            checks = self.strict_final._preflight_checks(args)

        by_label = {check["label"]: check for check in checks}
        self.assertFalse(by_label["artifacts_2k"]["ok"])
        self.assertFalse(by_label["artifacts_2k"]["nonempty"])

    def test_preflight_report_is_written_to_default_output_path(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            args = _args(root / "out")
            args.model_path = str(root / "model")
            args.artifacts_2k = str(root / "artifacts-2k")
            args.artifacts_8k = str(root / "artifacts-8k")
            args.artifacts_32k = str(root / "artifacts-32k")
            args.artifacts_128k = str(root / "artifacts-128k")
            args.artifacts_262k = str(root / "artifacts-262k")
            for path in (
                args.model_path,
                args.artifacts_2k,
                args.artifacts_8k,
                args.artifacts_32k,
                args.artifacts_128k,
                args.artifacts_262k,
            ):
                path_obj = Path(path)
                path_obj.mkdir()
                (path_obj / "marker").write_text("x", encoding="utf-8")
            checks = self.strict_final._preflight_checks(args)
            report = self.strict_final._preflight_report(args, checks)

            report_path = self.strict_final._write_preflight_report(args, report)
            payload = json.loads(report_path.read_text(encoding="utf-8"))

        self.assertEqual(report_path.name, "strict_preflight.json")
        self.assertTrue(payload["passed"])
        self.assertEqual(payload["expected_branch"], "qwen36-cold-prefill-perf")
        self.assertEqual(payload["runtime_shape"]["tensor_parallel_size"], 4)
        self.assertIn("acceptance", payload["expected_outputs"])
        self.assertIn("acceptance", payload["expected_logs"])
        self.assertTrue(payload["run_manifest"].endswith("strict_run_manifest.json"))

    def test_preflight_report_uses_explicit_json_path(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            args = _args(root / "out")
            args.preflight_json = root / "preflight-report.json"

            path = self.strict_final._preflight_json_path(args)

        self.assertEqual(path.name, "preflight-report.json")

    def test_run_logged_command_writes_phase_log(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = Path(tmpdir) / "phase.log"
            returncode = self.strict_final._run_logged_command(
                "phase",
                [sys.executable, "-c", "print('phase ok')"],
                log_path,
            )
            text = log_path.read_text(encoding="utf-8")

        self.assertEqual(returncode, 0)
        self.assertIn("phase ok", text)
        self.assertIn("returncode=0", text)

    def test_run_manifest_records_commands_outputs_and_logs(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            args = _args(Path(tmpdir))
            commands = self.strict_final._build_commands(args)

            manifest = self.strict_final._run_manifest(args, commands)
            manifest_path = self.strict_final._write_run_manifest(args, manifest)
            payload = json.loads(manifest_path.read_text(encoding="utf-8"))

        self.assertEqual(manifest_path.name, "strict_run_manifest.json")
        self.assertTrue(payload["strict_final"])
        self.assertIn("matrix", payload["commands"])
        self.assertIn("acceptance", payload["command_strings"])
        self.assertIn("matrix", payload["expected_outputs"])
        self.assertIn("hybrid_apc", payload["logs"])
        self.assertIn("matrix", payload["evidence_status"]["outputs"])
        self.assertIn("preflight", payload["evidence_status"])
        self.assertFalse(payload["evidence_status"]["outputs"]["matrix"]["exists"])
        self.assertEqual(payload["phases"], {})

    def test_evidence_status_reports_existing_output_and_log_files(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            args = _args(Path(tmpdir))
            preflight = Path(tmpdir) / "strict_preflight.json"
            matrix_output = Path(tmpdir) / "strict_matrix.json"
            matrix_log = Path(tmpdir) / "strict_matrix.log"
            preflight.write_text("{}\n", encoding="utf-8")
            matrix_output.write_text("[]\n", encoding="utf-8")
            matrix_log.write_text("matrix log\n", encoding="utf-8")

            status = self.strict_final._evidence_status(args)

        self.assertTrue(status["preflight"]["exists"])
        self.assertTrue(status["preflight"]["nonempty"])
        self.assertTrue(status["outputs"]["matrix"]["exists"])
        self.assertTrue(status["outputs"]["matrix"]["nonempty"])
        self.assertTrue(status["logs"]["matrix"]["exists"])
        self.assertTrue(status["logs"]["matrix"]["nonempty"])
        self.assertFalse(status["outputs"]["acceptance"]["exists"])

    def test_evidence_check_passes_for_complete_strict_final_bundle(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            args = _args(Path(tmpdir))
            commands = self.strict_final._build_commands(args)
            _write_preflight_fixture(self.strict_final, args)
            for name, path in self.strict_final._expected_output_paths(args).items():
                payload = (
                    _strict_acceptance_payload()
                    if name == "acceptance"
                    else {}
                )
                Path(path).write_text(json.dumps(payload) + "\n", encoding="utf-8")
            for path in self.strict_final._phase_log_paths(args).values():
                Path(path).write_text("phase log\n", encoding="utf-8")
            manifest = self.strict_final._run_manifest(args, commands)
            manifest["phases"] = {
                phase: {
                    "returncode": 0,
                    "elapsed_seconds": 1.0,
                    "log": self.strict_final._phase_log_paths(args)[phase],
                }
                for phase in self.strict_final.RUN_PHASES
            }
            manifest["returncode"] = 0
            manifest["passed"] = True
            self.strict_final._write_run_manifest(args, manifest)

            report = self.strict_final._evidence_check_report(args)

        self.assertTrue(report["passed"], report)
        self.assertTrue(report["preflight"]["passed"])
        self.assertTrue(report["acceptance"]["passed"])
        self.assertTrue(report["acceptance"]["strict_final"])
        self.assertTrue(report["acceptance"]["audit_checklist_present"])
        self.assertTrue(report["manifest"]["passed"])
        self.assertTrue(report["preflight"]["expected_path_check"]["ok"])
        self.assertTrue(
            all(check["ok"] for check in report["manifest"]["command_checks"])
        )
        self.assertTrue(
            all(check["ok"] for check in report["manifest"]["phase_checks"])
        )

    def test_evidence_check_fails_when_preflight_did_not_pass(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            args = _args(Path(tmpdir))
            commands = self.strict_final._build_commands(args)
            _write_preflight_fixture(
                self.strict_final,
                args,
                passed=False,
                checks=[
                    {
                        "label": "git_branch",
                        "ok": False,
                        "actual_branch": "experimental",
                    }
                ],
            )
            for name, path in self.strict_final._expected_output_paths(args).items():
                payload = (
                    _strict_acceptance_payload()
                    if name == "acceptance"
                    else {}
                )
                Path(path).write_text(json.dumps(payload) + "\n", encoding="utf-8")
            for path in self.strict_final._phase_log_paths(args).values():
                Path(path).write_text("phase log\n", encoding="utf-8")
            manifest = self.strict_final._run_manifest(args, commands)
            manifest["phases"] = {
                phase: {
                    "returncode": 0,
                    "elapsed_seconds": 1.0,
                    "log": self.strict_final._phase_log_paths(args)[phase],
                }
                for phase in self.strict_final.RUN_PHASES
            }
            manifest["returncode"] = 0
            manifest["passed"] = True
            self.strict_final._write_run_manifest(args, manifest)

            report = self.strict_final._evidence_check_report(args)

        self.assertFalse(report["passed"])
        self.assertFalse(report["preflight"]["passed"])
        self.assertEqual(report["preflight"]["failed_checks"][0]["label"], "git_branch")

    def test_evidence_check_fails_when_preflight_paths_do_not_match(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            args = _args(Path(tmpdir))
            commands = self.strict_final._build_commands(args)
            _write_preflight_fixture(self.strict_final, args)
            preflight_path = self.strict_final._preflight_json_path(args)
            preflight = json.loads(preflight_path.read_text(encoding="utf-8"))
            preflight["expected_outputs"]["matrix"] = "/tmp/other_matrix.json"
            preflight_path.write_text(json.dumps(preflight) + "\n", encoding="utf-8")
            for name, path in self.strict_final._expected_output_paths(args).items():
                payload = (
                    _strict_acceptance_payload()
                    if name == "acceptance"
                    else {}
                )
                Path(path).write_text(json.dumps(payload) + "\n", encoding="utf-8")
            for path in self.strict_final._phase_log_paths(args).values():
                Path(path).write_text("phase log\n", encoding="utf-8")
            manifest = self.strict_final._run_manifest(args, commands)
            manifest["phases"] = {
                phase: {
                    "returncode": 0,
                    "elapsed_seconds": 1.0,
                    "log": self.strict_final._phase_log_paths(args)[phase],
                }
                for phase in self.strict_final.RUN_PHASES
            }
            manifest["returncode"] = 0
            manifest["passed"] = True
            self.strict_final._write_run_manifest(args, manifest)

            report = self.strict_final._evidence_check_report(args)

        self.assertFalse(report["passed"])
        self.assertFalse(
            report["preflight"]["expected_path_check"]["expected_outputs_match"]
        )

    def test_evidence_check_fails_when_manifest_commands_are_missing(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            args = _args(Path(tmpdir))
            commands = self.strict_final._build_commands(args)
            _write_preflight_fixture(self.strict_final, args)
            for name, path in self.strict_final._expected_output_paths(args).items():
                payload = (
                    _strict_acceptance_payload()
                    if name == "acceptance"
                    else {}
                )
                Path(path).write_text(json.dumps(payload) + "\n", encoding="utf-8")
            for path in self.strict_final._phase_log_paths(args).values():
                Path(path).write_text("phase log\n", encoding="utf-8")
            manifest = self.strict_final._run_manifest(args, commands)
            manifest["commands"] = {}
            manifest["command_strings"] = {}
            manifest["phases"] = {
                phase: {
                    "returncode": 0,
                    "elapsed_seconds": 1.0,
                    "log": self.strict_final._phase_log_paths(args)[phase],
                }
                for phase in self.strict_final.RUN_PHASES
            }
            manifest["returncode"] = 0
            manifest["passed"] = True
            self.strict_final._write_run_manifest(args, manifest)

            report = self.strict_final._evidence_check_report(args)

        self.assertFalse(report["passed"])
        self.assertFalse(report["manifest"]["command_checks"][0]["ok"])

    def test_evidence_check_fails_when_acceptance_was_not_strict_final(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            args = _args(Path(tmpdir))
            commands = self.strict_final._build_commands(args)
            _write_preflight_fixture(self.strict_final, args)
            for name, path in self.strict_final._expected_output_paths(args).items():
                payload = (
                    _strict_acceptance_payload(strict_final=False)
                    if name == "acceptance"
                    else {}
                )
                Path(path).write_text(json.dumps(payload) + "\n", encoding="utf-8")
            for path in self.strict_final._phase_log_paths(args).values():
                Path(path).write_text("phase log\n", encoding="utf-8")
            manifest = self.strict_final._run_manifest(args, commands)
            manifest["phases"] = {
                phase: {
                    "returncode": 0,
                    "elapsed_seconds": 1.0,
                    "log": self.strict_final._phase_log_paths(args)[phase],
                }
                for phase in self.strict_final.RUN_PHASES
            }
            manifest["returncode"] = 0
            manifest["passed"] = True
            self.strict_final._write_run_manifest(args, manifest)

            report = self.strict_final._evidence_check_report(args)

        self.assertFalse(report["passed"])
        self.assertFalse(report["acceptance"]["strict_final"])

    def test_evidence_check_fails_when_acceptance_audit_has_failures(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            args = _args(Path(tmpdir))
            commands = self.strict_final._build_commands(args)
            _write_preflight_fixture(self.strict_final, args)
            for name, path in self.strict_final._expected_output_paths(args).items():
                payload = (
                    _strict_acceptance_payload(audit_status="fail")
                    if name == "acceptance"
                    else {}
                )
                Path(path).write_text(json.dumps(payload) + "\n", encoding="utf-8")
            for path in self.strict_final._phase_log_paths(args).values():
                Path(path).write_text("phase log\n", encoding="utf-8")
            manifest = self.strict_final._run_manifest(args, commands)
            manifest["phases"] = {
                phase: {
                    "returncode": 0,
                    "elapsed_seconds": 1.0,
                    "log": self.strict_final._phase_log_paths(args)[phase],
                }
                for phase in self.strict_final.RUN_PHASES
            }
            manifest["returncode"] = 0
            manifest["passed"] = True
            self.strict_final._write_run_manifest(args, manifest)

            report = self.strict_final._evidence_check_report(args)

        self.assertFalse(report["passed"])
        self.assertEqual(
            report["acceptance"]["failed_audit_items"][0]["name"],
            "strict-final",
        )

    def test_evidence_check_fails_when_acceptance_did_not_pass(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            args = _args(Path(tmpdir))
            commands = self.strict_final._build_commands(args)
            _write_preflight_fixture(self.strict_final, args)
            for name, path in self.strict_final._expected_output_paths(args).items():
                payload = (
                    _strict_acceptance_payload(passed=False, audit_status="fail")
                    if name == "acceptance"
                    else {}
                )
                Path(path).write_text(json.dumps(payload) + "\n", encoding="utf-8")
            for path in self.strict_final._phase_log_paths(args).values():
                Path(path).write_text("phase log\n", encoding="utf-8")
            manifest = self.strict_final._run_manifest(args, commands)
            manifest["phases"] = {
                phase: {
                    "returncode": 0,
                    "elapsed_seconds": 1.0,
                    "log": self.strict_final._phase_log_paths(args)[phase],
                }
                for phase in self.strict_final.RUN_PHASES
            }
            manifest["returncode"] = 0
            manifest["passed"] = True
            self.strict_final._write_run_manifest(args, manifest)

            report = self.strict_final._evidence_check_report(args)

        self.assertFalse(report["passed"])
        self.assertFalse(report["acceptance"]["passed"])
        self.assertEqual(report["acceptance"]["failure_count"], 1)

    def test_trainium_wrapper_resolves_repo_from_script_path(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            fake_python = root / "fake_python.py"
            call_log = root / "calls.jsonl"
            fake_python.write_text(
                "\n".join(
                    [
                        "#!/usr/bin/env python3",
                        "import json",
                        "import os",
                        "import sys",
                        "with open(os.environ['FAKE_PYTHON_LOG'], 'a', encoding='utf-8') as handle:",
                        "    handle.write(json.dumps(sys.argv[1:]) + '\\n')",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            fake_python.chmod(0o755)
            env = {
                "PATH": os.environ.get("PATH", ""),
                "PYTHON": str(fake_python),
                "FAKE_PYTHON_LOG": str(call_log),
                "QWEN36_MODEL": str(root / "model"),
                "QWEN36_ARTIFACT_2K": str(root / "artifacts-2k"),
                "QWEN36_ARTIFACT_8K": str(root / "artifacts-8k"),
                "QWEN36_ARTIFACT_32K": str(root / "artifacts-32k"),
                "QWEN36_ARTIFACT_128K": str(root / "artifacts-128k"),
                "QWEN36_ARTIFACT_262K": str(root / "artifacts-262k"),
                "QWEN36_OUT": str(root / "out"),
                "QWEN36_REPETITIONS": "5",
                "QWEN36_STRICT_MIN_SAMPLES": "4",
                "QWEN36_EXPECTED_BRANCH": "qwen36-cold-prefill-perf",
                "QWEN36_OUTPUT_PREFIX": "custom_prefix",
                "QWEN36_FAIL_FAST": "1",
            }

            proc = subprocess.run(
                ["bash", str(_WRAPPER_PATH)],
                cwd=root,
                env=env,
                check=False,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            calls = [
                json.loads(line)
                for line in call_log.read_text(encoding="utf-8").splitlines()
            ]

        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertIn("== strict-final preflight ==", proc.stdout)
        self.assertEqual(len(calls), 4)
        self.assertTrue(all(call[0] == str(_STRICT_FINAL_PATH) for call in calls))
        self.assertIn("--preflight", calls[0])
        self.assertIn("--dry-run", calls[1])
        self.assertIn("--repetitions", calls[2])
        self.assertIn("5", calls[2])
        self.assertIn("--strict-min-samples", calls[2])
        self.assertIn("4", calls[2])
        self.assertIn("--output-prefix", calls[2])
        self.assertIn("custom_prefix", calls[2])
        self.assertIn("--fail-fast", calls[2])
        self.assertIn("--evidence-check", calls[3])
        self.assertIn("--output-prefix", calls[3])
        self.assertIn("custom_prefix", calls[3])


if __name__ == "__main__":
    unittest.main()
