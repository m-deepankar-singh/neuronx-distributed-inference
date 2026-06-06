import importlib.util
import json
import subprocess
import sys
from pathlib import Path


_REPO_ROOT = Path(__file__).resolve().parents[3]
_SCRIPT_PATH = _REPO_ROOT / "validation_scripts/qwen36_launch_env_audit.py"
_SPEC = importlib.util.spec_from_file_location("qwen36_launch_env_audit", _SCRIPT_PATH)
_SCRIPT = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _SCRIPT
_SPEC.loader.exec_module(_SCRIPT)


def _compile_env(**overrides):
    values = {
        "ENVLOG": "/logs/compile_env.txt",
        "ARTIFACT": "/artifacts/qwen",
        "SOURCE_COMMIT": "abc1234",
        "SOURCE_BRANCH": "codex/qwen36-prefill-speed-coherent",
        "MAX_CONTEXT_LENGTH": "32768",
        "SEQ_LEN": "32768",
        "CTE_BUCKETS": "2048",
        "TOKEN_GENERATION_BUCKETS": "512 16384 16640 32768",
        "CONTEXT_ENCODING_BUCKET_PAIRS": "2048:256 2048:512 2048:1024 2048:2048 2048:4096 2048:8192 2048:16384 2048:32768",
        "MAX_GDN_CHECKPOINT_SLOTS": "64",
        "GDN_RECURRENT_CACHE_DTYPE": "bfloat16",
        "GDN_CONV_CACHE_DTYPE": "bfloat16",
        "QWEN36_DELTANET_MULTIHEAD_CTE": "0",
        "QWEN36_SPLIT_QKV_TKG_ROW_SCALES": "0",
        "QWEN36_DELTANET_FUSED_SEGMENT_TOKENS": "0",
        "QWEN36_DELTANET_SOLVE_MODE": "direct",
        "QWEN36_DELTANET_SOLVE_SCAN_STEPS": "0",
        "ENABLE_KV_CACHE_QUANT": "0",
    }
    values.update(overrides)
    return values


def _launch_env(**overrides):
    values = {
        "ENVLOG": "/logs/launch_env.txt",
        "ARTIFACT": "/artifacts/qwen",
        "LAUNCH_DRY_RUN": "1",
        "MAX_MODEL_LEN": "32768",
        "SEQ_LEN": "32768",
        "CTE_BUCKETS": "2048",
        "TOKEN_GENERATION_BUCKETS": "512 16384 16640 32768",
        "CONTEXT_ENCODING_BUCKET_PAIRS": "2048:0 2048:256 2048:512 2048:1024 2048:2048 2048:4096 2048:8192 2048:16384 2048:32768",
        "MAX_GDN_CHECKPOINT_SLOTS": "64",
        "GDN_RECURRENT_CACHE_DTYPE": "bfloat16",
        "GDN_CONV_CACHE_DTYPE": "torch.bfloat16",
        "QWEN36_DELTANET_MULTIHEAD_CTE": "0",
        "QWEN36_SPLIT_QKV_TKG_ROW_SCALES": "0",
        "QWEN36_DELTANET_FUSED_SEGMENT_TOKENS": "0",
        "QWEN36_DELTANET_SOLVE_MODE": "direct",
        "QWEN36_DELTANET_SOLVE_SCAN_STEPS": "0",
        "DISABLE_HYBRID_KV_CACHE_MANAGER": "0",
        "ENABLE_HYBRID_APC": "1",
        "ENABLE_PREFIX_CACHING": "1",
        "ENABLE_VLLM_CHUNKED_PREFILL": "1",
        "HYBRID_APC_REQUIRE_VLLM_METADATA": "1",
        "HYBRID_APC_ENABLE_BACKED_PREFIX_READS": "1",
        "BLOCK_SIZE": "256",
        "GDN_CHECKPOINT_INTERVAL": "256",
        "KV_CACHE_DTYPE": "",
        "KV_CACHE_DTYPE_ARGS": "",
    }
    values.update(overrides)
    return values


def _write_env(path, values):
    path.write_text("".join(f"{key}={value}\n" for key, value in values.items()))


def test_audit_passes_for_matching_launch_with_prefix_zero_pair():
    result = _SCRIPT.audit(compile_env=_compile_env(), launch_env=_launch_env())

    assert result["passed"]
    assert result["warnings"] == [
        "launch adds prefix-0 context pair(s), which are allowed: 2048:0"
    ]


def test_audit_fails_when_launch_advertises_uncompiled_256k_buckets():
    result = _SCRIPT.audit(
        compile_env=_compile_env(),
        launch_env=_launch_env(
            MAX_MODEL_LEN="262144",
            SEQ_LEN="262144",
            TOKEN_GENERATION_BUCKETS="512 16384 16640 32768 65536 131072 262144",
            CONTEXT_ENCODING_BUCKET_PAIRS=(
                "2048:0 2048:256 2048:512 2048:1024 2048:2048 2048:4096 "
                "2048:8192 2048:16384 2048:32768 2048:65536 2048:131072 2048:262144"
            ),
        ),
    )

    assert not result["passed"]
    joined = "\n".join(result["errors"])
    assert "MAX_MODEL_LEN must match compiled MAX_CONTEXT_LENGTH" in joined
    assert "SEQ_LEN must match compiled SEQ_LEN" in joined
    assert "TOKEN_GENERATION_BUCKETS must match compiled buckets" in joined
    assert "extra_not_prefix0=['2048:65536', '2048:131072', '2048:262144']" in joined


def test_audit_fails_on_gdn_dtype_or_hybrid_kv_drift():
    result = _SCRIPT.audit(
        compile_env=_compile_env(),
        launch_env=_launch_env(
            GDN_RECURRENT_CACHE_DTYPE="float32",
            DISABLE_HYBRID_KV_CACHE_MANAGER="1",
        ),
    )

    assert not result["passed"]
    assert any("GDN_RECURRENT_CACHE_DTYPE mismatch" in error for error in result["errors"])
    assert any(
        "DISABLE_HYBRID_KV_CACHE_MANAGER must be '0'" in error
        for error in result["errors"]
    )


def test_audit_rejects_fp8_runtime_kv_when_compile_uses_bf16_kv():
    result = _SCRIPT.audit(
        compile_env=_compile_env(ENABLE_KV_CACHE_QUANT="0"),
        launch_env=_launch_env(
            KV_CACHE_DTYPE="fp8",
            KV_CACHE_DTYPE_ARGS="--kv-cache-dtype fp8",
        ),
    )

    assert not result["passed"]
    assert any("KV_CACHE_DTYPE must not be fp8" in error for error in result["errors"])


def test_audit_allows_auto_or_bf16_runtime_kv_when_compile_uses_bf16_kv():
    for dtype, args in (
        ("", ""),
        ("auto", "--kv-cache-dtype auto"),
        ("bfloat16", "--kv-cache-dtype bfloat16"),
        ("bf16", "--kv-cache-dtype bf16"),
    ):
        result = _SCRIPT.audit(
            compile_env=_compile_env(ENABLE_KV_CACHE_QUANT="0"),
            launch_env=_launch_env(KV_CACHE_DTYPE=dtype, KV_CACHE_DTYPE_ARGS=args),
        )

        assert result["passed"], (dtype, args, result["errors"])


def test_audit_requires_fp8_runtime_kv_when_compile_uses_fp8_kv():
    missing_runtime_fp8 = _SCRIPT.audit(
        compile_env=_compile_env(ENABLE_KV_CACHE_QUANT="1"),
        launch_env=_launch_env(KV_CACHE_DTYPE="", KV_CACHE_DTYPE_ARGS=""),
    )
    matching_runtime_fp8 = _SCRIPT.audit(
        compile_env=_compile_env(ENABLE_KV_CACHE_QUANT="1"),
        launch_env=_launch_env(
            KV_CACHE_DTYPE="fp8_e5m2",
            KV_CACHE_DTYPE_ARGS="--kv-cache-dtype fp8_e5m2",
        ),
    )

    assert not missing_runtime_fp8["passed"]
    assert any(
        "KV_CACHE_DTYPE must be fp8" in error
        for error in missing_runtime_fp8["errors"]
    )
    assert matching_runtime_fp8["passed"]


def test_audit_rejects_kv_dtype_env_and_args_disagreement():
    result = _SCRIPT.audit(
        compile_env=_compile_env(),
        launch_env=_launch_env(
            KV_CACHE_DTYPE="auto",
            KV_CACHE_DTYPE_ARGS="--kv-cache-dtype fp8",
        ),
    )

    assert not result["passed"]
    assert any(
        "KV_CACHE_DTYPE and KV_CACHE_DTYPE_ARGS disagree" in error
        for error in result["errors"]
    )


def test_cli_writes_json_and_uses_nonzero_exit_for_mismatch(tmp_path):
    compile_env_path = tmp_path / "compile_env.txt"
    launch_env_path = tmp_path / "launch_env.txt"
    output_json = tmp_path / "audit.json"
    _write_env(compile_env_path, _compile_env())
    _write_env(launch_env_path, _launch_env(ARTIFACT="/wrong/artifact"))

    completed = subprocess.run(
        [
            sys.executable,
            str(_SCRIPT_PATH),
            "--compile-env-log",
            str(compile_env_path),
            "--launch-env-log",
            str(launch_env_path),
            "--output-json",
            str(output_json),
            "--require-launch-dry-run",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 1
    payload = json.loads(output_json.read_text())
    assert not payload["passed"]
    assert payload == json.loads(completed.stdout)
    assert any("ARTIFACT mismatch" in error for error in payload["errors"])
