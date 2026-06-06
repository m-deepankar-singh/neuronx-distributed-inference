import os
import subprocess
from pathlib import Path


_REPO_ROOT = Path(__file__).resolve().parents[3]
_DRIVER = _REPO_ROOT / "tmp_launch_qwen36_segcte2048.sh"


def _parse_key_values(text):
    parsed = {}
    for line in text.splitlines():
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        parsed[key] = value
    return parsed


def test_launch_dry_run_writes_runtime_identity_without_pid(tmp_path):
    artifact = tmp_path / "artifact"
    artifact.mkdir()
    log = tmp_path / "runtime.log"
    repo = tmp_path / "repo"
    model = tmp_path / "model"
    venv = tmp_path / "venv"
    repo.mkdir()
    model.mkdir()
    venv.mkdir()
    env = os.environ.copy()
    env.update(
        {
            "LAUNCH_DRY_RUN": "1",
            "REPO": str(repo),
            "MODEL": str(model),
            "VENV": str(venv),
            "MAX_MODEL_LEN": "32768",
            "SEQ_LEN": "32768",
            "CONTEXT_ENCODING_BUCKET_PAIRS": "2048:0 2048:16384 2048:32768",
            "TOKEN_GENERATION_BUCKETS": "512 16384 32768",
            "GDN_RECURRENT_CACHE_DTYPE": "bfloat16",
            "GDN_CONV_CACHE_DTYPE": "bfloat16",
            "GPU_MEMORY_UTILIZATION": "0.90",
            "KV_CACHE_DTYPE": "auto",
            "KV_CACHE_MEMORY_BYTES": "123456",
        }
    )

    completed = subprocess.run(
        ["bash", str(_DRIVER), str(artifact), str(log)],
        check=True,
        capture_output=True,
        env=env,
        text=True,
    )

    stdout = _parse_key_values(completed.stdout)
    envlog = _parse_key_values(Path(stdout["ENVLOG"]).read_text())
    assert stdout["ARTIFACT"] == str(artifact)
    assert stdout["LOG"] == str(log)
    assert stdout["LAUNCH_DRY_RUN"] == "1"
    assert envlog["ARTIFACT"] == str(artifact)
    assert envlog["LAUNCH_DRY_RUN"] == "1"
    assert envlog["MAX_MODEL_LEN"] == "32768"
    assert envlog["SEQ_LEN"] == "32768"
    assert envlog["DISABLE_HYBRID_KV_CACHE_MANAGER"] == "0"
    assert envlog["ENABLE_HYBRID_APC"] == "1"
    assert envlog["ENABLE_PREFIX_CACHING"] == "1"
    assert envlog["ENABLE_VLLM_CHUNKED_PREFILL"] == "1"
    assert envlog["HYBRID_APC_REQUIRE_VLLM_METADATA"] == "1"
    assert envlog["HYBRID_APC_ENABLE_BACKED_PREFIX_READS"] == "1"
    assert envlog["BLOCK_SIZE"] == "256"
    assert envlog["GDN_CHECKPOINT_INTERVAL"] == "256"
    assert "--gpu-memory-utilization 0.90" == envlog["GPU_MEMORY_ARGS"]
    assert "--kv-cache-dtype auto" == envlog["KV_CACHE_DTYPE_ARGS"]
    assert "--kv-cache-memory-bytes 123456" == envlog["KV_CACHE_MEMORY_ARGS"]
    assert not Path(stdout["PIDFILE"]).exists()


def test_launch_rejects_disabled_hybrid_kv_cache_manager(tmp_path):
    artifact = tmp_path / "artifact"
    artifact.mkdir()
    log = tmp_path / "runtime.log"
    env = os.environ.copy()
    env.update(
        {
            "LAUNCH_DRY_RUN": "1",
            "DISABLE_HYBRID_KV_CACHE_MANAGER": "1",
        }
    )

    completed = subprocess.run(
        ["bash", str(_DRIVER), str(artifact), str(log)],
        check=False,
        capture_output=True,
        env=env,
        text=True,
    )

    assert completed.returncode == 2
    assert "DISABLE_HYBRID_KV_CACHE_MANAGER must remain 0/absent" in completed.stderr
