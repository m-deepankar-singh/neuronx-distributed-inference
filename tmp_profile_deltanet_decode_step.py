#!/usr/bin/env python3
"""Profile the Qwen3.6 DeltaNet one-token NKI decode kernel on Neuron."""

import os
import importlib.util
from pathlib import Path


def _setdefault_env() -> None:
    out_dir = os.environ.get("NEURON_RT_INSPECT_OUTPUT_DIR")
    if not out_dir:
        out_dir = "/tmp/deltanet_decode_step_profile"
        os.environ["NEURON_RT_INSPECT_OUTPUT_DIR"] = out_dir
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    os.environ.setdefault("XLA_IR_DEBUG", "1")
    os.environ.setdefault("XLA_HLO_DEBUG", "1")
    os.environ.setdefault("NEURON_FRAMEWORK_DEBUG", "1")
    os.environ.setdefault("NEURON_RT_INSPECT_ENABLE", "1")
    os.environ.setdefault("NEURON_RT_INSPECT_DEVICE_PROFILE", "1")
    os.environ.setdefault("NEURON_RT_INSPECT_SYSTEM_PROFILE", "0")
    os.environ.setdefault("NEURON_RT_VISIBLE_CORES", "0")


_setdefault_env()

import torch
import torch_xla.core.xla_model as xm


def _load_deltanet_step():
    repo = Path(__file__).resolve().parent
    kernel_path = (
        repo / "contrib/models/Qwen3.6-27B/src/nki_kernels/nki_deltanet.py"
    )
    spec = importlib.util.spec_from_file_location("profile_nki_deltanet", kernel_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load NKI kernel module from {kernel_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.deltanet_recurrent_step_batched


deltanet_recurrent_step_batched = _load_deltanet_step()


def main() -> None:
    batch_heads = int(os.environ.get("PROFILE_BATCH_HEADS", "12"))
    state_dtype_name = os.environ.get("PROFILE_STATE_DTYPE", "float32")
    state_dtype = {
        "bf16": torch.bfloat16,
        "bfloat16": torch.bfloat16,
        "fp32": torch.float32,
        "float32": torch.float32,
    }[state_dtype_name]
    dim = 128
    torch.manual_seed(0)

    device = xm.xla_device()
    query = torch.randn(batch_heads, dim, dtype=torch.float32).to(device)
    key = torch.randn(batch_heads, dim, dtype=torch.float32).to(device)
    value = torch.randn(batch_heads, dim, dtype=torch.float32).to(device)
    g = torch.randn(batch_heads, 1, dtype=torch.float32).mul(-0.01).to(device)
    beta = torch.rand(batch_heads, 1, dtype=torch.float32).to(device)
    state = torch.randn(batch_heads * dim, dim, dtype=state_dtype).to(device)

    # Compile/warm, then execute again so the runtime profile captures a steady call.
    out, state_out = deltanet_recurrent_step_batched(query, key, value, g, beta, state)
    xm.mark_step()
    out, state_out = deltanet_recurrent_step_batched(query, key, value, g, beta, state_out)
    xm.mark_step()

    print(
        {
            "batch_heads": batch_heads,
            "state_dtype": str(state_dtype),
            "state_out_dtype": str(state_out.dtype),
            "output_shape": tuple(out.shape),
            "state_shape": tuple(state_out.shape),
            "inspect_output": os.environ["NEURON_RT_INSPECT_OUTPUT_DIR"],
        }
    )


if __name__ == "__main__":
    main()
