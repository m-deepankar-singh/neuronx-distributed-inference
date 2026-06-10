from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Tuple


def _compiled_seq_len(compiled_model_path: str) -> int:
    config_path = Path(compiled_model_path) / "neuron_config.json"
    with open(config_path) as f:
        config = json.load(f)
    neuron_config = config.get("neuron_config", config)
    return int(neuron_config.get("seq_len") or neuron_config.get("max_context_length") or 131072)


def load_model(model_config: Dict[str, Any]) -> Tuple[Any, Any, Any]:
    from validator.main import create_model
    from validator.patches import ensure_generation_config_version, patch_generation_mixin

    config = {
        "model_name": Path(model_config["model_path"]).name,
        "model_path": model_config["model_path"],
        "compiled_model_path": model_config["compiled_model_path"],
        "model_class": model_config["model_class"],
        "config_class": model_config["config_class"],
        "test_parameters": [{"batch_size": 1, "seq_len": _compiled_seq_len(model_config["compiled_model_path"])}],
    }
    seq_len = _compiled_seq_len(model_config["compiled_model_path"])
    model, tokenizer, generation_config = create_model(config, batch_size=1, seq_len=seq_len)
    model.load(model_config["compiled_model_path"])
    ensure_generation_config_version(model)
    patch_generation_mixin()
    return model, tokenizer, generation_config
