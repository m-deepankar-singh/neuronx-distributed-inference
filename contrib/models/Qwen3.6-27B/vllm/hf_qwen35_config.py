"""Minimal Hugging Face config registration for Qwen3.5/Qwen3.6 vLLM smoke.

The Neuron vLLM environment can lag upstream Transformers. vLLM validates the
HF config before the NxDI model registry gets a chance to instantiate the
contrib model, so register a permissive config class for the new model_type.
"""

from __future__ import annotations

from transformers import AutoConfig, PretrainedConfig


class Qwen35TextConfig(PretrainedConfig):
    model_type = "qwen3_5_text"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)


class Qwen35Config(PretrainedConfig):
    model_type = "qwen3_5"
    sub_configs = {"text_config": Qwen35TextConfig}

    def __init__(self, text_config=None, **kwargs):
        if isinstance(text_config, dict):
            text_config = Qwen35TextConfig(**text_config)
        self.text_config = text_config
        super().__init__(**kwargs)


def _is_registered(model_type: str) -> bool:
    try:
        AutoConfig.for_model(model_type)
    except ValueError:
        return False
    return True


def register_qwen35_config() -> None:
    if not _is_registered(Qwen35TextConfig.model_type):
        AutoConfig.register(Qwen35TextConfig.model_type, Qwen35TextConfig)
    if not _is_registered(Qwen35Config.model_type):
        AutoConfig.register(Qwen35Config.model_type, Qwen35Config)
