from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

import yaml


@dataclass
class BenchConfig:
    model: Dict[str, Any]
    lm_eval: Dict[str, Any]
    tolerance: Dict[str, float]


def parse_config(path: str) -> BenchConfig:
    with open(path) as f:
        raw = yaml.safe_load(f)

    lm_eval = (raw.get("benchmarks") or {}).get("lm_eval") or {}
    tolerance = raw.get("tolerance") or {"default": 0.02}
    if isinstance(tolerance, (float, int)):
        tolerance = {"default": float(tolerance)}

    return BenchConfig(
        model=raw["model"],
        lm_eval=lm_eval,
        tolerance=tolerance,
    )
