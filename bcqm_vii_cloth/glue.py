\
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any, List


@dataclass(frozen=True)
class GlueParams:
    shared_bias: float
    phase_lock: float
    domains: float
    cadence_disorder: float


def linear_map(minv: float, maxv: float, n: float) -> float:
    return minv + n * (maxv - minv)


def resolve_glue_params(glue_cfg: Dict[str, Any], n: float) -> GlueParams:
    profile = glue_cfg["profile"]
    axes = glue_cfg["axes"]
    profile_axes: List[str] = glue_cfg.get("profile_axes", [])

    def val(ax: str) -> float:
        rng = axes[ax]
        if profile == "composite_all":
            return float(linear_map(rng["min"], rng["max"], n))
        if profile == "single_axis":
            if ax == profile_axes[0]:
                return float(linear_map(rng["min"], rng["max"], n))
            return float(rng["min"])
        if profile == "composite_subset":
            if ax in profile_axes:
                return float(linear_map(rng["min"], rng["max"], n))
            return float(rng["min"])
        return float(linear_map(rng["min"], rng["max"], n))

    return GlueParams(
        shared_bias=val("shared_bias"),
        phase_lock=val("phase_lock"),
        domains=val("domains"),
        cadence_disorder=val("cadence_disorder"),
    )


def sync_strength(g: GlueParams) -> float:
    base = (g.shared_bias + g.phase_lock + g.domains) / 3.0
    return max(0.0, min(1.0, base * (1.0 - g.cadence_disorder)))
