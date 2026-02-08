\
from __future__ import annotations

from typing import Any, Dict, List
# PROVENANCE: BCQM_VII stage2 cloth; schema add store_lists; 2026-01-30


class ConfigError(ValueError):
    pass


def _req(d: Dict[str, Any], key: str) -> Any:
    if key not in d:
        raise ConfigError(f"Missing required key: {key}")
    return d[key]


def _as_float(x: Any, name: str) -> float:
    try:
        return float(x)
    except Exception as e:
        raise ConfigError(f"Expected float for {name}, got {type(x)}") from e


def _as_int(x: Any, name: str) -> int:
    try:
        return int(x)
    except Exception as e:
        raise ConfigError(f"Expected int for {name}, got {type(x)}") from e


def resolve_seeds(seeds_cfg: Any) -> List[int]:
    if isinstance(seeds_cfg, list):
        return [int(s) for s in seeds_cfg]
    if isinstance(seeds_cfg, dict):
        count = _as_int(_req(seeds_cfg, "count"), "seeds.count")
        start = _as_int(seeds_cfg.get("start", 1), "seeds.start")
        return list(range(start, start + count))
    raise ConfigError("seeds must be a list[int] or {count,start}")


def resolve_n_values(scan_cfg: Any) -> List[float]:
    if not isinstance(scan_cfg, dict):
        raise ConfigError("scan must be a mapping")
    if "n_values" in scan_cfg:
        vals = scan_cfg["n_values"]
        if not isinstance(vals, list):
            raise ConfigError("scan.n_values must be a list")
        return [float(v) for v in vals]
    n_min = _as_float(_req(scan_cfg, "n_min"), "scan.n_min")
    n_max = _as_float(_req(scan_cfg, "n_max"), "scan.n_max")
    n_step = _as_float(_req(scan_cfg, "n_step"), "scan.n_step")
    if n_step <= 0:
        raise ConfigError("scan.n_step must be > 0")
    n = n_min
    out: List[float] = []
    while n <= n_max + 1e-12:
        out.append(round(n, 12))
        n += n_step
    return out


def validate(cfg: Dict[str, Any]) -> None:
    _req(cfg, "schema_version")
    _req(cfg, "experiment_id")
    variant = _req(cfg, "variant")
    if variant not in ("full", "lockstep_only", "crosslink_only"):
        raise ConfigError("variant must be full | lockstep_only | crosslink_only")

    sizes = _req(cfg, "sizes")
    if not isinstance(sizes, list) or not sizes:
        raise ConfigError("sizes must be a non-empty list[int]")
    for N in sizes:
        if int(N) <= 0:
            raise ConfigError("sizes entries must be positive")

    steps_total = _as_int(_req(cfg, "steps_total"), "steps_total")
    burn_in = _as_int(_req(cfg, "burn_in_epochs"), "burn_in_epochs")
    measure = _as_int(_req(cfg, "measure_epochs"), "measure_epochs")
    if burn_in + measure != steps_total:
        raise ConfigError("burn_in_epochs + measure_epochs must equal steps_total")

    _as_int(_req(cfg, "W_coh"), "W_coh")

    aw = _req(cfg, "active_window")
    if not isinstance(aw, dict):
        raise ConfigError("active_window must be a mapping")
    mode = _req(aw, "mode")
    if mode not in ("recency", "horizon_ball"):
        raise ConfigError("active_window.mode must be recency | horizon_ball")
    if mode == "recency":
        _as_int(_req(aw, "hops"), "active_window.hops")
    else:
        _as_int(_req(aw, "d_horizon"), "active_window.d_horizon")

    glue = _req(cfg, "glue")
    if not isinstance(glue, dict):
        raise ConfigError("glue must be a mapping")
    if glue.get("mapping_mode", "linear") != "linear":
        raise ConfigError("Only glue.mapping_mode=linear is supported in v0.1.0")
    profile = _req(glue, "profile")
    if profile not in ("single_axis", "composite_all", "composite_subset"):
        raise ConfigError("glue.profile invalid")
    axes = _req(glue, "axes")
    if not isinstance(axes, dict):
        raise ConfigError("glue.axes must be a mapping")
    for ax in ("shared_bias", "phase_lock", "domains", "cadence_disorder"):
        if ax not in axes:
            raise ConfigError(f"glue.axes missing: {ax}")
        rng = axes[ax]
        if not isinstance(rng, dict) or "min" not in rng or "max" not in rng:
            raise ConfigError(f"glue.axes.{ax} must be {{min,max}}")
        _as_float(rng["min"], f"glue.axes.{ax}.min")
        _as_float(rng["max"], f"glue.axes.{ax}.max")
    if profile in ("single_axis", "composite_subset"):
        pa = _req(glue, "profile_axes")
        if not isinstance(pa, list) or not pa:
            raise ConfigError("glue.profile_axes must be a non-empty list for this profile")

    x = _req(cfg, "crosslinks")
    if not isinstance(x, dict):
        raise ConfigError("crosslinks must be a mapping")
    allow = bool(_req(x, "allow_junctions"))
    if variant == "lockstep_only" and allow:
        raise ConfigError("lockstep_only requires crosslinks.allow_junctions=false")
    if variant in ("full", "crosslink_only") and not allow:
        raise ConfigError("full/crosslink_only require crosslinks.allow_junctions=true")
    _req(x, "new_event_policy")
    if variant == "lockstep_only":
        _req(x, "lockstep_only_policy")

    out = _req(cfg, "output")
    if not isinstance(out, dict) or "out_dir" not in out:
        raise ConfigError("output.out_dir required")

    snaps = _req(cfg, "snapshots")
    if not isinstance(snaps, dict) or "enabled" not in snaps:
        raise ConfigError("snapshots.enabled required")

    resolve_seeds(_req(cfg, "seeds"))
    resolve_n_values(_req(cfg, "scan"))

    # Optional Stage-2 cloth config
    if "cloth" in cfg:
        cloth = cfg.get("cloth")
        if not isinstance(cloth, dict):
            raise ConfigError("cloth must be a mapping")
        if "enabled" in cloth:
            bool(cloth.get("enabled"))
        if "bins" in cloth:
            _as_int(cloth["bins"], "cloth.bins")
        if "w_lock" in cloth:
            _as_float(cloth["w_lock"], "cloth.w_lock")
        if "min_concurrency" in cloth:
            _as_int(cloth["min_concurrency"], "cloth.min_concurrency")
        if "min_bin_hits" in cloth:
            _as_int(cloth["min_bin_hits"], "cloth.min_bin_hits")
        if "include_ledger" in cloth:
            bool(cloth.get("include_ledger"))

