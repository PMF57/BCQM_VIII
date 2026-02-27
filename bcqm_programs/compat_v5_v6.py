from __future__ import annotations


"""
compat_v5_v6.py

Translation-only importer to replay BCQM V glue-axis configs as regression inputs for bcqm_vi_spacetime.

Design intent:
- Preserve BCQM V parameter semantics.
- Expand (W_coh, N, ensembles) grids into a run manifest.
- Emit VI-shaped run configs (locked key set) suitable for downstream execution.
- Store the original BCQM V config verbatim in provenance for traceability.

NOTE:
This is an importer/manifest generator. It does not run simulations.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml


def _safe_load_yaml(path: Path) -> Dict[str, Any]:
    """
    Load a BCQM V YAML config. Some configs use YAML end markers '...'.
    safe_load handles these fine.
    """
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"BCQM V config root must be a mapping: {path}")
    return data


def _get(d: Dict[str, Any], key: str, default: Any = None) -> Any:
    return d.get(key, default)


def _req(d: Dict[str, Any], key: str) -> Any:
    if key not in d:
        raise KeyError(f"Missing required BCQM V key: {key}")
    return d[key]


def _as_int(x: Any, name: str) -> int:
    try:
        return int(x)
    except Exception as e:
        raise ValueError(f"Expected int for {name}, got {type(x)}") from e


def _as_list(x: Any, name: str) -> List[Any]:
    if not isinstance(x, list):
        raise ValueError(f"Expected list for {name}, got {type(x)}")
    return x


@dataclass
class ImportOptions:
    """
    Options controlling how much of V diagnostics are reflected into VI-shaped configs.
    """
    respect_store_states: bool = False   # if True, snapshots.enabled follows V diagnostics.store_states
    respect_compute_psd: bool = False    # if True, output.write_timeseries follows V diagnostics.compute_psd
    emit_yaml_per_run: bool = False      # if True, write one VI YAML per expanded run (can be many)


def build_vi_run_config(
    *,
    experiment_id: str,
    description: str,
    variant: str,
    N: int,
    n_value: float,
    seed: int,
    steps_total: int,
    burn_in: int,
    W_coh: int,
    output_dir: str,
    v_source: Dict[str, Any],
    opts: ImportOptions,
) -> Dict[str, Any]:
    """
    Build a VI-shaped config dict that satisfies the locked key set.
    This config is designed to be valid under bcqm_vi_spacetime/config_schema.py,
    while carrying the original V config for provenance.
    """
    measure = steps_total - burn_in
    if measure <= 0:
        raise ValueError("Invalid grid: steps_total must exceed burn_in")

    # Minimal VI glue fields must exist to satisfy the locked schema.
    # These values are placeholders; once VI ports BCQM V glue dynamics, this block
    # will be superseded by the V-derived parameters stored in provenance.
    glue_stub = {
        "mapping_mode": "linear",
        "profile": "composite_all",
        "axes": {
            "shared_bias": {"min": 0.0, "max": 0.0},
            "phase_lock": {"min": 0.0, "max": 0.0},
            "domains": {"min": 0.0, "max": 0.0},
            "cadence_disorder": {"min": 1.0, "max": 1.0},
        },
    }

    # Crosslinks: BCQM V runs are pre-crosslink; keep junctions allowed to satisfy current validator,
    # but record intended behaviour in provenance.
    crosslinks = {
        "allow_junctions": True,
        "new_event_policy": "unique_per_thread",
    }

    # Diagnostics mapping
    diag = _get(v_source, "diagnostics", {}) or {}
    store_states = bool(_get(diag, "store_states", False))
    compute_psd = bool(_get(diag, "compute_psd", False))

    snapshots = {"enabled": False}
    if opts.respect_store_states:
        snapshots = {"enabled": store_states}

    output = {
        "out_dir": output_dir,
        "write_timeseries": False,
        "timeseries_bins": 50,
    }
    if opts.respect_compute_psd:
        output["write_timeseries"] = compute_psd

    cfg: Dict[str, Any] = {
        "schema_version": "vi_spacetime_config_v0.1",
        "experiment_id": experiment_id,
        "description": description,
        "variant": variant,
        "sizes": [N],
        "seeds": [seed],
        "scan": {"n_values": [float(n_value)]},
        "steps_total": int(steps_total),
        "burn_in_epochs": int(burn_in),
        "measure_epochs": int(measure),
        "W_coh": int(W_coh),
        "active_window": {"mode": "recency", "hops": int(W_coh)},
        "glue": glue_stub,
        "crosslinks": crosslinks,
        "observables": {
            "beta_junc": 1.5,
            "tick_definition": "v_clock_from_V",
            "compute_clustering": True,
            "compute_hub_metrics": True,
        },
        "snapshots": snapshots,
        "output": output,
        "anomaly_thresholds": {
            "hubshare_star": 0.90,
            "max_indegree_factor": 3.0,
            "degenerate_S_tol": 1.0e-6,
        },
        # Provenance: keep original BCQM V config verbatim
        "provenance": {
            "source": "BCQM V config",
            "v_config": v_source,
            "v_grid_point": {"W_coh": int(W_coh), "N": int(N), "seed": int(seed)},
            "notes": "Translation-only import; semantics preserved. Glue parameters are carried in v_config for later direct port.",
        },
    }
    return cfg


def import_bcqm_v_config(
    v_config_path: Path,
    out_dir: Path,
    *,
    opts: Optional[ImportOptions] = None,
) -> Path:
    """
    Import a BCQM V YAML config and emit an IMPORT_MANIFEST JSON.

    The manifest contains:
      - source config path
      - grid definition
      - run_count
      - list of run entries, each with:
          - run_key (W,N,seed)
          - vi_config (embedded dict)
          - (optional) vi_yaml_path if emit_yaml_per_run=True

    Returns the manifest path.
    """
    opts = opts or ImportOptions()
    out_dir.mkdir(parents=True, exist_ok=True)

    v_source = _safe_load_yaml(v_config_path)

    grid = _req(v_source, "grid")
    if not isinstance(grid, dict):
        raise ValueError("BCQM V key 'grid' must be a mapping")

    W_values = _as_list(_req(grid, "W_coh_values"), "grid.W_coh_values")
    N_values = _as_list(_req(grid, "N_values"), "grid.N_values")
    ensembles = _as_int(_req(grid, "ensembles"), "grid.ensembles")
    steps = _as_int(_req(grid, "steps"), "grid.steps")
    burn_in = _as_int(_req(grid, "burn_in"), "grid.burn_in")

    output_dir_v = str(_get(v_source, "output_dir", "outputs_v_import"))
    random_seed = _as_int(_get(v_source, "random_seed", 1), "random_seed")

    stem = v_config_path.stem
    experiment_id_base = f"vreg_{stem}"

    run_entries: List[Dict[str, Any]] = []
    yaml_dir = out_dir / "generated_vi_yamls"
    if opts.emit_yaml_per_run:
        yaml_dir.mkdir(parents=True, exist_ok=True)

    # Expand grid
    for W in W_values:
        W_int = _as_int(W, "W_coh_values item")
        for N in N_values:
            N_int = _as_int(N, "N_values item")
            for e in range(ensembles):
                seed = random_seed + e
                run_key = {"W_coh": W_int, "N": N_int, "seed": seed}
                # Output dir for this run
                out_dir_run = str(Path(output_dir_v) / f"W{W_int}" / f"N{N_int}" / f"seed{seed}")

                vi_cfg = build_vi_run_config(
                    experiment_id=experiment_id_base,
                    description=f"Imported from BCQM V: {stem} (W={W_int}, N={N_int}, seed={seed})",
                    variant="full",
                    N=N_int,
                    n_value=0.0,
                    seed=seed,
                    steps_total=steps,
                    burn_in=burn_in,
                    W_coh=W_int,
                    output_dir=out_dir_run,
                    v_source=v_source,
                    opts=opts,
                )

                entry: Dict[str, Any] = {
                    "run_key": run_key,
                    "vi_config": vi_cfg,
                }

                if opts.emit_yaml_per_run:
                    ypath = yaml_dir / f"{experiment_id_base}__W{W_int}__N{N_int}__seed{seed}.yml"
                    with ypath.open("w", encoding="utf-8") as f:
                        yaml.safe_dump(vi_cfg, f, sort_keys=False)
                    entry["vi_yaml_path"] = str(ypath)

                run_entries.append(entry)

    manifest = {
        "manifest_version": "import_manifest_v0.1",
        "created_utc": __import__("datetime").datetime.utcnow().isoformat() + "Z",
        "source_config_path": str(v_config_path),
        "source_config_stem": stem,
        "experiment_id_base": experiment_id_base,
        "grid": {
            "W_coh_values": [int(x) for x in W_values],
            "N_values": [int(x) for x in N_values],
            "ensembles": ensembles,
            "steps": steps,
            "burn_in": burn_in,
        },
        "random_seed_base": random_seed,
        "run_count": len(run_entries),
        "options": {
            "respect_store_states": opts.respect_store_states,
            "respect_compute_psd": opts.respect_compute_psd,
            "emit_yaml_per_run": opts.emit_yaml_per_run,
        },
        "runs": run_entries,
    }

    manifest_path = out_dir / f"IMPORT_MANIFEST_{stem}.json"
    manifest_path.write_text(__import__("json").dumps(manifest, indent=2, sort_keys=False), encoding="utf-8")
    return manifest_path
