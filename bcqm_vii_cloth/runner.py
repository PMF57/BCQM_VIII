from __future__ import annotations

import math
import os
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from .config_schema import validate, resolve_seeds, resolve_n_values
from .glue import resolve_glue_params
from .graph_store import GraphStore
from .io import ensure_dir, write_json
from .observables import (
    induced_active_set,
    s_perc,
    weighted_junction_stat,
    hubshare as hubshare_fn,
    clustering_coefficient_undirected,
    compute_Q_clock,
)
from .selection import choose_targets
from .snapshots import write_edges_csv, write_nodes_json
from .engine_vglue import run_single_v_glue


def _run_id(experiment_id: str, variant: str, N: int, n: float, seed: int) -> str:
    return f"{experiment_id}__{variant}__N{N}__n{n:.3f}__seed{seed}"


def _resolve_snapshot_epochs(cfg: Dict[str, Any]) -> List[int]:
    snaps = cfg["snapshots"]
    if not snaps.get("enabled", False):
        return []
    total = int(cfg["steps_total"])

    if "epochs" in snaps and isinstance(snaps["epochs"], list):
        eps = [int(e) for e in snaps["epochs"]]
        return [e for e in eps if e < total]

    cad = snaps.get("cadence")
    if isinstance(cad, dict) and "every" in cad:
        every = int(cad["every"])
        start = int(cad.get("start", 0))
        eps = list(range(start, total + 1, every))
        return [e for e in eps if e < total]

    return []


def _build_candidates(cfg: Dict[str, Any], g: GraphStore, frontier: List[int], epoch: int) -> List[List[int]]:
    active = induced_active_set(cfg, g, frontier, epoch)
    cap = int(cfg.get("max_candidates_per_thread", 2000))
    if len(active) > cap:
        active = active[:cap]
    return [active for _ in frontier]


def _choose_preferred_existing(
    rng: np.random.Generator,
    cfg: Dict[str, Any],
    g: GraphStore,
    active: List[int],
    frontier: List[int],
    epoch: int,
) -> int | None:
    if not active:
        return None
    frontier_set = set(frontier)
    pool = [v for v in active if v not in frontier_set] or active

    W_coh = int(cfg.get("W_coh", 256))
    tau = max(1, W_coh // 4)
    alpha = 0.75

    def score(vid: int) -> float:
        nd = g.nodes[vid]
        age = max(0, epoch - int(nd.created_at))
        return math.exp(-age / float(tau)) / ((1.0 + float(nd.indeg)) ** float(alpha))

    k = min(10, len(pool))
    top = sorted(pool, key=score, reverse=True)[:k]
    return int(rng.choice(top))


def run_single(cfg: Dict[str, Any], N: int, n: float, seed: int) -> None:
    validate(cfg)

    # Engine dispatch
    engine = cfg.get("engine", {}) or {}
    mode = engine.get("mode", "scaffold")
    if mode == "v_glue":
        run_single_v_glue(cfg, N, n, seed)
        return

    # Default: existing scaffold engine
    variant = cfg["variant"]
    experiment_id = cfg["experiment_id"]
    out_dir = Path(cfg["output"]["out_dir"])
    ensure_dir(out_dir)

    run_id = _run_id(experiment_id, variant, N, n, seed)

    glue_params = resolve_glue_params(cfg["glue"], n)
    allow_junctions = bool(cfg["crosslinks"]["allow_junctions"])

    steps_total = int(cfg["steps_total"])
    burn_in = int(cfg["burn_in_epochs"])
    measure = int(cfg["measure_epochs"])
    assert burn_in + measure == steps_total

    rng = np.random.default_rng(int(seed))

    g = GraphStore()
    frontier = [g.new_node(created_at=0) for _ in range(N)]

    snapshot_epochs = _resolve_snapshot_epochs(cfg)

    out_cfg = cfg.get("output", {})
    ts_enabled = bool(out_cfg.get("write_timeseries", False))
    ts_bins = int(out_cfg.get("timeseries_bins", 100)) if ts_enabled else 0
    bin_ends: List[int] = []
    if ts_enabled:
        start = burn_in + 1
        end = steps_total
        if ts_bins <= 1:
            bin_ends = [end]
        else:
            step = max(1, (end - start) // ts_bins)
            e = start + step
            while e < end:
                bin_ends.append(e)
                e += step
            bin_ends.append(end)

    ts_epochs: List[int] = []
    L_series: List[float] = []
    Sperc_series: List[float] = []
    Sjuncw_series: List[float] = []
    hubshare_series: List[float] = []

    tick_epochs: List[int] = []
    lockstep_run = 0
    ell_lock = 0

    beta_junc = float(cfg.get("observables", {}).get("beta_junc", 1.5))
    compute_clust = bool(cfg.get("observables", {}).get("compute_clustering", True))

    t0 = time.time()

    for epoch in range(1, steps_total + 1):
        cand = _build_candidates(cfg, g, frontier, epoch)

        active = induced_active_set(cfg, g, frontier, epoch)
        preferred_existing = _choose_preferred_existing(rng, cfg, g, active, frontier, epoch)

        choices = choose_targets(rng, cand, allow_new=True, g=glue_params, preferred_existing=preferred_existing)

        for i, (kind, target) in enumerate(choices):
            u = frontier[i]

            if kind == "new":
                v = g.new_node(created_at=epoch)
                g.add_edge(u, v, w=1.0, epoch=epoch)
                frontier[i] = v
                continue

            v = int(target)  # type: ignore[arg-type]

            if v == u:
                v2 = g.new_node(created_at=epoch)
                g.add_edge(u, v2, w=1.0, epoch=epoch)
                frontier[i] = v2
                continue

            if (not allow_junctions) and (g.nodes[v].indeg >= 1):
                v2 = g.new_node(created_at=epoch)
                g.add_edge(u, v2, w=1.0, epoch=epoch)
                frontier[i] = v2
            else:
                g.add_edge(u, v, w=1.0, epoch=epoch)
                frontier[i] = v

        if epoch in snapshot_epochs:
            write_edges_csv(out_dir / f"SNAPSHOT_{run_id}_edges_epoch{epoch}.csv", g.edges)
            write_nodes_json(out_dir / f"SNAPSHOT_{run_id}_nodes_epoch{epoch}.json", g)

        if epoch > burn_in:
            counts: Dict[int, int] = {}
            for v in frontier:
                counts[v] = counts.get(v, 0) + 1
            max_count = max(counts.values()) if counts else 0
            sync_level = max_count / float(N) if N > 0 else 0.0

            if sync_level >= 0.8:
                lockstep_run += 1
            else:
                lockstep_run = 0
            ell_lock = max(ell_lock, lockstep_run)

            if sync_level >= 0.8:
                if len(tick_epochs) == 0 or (epoch - tick_epochs[-1] >= 5):
                    tick_epochs.append(epoch)

            if ts_enabled and epoch in bin_ends:
                active_now = induced_active_set(cfg, g, frontier, epoch)
                indegs = [g.nodes[v].indeg for v in active_now]
                sperc_val = s_perc(g, active_now)
                sjw_val = weighted_junction_stat(indegs, beta_junc)
                hub = hubshare_fn(indegs)
                Q = compute_Q_clock(tick_epochs)
                L_val = float("inf") if Q == float("inf") else (Q / math.sqrt(N) if N > 0 else 0.0)

                ts_epochs.append(epoch)
                L_series.append(float(L_val) if L_val != float("inf") else 1e18)
                Sperc_series.append(float(sperc_val))
                Sjuncw_series.append(float(sjw_val))
                hubshare_series.append(float(hub))

    elapsed = time.time() - t0

    active_final = induced_active_set(cfg, g, frontier, steps_total)
    indegs_final = [g.nodes[v].indeg for v in active_final]
    sperc_final = s_perc(g, active_final)
    sjw_final = weighted_junction_stat(indegs_final, beta_junc)
    hub_final = hubshare_fn(indegs_final)
    max_indeg = int(max(indegs_final)) if indegs_final else 0

    clust_val = None
    if compute_clust and active_final:
        aset = set(active_final)
        adj: Dict[int, set[int]] = {vv: set() for vv in aset}
        for uu, vv, w, t in g.edges:
            if uu in aset and vv in aset:
                adj[uu].add(vv)
                adj[vv].add(uu)
        clust_val = clustering_coefficient_undirected(adj, sample=500)

    Q = compute_Q_clock(tick_epochs)
    L_val = float("inf") if Q == float("inf") else (Q / (math.sqrt(N) if N > 0 else 1.0))

    thresh = cfg.get("anomaly_thresholds", {})
    hub_star = float(thresh.get("hubshare_star", 0.90))
    max_indeg_factor = float(thresh.get("max_indegree_factor", 3.0))
    star_collapse = bool(hub_final >= hub_star)
    runaway_hubbing = bool(max_indeg > int(max_indeg_factor * N))

    metrics_obj: Dict[str, Any] = {
        "run_id": run_id,
        "variant": variant,
        "engine_mode": "scaffold",
        "N": int(N),
        "n": float(n),
        "seed": int(seed),
        "steps_total": steps_total,
        "burn_in_epochs": burn_in,
        "measure_epochs": measure,
        "Q_clock": float(Q) if Q != float("inf") else "inf",
        "L": float(L_val) if L_val != float("inf") else "inf",
        "ell_lock": int(ell_lock),
        "S_perc": float(sperc_final),
        "S_junc_w": float(sjw_final),
        "beta_junc": float(beta_junc),
        "max_indegree": int(max_indeg),
        "hubshare": float(hub_final),
        "clustering": None if clust_val is None else float(clust_val),
        "anomaly_flags": {
            "star_collapse": star_collapse,
            "runaway_hubbing": runaway_hubbing,
            "tierB_evaluated": bool(ts_enabled),
        },
        "elapsed_seconds": float(elapsed),
    }

    if ts_enabled:
        metrics_obj["timeseries"] = {
            "ts_epochs": ts_epochs,
            "L_series": L_series,
            "Sperc_series": Sperc_series,
            "Sjuncw_series": Sjuncw_series,
            "hubshare_series": hubshare_series,
            "bins": int(ts_bins),
        }

    cfg_obj: Dict[str, Any] = {
        "run_id": run_id,
        "schema_version": cfg.get("schema_version"),
        "experiment_id": experiment_id,
        "description": cfg.get("description"),
        "variant": variant,
        "engine_mode": "scaffold",
        "N": int(N),
        "n": float(n),
        "seed": int(seed),
        "resolved": cfg,
        "resolved_glue_params": {
            "shared_bias": glue_params.shared_bias,
            "phase_lock": glue_params.phase_lock,
            "domains": glue_params.domains,
            "cadence_disorder": glue_params.cadence_disorder,
        },
        "platform": {"python": os.sys.version.split()[0]},
    }

    write_json(out_dir / f"RUN_CONFIG_{run_id}.json", cfg_obj)
    write_json(out_dir / f"RUN_METRICS_{run_id}.json", metrics_obj)


def run_from_config(cfg: Dict[str, Any]) -> None:
    validate(cfg)
    seeds = resolve_seeds(cfg["seeds"])
    n_vals = resolve_n_values(cfg["scan"])
    for N in cfg["sizes"]:
        for n in n_vals:
            for seed in seeds:
                run_single(cfg, int(N), float(n), int(seed))


def scan_from_config(cfg: Dict[str, Any]) -> None:
    run_from_config(cfg)
