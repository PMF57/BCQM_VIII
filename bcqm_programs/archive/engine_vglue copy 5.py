from __future__ import annotations

"""
engine_vglue.py (BCQM VI)

V-glue engine runner using direct-ancestor BCQM V modules (state/kernels/metrics).

Path A extension (v0.1.1):
- Optional event-graph bookkeeping + cross-links (co-selection reuse) on top of v_glue.
- Spatial observables computed post-hoc from the realised event graph (S_perc, S_junc_w, etc.).
- Island/bundle observables computed from per-thread event histories (optional).

Default behaviour is unchanged (space layer disabled unless cfg.space.enabled == true).

Assumes package layout with sibling modules imported via relative imports.
"""

import math
import os
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Optional, List, Set

import numpy as np

from .io import ensure_dir, write_json
from .state import ThreadState, BundleState
from .glue_dynamics import (
    hop_coherence_step,
    shared_bias_step,
    phase_lock_step,
    domain_glue_step,
    initialise_cadence,
    cadence_step,
)
from .metrics import compute_lockstep_metrics
from .event_graph import EventGraph
# PROVENANCE: BCQM_VII stage2 cloth (bin-based); store_lists v0.3.1; 2026-01-30


_DEFAULT_HOP = {"form": "power_law", "alpha": 1.0, "k_prefactor": 2.0, "memory_depth": 1}
_DEFAULT_SHARED = {"enabled": False, "lambda_bias": 0.0}
_DEFAULT_PHASE = {"enabled": False, "lambda_phase": 0.0, "theta_join": 0.3, "theta_break": 1.5, "omega_0": 0.1, "noise_sigma": 0.0}
_DEFAULT_DOMAINS = {"enabled": False, "n_initial_domains": 1, "lambda_domain": 0.0, "merge_threshold": 0.8, "split_threshold": 0.3, "min_domain_size": 1}
_DEFAULT_CADENCE = {"enabled": False, "distribution": "lognormal", "mean_T": 1.0, "sigma_T": 0.0, "lambda_cadence": 0.0}

_FALLBACK_C5 = {
    "hop_coherence": {"form": "power_law", "alpha": 1.0, "k_prefactor": 2.0, "memory_depth": 1},
    "shared_bias": {"enabled": False, "lambda_bias": 0.0},
    "phase_lock": {"enabled": True, "lambda_phase": 0.25, "omega_0": 0.1},
    "domains": {"enabled": False, "n_initial_domains": 1, "lambda_domain": 0.0},
    "cadence": {"enabled": True, "distribution": "lognormal", "mean_T": 1.0, "sigma_T": 0.2, "lambda_cadence": 0.15},
}


def _run_id(experiment_id: str, variant: str, N: int, n: float, seed: int) -> str:
    return f"{experiment_id}__{variant}__N{N}__n{n:.3f}__seed{seed}"


def _ns(block: Optional[Dict[str, Any]], defaults: Dict[str, Any]) -> SimpleNamespace:
    d = dict(defaults)
    if block:
        d.update(block)
    return SimpleNamespace(**d)


def _compute_q_base(cfg_hop: SimpleNamespace, W_coh: float) -> float:
    form = getattr(cfg_hop, "form", "power_law")
    alpha = float(getattr(cfg_hop, "alpha", 1.0))
    k_pref = float(getattr(cfg_hop, "k_prefactor", 2.0))
    if form == "power_law":
        return float(min(0.5, k_pref / (float(W_coh) ** alpha)))
    if form == "exp":
        return float(min(0.5, k_pref * math.exp(-float(W_coh))))
    return float(min(0.5, k_pref / max(float(W_coh), 1.0)))


def _load_v_blocks(cfg: Dict[str, Any], n: float) -> Dict[str, SimpleNamespace]:
    prov = cfg.get("provenance", {})
    vcfg = prov.get("v_config", None)
    scaled = False

    if isinstance(vcfg, dict):
        blocks = vcfg
    else:
        blocks = _FALLBACK_C5
        scaled = True

    hop = _ns(blocks.get("hop_coherence"), _DEFAULT_HOP)
    shared = _ns(blocks.get("shared_bias"), _DEFAULT_SHARED)
    phase = _ns(blocks.get("phase_lock"), _DEFAULT_PHASE)
    domains = _ns(blocks.get("domains"), _DEFAULT_DOMAINS)
    cadence = _ns(blocks.get("cadence"), _DEFAULT_CADENCE)

    if not hasattr(phase, "theta_join"):
        phase.theta_join = _DEFAULT_PHASE["theta_join"]
    if not hasattr(phase, "theta_break"):
        phase.theta_break = _DEFAULT_PHASE["theta_break"]
    if not hasattr(phase, "noise_sigma"):
        phase.noise_sigma = _DEFAULT_PHASE["noise_sigma"]

    if scaled:
        if getattr(phase, "enabled", False):
            phase.lambda_phase = float(n) * float(getattr(phase, "lambda_phase", 0.0))
        if getattr(cadence, "enabled", False):
            cadence.lambda_cadence = float(n) * float(getattr(cadence, "lambda_cadence", 0.0))


    # Ablation hook: optionally destroy glue coherence while keeping space enabled.

    # YAML:

    #   ablation:

    #     glue_decohere: true

    abl = cfg.get('ablation', {}) or {}

    if bool(abl.get('glue_decohere', False)):

        try:

            phase.enabled = False

        except Exception:

            pass

        try:

            phase.lambda_phase = 0.0

        except Exception:

            pass

        try:

            cadence.lambda_cadence = 0.0

        except Exception:

            pass

        try:

            shared.enabled = False

        except Exception:

            pass

        try:

            shared.lambda_bias = 0.0

        except Exception:

            pass

        try:

            domains.enabled = False

        except Exception:

            pass

        try:

            domains.lambda_domain = 0.0

        except Exception:

            pass

        # Increase hop-noise by overriding q_base unless overridden by YAML.

        try:

            hop.q_base_override = float(abl.get('q_base_override', 0.45))

        except Exception:

            hop.q_base_override = 0.45

    return {
        "hop": hop,
        "shared": shared,
        "phase": phase,
        "domains": domains,
        "cadence": cadence,
        "used_provenance": bool(isinstance(vcfg, dict)),
        "scaled_fallback_by_n": scaled,
    }


def _kuramoto_R(theta: np.ndarray) -> float:
    if theta.size == 0:
        return 0.0
    z = np.mean(np.exp(1j * theta))
    return float(np.abs(z))


def _domain_stats(domain: np.ndarray) -> Dict[str, Any]:
    if domain.size == 0:
        return {"counts": {}, "entropy": 0.0}
    vals, counts = np.unique(domain, return_counts=True)
    total = float(np.sum(counts))
    p = counts / total
    entropy = float(-np.sum(p * np.log(p + 1e-12)))
    return {"counts": {str(int(v)): int(c) for v, c in zip(vals, counts)}, "entropy": entropy}


def _cadence_stats(threads: ThreadState) -> Dict[str, Any]:
    for name in ("T", "L", "cadence", "tau"):
        if hasattr(threads, name):
            arr = np.asarray(getattr(threads, name), dtype=float)
            return {"field": name, "mean": float(arr.mean()), "var": float(arr.var())}
    if hasattr(threads, "active"):
        act = np.asarray(getattr(threads, "active")).astype(bool)
        return {"field": "active", "mean": float(act.mean()), "var": float(act.var())}
    return {"field": None, "mean": None, "var": None}


def _space_cfg(cfg: Dict[str, Any]) -> Dict[str, Any]:
    sp = cfg.get("space", {}) or {}
    return {
        "enabled": bool(sp.get("enabled", False)),
        "p_reuse": sp.get("p_reuse", None),  # fixed value if provided
        "p_reuse_mode": sp.get("p_reuse_mode", "fixed"),  # fixed|from_n|derived
        "domain_match": bool(sp.get("domain_match", True)),
        "allow_cocreate_merge": bool(sp.get("allow_cocreate_merge", False)),
        "beta_junc": float(sp.get("beta_junc", 1.5)),
        "w_star": float(sp.get("w_star", 0.3)),
        "log_island_timeseries": bool(sp.get("log_island_timeseries", False)),
    }


def _derive_p_reuse(space: Dict[str, Any], threads: ThreadState, n_value: float) -> float:
    mode = str(space.get("p_reuse_mode", "fixed")).lower()
    # Fixed: honour explicit p_reuse if provided
    if mode == "fixed" and space.get("p_reuse") is not None:
        return float(space["p_reuse"])
    # From_n: use the scan parameter n (clipped)
    if mode == "from_n":
        return float(max(0.0, min(0.95, float(n_value))))
    # Derived: reuse increases with phase coherence and cadence activity mean
    R = _kuramoto_R(np.asarray(threads.theta, dtype=float))
    if hasattr(threads, "active"):
        act = np.asarray(getattr(threads, "active")).astype(bool)
        act_mean = float(act.mean())
    else:
        act_mean = 1.0
    p = 0.05 + 0.90 * (R * act_mean)
    return float(max(0.0, min(0.95, p)))


def _histories_init(N: int, W: int, init_ids: List[int]) -> List[List[int]]:
    return [[init_ids[i]] for i in range(N)]


def _histories_push(hist: List[List[int]], W: int, events: List[int]) -> None:
    for i, e in enumerate(events):
        h = hist[i]
        h.append(int(e))
        if len(h) > W:
            del h[0]


def _overlap_w(hist_i: Set[int], hist_j: Set[int]) -> float:
    if not hist_i and not hist_j:
        return 0.0
    inter = len(hist_i.intersection(hist_j))
    uni = len(hist_i.union(hist_j))
    return inter / float(uni) if uni else 0.0


def _bundles_from_histories(hist: List[List[int]], w_star: float) -> Dict[str, Any]:
    N = len(hist)
    sets = [set(h) for h in hist]
    # build adjacency on threads
    adj = {i: set() for i in range(N)}
    for i in range(N):
        for j in range(i + 1, N):
            w = _overlap_w(sets[i], sets[j])
            if w > w_star:
                adj[i].add(j)
                adj[j].add(i)
    # connected components
    seen = set()
    sizes = []
    for i in range(N):
        if i in seen:
            continue
        stack = [i]
        seen.add(i)
        size = 0
        while stack:
            x = stack.pop()
            size += 1
            for nb in adj[x]:
                if nb not in seen:
                    seen.add(nb)
                    stack.append(nb)
        sizes.append(size)
    sizes.sort(reverse=True)
    fmax = (sizes[0] / float(N)) if sizes else 0.0
    # histogram
    histo = {}
    for sz in sizes:
        histo[str(sz)] = histo.get(str(sz), 0) + 1
    return {"F_max": fmax, "bundle_sizes": sizes, "bundle_hist": histo}


def _largest_bundle_members(hist: List[List[int]], w_star: float) -> List[int]:
    """Return indices of the largest overlap-bundle component at threshold w_star."""
    N = len(hist)
    if N == 0:
        return []
    sets = [set(h) for h in hist]
    adj = {i: set() for i in range(N)}
    for i in range(N):
        for j in range(i + 1, N):
            w = _overlap_w(sets[i], sets[j])
            if w > w_star:
                adj[i].add(j)
                adj[j].add(i)
    seen: Set[int] = set()
    best: List[int] = []
    for i in range(N):
        if i in seen:
            continue
        stack = [i]
        comp = []
        seen.add(i)
        while stack:
            x = stack.pop()
            comp.append(x)
            for nb in adj[x]:
                if nb not in seen:
                    seen.add(nb)
                    stack.append(nb)
        if len(comp) > len(best):
            best = comp
    best.sort()
    return best


def _cloth_cfg(cfg: Dict[str, Any], steps_total: int, burn_in: int) -> Dict[str, Any]:
    """Resolve cloth logging/extraction config (bin-based, Stage-2)."""
    c = cfg.get("cloth", {}) or {}
    enabled = bool(c.get("enabled", False))
    out = cfg.get("output", {}) or {}
    bins = int(c.get("bins", out.get("timeseries_bins", 80)))
    bins = max(10, min(2000, bins))
    T_eff = max(1, steps_total - burn_in)
    interval = max(1, T_eff // bins)
    w_lock = float(c.get("w_lock", 0.10))
    min_concurrency = int(c.get("min_concurrency", 2))
    min_bin_hits = int(c.get("min_bin_hits", 3))
    include_ledger = bool(c.get("include_ledger", True))
    return {
        "enabled": enabled,
        "bins": bins,
        "interval": interval,
        "w_lock": w_lock,
        "min_concurrency": min_concurrency,
        "min_bin_hits": min_bin_hits,
        "include_ledger": include_ledger,
    }


def _ball_growth_from_edges(nodes: Set[int], edges: Set[tuple], r_max: int = 30, samples: int = 40, seed: int = 0) -> Dict[str, Any]:
    """Ball growth on an undirected graph defined by (nodes, edges)."""
    import random
    rng = random.Random(int(seed))
    if not nodes:
        return {"comp_size": 0, "r_max": int(r_max), "samples": 0, "mean_ball": []}
    adj = {u: set() for u in nodes}
    for (u, v) in edges:
        if u in adj and v in adj:
            adj[u].add(v)
            adj[v].add(u)
    # largest component
    best = set()
    seen = set()
    for u in nodes:
        if u in seen:
            continue
        stack = [u]
        comp = set([u])
        seen.add(u)
        while stack:
            x = stack.pop()
            for nb in adj.get(x, ()):
                if nb not in seen:
                    seen.add(nb)
                    comp.add(nb)
                    stack.append(nb)
        if len(comp) > len(best):
            best = comp
    comp_size = len(best)
    if comp_size == 0:
        return {"comp_size": 0, "r_max": int(r_max), "samples": 0, "mean_ball": []}
    nodes_list = list(best)
    k = min(int(samples), len(nodes_list))
    roots = [nodes_list[rng.randrange(len(nodes_list))] for _ in range(k)]
    mean_ball = [0.0] * (int(r_max) + 1)
    for root in roots:
        dist = {root: 0}
        frontier = [root]
        while frontier:
            x = frontier.pop()
            dx = dist[x]
            if dx >= r_max:
                continue
            for nb in adj.get(x, ()):
                if nb not in dist:
                    dist[nb] = dx + 1
                    if dist[nb] <= r_max:
                        frontier.append(nb)
        counts = [0] * (int(r_max) + 1)
        for d in dist.values():
            if d <= r_max:
                counts[d] += 1
        cum = 0
        for r in range(r_max + 1):
            cum += counts[r]
            mean_ball[r] += cum
    mean_ball = [x / float(k) for x in mean_ball]
    return {"comp_size": int(comp_size), "r_max": int(r_max), "samples": int(k), "mean_ball": mean_ball}


def _ts_config(cfg: Dict[str, Any], steps_total: int, burn_in: int) -> Dict[str, Any]:
    """
    Time-series logging config.
    Controlled by cfg["output"]["write_timeseries"] and cfg["output"]["timeseries_bins"].
    Bins are placed across the measurement window [burn_in, steps_total).
    """
    out = cfg.get("output", {}) or {}
    enabled = bool(out.get("write_timeseries", False))
    bins = int(out.get("timeseries_bins", 50))
    bins = max(10, min(500, bins))
    T_eff = max(1, steps_total - burn_in)
    interval = max(1, T_eff // bins)
    return {"enabled": enabled, "bins": bins, "interval": interval}


def run_single_v_glue(cfg: Dict[str, Any], N: int, n: float, seed: int) -> None:
    variant = cfg["variant"]
    experiment_id = cfg["experiment_id"]
    out_dir = Path(cfg["output"]["out_dir"])
    ensure_dir(out_dir)

    run_id = _run_id(experiment_id, variant, N, n, seed)

    steps_total = int(cfg["steps_total"])
    burn_in = int(cfg["burn_in_epochs"])
    measure = int(cfg["measure_epochs"])
    if burn_in + measure != steps_total:
        raise ValueError("burn_in_epochs + measure_epochs must equal steps_total")
    T_eff = steps_total - burn_in
    if T_eff <= 0:
        raise ValueError("burn_in must be < steps_total")

    W_coh = int(cfg.get("W_coh", cfg.get("active_window", {}).get("hops", 256)))

    rng = np.random.default_rng(int(seed))

    blocks = _load_v_blocks(cfg, n)
    cfg_hop = blocks["hop"]
    cfg_shared = blocks["shared"]
    cfg_phase = blocks["phase"]
    cfg_domains = blocks["domains"]
    cfg_cadence = blocks["cadence"]

    cfg_hop.q_base = _compute_q_base(cfg_hop, W_coh)
    if hasattr(cfg_hop, 'q_base_override'):
        cfg_hop.q_base = float(getattr(cfg_hop, 'q_base_override'))

    n_domains = int(getattr(cfg_domains, "n_initial_domains", 1))
    if n_domains < 1:
        n_domains = 1

    v0 = rng.choice([-1.0, 1.0], size=N)
    theta0 = rng.uniform(0.0, 2.0 * np.pi, size=N)
    domain0 = rng.integers(0, n_domains, size=N, endpoint=False)

    threads = ThreadState(v=v0, theta=theta0, domain=domain0, history_v=None)
    initialise_cadence(rng, threads, cfg_cadence)

    bundle = BundleState(X=0.0, m=float(np.mean(threads.v)),
                         theta_mean=float(np.angle(np.mean(np.exp(1j * threads.theta)))))

    m_all = np.zeros((1, T_eff), dtype=float)
    dX_all = np.zeros((1, T_eff), dtype=float)

    # Optional space layer
    space = _space_cfg(cfg)
    g = EventGraph() if space["enabled"] else None
    frontiers: List[int] = []
    histories: List[List[int]] = []
    
    # Time-series logging (binned) across measurement window
    ts_cfg = _ts_config(cfg, steps_total, burn_in)
    ts = {"enabled": bool(ts_cfg["enabled"])}
    if ts_cfg["enabled"]:
        ts.update({
            "interval": int(ts_cfg["interval"]),
            "records": [],  # list of per-sample dicts
        })

    # Cloth logging / extraction (Stage-2): concurrency per bin + lockstep-supported core.
    cloth_cfg = _cloth_cfg(cfg, steps_total, burn_in)
    cloth = {"enabled": bool(cloth_cfg["enabled"])}
    cloth_ledger: List[Dict[str, Any]] = []
    # Stage-2 cloth: per-bin presence hits (no concurrency filter) for edges/events used by all threads and by core members.
    # These are derived counters (not stored on primitives) and support persistence filters at end-of-run.
    edge_bin_hits_all_used: Dict[tuple, int] = {}
    edge_bin_hits_core_used: Dict[tuple, int] = {}
    event_bin_hits_all_used: Dict[int, int] = {}
    event_bin_hits_core_used: Dict[int, int] = {}

    if cloth_cfg["enabled"]:
        cloth.update({
            "interval": int(cloth_cfg["interval"]),
            "w_lock": float(cloth_cfg["w_lock"]),
            "min_concurrency": int(cloth_cfg["min_concurrency"]),
            "min_bin_hits": int(cloth_cfg["min_bin_hits"]),
        })
        # per-bin per-thread ledgers (reset each bin)
        bin_thread_edges = [dict() for _ in range(N)]   # (u,v) -> count
        bin_thread_events = [dict() for _ in range(N)]  # e -> count
        bin_index = 0
    if space["enabled"]:
        # one initial event per thread
        frontiers = [g.new_event(0, domain=int(threads.domain[i])) for i in range(N)]  # type: ignore
        histories = _histories_init(N, W_coh, frontiers)

    t_eff = 0
    t0_wall = time.time()

    # Optional island time-series (binned) — minimal: record at end-of-run unless enabled
    island_ts = {"t": [], "F_max": [], "N_bund": []}

    for t in range(steps_total):
        cadence_step(rng, threads, cfg_cadence)
        hop_coherence_step(rng, threads, cfg_hop)

        bundle.m = float(np.mean(threads.v))
        bundle.theta_mean = float(np.angle(np.mean(np.exp(1j * threads.theta))))

        shared_bias_step(rng, threads, bundle, cfg_shared)
        phase_lock_step(rng, threads, bundle, cfg_phase)
        domain_glue_step(rng, threads, cfg_domains)

        bundle.m = float(np.mean(threads.v))
        bundle.theta_mean = float(np.angle(np.mean(np.exp(1j * threads.theta))))
        dX = float(np.mean(threads.v))
        bundle.X += dX

        # Space layer step: select next events and add edges
        if space["enabled"]:
            assert g is not None
            p_reuse = _derive_p_reuse(space, threads, n)
            # Build V_active from recency
            active = g.v_active(t, W_coh, frontiers)
            active_list = list(active)

            next_events: List[int] = []
            for i in range(N):
                use_reuse = (rng.random() < p_reuse) and len(active_list) > 0
                if use_reuse:
                    # domain match filter if requested
                    if space["domain_match"]:
                        dom_i = int(threads.domain[i])
                        cand = [e for e in active_list if g.nodes[e].domain == dom_i]  # type: ignore
                        if cand:
                            e_next = int(rng.choice(cand))
                        else:
                            e_next = g.new_event(t + 1, domain=int(dom_i))
                    else:
                        e_next = int(rng.choice(active_list))
                else:
                    e_next = g.new_event(t + 1, domain=int(threads.domain[i]))

                # Self-loop guard: if chosen equals current frontier, force new
                if e_next == frontiers[i]:
                    e_next = g.new_event(t + 1, domain=int(threads.domain[i]))

                next_events.append(e_next)

            # Optional co-create merge (default off): if enabled, merge all NEW choices into one node per domain
            if space["allow_cocreate_merge"]:
                # Simple conservative merge: for each domain, if multiple threads created NEW at this tick, merge them.
                dom_to_new = {}
                for i in range(N):
                    # identify if it was created at t+1 (heuristic)
                    if g.nodes[next_events[i]].created_at == (t + 1):  # type: ignore
                        dom = g.nodes[next_events[i]].domain  # type: ignore
                        if dom not in dom_to_new:
                            dom_to_new[dom] = next_events[i]
                        else:
                            next_events[i] = dom_to_new[dom]


            # Cloth bin ledger: record selected events and traversed edges per thread (before updating frontiers).
            if cloth_cfg["enabled"] and (t >= burn_in):
                for i in range(N):
                    e = next_events[i]
                    u = frontiers[i]
                    v = e
                    bin_thread_events[i][e] = bin_thread_events[i].get(e, 0) + 1
                    key = (int(u), int(v))
                    bin_thread_edges[i][key] = bin_thread_edges[i].get(key, 0) + 1
            # Commit edges and update frontiers/histories
            for i in range(N):
                g.add_edge(frontiers[i], next_events[i], t + 1)
            frontiers = next_events
            _histories_push(histories, W_coh, frontiers)

            # Cloth bin finalisation: at bin boundaries, compute concurrency (all vs core bundle) and store compact ledger.
            if cloth_cfg["enabled"] and (t >= burn_in) and ((t - burn_in) % cloth_cfg["interval"] == 0):
                core_members = set(_largest_bundle_members(histories, float(cloth_cfg["w_lock"])))
                ev_all: Dict[int, int] = {}
                ev_core: Dict[int, int] = {}
                ed_all: Dict[tuple, int] = {}
                ed_core: Dict[tuple, int] = {}
                for i in range(N):
                    for e, c in bin_thread_events[i].items():
                        ev_all[int(e)] = ev_all.get(int(e), 0) + int(c)
                        if i in core_members:
                            ev_core[int(e)] = ev_core.get(int(e), 0) + int(c)
                    for (u,v), c in bin_thread_edges[i].items():
                        ed_all[(int(u), int(v))] = ed_all.get((int(u), int(v)), 0) + int(c)
                        if i in core_members:
                            ed_core[(int(u), int(v))] = ed_core.get((int(u), int(v)), 0) + int(c)
                minc = int(cloth_cfg["min_concurrency"])
                events_all = [[int(e), int(c)] for e,c in ev_all.items() if int(c) >= minc]
                events_core = [[int(e), int(c)] for e,c in ev_core.items() if int(c) >= minc]
                edges_all = [[int(u), int(v), int(c)] for (u,v),c in ed_all.items() if int(c) >= minc]
                edges_core = [[int(u), int(v), int(c)] for (u,v),c in ed_core.items() if int(c) >= minc]

                # Update per-bin presence hits (used edges/events) without concurrency threshold.
                # Presence is counted once per bin if an item was used at least once in that bin.
                for e, c in ev_all.items():
                    event_bin_hits_all_used[int(e)] = event_bin_hits_all_used.get(int(e), 0) + 1
                for e, c in ev_core.items():
                    event_bin_hits_core_used[int(e)] = event_bin_hits_core_used.get(int(e), 0) + 1
                for (u, v), c in ed_all.items():
                    edge_bin_hits_all_used[(int(u), int(v))] = edge_bin_hits_all_used.get((int(u), int(v)), 0) + 1
                for (u, v), c in ed_core.items():
                    edge_bin_hits_core_used[(int(u), int(v))] = edge_bin_hits_core_used.get((int(u), int(v)), 0) + 1
                if bool(cloth_cfg.get("include_ledger", True)):
                    cloth_ledger.append({
                        "bin": int(bin_index),
                        "t_end": int(t),
                        "core_size": int(len(core_members)),
                        "events_all": events_all,
                        "events_core": events_core,
                        "edges_all": edges_all,
                        "edges_core": edges_core,
                    })
                for i in range(N):
                    bin_thread_events[i].clear()
                    bin_thread_edges[i].clear()
                bin_index += 1

            # Optional binned time series record
            if ts_cfg["enabled"] and (t >= burn_in) and ((t - burn_in) % ts_cfg["interval"] == 0):
                active_ts = g.v_active(t, W_coh, frontiers)
                s_perc_t = g.s_perc(active_ts)
                s_junc_t = g.s_junc_w(active_ts, beta=space["beta_junc"])
                comp_ts = g.largest_component_nodes(active_ts)
                comp_size_t = len(comp_ts)
                wstars = [0.10, 0.20, 0.30]
                f_by_w = {}
                for w in wstars:
                    f_by_w[f"{w:.2f}"] = float(_bundles_from_histories(histories, w)["F_max"])
                ts["records"].append({
                    "t": int(t),
                    "V_active_size": int(len(active_ts)),
                    "comp_size": int(comp_size_t),
                    "S_perc": float(s_perc_t),
                    "S_junc_w": float(s_junc_t),
                    "F_max_by_wstar": f_by_w,
                })


            if space["log_island_timeseries"] and (t >= burn_in) and (t % max(1, W_coh // 4) == 0):
                b = _bundles_from_histories(histories, space["w_star"])
                island_ts["t"].append(int(t))
                island_ts["F_max"].append(float(b["F_max"]))
                island_ts["N_bund"].append(int(len(b["bundle_sizes"])))

        if t >= burn_in:
            m_all[0, t_eff] = bundle.m
            dX_all[0, t_eff] = dX
            t_eff += 1

    elapsed = time.time() - t0_wall
    assert t_eff == T_eff

    met = compute_lockstep_metrics(m_all, dX_all)
    Q_clock = float(met.get("Q_clock", 0.0))
    ell_lock = float(met.get("ell_lock", 0.0))
    L_inst = float(met.get("L_inst", 0.0))
    L = Q_clock / (math.sqrt(N) if N > 0 else 1.0)

    diag = {
        "cadence": _cadence_stats(threads),
        "phase": {"R": _kuramoto_R(np.asarray(threads.theta, dtype=float))},
        "domains": _domain_stats(np.asarray(threads.domain)),
    }

    # Spatial metrics (end-of-run)
    geometry_out = {"enabled": False}
    space_out = {"enabled": bool(space["enabled"])}
    if space["enabled"]:
        assert g is not None
        active_end = g.v_active(steps_total, W_coh, frontiers)
        S_perc = g.s_perc(active_end)
        S_junc_w = g.s_junc_w(active_end, beta=space["beta_junc"])
        hub = g.hubshare(active_end)
        max_indeg = g.max_indegree(active_end)
        clust = g.clustering_coeff(active_end)
        # Geometry probe: spectral dimension estimate on largest component (optional)
        geom_cfg = cfg.get('geometry', {}) or {}
        geom_enabled = bool(geom_cfg.get('enabled', False))
        geom_mode = str(geom_cfg.get('mode', 'full'))
        geometry_out = {'enabled': geom_enabled, 'mode': geom_mode}
        if geom_enabled:
            if geom_mode == 'ball_growth_only':
                # Ball-growth only: skip ds fitting; always attach ball-growth profile for structural geometry.
                try:
                    geometry_out['comp_size'] = int(len(g.largest_component_nodes(active_end)))
                except Exception:
                    geometry_out['comp_size'] = None
                try:
                    geometry_out['ball_growth'] = g.ball_growth_profile(
                        active_end,
                        r_max=int(geom_cfg.get('ball_r_max', 30)),
                        samples=int(geom_cfg.get('ball_samples', 40)),
                        seed=int(seed) + 54321,
                    )
                except Exception as e:
                    geometry_out['ball_growth'] = {'error': type(e).__name__}
            else:
                req = float(geom_cfg.get('require_sperc', 0.8))
                if float(S_perc) >= req:
                    geometry_out.update(g.estimate_spectral_dimension(
                        active_end,
                        t_max=int(geom_cfg.get('t_max', 60)),
                        n_walkers=int(geom_cfg.get('n_walkers', 300)),
                        fit_t_min=int(geom_cfg.get('fit_t_min', 5)),
                        fit_t_max=int(geom_cfg.get('fit_t_max', 30)),
                        seed=int(seed) + 12345,
                    ))
                    # Also attach ball-growth profile if available (diagnostic)
                    try:
                        geometry_out['ball_growth'] = g.ball_growth_profile(
                            active_end,
                            r_max=int(geom_cfg.get('ball_r_max', 30)),
                            samples=int(geom_cfg.get('ball_samples', 40)),
                            seed=int(seed) + 54321,
                        )
                    except Exception as e:
                        geometry_out['ball_growth'] = {'error': type(e).__name__}
                else:
                    geometry_out['reason'] = f'below_sperc_threshold({req})'
        else:
            geometry_out['reason'] = 'disabled'
                # Multi-threshold island diagnostics (Option A): report F_max at w_star in {0.10, 0.20, 0.30}
        wstars = [0.10, 0.20, 0.30]
        bundles_by_w = {f"{w:.2f}": _bundles_from_histories(histories, w) for w in wstars}
        # Primary (configured) threshold
        bundles = _bundles_from_histories(histories, space["w_star"])
        space_out.update({
            "V_active_size": int(len(active_end)),
            "S_perc": float(S_perc),
            "S_junc_w": float(S_junc_w),
            "hubshare": float(hub),
            "max_indegree": int(max_indeg),
            "clustering": float(clust),
            "p_reuse_policy": str(space.get("p_reuse_mode","fixed")),
            "p_reuse_value_last": float(_derive_p_reuse(space, threads, n)),
            "domain_match": bool(space["domain_match"]),
            "allow_cocreate_merge": bool(space["allow_cocreate_merge"]),
        })
        islands_out = {
            "w_star": float(space["w_star"]),
            "F_max_by_wstar": {k: float(v["F_max"]) for k, v in bundles_by_w.items()},
            "bundle_hist_by_wstar": {k: v["bundle_hist"] for k, v in bundles_by_w.items()},
            "F_max": float(bundles["F_max"]),
            "bundle_hist": bundles["bundle_hist"],
            "bundle_sizes": bundles["bundle_sizes"],
        }
        if space["log_island_timeseries"]:
            islands_out["timeseries"] = island_ts
    else:
        S_perc = 0.0
        S_junc_w = 0.0
        hub = 0.0
        max_indeg = 0
        clust = None
        islands_out = {"enabled": False}
    # Cloth summary (end-of-run): persistent hits across bins.
    # We define a Stage-2 "used" cloth core from edges/events used by core members (presence per bin, no concurrency filter),
    # and retain concurrent-only ledgers as a reinforcement diagnostic.
    if cloth_cfg["enabled"]:
        min_hits = int(cloth_cfg["min_bin_hits"])
        # Core/halo based on per-bin usage (presence) accumulated during the run.
        core_edges = {k for k,v in edge_bin_hits_core_used.items() if int(v) >= min_hits}
        core_events = {k for k,v in event_bin_hits_core_used.items() if int(v) >= min_hits}
        halo_edges = {k for k,v in edge_bin_hits_all_used.items() if int(v) >= min_hits and k not in core_edges}
        halo_events = {k for k,v in event_bin_hits_all_used.items() if int(v) >= min_hits and k not in core_events}

        # Ball growth on the core (edge-based) cloth.
        bg = _ball_growth_from_edges(core_events if core_events else set(), core_edges, r_max=30, samples=40, seed=int(seed)+999)

        # Diagnostic: persistent concurrent hits (min_concurrency-filtered) for comparison.
        edge_hits_conc_core: Dict[tuple, int] = {}
        event_hits_conc_core: Dict[int, int] = {}
        for rec in cloth_ledger:
            for u,v,c in rec.get("edges_core", []):
                edge_hits_conc_core[(int(u), int(v))] = edge_hits_conc_core.get((int(u), int(v)), 0) + 1
            for e,c in rec.get("events_core", []):
                event_hits_conc_core[int(e)] = event_hits_conc_core.get(int(e), 0) + 1
        core_edges_concurrent = {k for k,v in edge_hits_conc_core.items() if int(v) >= min_hits}
        core_events_concurrent = {k for k,v in event_hits_conc_core.items() if int(v) >= min_hits}

        cloth.update({
            "core_edges_count": int(len(core_edges)),
            "core_events_count": int(len(core_events)),
            "halo_edges_count": int(len(halo_edges)),
            "halo_events_count": int(len(halo_events)),
            # Explicit core/halo lists for survival analysis (Jaccard across seeds).
            # Toggle with cloth.store_lists (default: true).
            "store_lists": bool(cloth_cfg.get("store_lists", True)),
            "core_edges_used": sorted([[int(u), int(v)] for (u, v) in core_edges]) if bool(cloth_cfg.get("store_lists", True)) else None,
            "halo_edges_used": sorted([[int(u), int(v)] for (u, v) in halo_edges]) if bool(cloth_cfg.get("store_lists", True)) else None,
            "core_events_used": sorted([int(e) for e in core_events]) if bool(cloth_cfg.get("store_lists", True)) else None,
            "halo_events_used": sorted([int(e) for e in halo_events]) if bool(cloth_cfg.get("store_lists", True)) else None,
            "core_edges_concurrent_count": int(len(core_edges_concurrent)),
            "core_events_concurrent_count": int(len(core_events_concurrent)),
            "ball_growth": bg,
            "ledger_bins": int(len(cloth_ledger)) if bool(cloth_cfg.get("include_ledger", True)) else None,
        })
    else:
        cloth_ledger = []



    metrics_obj: Dict[str, Any] = {
        "run_id": run_id,
        "variant": variant,
        "engine_mode": "v_glue",
        "N": int(N),
        "n": float(n),
        "seed": int(seed),
        "steps_total": steps_total,
        "burn_in_epochs": burn_in,
        "measure_epochs": measure,
        "Q_clock": Q_clock,
        "L": float(L),
        "ell_lock": ell_lock,
        "L_inst": L_inst,
        "glue_state": diag,
        "space_state": space_out,
        "geometry": geometry_out,
        "islands": islands_out,
        "timeseries": ts,
        "cloth": cloth,
        "cloth_ledger": cloth_ledger,
        "S_perc": float(S_perc),
        "S_junc_w": float(S_junc_w),
        "hubshare": float(hub),
        "max_indegree": int(max_indeg),
        "clustering": None if clust is None else float(clust),
        "anomaly_flags": {
            "star_collapse": False,
            "runaway_hubbing": False,
            "tierB_evaluated": False,
        },
        "elapsed_seconds": float(elapsed),
    }

    cfg_obj: Dict[str, Any] = {
        "run_id": run_id,
        "schema_version": cfg.get("schema_version"),
        "experiment_id": experiment_id,
        "description": cfg.get("description"),
        "variant": variant,
        "engine_mode": "v_glue",
        "N": int(N),
        "n": float(n),
        "seed": int(seed),
        "resolved": cfg,
        "v_glue": {
            "used_provenance_v_config": bool(blocks["used_provenance"]),
            "scaled_fallback_by_n": bool(blocks["scaled_fallback_by_n"]),
            "W_coh": int(W_coh),
            "ablation": cfg.get("ablation", {}),
            "hop_coherence": vars(cfg_hop),
            "shared_bias": vars(cfg_shared),
            "phase_lock": vars(cfg_phase),
            "domains": vars(cfg_domains),
            "cadence": vars(cfg_cadence),
        },
        "platform": {"python": os.sys.version.split()[0]},
    }

    write_json(out_dir / f"RUN_CONFIG_{run_id}.json", cfg_obj)
    write_json(out_dir / f"RUN_METRICS_{run_id}.json", metrics_obj)
