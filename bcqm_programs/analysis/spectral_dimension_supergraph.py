#!/usr/bin/env python3
# PROVENANCE: BCQM_VII Stage-2 analysis; Test 2.1 spectral dimension on super-graph; v0.2 (adds exact mode); 2026-02-03

"""Estimate spectral dimension d_s(t) on the community super-graph.

We estimate return probability P0(t) for a random walk on the *undirected* community super-graph and compute:
    d_s(t) = -2 * d log P0(t) / d log t

This is analysis-only and uses existing RUN_METRICS_*.json files that provide:
  cloth.core_edges_used (and optionally cloth.halo_edges_used)

Pipeline per RUN_METRICS file:
  1) build undirected cloth graph from chosen edge_source (core or all)
  2) Louvain partition on that undirected graph (networkx.louvain_communities)
  3) build undirected community super-graph using chosen edge_source
  4) estimate P0(t) by either:
       (a) Monte Carlo random walks (mode=mc), or
       (b) exact transition-matrix trace (mode=exact): P0(t)=(1/K) tr(P^t) = (1/K) sum_i lambda_i^t
  5) compute d_s(t) via finite differences in log space

Outputs:
  - <out_dir>/<tag>_spectral_dim_curves.csv   (seedwise P0(t) and d_s(t))
  - <out_dir>/<tag>_spectral_dim_runs.csv     (plateau estimate per run)
  - <out_dir>/<tag>_spectral_dim_summary.csv  (per (N,n): aggregate)

Notes:
- Finite-size effects can dominate when K is small; plateau may be absent.
- Treat results as diagnostic; do not overclaim asymptotic dimension.
"""

import argparse, json, re, math
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import networkx as nx
from networkx.algorithms.community import louvain_communities

NAME_PAT = re.compile(r"__full__N(\d+)__n([0-9.]+)__seed(\d+)\.json$")

def parse_name(p: Path):
    m = NAME_PAT.search(p.name)
    if not m:
        return (-1, float("nan"), -1)
    return int(m.group(1)), float(m.group(2)), int(m.group(3))

def load_edges(data: dict, edge_source: str) -> List[Tuple[int,int]]:
    cloth = data.get("cloth", {})
    core = cloth.get("core_edges_used")
    if not isinstance(core, list):
        raise ValueError("Missing cloth.core_edges_used; ensure store_lists=true.")
    edges = [(int(u), int(v)) for u,v in core]
    if edge_source == "all":
        halo = cloth.get("halo_edges_used")
        if isinstance(halo, list):
            edges.extend([(int(u), int(v)) for u,v in halo])
    return edges

def build_partition(edges: List[Tuple[int,int]], resolution: float, seed: int = 0) -> Dict[int,int]:
    G = nx.Graph()
    for u,v in edges:
        if u!=v:
            G.add_edge(int(u), int(v))
    if G.number_of_nodes()==0:
        return {}
    comms = louvain_communities(G, seed=seed, resolution=float(resolution))
    part={}
    for cid, nodes in enumerate(comms):
        for node in nodes:
            part[int(node)] = int(cid)
    return part

def build_supergraph(edges: List[Tuple[int,int]], part: Dict[int,int]) -> nx.Graph:
    SG = nx.Graph()
    for u,v in edges:
        cu = part.get(int(u)); cv = part.get(int(v))
        if cu is None or cv is None:
            continue
        if cu!=cv:
            SG.add_edge(int(cu), int(cv))
    return SG

def gcc_subgraph(G: nx.Graph) -> nx.Graph:
    if G.number_of_nodes()==0:
        return nx.Graph()
    comp = max(nx.connected_components(G), key=len)
    return G.subgraph(comp).copy()

def random_walk_return_probs_mc(G: nx.Graph, t_max: int, n_walks: int, rng: np.random.Generator) -> np.ndarray:
    """Estimate P0(t): return probability at step t, averaged over random start nodes."""
    nodes = list(G.nodes())
    K = len(nodes)
    if K == 0:
        return np.full(t_max+1, np.nan)
    nbrs = {u: list(G.neighbors(u)) for u in nodes}
    P0 = np.zeros(t_max+1, dtype=float)
    P0[0] = 1.0
    for _ in range(n_walks):
        start = nodes[int(rng.integers(0, K))]
        cur = start
        for t in range(1, t_max+1):
            neigh = nbrs[cur]
            if not neigh:
                break
            cur = neigh[int(rng.integers(0, len(neigh)))]
            if cur == start:
                P0[t] += 1.0
    P0[1:] /= float(n_walks)
    return P0

def random_walk_return_probs_exact(G: nx.Graph, t_max: int) -> np.ndarray:
    """Exact P0(t) via transition matrix eigenvalues: P0(t)=(1/K) tr(P^t)=(1/K) sum_i lambda_i^t."""
    nodes = list(G.nodes())
    K = len(nodes)
    if K == 0:
        return np.full(t_max+1, np.nan)
    idx = {nodes[i]: i for i in range(K)}
    deg = dict(G.degree())
    P = np.zeros((K, K), dtype=float)
    for u in nodes:
        du = deg[u]
        if du == 0:
            continue
        i = idx[u]
        w = 1.0 / float(du)
        for v in G.neighbors(u):
            j = idx[v]
            P[i, j] = w
    # eigenvalues of P (may have tiny imaginary parts due to numerical error)
    evals = np.linalg.eigvals(P)
    evals = np.real_if_close(evals, tol=1e-9)
    evals = np.asarray(evals, dtype=complex)
    P0 = np.zeros(t_max+1, dtype=float)
    P0[0] = 1.0
    for t in range(1, t_max+1):
        val = np.sum(evals ** t) / float(K)
        P0[t] = float(np.real(val))
    # numerical cleanup
    P0 = np.maximum(P0, 0.0)
    return P0

def spectral_dim_from_P0(P0: np.ndarray) -> np.ndarray:
    """Compute d_s(t) via central finite differences in log space."""
    t = np.arange(len(P0), dtype=float)
    ds = np.full_like(P0, np.nan, dtype=float)
    eps = 1e-15
    logt = np.log(np.maximum(t, 1.0))
    logP = np.log(np.maximum(P0, eps))
    for i in range(2, len(P0)-1):
        num = logP[i+1] - logP[i-1]
        den = logt[i+1] - logt[i-1]
        if den != 0:
            ds[i] = float(-2.0 * num / den)
    return ds

def plateau_estimate(ds: np.ndarray, t_min: int, t_max: int) -> Tuple[float,float,int]:
    sl = ds[t_min:t_max+1]
    sl = sl[np.isfinite(sl)]
    if len(sl) < 3:
        return (float("nan"), float("nan"), int(len(sl)))
    return (float(np.mean(sl)), float(np.std(sl, ddof=1)), int(len(sl)))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True)
    ap.add_argument("--pattern", default="RUN_METRICS_*.json")
    ap.add_argument("--out_dir", default="csv")
    ap.add_argument("--tag", default=None)
    ap.add_argument("--edge_source", choices=["core","all"], default="core")
    ap.add_argument("--resolution", type=float, default=1.0)
    ap.add_argument("--mode", choices=["mc","exact"], default="mc",
                    help="mc: Monte-Carlo return probability; exact: transition-matrix trace via eigenvalues.")
    ap.add_argument("--t_max", type=int, default=200)
    ap.add_argument("--n_walks", type=int, default=20000)
    ap.add_argument("--plateau_tmin", type=int, default=10)
    ap.add_argument("--plateau_tmax", type=int, default=80)
    ap.add_argument("--rng_seed", type=int, default=0)
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = args.tag or run_dir.name

    files = sorted(run_dir.glob(args.pattern))
    if not files:
        raise SystemExit(f"No files matched {args.pattern} in {run_dir}")

    rng = np.random.default_rng(args.rng_seed)

    curve_rows=[]
    run_rows=[]
    for f in files:
        N, n, seed = parse_name(f)
        data = json.loads(f.read_text(encoding="utf-8"))
        edges = load_edges(data, args.edge_source)
        part = build_partition(edges, resolution=args.resolution, seed=0)
        SG = build_supergraph(edges, part)
        GCC = gcc_subgraph(SG)
        K = GCC.number_of_nodes()
        E = GCC.number_of_edges()

        if args.mode == "exact":
            P0 = random_walk_return_probs_exact(GCC, args.t_max)
        else:
            P0 = random_walk_return_probs_mc(GCC, args.t_max, args.n_walks, rng)

        ds = spectral_dim_from_P0(P0)
        ds_mean, ds_std, ds_count = plateau_estimate(ds, args.plateau_tmin, min(args.plateau_tmax, args.t_max))

        run_rows.append({
            "N": N, "n": n, "seed": seed,
            "edge_source": args.edge_source,
            "resolution": args.resolution,
            "mode": args.mode,
            "K_super": K, "E_super": E,
            "ds_plateau_mean": ds_mean, "ds_plateau_std": ds_std, "ds_plateau_count": ds_count,
            "plateau_window": f"{args.plateau_tmin}-{args.plateau_tmax}",
        })

        for t in range(0, args.t_max+1):
            curve_rows.append({
                "N": N, "n": n, "seed": seed,
                "edge_source": args.edge_source,
                "resolution": args.resolution,
                "mode": args.mode,
                "t": t,
                "P0": float(P0[t]) if np.isfinite(P0[t]) else np.nan,
                "d_s": float(ds[t]) if np.isfinite(ds[t]) else np.nan,
            })

    df_runs = pd.DataFrame(run_rows).sort_values(["N","n","seed"])
    df_curves = pd.DataFrame(curve_rows)

    out_curves = out_dir / f"{tag}_spectral_dim_curves.csv"
    out_runs_csv = out_dir / f"{tag}_spectral_dim_runs.csv"
    df_curves.to_csv(out_curves, index=False)
    df_runs.to_csv(out_runs_csv, index=False)

    if not df_runs.empty:
        summary = df_runs.groupby(["N","n","edge_source","mode"]).agg(
            runs=("seed","count"),
            K_mean=("K_super","mean"),
            ds_mean=("ds_plateau_mean","mean"),
            ds_std=("ds_plateau_mean","std"),
            ds_within_mean=("ds_plateau_std","mean"),
        ).reset_index()
    else:
        summary = pd.DataFrame()
    out_sum = out_dir / f"{tag}_spectral_dim_summary.csv"
    summary.to_csv(out_sum, index=False)

    print("Wrote:")
    print(" -", out_curves)
    print(" -", out_runs_csv)
    print(" -", out_sum)

if __name__ == "__main__":
    main()
