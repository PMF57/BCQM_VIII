#!/usr/bin/env python3
# PROVENANCE: BCQM_VII Stage-2 analysis; effective dimensionality from ball growth (d_eff); v0.1; 2026-02-03

"""Compute effective dimensionality d_eff from ball-growth curves.

This is analysis-only. It reads existing RUN_METRICS_*.json files and estimates an
effective scaling exponent from ball growth:

    |B(r)| ~ r^{d_eff}

We select an intermediate scaling window automatically (avoid r too small and
avoid saturation near |B(r)| ~ |C|), then fit a line to:
    log |B(r)| vs log r.

Supports two objects:
  1) cloth: uses RUN_METRICS["cloth"]["ball_growth"] (mean_ball, comp_size)
  2) supergraph: builds Louvain communities from edges and computes ball growth on
     the community super-graph (undirected projection). Requires core/halo edge lists.

Outputs:
  - <out_dir>/<tag>_d_eff_runs.csv        (per RUN_METRICS file)
  - <out_dir>/<tag>_d_eff_summary.csv     (grouped by N,n: mean±std)

Example (cloth only):
  python3 bcqm_vii_cloth/analysis/d_eff_ball_growth.py \
    --run_dir outputs_cloth/gateA3_N32_hits1_x10_bins20_n0p8 \
    --object cloth \
    --out_dir csv/gateA3_N32 \
    --tag gateA3_N32_cloth

Example (supergraph, all/all):
  python3 bcqm_vii_cloth/analysis/d_eff_ball_growth.py \
    --run_dir outputs_cloth/ensemble_W100_N4N8_hits1_x10_bins20 \
    --object supergraph \
    --partition_source all --supergraph_source all \
    --out_dir csv/pivot_base \
    --tag pivot_supergraph

Notes:
- This produces an "effective" exponent for a finite object; it is not a proof of
  an asymptotic dimension. Use R^2 and the chosen window [r_lo, r_hi] to judge quality.
- For small super-graphs (K ~ 20-30), expect finite-size effects; scaling windows may be short.
"""
import argparse
import json
import math
import re
from itertools import combinations
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


def linfit(x: np.ndarray, y: np.ndarray):
    # returns slope, intercept, r2
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if len(x) < 2:
        return (float("nan"), float("nan"), float("nan"))
    A = np.vstack([x, np.ones_like(x)]).T
    slope, intercept = np.linalg.lstsq(A, y, rcond=None)[0]
    yhat = slope * x + intercept
    ss_res = float(np.sum((y - yhat) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return float(slope), float(intercept), float(r2)


def choose_window(r: np.ndarray, B: np.ndarray, C: float,
                  min_points: int = 4,
                  frac_lo: float = 0.05,
                  frac_hi: float = 0.80):
    """Choose best log-log window by max R^2, constrained by occupancy fraction."""
    # r starts at 1
    frac = B / C
    mask = (frac >= frac_lo) & (frac <= frac_hi) & np.isfinite(B) & (B > 1.0)
    r2_best = -1.0
    best = None

    idx = np.where(mask)[0]
    if len(idx) < min_points:
        return None  # no viable window

    # consider all contiguous windows inside mask range
    # build maximal contiguous segments
    segments = []
    start = idx[0]
    prev = idx[0]
    for k in idx[1:]:
        if k == prev + 1:
            prev = k
        else:
            segments.append((start, prev))
            start = prev = k
    segments.append((start, prev))

    for a, b in segments:
        for i in range(a, b + 1):
            for j in range(i + min_points - 1, b + 1):
                rr = r[i:j+1]
                BB = B[i:j+1]
                slope, intercept, r2 = linfit(np.log(rr), np.log(BB))
                if not np.isfinite(r2):
                    continue
                if r2 > r2_best:
                    r2_best = r2
                    best = (int(rr[0]), int(rr[-1]), float(slope), float(r2))
    return best  # (r_lo, r_hi, slope, r2)


def ball_growth_on_graph(G: nx.Graph, r_max: int = 20):
    """Mean fraction |B(r)|/|C| on largest connected component."""
    if G.number_of_nodes() == 0:
        return 0, [0.0] * (r_max + 1)
    comp_nodes = max(nx.connected_components(G), key=len)
    H = G.subgraph(comp_nodes).copy()
    C = H.number_of_nodes()
    nodes = list(H.nodes())
    mean_ball = []
    for r in range(r_max + 1):
        sizes = []
        for root in nodes:
            seen = {root}
            frontier = {root}
            for _ in range(r):
                nxt = set()
                for x in frontier:
                    for y in H.neighbors(x):
                        if y not in seen:
                            seen.add(y)
                            nxt.add(y)
                frontier = nxt
                if not frontier:
                    break
            sizes.append(len(seen) / C)
        mean_ball.append(float(np.mean(sizes)))
    return C, mean_ball


def build_partition_and_supergraph(edges_directed: List[Tuple[int, int]],
                                  resolution: float = 1.0,
                                  seed: int = 0):
    """Partition on undirected projection, then return supergraph undirected graph for ball growth."""
    UG = nx.Graph()
    for u, v in edges_directed:
        if u != v:
            UG.add_edge(int(u), int(v))
    if UG.number_of_nodes() == 0:
        return {}, nx.Graph()

    comms = louvain_communities(UG, seed=seed, resolution=resolution)
    part = {}
    for cid, nodes in enumerate(comms):
        for node in nodes:
            part[int(node)] = int(cid)

    # build community supergraph (undirected projection for geometry)
    SG = nx.Graph()
    for u, v in edges_directed:
        cu = part.get(int(u))
        cv = part.get(int(v))
        if cu is None or cv is None:
            continue
        if cu != cv:
            SG.add_edge(cu, cv)
    return part, SG


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True)
    ap.add_argument("--pattern", default="RUN_METRICS_*.json")
    ap.add_argument("--out_dir", default="csv")
    ap.add_argument("--tag", default=None)
    ap.add_argument("--object", choices=["cloth", "supergraph"], default="cloth")
    ap.add_argument("--partition_source", choices=["core", "all"], default="all")
    ap.add_argument("--supergraph_source", choices=["core", "all"], default="all")
    ap.add_argument("--resolution", type=float, default=1.0)
    ap.add_argument("--r_max", type=int, default=20)
    ap.add_argument("--min_points", type=int, default=4)
    ap.add_argument("--frac_lo", type=float, default=0.05)
    ap.add_argument("--frac_hi", type=float, default=0.80)
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = args.tag or run_dir.name

    files = sorted(run_dir.glob(args.pattern))
    if not files:
        raise SystemExit(f"No files matched {args.pattern} in {run_dir}")

    rows = []
    for f in files:
        N, n, seed = parse_name(f)
        data = json.loads(f.read_text(encoding="utf-8"))

        if args.object == "cloth":
            cloth = data.get("cloth", {})
            bg = (cloth.get("ball_growth") or {})
            C = float(bg.get("comp_size") or 0.0)
            mean_ball = bg.get("mean_ball")
            if not isinstance(mean_ball, list) or C <= 0:
                continue
            B_abs = np.asarray(mean_ball, dtype=float) * C
            r = np.arange(len(B_abs), dtype=int)
            # fit uses r>=1
            rr = r[1:]
            BB = B_abs[1:]
            win = choose_window(rr, BB, C, min_points=args.min_points, frac_lo=args.frac_lo, frac_hi=args.frac_hi)
            if win is None:
                r_lo=r_hi=d_eff=r2=float("nan")
            else:
                r_lo, r_hi, d_eff, r2 = win

            rows.append({
                "N": N, "n": n, "seed": seed,
                "object": "cloth",
                "comp_size": C,
                "r_lo": r_lo, "r_hi": r_hi,
                "d_eff": d_eff, "r2": r2,
            })

        else:  # supergraph
            cloth = data.get("cloth", {})
            core_edges = cloth.get("core_edges_used")
            halo_edges = cloth.get("halo_edges_used")
            if not isinstance(core_edges, list):
                raise SystemExit(f"Missing core_edges_used in {f.name}. Ensure store_lists=true.")
            edges_core = [(int(u), int(v)) for u, v in core_edges]
            edges_all = list(edges_core)
            if args.partition_source == "all" or args.supergraph_source == "all":
                if isinstance(halo_edges, list):
                    edges_all.extend([(int(u), int(v)) for u, v in halo_edges])

            edges_for_partition = edges_core if args.partition_source == "core" else edges_all
            edges_for_super = edges_core if args.supergraph_source == "core" else edges_all

            part, SG = build_partition_and_supergraph(edges_for_partition, resolution=args.resolution, seed=0)
            # build supergraph using edges_for_super but same partition
            SG2 = nx.Graph()
            for u, v in edges_for_super:
                cu = part.get(int(u)); cv = part.get(int(v))
                if cu is None or cv is None:
                    continue
                if cu != cv:
                    SG2.add_edge(cu, cv)

            C, mean_ball = ball_growth_on_graph(SG2, r_max=args.r_max)
            if C <= 0:
                continue
            B_abs = np.asarray(mean_ball, dtype=float) * C
            rr = np.arange(len(B_abs), dtype=int)[1:]
            BB = B_abs[1:]
            win = choose_window(rr, BB, float(C), min_points=args.min_points, frac_lo=args.frac_lo, frac_hi=args.frac_hi)
            if win is None:
                r_lo=r_hi=d_eff=r2=float("nan")
            else:
                r_lo, r_hi, d_eff, r2 = win

            rows.append({
                "N": N, "n": n, "seed": seed,
                "object": "supergraph",
                "comp_size": float(C),
                "r_lo": r_lo, "r_hi": r_hi,
                "d_eff": d_eff, "r2": r2,
                "partition_source": args.partition_source,
                "supergraph_source": args.supergraph_source,
                "resolution": args.resolution,
            })

    df = pd.DataFrame(rows).sort_values(["N","n","seed"])
    out_runs = out_dir / f"{tag}_d_eff_runs.csv"
    df.to_csv(out_runs, index=False)

    # summary by N,n
    if not df.empty:
        summary = df.groupby(["N","n","object"]).agg(
            runs=("seed","count"),
            d_eff_mean=("d_eff","mean"),
            d_eff_std=("d_eff","std"),
            r2_mean=("r2","mean"),
            comp_mean=("comp_size","mean"),
        ).reset_index()
    else:
        summary = pd.DataFrame()

    out_sum = out_dir / f"{tag}_d_eff_summary.csv"
    summary.to_csv(out_sum, index=False)

    print("Wrote:")
    print(" -", out_runs)
    print(" -", out_sum)


if __name__ == "__main__":
    main()
