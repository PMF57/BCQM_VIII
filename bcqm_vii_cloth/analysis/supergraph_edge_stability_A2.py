#!/usr/bin/env python3
# PROVENANCE: BCQM_VII A2 analysis; add --edge_source core|all; v0.1; 2026-02-03
"""Super-graph edge stability (Gate-2 pivot check).

Uses the same idea as community_cloth_stability.py:
- takes RUN_METRICS_*.json with cloth.core_edges_used + partition via Louvain
- builds directed community super-graph edges (Ci->Cj) with weights = crossing counts
- computes pairwise Jaccard on the *edge set* of the super-graph (ignoring weights)
- computes optional weight correlation on the intersection

Outputs:
- csv/supergraph_edge_stability.csv

Run (example):
  python3 bcqm_vii_cloth/analysis/supergraph_edge_stability.py \
    --run_dir outputs_cloth/ensemble_W100_N4N8_hits1_x10_bins20 \
    --out_dir csv \
    --method louvain
"""

from __future__ import annotations
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

try:
    from community import community_louvain
except Exception:
    community_louvain = None

_name_pat = re.compile(r"__full__N(\d+)__n([0-9.]+)__seed(\d+)\.json$")

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True)
    ap.add_argument("--pattern", default="RUN_METRICS_*.json")
    ap.add_argument("--out_dir", default="csv")
    ap.add_argument("--method", default="louvain", choices=["louvain"])
    ap.add_argument("--edge_source", choices=["core","all"], default="core",
                    help="Which edge set to use: core edges only, or core+halo (all-used).")
    ap.add_argument("--n_filter", default=None, type=float)
    ap.add_argument("--N_filter", default=None, type=int)
    return ap.parse_args()

def parse_name(p: Path):
    m = _name_pat.search(p.name)
    if not m:
        return (-1, float("nan"), -1)
    return int(m.group(1)), float(m.group(2)), int(m.group(3))

def load_edges(p: Path, edge_source: str) -> List[Tuple[int,int]]:
    data = json.loads(p.read_text(encoding="utf-8"))
    cloth = data.get("cloth", {})
    core_edges = cloth.get("core_edges_used")
    if not isinstance(core_edges, list):
        raise ValueError(f"Missing core_edges_used in {p.name}")
    return [(int(u), int(v)) for u, v in core_edges]

def undirected_graph(edges: List[Tuple[int,int]]) -> nx.Graph:
    G = nx.Graph()
    for u,v in edges:
        if u==v:
            continue
        G.add_edge(u,v)
    return G

def partition(G: nx.Graph) -> Dict[int,int]:
    if community_louvain is None:
        raise RuntimeError("python-louvain not installed")
    return community_louvain.best_partition(G, random_state=0)

def build_super_edges(edges: List[Tuple[int,int]], part: Dict[int,int]) -> Dict[Tuple[int,int], int]:
    flows: Dict[Tuple[int,int], int] = {}
    for u,v in edges:
        cu = part.get(u)
        cv = part.get(v)
        if cu is None or cv is None:
            continue
        key = (int(cu), int(cv))
        flows[key] = flows.get(key, 0) + 1
    return flows

def jaccard(a: set, b: set) -> float:
    if len(a)==0 and len(b)==0:
        return 1.0
    if len(a)==0 or len(b)==0:
        return 0.0
    return len(a & b) / len(a | b)

def weight_corr(a: Dict[Tuple[int,int], int], b: Dict[Tuple[int,int], int]) -> float:
    common = sorted(set(a.keys()) & set(b.keys()))
    if len(common) < 3:
        return float("nan")
    x = np.asarray([a[k] for k in common], dtype=float)
    y = np.asarray([b[k] for k in common], dtype=float)
    if np.std(x) == 0 or np.std(y) == 0:
        return float("nan")
    return float(np.corrcoef(x,y)[0,1])

def main():
    args = parse_args()
    run_dir = Path(args.run_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(run_dir.glob(args.pattern))
    if not files:
        raise SystemExit(f"No files matched {args.pattern} in {run_dir}")

    runs = []
    for p in files:
        N, n, seed = parse_name(p)
        if args.N_filter is not None and N != args.N_filter:
            continue
        if args.n_filter is not None and (math.isnan(n) or abs(n - args.n_filter) > 1e-12):
            continue
        edges = load_edges(p, args.edge_source)
        G = undirected_graph(edges)
        part = partition(G)
        flows = build_super_edges(edges, part)
        runs.append((N, n, seed, flows))

    if not runs:
        raise SystemExit("No runs left after filters.")

    rows=[]
    for (N,n) in sorted({(r[0], r[1]) for r in runs}):
        subset = [(seed, flows) for (NN, nn, seed, flows) in runs if NN==N and abs(nn-n) < 1e-12]
        for (s1, f1), (s2, f2) in combinations(subset, 2):
            e1 = set(f1.keys())
            e2 = set(f2.keys())
            rows.append({
                "N": N, "n": n,
                "seed_a": s1, "seed_b": s2,
                "J_super_edges": jaccard(e1, e2),
                "corr_weights": weight_corr(f1, f2),
                "edges_a": len(e1), "edges_b": len(e2),
                "common_edges": len(e1 & e2),
            })

    df = pd.DataFrame(rows)
    out = out_dir / "supergraph_edge_stability.csv"
    df.to_csv(out, index=False)
    print("Wrote:", out)

if __name__ == "__main__":
    main()
