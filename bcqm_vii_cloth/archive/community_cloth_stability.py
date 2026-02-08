#!/usr/bin/env python3
"""Community cloth stability analysis (Stage-2 pivot).

Runs on an EXISTING ensemble output folder containing RUN_METRICS_*.json files
with Stage-2 cloth lists:
  cloth.core_edges_used (list of [u,v])
  cloth.core_events_used (list of event ids)

Outputs CSVs to ./csv/ by default:
  - community_partition_stability.csv   (pairwise NMI/ARI + K stats)
  - supergraph_ballgrowth_stability.csv (pairwise ball-growth curve distances)

Usage:
  python3 bcqm_vii_cloth/analysis/community_cloth_stability.py \
      --run_dir outputs_cloth/ensemble_W100_N4N8_hits1_x10_bins20 \
      --pattern 'RUN_METRICS_*.json' \
      --out_dir csv \
      --method louvain

Notes:
- This does NOT re-run simulations.
- Community detection is on the UNDIRECTED projection of core_edges_used.
- Super-graph is DIRECTED (community->community) with weight = number of crossing edges.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import networkx as nx
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score

try:
    from community import community_louvain  # python-louvain
except Exception:
    community_louvain = None


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True, help="Folder containing RUN_METRICS_*.json")
    ap.add_argument("--pattern", default="RUN_METRICS_*.json", help="Glob for metrics files")
    ap.add_argument("--out_dir", default="csv", help="Where to write output CSVs")
    ap.add_argument("--method", default="louvain", choices=["louvain"], help="Community method")
    ap.add_argument("--seed_filter", default=None, help="Optional regex to filter filenames")
    ap.add_argument("--n_filter", default=None, type=float, help="Optional filter for n (exact float match)")
    ap.add_argument("--N_filter", default=None, type=int, help="Optional filter for N (exact match)")
    ap.add_argument("--ball_r_max", default=20, type=int, help="Max radius for ball growth on supergraph")
    return ap.parse_args()


@dataclass
class Run:
    path: Path
    N: int
    n: float
    seed: int
    core_edges: List[Tuple[int, int]]
    core_events: List[int]
    part: Dict[int, int]  # node->community
    K: int
    super_edges: Dict[Tuple[int, int], int]  # (ci,cj)->weight


_name_pat = re.compile(r"__full__N(\d+)__n([0-9.]+)__seed(\d+)\.json$")


def parse_name(p: Path) -> Tuple[int, float, int]:
    m = _name_pat.search(p.name)
    if not m:
        return (-1, float("nan"), -1)
    return int(m.group(1)), float(m.group(2)), int(m.group(3))


def load_run(p: Path) -> Tuple[List[Tuple[int, int]], List[int]]:
    data = json.loads(p.read_text(encoding="utf-8"))
    cloth = data.get("cloth", {})
    core_edges = cloth.get("core_edges_used")
    core_events = cloth.get("core_events_used")
    if not isinstance(core_edges, list) or not isinstance(core_events, list):
        raise ValueError(f"Missing core lists in {p.name}. Ensure cloth.store_lists=true.")
    edges = [(int(u), int(v)) for u, v in core_edges]
    evs = [int(e) for e in core_events]
    return edges, evs


def undirected_core_graph(edges: List[Tuple[int, int]]) -> nx.Graph:
    G = nx.Graph()
    for u, v in edges:
        if u == v:
            continue
        G.add_edge(u, v)
    return G


def compute_partition(G: nx.Graph, method: str) -> Dict[int, int]:
    if method == "louvain":
        if community_louvain is None:
            raise RuntimeError("python-louvain not available. Install python-louvain.")
        return community_louvain.best_partition(G, random_state=0)
    raise ValueError(method)


def build_supergraph(edges: List[Tuple[int, int]], part: Dict[int, int]) -> Dict[Tuple[int, int], int]:
    """Directed community->community edges with weights (# crossing edges)."""
    flows: Dict[Tuple[int, int], int] = {}
    for u, v in edges:
        cu = part.get(u)
        cv = part.get(v)
        if cu is None or cv is None:
            continue
        key = (int(cu), int(cv))
        flows[key] = flows.get(key, 0) + 1
    return flows


def ball_growth_supergraph(flows: Dict[Tuple[int, int], int], r_max: int) -> Tuple[int, List[float]]:
    """Ball growth on undirected projection of supergraph (ignores weights for connectivity)."""
    UG = nx.Graph()
    for (cu, cv), w in flows.items():
        if cu == cv:
            continue
        UG.add_edge(cu, cv)
    if UG.number_of_nodes() == 0:
        return 0, [0.0] * (r_max + 1)

    comp_nodes = max(nx.connected_components(UG), key=len)
    H = UG.subgraph(comp_nodes).copy()
    comp_size = H.number_of_nodes()

    roots = list(H.nodes())
    mean_ball = []
    for r in range(r_max + 1):
        sizes = []
        for root in roots:
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
            sizes.append(len(seen) / comp_size)
        mean_ball.append(float(np.mean(sizes)))
    return comp_size, mean_ball


def curve_l2(a: List[float], b: List[float]) -> float:
    L = min(len(a), len(b))
    a = np.asarray(a[:L], dtype=float)
    b = np.asarray(b[:L], dtype=float)
    return float(np.sqrt(np.mean((a - b) ** 2)))


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(run_dir.glob(args.pattern))
    if args.seed_filter:
        rx = re.compile(args.seed_filter)
        files = [p for p in files if rx.search(p.name)]
    if not files:
        raise SystemExit(f"No files matched in {run_dir} with pattern {args.pattern}")

    runs: List[Run] = []
    for p in files:
        N, n, seed = parse_name(p)
        if args.N_filter is not None and N != args.N_filter:
            continue
        if args.n_filter is not None and (math.isnan(n) or abs(n - args.n_filter) > 1e-12):
            continue
        edges, evs = load_run(p)
        G = undirected_core_graph(edges)
        part = compute_partition(G, args.method)
        K = len(set(part.values())) if part else 0
        flows = build_supergraph(edges, part)
        runs.append(Run(p, N, n, seed, edges, evs, part, K, flows))

    if not runs:
        raise SystemExit("No runs left after filters.")

    part_rows = []
    for r1, r2 in combinations(runs, 2):
        if r1.N != r2.N or abs(r1.n - r2.n) > 1e-12:
            continue
        common = sorted(set(r1.part.keys()) & set(r2.part.keys()))
        if len(common) < 10:
            continue
        a = [r1.part[x] for x in common]
        b = [r2.part[x] for x in common]
        part_rows.append({
            "N": r1.N, "n": r1.n,
            "seed_a": r1.seed, "seed_b": r2.seed,
            "K_a": r1.K, "K_b": r2.K,
            "NMI": float(normalized_mutual_info_score(a, b)),
            "ARI": float(adjusted_rand_score(a, b)),
        })
    df_part = pd.DataFrame(part_rows)

    sg_rows = []
    bg_map = {}
    for r in runs:
        comp_size, mean_ball = ball_growth_supergraph(r.super_edges, args.ball_r_max)
        bg_map[(r.N, r.n, r.seed)] = (comp_size, mean_ball)

    for r1, r2 in combinations(runs, 2):
        if r1.N != r2.N or abs(r1.n - r2.n) > 1e-12:
            continue
        c1, b1 = bg_map[(r1.N, r1.n, r1.seed)]
        c2, b2 = bg_map[(r2.N, r2.n, r2.seed)]
        sg_rows.append({
            "N": r1.N, "n": r1.n,
            "seed_a": r1.seed, "seed_b": r2.seed,
            "comp_a": c1, "comp_b": c2,
            "d_l2": curve_l2(b1, b2),
        })
    df_sg = pd.DataFrame(sg_rows)

    out_part = out_dir / "community_partition_stability.csv"
    out_sg = out_dir / "supergraph_ballgrowth_stability.csv"
    df_part.to_csv(out_part, index=False)
    df_sg.to_csv(out_sg, index=False)

    print("Wrote:")
    print(" -", out_part)
    print(" -", out_sg)


if __name__ == "__main__":
    main()
