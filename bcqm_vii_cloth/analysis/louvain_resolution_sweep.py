#!/usr/bin/env python3
# PROVENANCE: BCQM_VII Gate-3 tightening; Louvain resolution sweep; analysis-only; 2026-02-02
"""Louvain resolution sweep for Gate-3 tightening (community + super-graph stability).

Analysis-only: runs on an existing ensemble output folder containing RUN_METRICS_*.json files
with Stage-2 cloth lists:
  cloth.core_edges_used : list of [u,v] directed edges

For each Louvain resolution value, it computes per (N,n):
- Partition stability: mean±std NMI/ARI across seed pairs, plus K (community count) stats
- Super-graph edge stability: Jaccard on (Ci->Cj) edge sets and weight correlation on common edges
- Super-graph geometry stability: L2 distance between normalised super-graph ball-growth curves

Outputs:
- <out_dir>/louvain_resolution_sweep_summary.csv
- <out_dir>/louvain_resolution_sweep_pairs.csv (optional, if --write_pairs)

Example:
  python3 bcqm_vii_cloth/analysis/louvain_resolution_sweep.py \
    --run_dir outputs_cloth/ensemble_W100_N4N8_hits1_x10_bins20 \
    --out_dir csv \
    --resolutions 0.5,1.0,1.5,2.0
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
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score

try:
    from community import community_louvain
except Exception:
    community_louvain = None

_NAME_PAT = re.compile(r"__full__N(\d+)__n([0-9.]+)__seed(\d+)\.json$")

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True)
    ap.add_argument("--pattern", default="RUN_METRICS_*.json")
    ap.add_argument("--out_dir", default="csv")
    ap.add_argument("--resolutions", default="0.5,1.0,1.5,2.0")
    ap.add_argument("--write_pairs", action="store_true")
    ap.add_argument("--N_filter", type=int, default=None)
    ap.add_argument("--n_filter", type=float, default=None)
    ap.add_argument("--ball_r_max", type=int, default=20)
    return ap.parse_args()

def parse_name(p: Path) -> Tuple[int, float, int]:
    m = _NAME_PAT.search(p.name)
    if not m:
        return (-1, float("nan"), -1)
    return int(m.group(1)), float(m.group(2)), int(m.group(3))

def load_edges(p: Path) -> List[Tuple[int, int]]:
    data = json.loads(p.read_text(encoding="utf-8"))
    cloth = data.get("cloth", {})
    core_edges = cloth.get("core_edges_used")
    if not isinstance(core_edges, list):
        raise ValueError(f"Missing cloth.core_edges_used in {p.name}. Ensure cloth.store_lists=true.")
    return [(int(u), int(v)) for u, v in core_edges]

def build_undirected(edges: List[Tuple[int, int]]) -> nx.Graph:
    G = nx.Graph()
    for u, v in edges:
        if u == v:
            continue
        G.add_edge(u, v)
    return G

def louvain_partition(G: nx.Graph, resolution: float) -> Dict[int, int]:
    if community_louvain is None:
        raise RuntimeError("python-louvain not installed (import community_louvain failed).")
    return community_louvain.best_partition(G, random_state=0, resolution=float(resolution))

def build_supergraph(edges: List[Tuple[int, int]], part: Dict[int, int]) -> Dict[Tuple[int, int], int]:
    flows: Dict[Tuple[int, int], int] = {}
    for u, v in edges:
        cu = part.get(u)
        cv = part.get(v)
        if cu is None or cv is None:
            continue
        key = (int(cu), int(cv))
        flows[key] = flows.get(key, 0) + 1
    return flows

def jaccard(a: set, b: set) -> float:
    if len(a) == 0 and len(b) == 0:
        return 1.0
    if len(a) == 0 or len(b) == 0:
        return 0.0
    return len(a & b) / len(a | b)

def weight_corr(a: Dict[Tuple[int, int], int], b: Dict[Tuple[int, int], int]) -> float:
    common = sorted(set(a.keys()) & set(b.keys()))
    if len(common) < 3:
        return float("nan")
    x = np.asarray([a[k] for k in common], dtype=float)
    y = np.asarray([b[k] for k in common], dtype=float)
    if np.std(x) == 0 or np.std(y) == 0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])

def ball_growth_supergraph(flows: Dict[Tuple[int, int], int], r_max: int) -> Tuple[int, List[float]]:
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
    mean_ball: List[float] = []
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
    x = np.asarray(a[:L], dtype=float)
    y = np.asarray(b[:L], dtype=float)
    return float(np.sqrt(np.mean((x - y) ** 2)))

def stat(arr: List[float]) -> Dict[str, float]:
    v = np.asarray([x for x in arr if not (isinstance(x, float) and math.isnan(x))], dtype=float)
    if len(v) == 0:
        return {"mean": float("nan"), "std": float("nan"), "median": float("nan")}
    return {"mean": float(np.mean(v)), "std": float(np.std(v, ddof=1)) if len(v) > 1 else 0.0, "median": float(np.median(v))}

def main() -> None:
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
        edges = load_edges(p)
        G = build_undirected(edges)
        runs.append((N, n, seed, edges, G))

    if not runs:
        raise SystemExit("No runs left after filters.")

    resolutions = [float(x.strip()) for x in args.resolutions.split(",") if x.strip()]
    quadrants = sorted({(N, n) for (N, n, seed, edges, G) in runs})

    summary_rows = []
    pair_rows = []

    for res in resolutions:
        parts = {}
        Ks = {}
        flows = {}
        bg = {}

        for (N, n, seed, edges, G) in runs:
            key = (N, n, seed)
            part = louvain_partition(G, res)
            parts[key] = part
            Ks[key] = len(set(part.values())) if part else 0
            fl = build_supergraph(edges, part)
            flows[key] = fl
            bg[key] = ball_growth_supergraph(fl, args.ball_r_max)

        for (N, n) in quadrants:
            seeds = sorted({seed for (NN, nn, seed, edges, G) in runs if NN == N and abs(nn - n) < 1e-12})
            if len(seeds) < 2:
                continue

            nmi_vals, ari_vals, J_vals, corr_vals, d_vals = [], [], [], [], []
            K_vals = [Ks[(N, n, s)] for s in seeds]
            comp_vals = [bg[(N, n, s)][0] for s in seeds]

            for s1, s2 in combinations(seeds, 2):
                p1 = parts[(N, n, s1)]
                p2 = parts[(N, n, s2)]
                common = sorted(set(p1.keys()) & set(p2.keys()))
                if len(common) >= 10:
                    a = [p1[x] for x in common]
                    b = [p2[x] for x in common]
                    nmi = float(normalized_mutual_info_score(a, b))
                    ari = float(adjusted_rand_score(a, b))
                    nmi_vals.append(nmi)
                    ari_vals.append(ari)
                else:
                    nmi = float("nan"); ari = float("nan")

                f1 = flows[(N, n, s1)]
                f2 = flows[(N, n, s2)]
                J = jaccard(set(f1.keys()), set(f2.keys()))
                corr = weight_corr(f1, f2)

                c1, b1 = bg[(N, n, s1)]
                c2, b2 = bg[(N, n, s2)]
                d = curve_l2(b1, b2)

                J_vals.append(J); corr_vals.append(corr); d_vals.append(d)

                if args.write_pairs:
                    pair_rows.append({
                        "resolution": res, "N": N, "n": n,
                        "seed_a": s1, "seed_b": s2,
                        "NMI": nmi, "ARI": ari,
                        "J_super_edges": J, "corr_weights": corr, "d_l2": d,
                        "K_a": Ks[(N, n, s1)], "K_b": Ks[(N, n, s2)],
                        "comp_a": c1, "comp_b": c2,
                    })

            s_nmi = stat(nmi_vals); s_ari = stat(ari_vals); s_J = stat(J_vals); s_corr = stat(corr_vals); s_d = stat(d_vals)

            summary_rows.append({
                "resolution": res, "N": N, "n": n,
                "NMI_mean": s_nmi["mean"], "NMI_std": s_nmi["std"],
                "ARI_mean": s_ari["mean"], "ARI_std": s_ari["std"],
                "K_mean": float(np.mean(K_vals)), "K_std": float(np.std(K_vals, ddof=1)) if len(K_vals) > 1 else 0.0,
                "K_min": int(np.min(K_vals)), "K_max": int(np.max(K_vals)),
                "J_super_mean": s_J["mean"], "J_super_std": s_J["std"],
                "corr_mean": s_corr["mean"], "corr_std": s_corr["std"],
                "d_l2_mean": s_d["mean"], "d_l2_std": s_d["std"],
                "comp_mean": float(np.mean(comp_vals)), "comp_min": int(np.min(comp_vals)), "comp_max": int(np.max(comp_vals)),
                "pairs": int(len(seeds) * (len(seeds) - 1) / 2),
            })

    df_sum = pd.DataFrame(summary_rows).sort_values(["resolution", "N", "n"])
    out_sum = out_dir / "louvain_resolution_sweep_summary.csv"
    df_sum.to_csv(out_sum, index=False)

    if args.write_pairs:
        df_pairs = pd.DataFrame(pair_rows)
        out_pairs = out_dir / "louvain_resolution_sweep_pairs.csv"
        df_pairs.to_csv(out_pairs, index=False)

    print("Wrote:")
    print(" -", out_sum)
    if args.write_pairs:
        print(" -", out_pairs)

if __name__ == "__main__":
    main()
