#!/usr/bin/env python3
# PROVENANCE: BCQM_VII Stage-2 analysis; super-graph curvature (Forman-Ricci + clustering); v0.1; 2026-02-03

"""
Compute curvature proxies on the community super-graph:
  - Forman–Ricci curvature (unweighted)
  - Augmented Forman–Ricci (adds triangle contribution)
  - Clustering/transitivity sanity checks

Input: existing RUN_METRICS_*.json files containing:
  cloth.core_edges_used (and optionally cloth.halo_edges_used)

Method:
  1) Build undirected cloth graph from chosen edge_source (core or all).
  2) Louvain community detection on that undirected graph (networkx.louvain_communities).
  3) Build undirected community super-graph: add edge (Ci,Cj) if any cloth edge crosses communities.
  4) Compute curvature proxies on the super-graph.

Outputs:
  - <out_dir>/<tag>_supergraph_curvature_runs.csv
  - <out_dir>/<tag>_supergraph_curvature_summary.csv
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

def build_partition_from_edges(edges: List[Tuple[int,int]], resolution: float, seed: int = 0) -> Dict[int,int]:
    G = nx.Graph()
    for u,v in edges:
        if u != v:
            G.add_edge(int(u), int(v))
    if G.number_of_nodes() == 0:
        return {}
    comms = louvain_communities(G, seed=seed, resolution=float(resolution))
    part = {}
    for cid, nodes in enumerate(comms):
        for node in nodes:
            part[int(node)] = int(cid)
    return part

def build_supergraph(edges: List[Tuple[int,int]], part: Dict[int,int]) -> nx.Graph:
    SG = nx.Graph()
    for u,v in edges:
        cu = part.get(int(u))
        cv = part.get(int(v))
        if cu is None or cv is None:
            continue
        if cu != cv:
            SG.add_edge(int(cu), int(cv))
    return SG

def forman_edge_curvature(G: nx.Graph) -> Dict[Tuple[int,int], float]:
    """Unweighted Forman–Ricci on edges: F(e)=4 - deg(u) - deg(v)."""
    deg = dict(G.degree())
    out = {}
    for u,v in G.edges():
        out[(u,v)] = float(4 - deg[u] - deg[v])
    return out

def augmented_forman_edge_curvature(G: nx.Graph, triangles_per_edge: Dict[Tuple[int,int], int]) -> Dict[Tuple[int,int], float]:
    """Augmented Forman–Ricci: F_aug(e)=F(e)+3*T(e) where T(e) is #triangles containing edge e."""
    base = forman_edge_curvature(G)
    out = {}
    for (u,v), f in base.items():
        t = triangles_per_edge.get((u,v), triangles_per_edge.get((v,u), 0))
        out[(u,v)] = float(f + 3*t)
    return out

def triangle_counts_per_edge(G: nx.Graph) -> Dict[Tuple[int,int], int]:
    """Count triangles per edge (u,v) using common neighbours."""
    tri = {}
    nbrs = {n: set(G.neighbors(n)) for n in G.nodes()}
    for u,v in G.edges():
        tri[(u,v)] = len(nbrs[u].intersection(nbrs[v]))
    return tri

def summarise_edge_values(vals: np.ndarray):
    if len(vals)==0:
        return dict(mean=np.nan,std=np.nan,median=np.nan,q1=np.nan,q3=np.nan,min=np.nan,max=np.nan)
    return dict(
        mean=float(np.mean(vals)),
        std=float(np.std(vals, ddof=1)) if len(vals)>1 else 0.0,
        median=float(np.median(vals)),
        q1=float(np.percentile(vals,25)),
        q3=float(np.percentile(vals,75)),
        min=float(np.min(vals)),
        max=float(np.max(vals)),
    )

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True)
    ap.add_argument("--pattern", default="RUN_METRICS_*.json")
    ap.add_argument("--out_dir", default="csv")
    ap.add_argument("--tag", default=None)
    ap.add_argument("--edge_source", choices=["core","all"], default="core",
                    help="Use core edges only (core) or core+halo edges (all) to build partition and super-graph.")
    ap.add_argument("--resolution", type=float, default=1.0, help="Louvain resolution for partition.")
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = args.tag or run_dir.name

    files = sorted(run_dir.glob(args.pattern))
    if not files:
        raise SystemExit(f"No files matched {args.pattern} in {run_dir}")

    rows=[]
    for f in files:
        N, n, seed = parse_name(f)
        data = json.loads(f.read_text(encoding="utf-8"))

        edges = load_edges(data, args.edge_source)
        part = build_partition_from_edges(edges, resolution=args.resolution, seed=0)
        K = len(set(part.values())) if part else 0
        SG = build_supergraph(edges, part)

        # Super-graph connectivity
        comp_nodes = max(nx.connected_components(SG), key=len) if SG.number_of_nodes()>0 else set()
        GCC = SG.subgraph(comp_nodes).copy() if comp_nodes else nx.Graph()
        sg_nodes = GCC.number_of_nodes()
        sg_edges = GCC.number_of_edges()

        # Curvature + triangles
        tri = triangle_counts_per_edge(GCC) if sg_edges>0 else {}
        F = forman_edge_curvature(GCC) if sg_edges>0 else {}
        F_aug = augmented_forman_edge_curvature(GCC, tri) if sg_edges>0 else {}

        F_vals = np.array(list(F.values()), dtype=float) if F else np.array([], dtype=float)
        Faug_vals = np.array(list(F_aug.values()), dtype=float) if F_aug else np.array([], dtype=float)
        tri_vals = np.array(list(tri.values()), dtype=float) if tri else np.array([], dtype=float)

        # Clustering sanity checks
        trans = nx.transitivity(GCC) if sg_nodes>2 else float("nan")
        clust_mean = float(np.mean(list(nx.clustering(GCC).values()))) if sg_nodes>2 else float("nan")

        sF = summarise_edge_values(F_vals)
        sFa = summarise_edge_values(Faug_vals)
        sT = summarise_edge_values(tri_vals)

        rows.append({
            "N": N, "n": n, "seed": seed,
            "edge_source": args.edge_source,
            "resolution": args.resolution,
            "K": K,
            "super_nodes": sg_nodes,
            "super_edges": sg_edges,
            "transitivity": trans,
            "clustering_mean": clust_mean,

            "F_mean": sF["mean"], "F_std": sF["std"], "F_median": sF["median"], "F_q1": sF["q1"], "F_q3": sF["q3"],
            "F_min": sF["min"], "F_max": sF["max"],
            "F_pos_frac": float(np.mean(F_vals > 0)) if len(F_vals)>0 else np.nan,

            "Faug_mean": sFa["mean"], "Faug_std": sFa["std"], "Faug_median": sFa["median"], "Faug_q1": sFa["q1"], "Faug_q3": sFa["q3"],
            "Faug_min": sFa["min"], "Faug_max": sFa["max"],
            "Faug_pos_frac": float(np.mean(Faug_vals > 0)) if len(Faug_vals)>0 else np.nan,

            "tri_mean": sT["mean"], "tri_median": sT["median"],
        })

    df = pd.DataFrame(rows).sort_values(["N","n","seed"])
    out_runs = out_dir / f"{tag}_supergraph_curvature_runs.csv"
    df.to_csv(out_runs, index=False)

    # Summary by (N,n)
    if not df.empty:
        summ = df.groupby(["N","n","edge_source"]).agg(
            runs=("seed","count"),
            K_mean=("K","mean"),
            super_nodes_mean=("super_nodes","mean"),
            super_edges_mean=("super_edges","mean"),
            trans_mean=("transitivity","mean"),
            clust_mean=("clustering_mean","mean"),
            F_mean=("F_mean","mean"),
            F_pos=("F_pos_frac","mean"),
            Faug_mean=("Faug_mean","mean"),
            Faug_pos=("Faug_pos_frac","mean"),
        ).reset_index()
    else:
        summ = pd.DataFrame()
    out_sum = out_dir / f"{tag}_supergraph_curvature_summary.csv"
    summ.to_csv(out_sum, index=False)

    print("Wrote:")
    print(" -", out_runs)
    print(" -", out_sum)

if __name__ == "__main__":
    main()
