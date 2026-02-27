#!/usr/bin/env python3
# PROVENANCE: BCQM_VII Gate-4 analysis; thread localisation on super-graph; 2026-02-02

import argparse, json, math, re
from pathlib import Path
from itertools import combinations
from collections import Counter

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

def build_partition_from_edges(core_edges, resolution=1.0, seed=0):
    G = nx.Graph()
    for u, v in core_edges:
        if u != v:
            G.add_edge(int(u), int(v))
    comms = louvain_communities(G, seed=seed, resolution=resolution)
    part = {}
    for cid, nodes in enumerate(comms):
        for node in nodes:
            part[int(node)] = int(cid)
    return part

def build_supergraph(core_edges, part):
    flows = {}
    for u, v in core_edges:
        cu = part.get(int(u))
        cv = part.get(int(v))
        if cu is None or cv is None:
            continue
        flows[(cu, cv)] = flows.get((cu, cv), 0) + 1
    UG = nx.Graph()
    for (cu, cv), w in flows.items():
        if cu != cv:
            UG.add_edge(cu, cv)
    dmap = dict(nx.all_pairs_shortest_path_length(UG)) if UG.number_of_nodes() > 0 else {}
    return dmap

def hop_dist(dmap, a, b):
    if a == b:
        return 0
    try:
        return dmap[a][b]
    except Exception:
        return math.inf

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True)
    ap.add_argument("--pattern", default="RUN_METRICS_*.json")
    ap.add_argument("--out_dir", default="csv")
    ap.add_argument("--resolution", type=float, default=1.0)
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(run_dir.glob(args.pattern))
    if not files:
        raise SystemExit(f"No files matched in {run_dir}")

    stats_rows = []
    hop_rows = []

    for f in files:
        N, n, seed = parse_name(f)
        data = json.loads(f.read_text(encoding="utf-8"))
        cloth = data.get("cloth", {})
        trace = data.get("cloth_trace", None)

        if trace is None or not trace.get("enabled", False):
            raise SystemExit(f"Missing cloth_trace in {f.name}. Re-run with cloth.trace_threads=true.")

        core_edges = cloth.get("core_edges_used")
        if not isinstance(core_edges, list):
            raise SystemExit(f"Missing cloth.core_edges_used in {f.name}. Ensure cloth.store_lists=true.")
        core_edges = [(int(u), int(v)) for u, v in core_edges]

        part = build_partition_from_edges(core_edges, resolution=args.resolution, seed=0)
        dmap = build_supergraph(core_edges, part)

        ev_end = trace["event_at_end"]
        core_mask = trace["core_mask"]
        B = len(ev_end)
        Nthreads = len(ev_end[0])

        # map event -> community label per bin/thread
        comm = [[part.get(int(ev_end[b][i]), -1) for i in range(Nthreads)] for b in range(B)]

        hops = []
        for i in range(Nthreads):
            for b in range(B-1):
                c1 = comm[b][i]; c2 = comm[b+1][i]
                if c1 == -1 or c2 == -1:
                    continue
                hops.append(hop_dist(dmap, c1, c2))

        hops = np.asarray(hops, dtype=float)
        if len(hops) == 0:
            continue

        hop_rows.append({
            "N": N, "n": n, "seed": seed,
            "transitions": int(len(hops)),
            "frac_d0": float(np.mean(hops == 0)),
            "frac_d1": float(np.mean(hops == 1)),
            "frac_d2": float(np.mean(hops == 2)),
            "frac_dge3": float(np.mean(hops >= 3)),
            "mean_d": float(np.mean(hops)),
            "mean_d_cond_change": float(np.mean(hops[hops > 0])) if np.any(hops > 0) else np.nan,
        })

        stats_rows.append({
            "N": N, "n": n, "seed": seed,
            "bins_logged": B,
            "threads": Nthreads,
        })

    df_stats = pd.DataFrame(stats_rows).sort_values(["N","n","seed"])
    df_hops  = pd.DataFrame(hop_rows).sort_values(["N","n","seed"])

    df_stats.to_csv(out_dir / "gate4_thread_localisation_stats_seedwise.csv", index=False)
    df_hops.to_csv(out_dir / "gate4_thread_localisation_hopdist_seedwise.csv", index=False)

    print("Wrote:")
    print(" -", out_dir / "gate4_thread_localisation_stats_seedwise.csv")
    print(" -", out_dir / "gate4_thread_localisation_hopdist_seedwise.csv")

if __name__ == "__main__":
    main()
