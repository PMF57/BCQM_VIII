#!/usr/bin/env python3
# PROVENANCE: BCQM_VII Gate-4 analysis; thread localisation on super-graph; v0.2 (all-used partition + coverage + core/halo); 2026-02-02

import argparse, json, math, re
from pathlib import Path

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

def build_partition_from_edges(edges_undirected, resolution=1.0, seed=0):
    """Return node->community mapping using Louvain on an undirected graph."""
    G = nx.Graph()
    for u, v in edges_undirected:
        if u != v:
            G.add_edge(int(u), int(v))
    if G.number_of_nodes() == 0:
        return {}
    comms = louvain_communities(G, seed=seed, resolution=resolution)
    part = {}
    for cid, nodes in enumerate(comms):
        for node in nodes:
            part[int(node)] = int(cid)
    return part

def build_supergraph_dmap(edges_directed, part):
    """Build undirected projection of the community super-graph and return all-pairs hop distances."""
    flows = {}
    for u, v in edges_directed:
        cu = part.get(int(u))
        cv = part.get(int(v))
        if cu is None or cv is None:
            continue
        flows[(int(cu), int(cv))] = flows.get((int(cu), int(cv)), 0) + 1
    UG = nx.Graph()
    for (cu, cv), w in flows.items():
        if cu != cv:
            UG.add_edge(cu, cv)
    if UG.number_of_nodes() == 0:
        return {}, 0, 0
    dmap = dict(nx.all_pairs_shortest_path_length(UG))
    return dmap, UG.number_of_nodes(), UG.number_of_edges()

def hop_dist(dmap, a, b):
    if a == b:
        return 0
    try:
        return dmap[a][b]
    except Exception:
        return math.inf

def frac_stats(arr):
    arr = np.asarray(arr, dtype=float)
    if len(arr) == 0:
        return dict(frac_d0=np.nan, frac_d1=np.nan, frac_d2=np.nan, frac_dge3=np.nan, mean_d=np.nan, mean_d_cond_change=np.nan)
    return dict(
        frac_d0=float(np.mean(arr == 0)),
        frac_d1=float(np.mean(arr == 1)),
        frac_d2=float(np.mean(arr == 2)),
        frac_dge3=float(np.mean(arr >= 3)),
        mean_d=float(np.mean(arr)),
        mean_d_cond_change=float(np.mean(arr[arr > 0])) if np.any(arr > 0) else np.nan,
    )

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True)
    ap.add_argument("--pattern", default="RUN_METRICS_*.json")
    ap.add_argument("--out_dir", default="csv")
    ap.add_argument("--resolution", type=float, default=1.0)
    ap.add_argument("--partition_source", choices=["core", "all"], default="all",
                    help="Which edge set to use to build the community partition: core edges only, or core+halo (all-used).")
    ap.add_argument("--supergraph_source", choices=["core", "all"], default="all",
                    help="Which edge set to use to build the community super-graph distances.")
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
        core_events = cloth.get("core_events_used")
        halo_edges = cloth.get("halo_edges_used")  # may be None
        if not isinstance(core_edges, list) or not isinstance(core_events, list):
            raise SystemExit(f"Missing core lists in {f.name}. Ensure cloth.store_lists=true.")

        core_edges = [(int(u), int(v)) for u, v in core_edges]
        all_edges = list(core_edges)
        if isinstance(halo_edges, list):
            all_edges.extend([(int(u), int(v)) for u, v in halo_edges])

        # Partition graph edge set
        part_edges = core_edges if args.partition_source == "core" else all_edges
        part = build_partition_from_edges(part_edges, resolution=args.resolution, seed=0)

        # Supergraph distance map edge set
        sg_edges = core_edges if args.supergraph_source == "core" else all_edges
        dmap, sg_nodes, sg_edges_ct = build_supergraph_dmap(sg_edges, part)

        ev_end = trace["event_at_end"]
        core_mask = trace["core_mask"]
        B = len(ev_end)
        Nthreads = len(ev_end[0])

        total_possible = (B - 1) * Nthreads
        mapped = 0
        missing_labels = 0
        inf_hops = 0

        hops_total = []
        hops_core = []
        hops_halo = []

        # map event -> community label per bin/thread
        comm = [[part.get(int(ev_end[b][i]), -1) for i in range(Nthreads)] for b in range(B)]

        for i in range(Nthreads):
            for b in range(B - 1):
                c1 = comm[b][i]
                c2 = comm[b + 1][i]
                if c1 == -1 or c2 == -1:
                    missing_labels += 1
                    continue
                d = hop_dist(dmap, c1, c2)
                if math.isinf(d):
                    inf_hops += 1
                    continue
                mapped += 1
                hops_total.append(d)
                if int(core_mask[b][i]) == 1:
                    hops_core.append(d)
                else:
                    hops_halo.append(d)

        # seedwise hop distribution rows
        hops_total = np.asarray(hops_total, dtype=float)
        hops_core = np.asarray(hops_core, dtype=float)
        hops_halo = np.asarray(hops_halo, dtype=float)

        row = {"N": N, "n": n, "seed": seed,
               "bins_logged": B, "threads": Nthreads,
               "partition_source": args.partition_source,
               "supergraph_source": args.supergraph_source,
               "K": int(len(set(part.values()))) if part else 0,
               "super_nodes": int(sg_nodes),
               "super_edges": int(sg_edges_ct),
               "transitions_total": int(total_possible),
               "transitions_mapped": int(mapped),
               "coverage": float(mapped) / float(total_possible) if total_possible > 0 else np.nan,
               "missing_labels": int(missing_labels),
               "inf_hops": int(inf_hops),
        }

        # Add distributions
        row.update(frac_stats(hops_total))
        # Core/halo breakdown (may be empty)
        core_stats = frac_stats(hops_core)
        halo_stats = frac_stats(hops_halo)
        for k,v in core_stats.items():
            row["core_"+k] = v
        for k,v in halo_stats.items():
            row["halo_"+k] = v

        hop_rows.append(row)

        stats_rows.append({
            "N": N, "n": n, "seed": seed,
            "bins_logged": B, "threads": Nthreads,
            "partition_source": args.partition_source,
            "supergraph_source": args.supergraph_source,
            "K": int(len(set(part.values()))) if part else 0,
            "super_nodes": int(sg_nodes),
            "super_edges": int(sg_edges_ct),
            "transitions_total": int(total_possible),
            "transitions_mapped": int(mapped),
            "coverage": float(mapped) / float(total_possible) if total_possible > 0 else np.nan,
            "missing_labels": int(missing_labels),
            "inf_hops": int(inf_hops),
        })

    df_stats = pd.DataFrame(stats_rows).sort_values(["N", "n", "seed"])
    df_hops = pd.DataFrame(hop_rows).sort_values(["N", "n", "seed"])

    df_stats.to_csv(out_dir / "gate4_thread_localisation_stats_seedwise.csv", index=False)
    df_hops.to_csv(out_dir / "gate4_thread_localisation_hopdist_seedwise.csv", index=False)

    print("Wrote:")
    print(" -", out_dir / "gate4_thread_localisation_stats_seedwise.csv")
    print(" -", out_dir / "gate4_thread_localisation_hopdist_seedwise.csv")
    print("Note: coverage indicates how many (bin,thread) transitions were mappable into communities and finite hop distances.")

if __name__ == "__main__":
    main()
