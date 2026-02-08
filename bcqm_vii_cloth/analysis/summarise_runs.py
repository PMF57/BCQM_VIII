#!/usr/bin/env python3
# PROVENANCE: BCQM_VII analysis; summarise RUN_METRICS into compact CSVs; 2026-02-06

import argparse, json, re
from pathlib import Path
import numpy as np
import pandas as pd
from itertools import combinations

NAME_PAT = re.compile(r"__full__N(\d+)__n([0-9.]+)__seed(\d+)\.json$")

def parse_name(p: Path):
    m = NAME_PAT.search(p.name)
    if not m:
        return (-1, float("nan"), -1)
    return int(m.group(1)), float(m.group(2)), int(m.group(3))

def curve_l2(a, b):
    if a is None or b is None:
        return np.nan
    L = min(len(a), len(b))
    a = np.asarray(a[:L], dtype=float)
    b = np.asarray(b[:L], dtype=float)
    return float(np.sqrt(np.mean((a-b)**2)))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True, help="Folder with RUN_METRICS_*.json")
    ap.add_argument("--out_dir", default="csv", help="Where to write CSVs")
    ap.add_argument("--tag", default=None, help="Tag prefix for output files (optional)")
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(run_dir.glob("RUN_METRICS_*.json"))
    if not files:
        raise SystemExit(f"No RUN_METRICS_*.json in {run_dir}")

    rows = []
    curves = []  # (N, n, seed, frac_curve)
    for f in files:
        N, n, seed = parse_name(f)
        data = json.loads(f.read_text(encoding="utf-8"))
        cloth = data.get("cloth", {})
        bg = cloth.get("ball_growth") or {}
        comp = bg.get("comp_size")
        mean_ball = bg.get("mean_ball")
        frac = None
        if isinstance(mean_ball, list) and isinstance(comp,(int,float)) and comp and comp>0:
            frac = (np.asarray(mean_ball, dtype=float) / float(comp)).tolist()

        trace = data.get("cloth_trace") or {}
        rows.append({
            "N": N, "n": n, "seed": seed,
            "core_edges_count": cloth.get("core_edges_count"),
            "core_events_count": cloth.get("core_events_count"),
            "halo_edges_count": cloth.get("halo_edges_count"),
            "halo_events_count": cloth.get("halo_events_count"),
            "ball_comp_size": comp,
            "Q_clock": data.get("Q_clock"),
            "L": data.get("L"),
            "S_perc": data.get("S_perc"),
            "S_junc_w": data.get("S_junc_w"),
            "trace_enabled": bool(trace.get("enabled", False)),
            "bins_logged": len(trace.get("event_at_end", [])) if trace.get("enabled", False) else 0,
        })
        curves.append((N, n, seed, frac))

    df = pd.DataFrame(rows).sort_values(["N","n","seed"])

    # Pairwise ball-growth metric stability
    # IMPORTANT: in multi-N runs, seeds repeat across N, so pairwise rows must carry (N,n) to be interpretable.
    # We therefore compute:
    #   (i) a global pairwise table with N_a,n_a and N_b,n_b columns, and
    #   (ii) a per-(N,n) pairwise table, restricted to pairs within the same (N,n).
    pair_rows = []
    pair_rows_by_group = {}  # (N,n) -> list of rows

    for (N1, n1, s1, c1), (N2, n2, s2, c2) in combinations(curves, 2):
        d = curve_l2(c1, c2)
        pair_rows.append({
            "N_a": N1, "n_a": n1, "seed_a": s1,
            "N_b": N2, "n_b": n2, "seed_b": s2,
            "d_l2": d
        })
        if (N1 == N2) and (np.isfinite(n1) and np.isfinite(n2)) and (abs(float(n1) - float(n2)) < 1e-12):
            key = (N1, float(n1))
            pair_rows_by_group.setdefault(key, []).append({
                "N": N1, "n": float(n1),
                "seed_a": s1, "seed_b": s2,
                "d_l2": d
            })

    df_pairs = pd.DataFrame(pair_rows)


    tag = args.tag or Path(args.run_dir).name
    out_summary = out_dir / f"{tag}_run_summary.csv"
    out_pairs = out_dir / f"{tag}_ballgrowth_pairwise.csv"

    df.to_csv(out_summary, index=False)
    df_pairs.to_csv(out_pairs, index=False)

    # Per-(N,n) pairwise files (useful for per-N stability claims)
    for (NN, nn), rows_g in sorted(pair_rows_by_group.items()):
        df_g = pd.DataFrame(rows_g)
        # file tag includes N and n with safe formatting
        n_tag = str(nn).replace(".", "p")
        out_g = out_dir / f"{tag}_ballgrowth_pairwise_N{NN}_n{n_tag}.csv"
        df_g.to_csv(out_g, index=False)


    print("Wrote:")
    print(" -", out_summary)
    print(" -", out_pairs)
    if pair_rows_by_group:
        print(f" - (per-N files) {len(pair_rows_by_group)} file(s) written: {tag}_ballgrowth_pairwise_N*_n*.csv")

if __name__ == "__main__":
    main()
