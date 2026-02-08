#!/usr/bin/env python3
import json, re, itertools
from pathlib import Path
import numpy as np
import pandas as pd

# Where your x10 run outputs live (you said: outputs_cloth folder)
ROOT = Path("outputs_cloth")
# If you extracted the zip, use this folder:
RUN_DIR = ROOT / "ensemble_W100_N4N8_hits2_x10"

# Output folder you created to consolidate CSVs
OUT = Path("csv")
OUT.mkdir(parents=True, exist_ok=True)

pat = re.compile(r"__full__N(\d+)__n([0-9.]+)__seed(\d+)\.json$")

def parse_name(name: str):
    m = pat.search(name)
    if not m:
        return None
    return int(m.group(1)), float(m.group(2)), int(m.group(3))

def jaccard(a: set, b: set):
    if a is None or b is None:
        return np.nan
    if len(a)==0 and len(b)==0:
        return 1.0
    if len(a)==0 or len(b)==0:
        return 0.0
    return len(a & b) / len(a | b)

def curve_l2(a: np.ndarray, b: np.ndarray):
    if a is None or b is None:
        return np.nan
    L = min(len(a), len(b))
    a = a[:L]; b = b[:L]
    return float(np.sqrt(np.mean((a-b)**2)))

# Load runs
metrics_files = sorted(RUN_DIR.glob("RUN_METRICS_*.json"))
if not metrics_files:
    raise SystemExit(f"No RUN_METRICS_*.json found in {RUN_DIR}")

rows = []
for f in metrics_files:
    parsed = parse_name(f.name)
    if not parsed:
        continue
    N, n, seed = parsed
    m = json.loads(f.read_text(encoding="utf-8"))
    cloth = m.get("cloth", {})
    bg = cloth.get("ball_growth") or {}

    core_edges = cloth.get("core_edges_used")
    core_events = cloth.get("core_events_used")

    edge_set = set((int(u), int(v)) for u,v in core_edges) if isinstance(core_edges, list) else None
    event_set = set(int(e) for e in core_events) if isinstance(core_events, list) else None

    curve = bg.get("mean_ball")
    comp = bg.get("comp_size")
    frac = None
    if isinstance(curve, list) and isinstance(comp, (int,float)) and comp and comp > 0:
        frac = np.array(curve, dtype=float) / float(comp)

    rows.append({
        "N": N, "n": n, "seed": seed,
        "core_edges_count": cloth.get("core_edges_count"),
        "core_events_count": cloth.get("core_events_count"),
        "ball_comp_size": comp,
        "edge_set": edge_set,
        "event_set": event_set,
        "frac_curve": frac,
    })

df = pd.DataFrame(rows)

# Pairwise survival tables
pair_rows = []
metric_rows = []
for (N,n), g in df.groupby(["N","n"]):
    items = list(g.to_dict("records"))
    for r1, r2 in itertools.combinations(items, 2):
        pair_rows.append({
            "N": N, "n": n,
            "seed_a": r1["seed"], "seed_b": r2["seed"],
            "J_edges": jaccard(r1["edge_set"], r2["edge_set"]),
            "J_events": jaccard(r1["event_set"], r2["event_set"]),
            "edges_a": len(r1["edge_set"]) if r1["edge_set"] is not None else np.nan,
            "edges_b": len(r2["edge_set"]) if r2["edge_set"] is not None else np.nan,
            "events_a": len(r1["event_set"]) if r1["event_set"] is not None else np.nan,
            "events_b": len(r2["event_set"]) if r2["event_set"] is not None else np.nan,
            "comp_a": r1["ball_comp_size"], "comp_b": r2["ball_comp_size"],
        })
        metric_rows.append({
            "N": N, "n": n,
            "seed_a": r1["seed"], "seed_b": r2["seed"],
            "d_l2": curve_l2(r1["frac_curve"], r2["frac_curve"]),
        })

pairs = pd.DataFrame(pair_rows)
metric = pd.DataFrame(metric_rows)

def stats(series):
    s = series.dropna().to_numpy(dtype=float)
    if len(s)==0:
        return dict(count=0, mean=np.nan, std=np.nan, median=np.nan, q1=np.nan, q3=np.nan)
    return dict(
        count=len(s),
        mean=float(np.mean(s)),
        std=float(np.std(s, ddof=1)) if len(s)>1 else 0.0,
        median=float(np.median(s)),
        q1=float(np.percentile(s,25)),
        q3=float(np.percentile(s,75)),
    )

summary = []
for (N,n), g in pairs.groupby(["N","n"]):
    se = stats(g["J_edges"])
    sv = stats(g["J_events"])
    gd = stats(metric[(metric["N"]==N)&(metric["n"]==n)]["d_l2"])
    runs = df[(df["N"]==N)&(df["n"]==n)]
    summary.append({
        "N": N, "n": n,
        "pairs": se["count"],
        "J_edges_mean": se["mean"], "J_edges_std": se["std"], "J_edges_med": se["median"], "J_edges_q1": se["q1"], "J_edges_q3": se["q3"],
        "J_events_mean": sv["mean"], "J_events_std": sv["std"], "J_events_med": sv["median"], "J_events_q1": sv["q1"], "J_events_q3": sv["q3"],
        "d_l2_mean": gd["mean"], "d_l2_std": gd["std"], "d_l2_med": gd["median"], "d_l2_q1": gd["q1"], "d_l2_q3": gd["q3"],
        "core_edges_count_mean": float(np.mean(runs["core_edges_count"])),
        "core_events_count_mean": float(np.mean(runs["core_events_count"])),
        "ball_comp_size_mean": float(np.mean(runs["ball_comp_size"])),
    })

summary_df = pd.DataFrame(summary).sort_values(["N","n"])

# Write outputs
summary_df.to_csv(OUT / "cloth_survival_summary_hits2_x10_W100.csv", index=False)
pairs.to_csv(OUT / "cloth_survival_pairs_hits2_x10_W100.csv", index=False)
metric.to_csv(OUT / "cloth_metric_pairs_hits2_x10_W100.csv", index=False)

print("Wrote:")
print(" -", OUT / "cloth_survival_summary_hits2_x10_W100.csv")
print(" -", OUT / "cloth_survival_pairs_hits2_x10_W100.csv")
print(" -", OUT / "cloth_metric_pairs_hits2_x10_W100.csv")