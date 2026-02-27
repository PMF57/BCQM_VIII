#!/usr/bin/env python3
"""
cloth_ensemble_summary.py (Stage-2)

Summarises cloth core/halo sizes and computes simple survival (Jaccard) of core edge sets across seeds.

Usage:
  python3 -m bcqm_vii_cloth.analysis.cloth_ensemble_summary outputs_cloth/ensemble_W100_N4N8

Or:
  python3 bcqm_vii_cloth/analysis/cloth_ensemble_summary.py outputs_cloth/ensemble_W100_N4N8
"""
from __future__ import annotations
import json, sys
from pathlib import Path
from glob import glob

def load_metrics(path: Path):
    files = sorted(Path(p) for p in glob(str(path / "RUN_METRICS_*.json")))
    out = []
    for f in files:
        out.append(json.loads(f.read_text(encoding="utf-8")))
    return out

def jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return len(a & b) / float(len(a | b))

def edgeset_from_metrics(m) -> set:
    c = m.get("cloth", {}) or {}
    # we did not store explicit edge list; approximate survival via hashing? fallback to ledger summary not possible.
    # If you want explicit edge sets, set cloth.include_edge_list=true in future.
    return set()

def main():
    if len(sys.argv) < 2:
        print("Usage: cloth_ensemble_summary.py <out_dir>")
        raise SystemExit(2)
    root = Path(sys.argv[1])
    mets = load_metrics(root)
    if not mets:
        print("No RUN_METRICS found under", root)
        raise SystemExit(1)
    # Group by (N,n)
    groups = {}
    for m in mets:
        key = (int(m.get("N")), float(m.get("n")))
        groups.setdefault(key, []).append(m)
    print("Found groups:", sorted(groups.keys()))
    for (N,n), ms in sorted(groups.items()):
        core_e = [ (m.get("cloth") or {}).get("core_edges_count") for m in ms ]
        core_v = [ (m.get("cloth") or {}).get("core_events_count") for m in ms ]
        halo_e = [ (m.get("cloth") or {}).get("halo_edges_count") for m in ms ]
        print(f"\n== N={N} n={n:.3f} ({len(ms)} seeds) ==")
        print(f"core_edges_count: {core_e}")
        print(f"core_events_count: {core_v}")
        print(f"halo_edges_count: {halo_e}")
        # placeholder for survival; explicit edge sets not stored in v0.1
        print("survival(core_edges): not computed (edge list not stored; enable if needed).")
    print("\nOK")
if __name__ == "__main__":
    main()
