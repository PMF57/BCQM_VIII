#cat > compute_cloth_occupancy.py <<'PY'
#!/usr/bin/env python3
import json
import sys
from pathlib import Path
from collections import defaultdict

def main():
    if len(sys.argv) != 2:
        print("Usage: python3 compute_cloth_occupancy.py <RUN_METRICS_*.json>")
        sys.exit(2)

    fpath = Path(sys.argv[1])
    data = json.loads(fpath.read_text(encoding="utf-8"))

    cloth = data.get("cloth", {})
    ledger = data.get("cloth_ledger", None)

    if ledger is None:
        raise SystemExit("No cloth_ledger found in this RUN_METRICS. Re-run with cloth.include_ledger=true.")

    # Number of bins recorded in ledger
    B = len(ledger)
    if B == 0:
        raise SystemExit("cloth_ledger is empty.")

    # Count bin-occupancy for events/edges (core vs all), presence-per-bin (not counts)
    hits_ev_core = defaultdict(int)
    hits_ev_all  = defaultdict(int)
    hits_ed_core = defaultdict(int)
    hits_ed_all  = defaultdict(int)

    for rec in ledger:
        # events_core: [[event_id, count], ...]
        for e, c in rec.get("events_core", []):
            hits_ev_core[int(e)] += 1
        for e, c in rec.get("events_all", []):
            hits_ev_all[int(e)] += 1

        # edges_core: [[u, v, count], ...]
        for u, v, c in rec.get("edges_core", []):
            hits_ed_core[(int(u), int(v))] += 1
        for u, v, c in rec.get("edges_all", []):
            hits_ed_all[(int(u), int(v))] += 1

    out_dir = fpath.parent
    out_ev = out_dir / (fpath.stem + "_occupancy_events.csv")
    out_ed = out_dir / (fpath.stem + "_occupancy_edges.csv")

    # Write CSVs: id, hits, occupancy
    with out_ev.open("w", encoding="utf-8") as w:
        w.write("channel,event_id,bin_hits,occupancy\n")
        for e,h in sorted(hits_ev_core.items()):
            w.write(f"core,{e},{h},{h/B:.6f}\n")
        for e,h in sorted(hits_ev_all.items()):
            w.write(f"all,{e},{h},{h/B:.6f}\n")

    with out_ed.open("w", encoding="utf-8") as w:
        w.write("channel,u,v,bin_hits,occupancy\n")
        for (u,v),h in sorted(hits_ed_core.items()):
            w.write(f"core,{u},{v},{h},{h/B:.6f}\n")
        for (u,v),h in sorted(hits_ed_all.items()):
            w.write(f"all,{u},{v},{h},{h/B:.6f}\n")

    # Small console summary (top occupancy)
    def top_items(d, k=10):
        return sorted(d.items(), key=lambda kv: kv[1], reverse=True)[:k]

    print(f"Bins in ledger: {B}")
    print(f"Wrote: {out_ev.name}")
    print(f"Wrote: {out_ed.name}")

    tec = top_items(hits_ed_core, 10)
    if tec:
        print("\nTop core-edge occupancies (by bin hits):")
        for (u,v),h in tec:
            print(f"  ({u}->{v}) hits={h}  occ={h/B:.3f}")

if __name__ == "__main__":
    main()
#PY
