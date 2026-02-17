#!/usr/bin/env python3
"""
Fit VII_c knee contour from run_summary CSVs.

Looks for files matching: csv_s3kc/vii_c_*_run_summary.csv
Each file is expected to contain columns: N, n, core_events_count, halo_events_count, Q_clock.

Outputs:
- csv_s3kc/knee_contour_summary.csv : estimated N_k (Phi=0.5) per n
- csv_s3kc/lock_contour_summary.csv : optional N_l where p_lowQ < 0.05 (if data supports)
"""

from __future__ import annotations
import glob
from pathlib import Path
import numpy as np
import pandas as pd

CSV_DIR = Path("csv_s3kc")

def phi(df: pd.DataFrame) -> pd.Series:
    tot = df["core_events_count"] + df["halo_events_count"]
    return df["core_events_count"] / tot.replace(0, np.nan)

def estimate_knee_for_n(df: pd.DataFrame) -> dict:
    # df includes multiple N with multiple seeds
    df = df.copy()
    df["phi"] = phi(df)
    g = df.groupby("N")["phi"].agg(["mean","std","count"]).reset_index().sort_values("N")
    # find adjacent Ns where mean crosses 0.5
    vals = g["mean"].values
    Ns = g["N"].values
    idx = None
    for i in range(len(vals)-1):
        if (vals[i] >= 0.5 and vals[i+1] <= 0.5) or (vals[i] <= 0.5 and vals[i+1] >= 0.5):
            idx = i
            break
    if idx is None:
        return {"n": float(df["n"].iloc[0]), "Nk_est": np.nan, "Nk_note": "no bracket (Phi did not cross 0.5 in provided N set)"}
    # linear interpolation in N between means
    N0, N1 = float(Ns[idx]), float(Ns[idx+1])
    y0, y1 = float(vals[idx]), float(vals[idx+1])
    if y1 == y0:
        Nk = (N0+N1)/2
    else:
        Nk = N0 + (0.5 - y0) * (N1 - N0) / (y1 - y0)
    return {
        "n": float(df["n"].iloc[0]),
        "Nk_est": float(Nk),
        "N_bracket_low": int(N0),
        "N_bracket_high": int(N1),
        "phi_low": y0,
        "phi_high": y1,
        "phi_low_sd": float(g.loc[g["N"]==Ns[idx], "std"].iloc[0]),
        "phi_high_sd": float(g.loc[g["N"]==Ns[idx+1], "std"].iloc[0]),
        "Nk_note": "linear interp of Phi means"
    }

def estimate_lock_for_n(df: pd.DataFrame, p_thresh: float = 0.05) -> dict:
    # estimate N where p_lowQ falls below p_thresh
    df = df.copy()
    df["lowQ"] = (df["Q_clock"] < 3.0).astype(int)
    g = df.groupby("N")["lowQ"].mean().reset_index().sort_values("N")
    Ns = g["N"].values
    ps = g["lowQ"].values
    # find first N where <= p_thresh
    idx = None
    for i in range(len(ps)):
        if ps[i] <= p_thresh:
            idx = i
            break
    if idx is None:
        return {"n": float(df["n"].iloc[0]), "Nl_est": np.nan, "Nl_note": f"no N with p_lowQ <= {p_thresh}"}
    return {"n": float(df["n"].iloc[0]), "Nl_est": float(Ns[idx]), "p_lowQ_at_Nl": float(ps[idx]), "Nl_note": f"first N with p_lowQ <= {p_thresh}"}

def main():
    files = sorted(glob.glob(str(CSV_DIR / "vii_c_*_run_summary.csv")))
    if not files:
        raise SystemExit(f"No run_summary CSVs found in {CSV_DIR}.")
    df_all = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    # Ensure numeric
    df_all["n"] = df_all["n"].astype(float)
    out_k = []
    out_l = []
    for nval, sub in df_all.groupby("n"):
        out_k.append(estimate_knee_for_n(sub))
        out_l.append(estimate_lock_for_n(sub))
    knee = pd.DataFrame(out_k).sort_values("n")
    lock = pd.DataFrame(out_l).sort_values("n")
    knee.to_csv(CSV_DIR/"knee_contour_summary.csv", index=False)
    lock.to_csv(CSV_DIR/"lock_contour_summary.csv", index=False)
    print("[ok] wrote", CSV_DIR/"knee_contour_summary.csv")
    print("[ok] wrote", CSV_DIR/"lock_contour_summary.csv")

if __name__ == "__main__":
    main()
