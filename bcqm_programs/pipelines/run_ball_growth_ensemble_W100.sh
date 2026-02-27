#!/bin/bash
set -euo pipefail

TS=$(date +"%Y%m%d_%H%M%S")
ANADIR="outputs/analysis"
mkdir -p "$ANADIR"
OUTFILE="$ANADIR/${TS}_ball_growth_ensemble_W100.txt"

echo "BCQM VI ball-growth ensemble (W=100; N={4,8}; n={0.4,0.8}; 5 seeds) — $(date)" | tee "$OUTFILE"
echo "" | tee -a "$OUTFILE"

run_cfg () {
  local cfg="$1"
  echo "== RUN: $cfg ==" | tee -a "$OUTFILE"
  python3 -m bcqm_vi_spacetime.cli run --config "$cfg" 2>&1 | tee -a "$OUTFILE"
  echo "" | tee -a "$OUTFILE"
}

run_cfg "configs/generated_vreg_C5_subset/ball_growth_ens_C5_W100_N8_n0p4.yml"
run_cfg "configs/generated_vreg_C5_subset/ball_growth_ens_C5_W100_N8_n0p8.yml"
run_cfg "configs/generated_vreg_C5_subset/ball_growth_ens_C5_W100_N4_n0p4.yml"
run_cfg "configs/generated_vreg_C5_subset/ball_growth_ens_C5_W100_N4_n0p8.yml"

echo "== SUMMARY ==" | tee -a "$OUTFILE"
python3 bcqm_vi_spacetime/analysis/ball_growth_ensemble_summary.py 2>&1 | tee -a "$OUTFILE"
echo "" | tee -a "$OUTFILE"
echo "DONE. Wrote: $OUTFILE" | tee -a "$OUTFILE"
