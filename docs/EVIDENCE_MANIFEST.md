# Evidence Manifest — BCQM VII_b (finite‑N knee + clock metastability sweeps)

_Last updated: 6 February 2026_

This manifest indexes the sweep artefacts supporting the Stage‑3 “knee + metastability” result set.

## 1) Configs (inputs)
All configs are in `sweep_test_configs/`.

Key families:
- Broad sweep:
  - `sweep_N1_128_hits1_x10_bins20_n0p8_light.yml`
  - `sweep_N32_128_hits1_x10_bins20_n0p8_heavy.yml`
- Knee localisation:
  - `sweep_N12_48_hits1_x10_bins20_n0p8_heavy.yml`
  - `sweep_N21_24_hits1_x10_bins20_n0p8_heavy.yml`
- 20‑seed metastability confirmations:
  - `sweep_N22_hits1_x10_bins20_n0p8_20seeds_heavy.yml`
  - `sweep_N23_hits1_x10_bins20_n0p8_20seeds_heavy.yml`
  - `sweep_N24_hits1_x10_bins20_n0p8_20seeds_heavy.yml`
  - `sweep_N28_hits1_x10_bins20_n0p8_20seeds_heavy.yml`
- Robustness:
  - `sweep_bins10_N22_28_n0p8_wstar0p30_light.yml`
  - `sweep_bins20_N22_28_n0p7_wstar0p30_light.yml`
  - `sweep_bins20_N22_28_n0p8_wstar0p25_light.yml`
  - `sweep_bins20_N22_28_n0p8_wstar0p35_light.yml`
  - (optional) `sweep_bins20_N22_28_n0p9_wstar0p30_light.yml`
  - (optional) `sweep_n0p7_knee_smallN_bins20_wstar0p30_light.yml`

## 2) Local run outputs (not committed)
Raw run outputs are stored locally under:
- `sweep_test_output/outputs_cloth/<experiment_id>/`
and contain `RUN_METRICS_*.json` files.

These are intentionally not committed to keep the repo small.

## 3) CSV artefacts (outputs)
All committed CSVs are in `csv/`.

Types:
- `*_run_summary.csv` — per‑run scalars (Phi, Q_clock, counts, etc.)
- `*_ballgrowth_pairwise.csv` — global pairwise dL2 with N_a/N_b (patched format)
- `*_ballgrowth_pairwise_N<NN>_n<nn>.csv` — within‑N pairwise dL2 (190 rows for 20 seeds)

Key compiled comparisons:
- `robustness_all_conditions_summary_N22_28.csv`
- `robustness_phi_pivot.csv`
- `robustness_plowQ_pivot.csv`

## 4) Notes / documentation
- `docs/BCQM_lab_note_sweeps_knee_metastability_2026-02-06_v0.1.tex`
- `docs/BCQM_lab_note_candidate_IV_b_sweeps_and_pipeline_2026-02-06_v0.1.tex`
- `docs/BCQM_robustness_panel_note_v0.2_2026-02-06.tex` (and PDF)

## 5) Rebuild recipe (minimal)
Run a config:
```bash
python3 -m bcqm_vii_cloth.cli scan --config sweep_test_configs/<CONFIG>.yml
```

Summarise:
```bash
python3 bcqm_vii_cloth/analysis/summarise_runs.py \
  --run_dir sweep_test_output/outputs_cloth/<RUN_DIR> \
  --out_dir csv --tag <TAG>
```

Ensure `summarise_runs.py` is the patched version described in `PROVENANCE.md` if you want per‑N pairwise files.
