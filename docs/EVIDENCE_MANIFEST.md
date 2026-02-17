# Evidence Manifest — BCQM VII_b (finite‑N knee + clock metastability sweeps)

_Last updated: 9 February 2026_

The paper PDF and official versioned manuscript are archived on Zenodo (Concept DOI: 10.5281/zenodo.18549145; v1 DOI: 10.5281/zenodo.18549146). This repository contains the code, configs, CSV artefacts, and lab notes supporting the paper.

GitHub release : https://github.com/PMF57/BCQM_VII_b/releases/tag/v1.0.0

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
  - `sweep_minconc3_N22_28_n0p8_bins20_wstar0p30_light.yml` (control: min_concurrency=3 extraction-definition check)
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
- Control (min_concurrency=3): `minconc3_n0p8_bins20_wstar030_run_summary.csv` and `minconc3_n0p8_bins20_wstar030_ballgrowth_pairwise_N*.csv`.
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

## BCQM_VII_c — Knee contour sweeps (Stage-3)

This section documents the planned (and then executed) **knee contour** campaign: mapping the cloth knee
N_k(n) defined by **Phi = 0.5** (core fraction crossing) as a function of coherence **n**, and (optionally)
a companion “lock contour” N_l(n) defined by a chosen threshold on **p_lowQ**.

Run log: RUN_LOG_2026-02-17_VII_c_knee_contour.md (root)
Outputs: csv/stage3_knee_contour/knee_contour_summary.csv,
 csv/stage3_knee_contour/lock_contour_summary.csv,
 and PDFs in figures/stage3_knee_contour/

### Repo folders (canonical)
- Configs: `configs_stage3_knee_contour/`
- Outputs (consolidated CSV artefacts): `csv/stage3_knee_contour/`
- Figures (optional): `figures/stage3_knee_contour/`
- Run logs: `provenance/RUN_LOG_YYYY-MM-DD_VII_c_knee_contour.md`

### Local working folders (your Desktop)
You mentioned using:
- `configs_stage3_knee_counter/` (local Desktop working folder)
- `csv_s3kc/` (local Desktop working folder)

After each batch run:
1) copy the YAML configs into the repo folder `configs_stage3_knee_contour/`
2) copy consolidated CSV outputs into `csv/stage3_knee_contour/`
3) add/update a run log under `provenance/`
4) update the checklist below.

### Checklist (fill in as you go)
**Scaffold commit**
- [ ] Added repo folders `configs_stage3_knee_contour/` and `csv/stage3_knee_contour/`
- [ ] Added `docs/stage3_knee_contour_README.md`
- [ ] Added/updated this section

**Runs executed**
- [ ] n-values covered: (list)
- [ ] N bracketing strategy used: (describe)
- [ ] Seed counts per point: (e.g. 10 for bracket, 20 for refine)

**Artefacts committed (CSV)**
- [ ] `csv/stage3_knee_contour/*_run_summary.csv`
- [ ] `csv/stage3_knee_contour/knee_contour_summary.csv` (N_k vs n, uncertainty)
- [ ] `csv/stage3_knee_contour/lock_contour_summary.csv` (optional)
- [ ] `csv/stage3_knee_contour/README.md` (optional index)

**Run logs**
- [ ] `provenance/RUN_LOG_YYYY-MM-DD_VII_c_knee_contour.md` (at least one)

**Figures (optional)**
- [ ] `figures/stage3_knee_contour/knee_contour.pdf`
- [ ] `figures/stage3_knee_contour/lock_contour.pdf`

### Notes
- Do **not** commit raw `outputs_cloth/` directories.
- Commit only configs, consolidated CSV artefacts, small figures, and run logs.
- Keep BCQM VII and BCQM_VII_b results locked; VII_c is a new Stage-3 evidence set.
