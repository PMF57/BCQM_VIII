# BCQM VII_b — Finite‑N knee and clock metastability sweeps (Stage‑3 / candidate “BCQM_IV_b” note)

This repository collects the **post‑BCQM VII** sweep campaign investigating a sharp finite‑N crossover (“knee”) in the Stage‑2 cloth and a **probabilistic clock/ordering metastability band** near that crossover.

BCQM VII (paper) is **published and locked**; this repo is a **new workstream** intended for Stage‑3 analysis and a potential short standalone note (suggested label: **BCQM_IV_b**).

## Key results captured here (high level)
- A narrow finite‑N crossover band near **N ≈ 22–24** (at n=0.8) where the core fraction \(\Phi=N_{core}/N_{total}\) drops through 0.5.
- A **mixed/metastable clock band** near the same region, quantified by the fraction of seeds with low clock quality (e.g. \(Q_{clock}<3\)).
- Robustness checks across **binning**, **coherence n**, and **space strength w\***.

## Relationship to BCQM VII
This repo starts from the BCQM VII codebase (copied into `bcqm_vii_cloth/`). The upstream BCQM VII artefacts are:
- Paper DOI (v1): **10.5281/zenodo.18497217**
- Paper concept DOI: **10.5281/zenodo.18497216**
- Code archive DOI (v1): **10.5281/zenodo.18505934** (6 Feb 2026)

## Repository layout
- `bcqm_vii_cloth/` — simulation + analysis code (copied from BCQM VII baseline)
- `sweep_test_configs/` — YAML configs used for the sweep campaign
- `csv/` — consolidated run summaries and ball‑growth pairwise stability CSVs
- `docs/` — lab notes / robustness panel notes and evidence manifest
- `sweep_test_results/` — plots and intermediate artefacts (optional)

Raw run outputs (`outputs_cloth/`) are not committed here.

## How to reproduce the CSV artefacts
### Run a sweep
From repo root (example):
```bash
python3 -m bcqm_vii_cloth.cli scan --config sweep_test_configs/<CONFIG>.yml
```

### Summarise to CSV
```bash
python3 bcqm_vii_cloth/analysis/summarise_runs.py \
  --run_dir sweep_test_output/outputs_cloth/<RUN_DIR> \
  --out_dir csv --tag <TAG>
```

> Note: for publication‑grade per‑N ball‑growth stability, `summarise_runs.py` should be the **patched** version that writes
> per‑(N,n) pairwise files. See `PROVENANCE.md` for the required script versions.

## Start here
- `docs/EVIDENCE_MANIFEST.md` — authoritative mapping: runs/configs → CSVs → figures/notes.
- `docs/BCQM_lab_note_candidate_IV_b_sweeps_and_pipeline_2026-02-06_v0.1.tex` — full sweep log + interpretation.
- `docs/BCQM_robustness_panel_note_v0.2_2026-02-06.tex` — one‑page robustness panel.


## BCQM VII_b paper (this work)
The paper PDF and official versioned manuscript are archived on Zenodo. This repository contains the code, configs, CSV artefacts, and lab notes supporting the paper.
- Concept DOI: 10.5281/zenodo.18549145
- Version 1 DOI: 10.5281/zenodo.18549146

## BCQM_VII_c — knee contour (Stage-3)
A Stage-3 sweep campaign mapped the cloth knee contour **Nk(n)** (where Phi crosses 0.5) and an operational lock contour **Nl(n)** (first N where p_lowQ <= 0.05, where defined).
	•	Code/archive DOI (VII_c v1.0.0): 10.5281/zenodo.18671297

Start here:
- Evidence index: `docs/EVIDENCE_MANIFEST.md` (VII_c section)
- Run log: `RUN_LOG_2026-02-17_VII_c_knee_contour.md`

Artefacts:
- Configs: `configs_stage3_knee_contour/`
- CSVs: `csv/stage3_knee_contour/`
- Plots: `figures/stage3_knee_contour/`
- Fitter: `tools/fit_knee_contour.py`

Stage-3 VII_c knee contour: see docs/EVIDENCE_MANIFEST.md (VII_c section)


