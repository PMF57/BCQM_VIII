# configs_viii

Stage-3 run configurations for BCQM_VIII live here.

Conventions:
- One config file per run family (e.g. hero run, seed sweep, knee bracket).
- Filenames include the key parameters (n, N set, bins, seeds, logging flags).
- Each run folder under `runs_viii/` should contain a copy of the exact config used.

Stage-3 logging:
- `stage3_trace_enabled` must be explicit.
- Island/turnover threshold policy must be explicit (fraction or count).