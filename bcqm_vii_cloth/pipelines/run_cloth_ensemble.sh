#!/bin/bash
set -euo pipefail
python3 -m bcqm_vii_cloth.cli run --config configs_stage2/ensemble_cloth_W100_N4N8_n0p4_0p8.yml
