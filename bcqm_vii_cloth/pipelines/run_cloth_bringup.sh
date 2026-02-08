#!/bin/bash
set -euo pipefail
python3 -m bcqm_vii_cloth.cli run --config configs_stage2/bringup_cloth_W100_N8_n0p8.yml
