#!/usr/bin/env bash
set -uo pipefail

export LORI_CD_RUN_MODE=full
export LORI_CD_TASKS=10
export LORI_CD_CALIBRATION_EPOCHS=2
export LORI_CD_SPARSE_EPOCHS=20
export LORI_CD_CA_EPOCHS=5

bash scripts/9_21_imgr10_lori_s_cd_smoke.sh
