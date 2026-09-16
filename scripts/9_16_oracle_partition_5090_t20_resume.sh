#!/usr/bin/env bash
set -uo pipefail

export ORACLE_PARTITION_SKIP_T10=1
exec bash scripts/9_15_oracle_partition_5090_overnight.sh
