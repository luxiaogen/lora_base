#!/usr/bin/env bash
set -uo pipefail
# CUB: 9 independent full ten-task runs.
exec bash scripts/9_14_lora_inherit_branches.sh cub10
