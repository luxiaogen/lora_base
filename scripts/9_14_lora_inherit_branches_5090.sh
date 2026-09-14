#!/usr/bin/env bash
set -uo pipefail
# ImageNet-R and ImageNet-A: 18 independent full ten-task runs.
exec bash scripts/9_14_lora_inherit_branches.sh imgr10 imga10
