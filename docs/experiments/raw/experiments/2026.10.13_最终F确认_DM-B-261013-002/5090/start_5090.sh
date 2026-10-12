#!/usr/bin/env bash
set -euo pipefail
cd /mnt/disk1/lys/CIL/code/baseline/lora_base
test "$(git rev-parse HEAD)" = d9cb728476259881c7dd9a8039f01cb03953496f
git diff --quiet -- models methods utils scripts main.py trainer.py test
export PATH=/mnt/disk1/envs/lys_cil/bin:/usr/local/bin:/usr/bin:/bin
export LODA_PYTHON=/mnt/disk1/envs/lys_cil/bin/python
"$LODA_PYTHON" - <<'PY'
import subprocess
apps = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid,process_name',
                                '--format=csv,noheader'], text=True).splitlines()
busy = [line for line in apps if 'gnome-remote-desktop-daemon' not in line]
assert not busy, 'GPU is still occupied; start paused: ' + repr(busy)
PY
exec bash scripts/10_13_final_f_validation_5090.sh
