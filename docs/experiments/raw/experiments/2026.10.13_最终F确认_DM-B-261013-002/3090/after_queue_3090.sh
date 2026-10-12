#!/usr/bin/env bash
set -euo pipefail
cd /home/shengqin/lys/baseline/LoDA_ICML2026
PYTHON=/home/shengqin/anaconda3/envs/ICML2026_LoDA/bin/python
REVISION=d9cb728476259881c7dd9a8039f01cb03953496f
OLD_QUEUE=logs/shell_logs/depth_permissions_3090/20261013_004924_345406
"$PYTHON" - "$OLD_QUEUE" <<'PY'
import hashlib
import json
import math
from pathlib import Path
import re
import sys
sys.path.insert(0, 'scripts')
from analyze_core_evidence import METRICS, read_run
root = Path(sys.argv[1])
manifest = json.loads((root / 'manifest.json').read_text())
expected = ['imgr10_' + name + '_seed1993' for name in 'ABCD']
assert manifest['queue_pid'] == 1903745 and manifest['order'] == expected
assert manifest['revision'] == 'fa53a839d4a8df7edd28afdf230b577b548179b6'
records = json.loads((root / 'queue.json').read_text())
assert [r['mode'] for r in records] == expected
for record in records:
    assert record['status'] == 'completed' and record['exit_code'] == 0, record
    measured, snapshot, _ = read_run(root, record)
    assert measured['valid_performance'] and measured['epoch_records_complete'], measured
    assert all(math.isfinite(measured[k]) for k in METRICS)
    assert snapshot['code_revision'] == manifest['revision'] and snapshot['phase'] == 'formal'
    assert snapshot['machine'] == '3090'
    assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest() == digest
               for p, digest in snapshot['source_sha256'].items())
    content = (root / record['mode'] / 'training.log').read_text()
    epochs = {(int(t), int(e)) for t, e in re.findall(r'LoRA learning rates: task=(\d+), epoch=(\d+)', content)}
    assert epochs == {(t, e) for t in range(10) for e in range(1, 21)}
print('Old four-group queue: full T10 health and fingerprints verified.', flush=True)
PY
"$PYTHON" - <<'PY'
import subprocess
apps = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid,process_name',
                                '--format=csv,noheader'], text=True).strip()
assert not apps, 'GPU is still occupied; continuation paused: ' + apps
PY
test "$(git rev-parse HEAD)" = fa53a839d4a8df7edd28afdf230b577b548179b6
git diff --quiet -- models methods utils scripts main.py trainer.py test
git fetch origin codex/mask-budget-comparison-20260924
git merge --ff-only "$REVISION"
test "$(git rev-parse HEAD)" = "$REVISION"
export PATH=/home/shengqin/anaconda3/envs/ICML2026_LoDA/bin:/usr/local/bin:/usr/bin:/bin
CUDA_VISIBLE_DEVICES='' "$PYTHON" -m unittest discover -s test -q
export LODA_PYTHON="$PYTHON"
exec bash scripts/10_13_final_f_validation_3090.sh
