#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
# Reuse the exact preceding 3090 baseline recipe, not machine data paths.
python3 - "$@" <<'PY'
import datetime
import json
import subprocess
import sys

spec = json.load(open('scripts/sweeps/imgr10_head_balance_3090.json'))
settings = dict(spec['common_overrides'])
stamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
settings.update(seed=[1993], head_balance_weight=0, dual_mask_anchor_reg_weight=10,
                history_audit=True, prefix='imgr10_history_audit_3090_' + stamp,
                history_audit_dir='logs/history_audit/3090_' + stamp,
                wandb_group='imgr10_history_audit_3090')
if '--smoke' in sys.argv:
    settings.update(max_tasks=2, init_epoch=1, epochs=1, ca_epochs=1, wandb_mode='offline')
command = [sys.executable, 'main.py', '--config', 'exps/dlora/imgr10.json']
for key, value in settings.items():
    command += ['--set', key + '=' + (value if isinstance(value, str) else json.dumps(value))]
print('Code revision:', subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD'], text=True).strip(), flush=True)
print(' '.join(command), flush=True)
if '--dry-run' not in sys.argv:
    sys.exit(subprocess.call(command))
PY
