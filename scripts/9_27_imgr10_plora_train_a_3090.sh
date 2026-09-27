#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
python3 - "$@" <<'PY'
import datetime
import json
import subprocess
import sys

spec = json.load(open('scripts/sweeps/imgr10_plora_train_a_3090.json'))
stamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
print('Code revision:', subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD'], text=True).strip(), flush=True)
print(subprocess.check_output(['git', 'status', '--short'], text=True), flush=True)
failed = 0
for variant in spec['variants']:
    settings = dict(spec['common_overrides'])
    settings.update(variant['overrides'])
    settings.update(seed=[1993], prefix='imgr10_plora_train_a_' + variant['name'] + '_seed1993_' + stamp)
    if '--smoke' in sys.argv:
        settings.update(max_tasks=2, init_epoch=1, epochs=1, ca_epochs=1, wandb_mode='offline')
    command = [sys.executable, 'main.py', '--config', 'exps/dlora/imgr10.json']
    for key, value in settings.items():
        command += ['--set', key + '=' + (value if isinstance(value, str) else json.dumps(value))]
    print('Starting:', variant['name'], variant['description'], flush=True)
    print(' '.join(command), flush=True)
    if '--dry-run' not in sys.argv:
        result = subprocess.call(command)
        print('Finished:', variant['name'], 'exit_code=', result, flush=True)
        failed = failed or result
sys.exit(failed)
PY
