#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

mode="${1:---t3}"
case "$mode" in
    --smoke|--t3|--t10|--dry-run) ;;
    *) echo "Usage: bash $0 [--smoke|--t3|--t10|--dry-run]" >&2; exit 2 ;;
esac

python3 - "$mode" <<'PY'
import datetime
import json
from pathlib import Path
import shlex
import subprocess
import sys

mode = sys.argv[1]
spec = json.loads(Path('scripts/sweeps/imgr10_anchor2p5_save_t10_3090.json').read_text())
stamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f')
settings = dict(spec['common_overrides'])
settings.update(spec['variants'][0]['overrides'])
settings.update(seed=[1993],
                prefix=f'imgr10_late_average_3090_{mode[2:]}_{stamp}',
                max_tasks=10 if mode == '--t10' else 3,
                late_weight_average_epochs=5,
                wandb_group='imgr10_late_average_3090')
if mode == '--smoke':
    settings.update(max_tasks=2, init_epoch=1, epochs=2,
                    late_weight_average_epochs=2, ca_epochs=1, wandb_mode='offline')

command = [sys.executable, 'main.py', '--config', spec['datasets'][0]['config']]
for key, value in settings.items():
    command += ['--set', key + '=' + (value if isinstance(value, str) else json.dumps(value))]
print('Code revision:', subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(), flush=True)
print('Mode:', mode, 'Task0 unchanged; Task1+ final-five-epoch S/P B and current head average.', flush=True)
print('Command:', shlex.join(command), flush=True)
if mode != '--dry-run':
    directory = Path('logs/shell_logs/imgr10_late_average_3090') / stamp
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / 'candidate.log'
    with path.open('w') as stream:
        with subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                              text=True, bufsize=1) as process:
            for line in process.stdout:
                stream.write(line)
                stream.flush()
                print(line, end='', flush=True)
            status = process.wait()
    print('Log:', path, 'Exit code:', status, flush=True)
    sys.exit(status)
PY
