#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
python3 - "$@" <<'PY'
import datetime
import json
from pathlib import Path
import shlex
import subprocess
import sys

spec = json.loads(Path('scripts/sweeps/imgr10_anchor2p5_t10_3090.json').read_text())
stamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f')
directory = Path(spec['log_dir']) / stamp
print('Code revision:', subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(), flush=True)
print(subprocess.check_output(['git', 'status', '--short'], text=True), flush=True)
print('Full training data; official test; Task0–9. One anchor2.5 candidate; no screening or baseline rerun.', flush=True)
failed = 0
for variant in spec['variants']:
    settings = dict(spec['common_overrides'])
    settings.update(variant['overrides'])
    settings.update(seed=[1993], prefix=spec['name'] + '_' + variant['name'] + '_' + stamp)
    if '--smoke' in sys.argv:
        settings.update(max_tasks=2, init_epoch=1, epochs=1, ca_epochs=1, wandb_mode='offline')
    command = [sys.executable, 'main.py', '--config', spec['datasets'][0]['config']]
    for key, value in settings.items():
        command += ['--set', key + '=' + (value if isinstance(value, str) else json.dumps(value))]
    print('Starting:', variant['name'], variant['description'], flush=True)
    print(shlex.join(command), flush=True)
    if '--dry-run' not in sys.argv:
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / (variant['name'] + '.log')
        with path.open('w') as stream:
            stream.write('Command: ' + shlex.join(command) + '\n')
            with subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                  text=True, bufsize=1) as process:
                for line in process.stdout:
                    stream.write(line)
                    stream.flush()
                    print(line, end='', flush=True)
                result = process.wait()
            stream.write(f'ExitCode: {result}\n')
        print('Finished:', variant['name'], 'exit_code=', result, 'log=', path, flush=True)
        failed = failed or result
sys.exit(failed)
PY
