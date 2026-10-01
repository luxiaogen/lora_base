#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

case "${1:---run}" in
    --run|--smoke|--t10|--dry-run) ;;
    *) echo "Usage: bash $0 [--run|--smoke|--t10|--dry-run]" >&2; exit 2 ;;
esac
if (( $# > 1 )); then
    echo "Supply only one mode." >&2
    exit 2
fi

python3 - "${1:---run}" <<'PY'
import datetime
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import sys

mode = sys.argv[1]
spec_path = Path('scripts/sweeps/imgr10_two_expert_oracle_3090.json')
spec = json.loads(spec_path.read_text())
stamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f')
directory = Path(spec['log_dir']) / stamp
revision = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()
print('Code revision:', revision, flush=True)
print('Read-only oracle uses true class labels; legal signals use neither labels nor task ID.', flush=True)
print('No old training images, checkpoints, or saved dense features. One smoke + one T10.', flush=True)
stages = ['smoke'] if mode == '--smoke' else ['t10'] if mode == '--t10' else ['smoke', 't10']
for stage in stages:
    settings = dict(spec['common_overrides'])
    settings.update(spec['variants'][0]['overrides'])
    settings.update(seed=spec['seeds'], prefix=f'{spec["name"]}_{stage}_{stamp}',
                    two_expert_oracle_dir=str(directory / stage / 'diagnostics'))
    if stage == 'smoke':
        settings.update(max_tasks=2, init_epoch=1, epochs=1, ca_epochs=1, wandb_mode='offline')
    command = [sys.executable, 'main.py', '--config', spec['datasets'][0]['config']]
    for key, value in settings.items():
        command += ['--set', key + '=' + (value if isinstance(value, str) else json.dumps(value))]
    print('Stage:', stage, flush=True)
    print('Command:', shlex.join(command), flush=True)
    print('Diagnostics:', settings['two_expert_oracle_dir'], flush=True)
    if mode == '--dry-run':
        continue
    stage_dir = directory / stage
    stage_dir.mkdir(parents=True, exist_ok=True)
    effective = json.loads(Path(spec['datasets'][0]['config']).read_text())
    effective.update(settings)
    metadata = dict(code_revision=revision, command=command, effective_config=effective,
                    spec_sha256=hashlib.sha256(spec_path.read_bytes()).hexdigest(),
                    source_sha256={path: hashlib.sha256(Path(path).read_bytes()).hexdigest()
                                   for path in ('methods/dlora.py', 'utils/two_expert_oracle.py')})
    (stage_dir / 'run.json').write_text(json.dumps(metadata, indent=2) + '\n')
    with (stage_dir / 'training.log').open('w') as stream:
        with subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                              text=True, bufsize=1) as process:
            for line in process.stdout:
                stream.write(line)
                stream.flush()
                print(line, end='', flush=True)
            status = process.wait()
    print('Stage:', stage, 'Exit code:', status, flush=True)
    if status:
        sys.exit(status)
    status = subprocess.run([sys.executable, 'scripts/analyze_two_expert_oracle.py',
                             settings['two_expert_oracle_dir'], '--expected-tasks',
                             str(settings['max_tasks'])]).returncode
    if status:
        sys.exit(status)
print('Finished. Logs and compact diagnostics:', directory, flush=True)
PY
