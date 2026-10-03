"""Four paired protection-position / conflict-ranking experiments on the original 3090 project."""
import argparse
import datetime
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import shlex
import subprocess
import sys
import time

from analyze_protect_position import summarize


ROOT = Path(__file__).resolve().parents[1]
SPEC = ROOT / 'scripts/sweeps/imgr10_protect_position_3090.json'
MODES = ('A', 'B', 'C', 'D')


def settings_for(mode, smoke=False):
    spec = json.loads(SPEC.read_text())
    settings = dict(spec['common_overrides'])
    settings.update(next(row['overrides'] for row in spec['variants'] if row['name'] == mode))
    if smoke:
        settings.update(max_tasks=2, init_epoch=1, epochs=1, ca_epochs=1,
                        wandb_mode='offline', stage_audit=True)
    return settings


def command_for(mode, directory, smoke=False):
    settings = settings_for(mode, smoke)
    settings['prefix'] = 'imgr10_protect_position_' + mode + '_' + directory.parent.name + '_' + directory.name
    command = [sys.executable, 'main.py', '--config', 'exps/dlora/imgr10.json']
    for key, value in settings.items():
        command += ['--set', key + '=' + (value if isinstance(value, str) else json.dumps(value))]
    return command, settings


def run(mode, directory, revision, smoke=False, dry_run=False):
    command, settings = command_for(mode, directory, smoke)
    print('Starting', mode, 'GPU SMOKE, NOT PERFORMANCE' if smoke else 'REAL FULL T10', flush=True)
    print('Command:', shlex.join(command), flush=True)
    if dry_run:
        return dict(mode=mode, exit_code=0, minutes=0)
    directory.mkdir(parents=True)
    effective = json.loads((ROOT / 'exps/dlora/imgr10.json').read_text())
    effective.update(settings)
    paths = ['main.py', 'trainer.py', 'scripts/run_protect_position.py',
             'scripts/analyze_protect_position.py', str(SPEC.relative_to(ROOT))]
    paths += sorted(str(path.relative_to(ROOT)) for folder in ('models', 'methods', 'utils')
                    for path in (ROOT / folder).rglob('*.py'))
    hardware = dict(hostname=platform.node(), cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),
        gpus=subprocess.check_output(['nvidia-smi', '--query-gpu=index,uuid,name,pci.bus_id,driver_version',
                                     '--format=csv,noheader'], text=True).strip().splitlines())
    (directory / 'run.json').write_text(json.dumps(dict(code_revision=revision,
        effective_config=effective, command=command, machine='3090', phase='smoke' if smoke else 'formal',
        hardware=hardware,
        source_sha256={p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in paths},
        software=dict(python=sys.version, platform=platform.platform(),
                      packages={p: importlib.metadata.version(p) for p in ('torch', 'torchvision', 'timm', 'numpy')})), indent=2) + '\n')
    started = time.monotonic()
    with (directory / 'training.log').open('w') as stream:
        with subprocess.Popen(command, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                              text=True, bufsize=1) as process:
            for line in process.stdout:
                stream.write(line)
                stream.flush()
                print(line, end='', flush=True)
            code = process.wait()
    record = dict(mode=mode, exit_code=code, minutes=(time.monotonic() - started) / 60)
    print('Finished:', json.dumps(record), flush=True)
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=('run', 'smoke', 't10', 'dry-run'), default='run')
    parser.add_argument('--only', nargs='+', choices=MODES)
    args = parser.parse_args()
    spec = json.loads(SPEC.read_text())
    modes = args.only or list(MODES)
    directory = ROOT / spec['log_dir'] / datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    revision = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    print('Code revision:', revision, '\nOutputs:', directory, flush=True)
    print('3090; ImageNet-R T10 seed1993; anchor2.5; 20 epochs; CA5; math SDPA. '
          'A once, then B/C/D. Task0 unchanged; fixed row/column permutations from Task1; '
          'no saved checkpoints. Equal coverage does NOT mean equal removed update norm.', flush=True)
    dry_run = args.mode == 'dry-run'
    if not dry_run:
        directory.mkdir(parents=True)
        (directory / 'manifest.json').write_text(json.dumps(dict(revision=revision, modes=modes,
            options=vars(args), baseline_reuse='No matching source/position telemetry; A is run once.'), indent=2) + '\n')
    if args.mode in ('run', 'smoke', 'dry-run'):
        smoke_records = []
        for mode in modes:
            record = run(mode, directory / ('smoke_' + mode), revision, True, dry_run)
            smoke_records.append(record)
            if not dry_run:
                (directory / 'smoke_queue.json').write_text(json.dumps(smoke_records, indent=2) + '\n')
            if record['exit_code']:
                print('Smoke failed; formal runs not started.', flush=True)
                return record['exit_code']
        if args.mode == 'smoke':
            return 0
    records = []
    for mode in modes:
        record = run(mode, directory / mode, revision, dry_run=dry_run)
        records.append(record)
        if not dry_run:
            (directory / 'queue.json').write_text(json.dumps(records, indent=2) + '\n')
            summarize(directory, records)
        if record['exit_code']:
            return record['exit_code']
    print('Queue finished:', directory, flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
