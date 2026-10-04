"""Two real T10 queues: P permission feedback and task-control simplification."""
import argparse
import datetime
import json
import os
from pathlib import Path
import subprocess
import sys

import run_core_evidence_night as engine
import analyze_permission_release as analysis


ROOT = Path(__file__).resolve().parents[1]
SPEC = ROOT / 'scripts/sweeps/imgr10_permission_release_night.json'
BASE = ROOT / 'scripts/sweeps/imgr10_protect_position_3090.json'


def variants(machine):
    return json.loads(SPEC.read_text())[machine]


def settings_for(machine, name, smoke=False):
    spec = json.loads(SPEC.read_text())
    settings = dict(json.loads(BASE.read_text())['common_overrides'])
    settings.update(spec['common_overrides'])
    settings.update(next(row['overrides'] for row in spec[machine] if row['name'] == name))
    settings['wandb_group'] = 'imgr10_permission_release_' + machine
    if smoke:
        settings.update(max_tasks=2, init_epoch=1, epochs=2, ca_epochs=1,
                        wandb_mode='offline', stage_audit=True)
    return settings


def command_for(machine, name, directory, smoke=False):
    settings = settings_for(machine, name, smoke)
    settings['prefix'] = 'imgr10_release_' + machine + '_' + name + '_' + directory.parent.name
    command = [sys.executable, 'main.py', '--config', 'exps/dlora/imgr10.json']
    for key, value in settings.items():
        command += ['--set', key + '=' + (value if isinstance(value, str) else json.dumps(value))]
    return command, settings


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--machine', choices=('3090', '5090'), required=True)
    parser.add_argument('--mode', choices=('run', 'smoke', 't10', 'dry-run'), default='run')
    parser.add_argument('--hours', type=float, default=10)
    args = parser.parse_args()
    modes = [row['name'] for row in variants(args.machine)]
    revision = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    directory = ROOT / ('logs/shell_logs/imgr10_permission_release_' + args.machine) / datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    print('Code revision:', revision, '\nOutputs:', directory, '\nQueue PID:', os.getpid(), flush=True)
    print('Order:', modes, '; ImageNet-R T10 seed1993 anchor2.5 20 epochs CA5 math-SDPA; no checkpoints.', flush=True)
    if args.mode != 'dry-run':
        directory.mkdir(parents=True)
        (directory / 'manifest.json').write_text(json.dumps(dict(revision=revision,
            machine=args.machine, queue_pid=os.getpid(), modes=modes, options=vars(args)), indent=2) + '\n')
    engine.SPEC = SPEC
    engine.settings_for = settings_for
    engine.command_for = command_for
    engine.summarize = analysis.summarize
    engine.EXTRA_SOURCE_PATHS = ['scripts/run_permission_release_night.py',
                                'scripts/analyze_permission_release.py']
    return engine.execute_queue(args.machine, modes, directory, revision, args.mode, args.hours)


if __name__ == '__main__':
    raise SystemExit(main())
