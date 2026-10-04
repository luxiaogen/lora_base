"""Real T10 queues for the protection, permissions, and simplification evidence."""
import argparse
import datetime
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import shlex
import subprocess
import sys
import time

from analyze_core_evidence import METRICS, read_run, summarize


ROOT = Path(__file__).resolve().parents[1]
SPEC = ROOT / 'scripts/sweeps/imgr10_core_evidence_night.json'
BASE = ROOT / 'scripts/sweeps/imgr10_protect_position_3090.json'
EXTRA_SOURCE_PATHS = []


def variants(machine):
    return json.loads(SPEC.read_text())[machine]


def settings_for(machine, name, smoke=False):
    spec = json.loads(SPEC.read_text())
    settings = dict(json.loads(BASE.read_text())['common_overrides'])
    settings.update(spec['common_overrides'])
    settings.update(next(row['overrides'] for row in spec[machine] if row['name'] == name))
    settings['wandb_group'] = 'imgr10_core_evidence_' + machine
    if smoke:
        settings.update(max_tasks=2, init_epoch=1, epochs=1, ca_epochs=1,
                        wandb_mode='offline', stage_audit=True)
    return settings


def command_for(machine, name, directory, smoke=False):
    settings = settings_for(machine, name, smoke)
    settings['prefix'] = 'imgr10_core_' + machine + '_' + name + '_' + directory.parent.name
    command = [sys.executable, 'main.py', '--config', 'exps/dlora/imgr10.json']
    for key, value in settings.items():
        command += ['--set', key + '=' + (value if isinstance(value, str) else json.dumps(value))]
    return command, settings


def run(machine, name, directory, revision, smoke=False, dry_run=False):
    command, settings = command_for(machine, name, directory, smoke)
    print('Starting', name, 'GPU SMOKE, NOT PERFORMANCE' if smoke else 'REAL FULL T10', flush=True)
    print('Command:', shlex.join(command), flush=True)
    if dry_run:
        return dict(mode=name, status='dry_run', exit_code=0, minutes=0)
    directory.mkdir(parents=True)
    effective = json.loads((ROOT / 'exps/dlora/imgr10.json').read_text())
    effective.update(settings)
    paths = ['main.py', 'trainer.py', str(BASE.relative_to(ROOT)), str(SPEC.relative_to(ROOT)),
             'scripts/run_core_evidence_night.py', 'scripts/analyze_core_evidence.py']
    paths.append('scripts/analyze_protect_position.py')
    paths += EXTRA_SOURCE_PATHS
    paths += sorted(str(path.relative_to(ROOT)) for folder in ('models', 'methods', 'utils')
                    for path in (ROOT / folder).rglob('*.py'))
    hardware = dict(hostname=platform.node(), cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),
        gpus=subprocess.check_output(['nvidia-smi', '--query-gpu=index,uuid,name,pci.bus_id,driver_version',
                                     '--format=csv,noheader'], text=True).strip().splitlines())
    snapshot = dict(code_revision=revision, effective_config=effective, command=command,
        machine=machine, mode=name, phase='smoke' if smoke else 'formal', hardware=hardware,
        queue_pid=os.getpid(),
        source_sha256={p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in paths},
        software=dict(python=sys.version, platform=platform.platform(),
                      packages={p: importlib.metadata.version(p) for p in ('torch', 'torchvision', 'timm', 'numpy')}))
    (directory / 'run.json').write_text(json.dumps(snapshot, indent=2) + '\n')
    started = time.monotonic()
    with (directory / 'training.log').open('w') as stream:
        with subprocess.Popen(command, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                              text=True, bufsize=1) as process:
            active = dict(mode=name, phase=snapshot['phase'], queue_pid=os.getpid(),
                          training_pid=process.pid, log=str(directory / 'training.log'))
            (directory.parent / 'active.json').write_text(json.dumps(active, indent=2) + '\n')
            print('Active:', json.dumps(active), flush=True)
            for line in process.stdout:
                stream.write(line)
                stream.flush()
                print(line, end='', flush=True)
            code = process.wait()
    record = dict(mode=name, status='completed' if code == 0 else 'failed', exit_code=code,
                  training_pid=process.pid, minutes=(time.monotonic() - started) / 60)
    if code == 0:
        measured, _, _ = read_run(directory.parent, dict(record, mode=directory.name))
        expected_tasks = 2 if smoke else 10
        healthy = (measured['tasks_reported'] == expected_tasks and not measured['runtime_error']
                   and all(measured[key] is not None and math.isfinite(measured[key]) for key in METRICS))
        if not healthy:
            record.update(status='failed', exit_code=2, reason='incomplete_or_nonfinite_run',
                          process_exit_code=code)
    active.update(status=record['status'], exit_code=record['exit_code'])
    (directory.parent / 'active.json').write_text(json.dumps(active, indent=2) + '\n')
    print('Finished:', json.dumps(record), flush=True)
    return record


def execute_queue(machine, modes, directory, revision, mode='run', hours=10):
    dry_run = mode == 'dry-run'
    started = time.monotonic()
    if mode in ('run', 'smoke', 'dry-run'):
        smoke_records = []
        for name in modes:
            if not dry_run and time.monotonic() - started >= hours * 3600:
                remaining = modes[len(smoke_records):]
                smoke_records += [dict(mode=n, status='time_budget_pending') for n in remaining]
                (directory / 'smoke_queue.json').write_text(json.dumps(smoke_records, indent=2) + '\n')
                records = [dict(mode=n, status='time_budget_pending', reason='smoke_budget_incomplete') for n in modes]
                (directory / 'queue.json').write_text(json.dumps(records, indent=2) + '\n')
                summarize(directory, machine, records)
                print('Time budget reached during smoke phase; no formal run started.', flush=True)
                return 0
            record = run(machine, name, directory / ('smoke_' + name), revision, True, dry_run)
            smoke_records.append(record)
            if not dry_run:
                (directory / 'smoke_queue.json').write_text(json.dumps(smoke_records, indent=2) + '\n')
            if record['exit_code']:
                print('Smoke failed; formal queue not started.', flush=True)
                return record['exit_code']
        if mode == 'smoke':
            return 0
    records = []
    for name in modes:
        if not dry_run and time.monotonic() - started >= hours * 3600:
            records.append(dict(mode=name, status='time_budget_pending'))
            print('Time budget: not starting', name, flush=True)
        else:
            record = run(machine, name, directory / name, revision, dry_run=dry_run)
            records.append(record)
            completed_minutes = [row['minutes'] for row in records if row.get('exit_code') == 0]
            if completed_minutes and not dry_run:
                mean = sum(completed_minutes) / len(completed_minutes)
                remaining = len(modes) - len(records)
                print('ETA remaining planned jobs (minutes):', round(mean * remaining, 1),
                      '; budget never kills an active T10.', flush=True)
        if not dry_run:
            (directory / 'queue.json').write_text(json.dumps(records, indent=2) + '\n')
            summarize(directory, machine, records)
        if records[-1].get('exit_code', 0):
            print('Formal failure; preserving outputs and pausing this queue.', flush=True)
            return records[-1]['exit_code']
    print('Queue finished:', directory, flush=True)
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--machine', choices=('3090', '5090'), required=True)
    parser.add_argument('--mode', choices=('run', 'smoke', 't10', 'dry-run'), default='run')
    parser.add_argument('--hours', type=float, default=10)
    args = parser.parse_args()
    modes = [row['name'] for row in variants(args.machine)]
    directory = ROOT / ('logs/shell_logs/imgr10_core_evidence_' + args.machine) / datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    revision = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    print('Code revision:', revision, '\nOutputs:', directory, '\nQueue PID:', os.getpid(), flush=True)
    print('Order:', modes, '; seed1993; anchor2.5; 20 epochs; CA5; math-SDPA; no checkpoints.', flush=True)
    if args.mode != 'dry-run':
        directory.mkdir(parents=True)
        (directory / 'manifest.json').write_text(json.dumps(dict(revision=revision,
            machine=args.machine, queue_pid=os.getpid(), modes=modes, options=vars(args)), indent=2) + '\n')
    return execute_queue(args.machine, modes, directory, revision, args.mode, args.hours)


if __name__ == '__main__':
    raise SystemExit(main())
