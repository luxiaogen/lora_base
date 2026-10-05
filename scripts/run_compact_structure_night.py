"""Fixed, resumable nine-hour queues for compact DualMask and structural controls."""
import argparse
import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

import run_core_evidence_night as engine
import analyze_compact_structure as analysis


ROOT = Path(__file__).resolve().parents[1]
SPEC = ROOT / 'scripts/sweeps/imgr10_compact_structure_night.json'
BASE = engine.BASE


def variants(machine):
    return json.loads(SPEC.read_text())[machine]


def settings_for(machine, name, smoke=False):
    spec = json.loads(SPEC.read_text())
    settings = dict(json.loads(BASE.read_text())['common_overrides'])
    settings.update(spec['common_overrides'])
    settings.update(next(row['overrides'] for row in spec[machine] if row['name'] == name))
    settings['wandb_group'] = 'imgr10_compact_structure_' + machine
    if smoke:
        settings.update(max_tasks=2, init_epoch=1, epochs=1, ca_epochs=1,
                        wandb_mode='offline', stage_audit=True)
    return settings


def command_for(machine, name, directory, smoke=False):
    settings = settings_for(machine, name, smoke)
    settings['prefix'] = 'imgr10_compact_' + machine + '_' + name + '_' + directory.parent.name
    command = [sys.executable, 'main.py', '--config', 'exps/dlora/imgr10.json']
    for key, value in settings.items():
        command += ['--set', key + '=' + (value if isinstance(value, str) else json.dumps(value))]
    return command, settings


def summarize_safely(directory, machine, records):
    try:
        analysis.summarize(directory, machine, records)
    except Exception:
        error = dict(time=datetime.datetime.now().isoformat(), traceback=traceback.format_exc())
        with (directory / 'analysis_errors.jsonl').open('a') as stream:
            stream.write(json.dumps(error) + '\n')
        print('Analysis error saved; training queue continues:', error['traceback'], flush=True)


def check_resume(directory, machine, revision):
    manifest = json.loads((directory / 'manifest.json').read_text())
    if manifest['machine'] != machine or manifest['revision'] != revision:
        raise ValueError('Resume requires the original machine and code revision.')
    for filename, smoke in (('queue.json', False), ('smoke_queue.json', True)):
        path = directory / filename
        for row in json.loads(path.read_text()) if path.exists() else []:
            if row.get('status') == 'failed':
                raise ValueError('Failed runs are preserved; after fixing, start unfinished groups in a new queue.')
            if row.get('status') != 'completed':
                continue
            name = ('smoke_' if smoke else '') + row['mode']
            measured, snapshot, _ = analysis.read_run(directory, dict(row, mode=name))
            expected = json.loads((ROOT / 'exps/dlora/imgr10.json').read_text())
            expected.update(settings_for(machine, row['mode'], smoke))
            healthy = (measured['tasks_reported'] == (2 if smoke else 10)
                and not measured['runtime_error'] and row.get('exit_code') == 0
                and snapshot['machine'] == machine and snapshot['code_revision'] == revision
                and snapshot['phase'] == ('smoke' if smoke else 'formal')
                and all(measured.get(k) is not None and math.isfinite(measured[k]) for k in analysis.METRICS))
            hashes_match = all(hashlib.sha256((ROOT / p).read_bytes()).hexdigest() == digest
                               for p, digest in snapshot['source_sha256'].items())
            settings_match = all(snapshot['effective_config'].get(k) == v for k, v in expected.items() if k != 'prefix')
            if not healthy or not hashes_match or not settings_match:
                raise ValueError('Cannot reuse unverified result/config/source: ' + name)


def execute_queue(machine, modes, directory, revision, mode='run', hours=9):
    dry_run = mode == 'dry-run'
    started = time.monotonic()
    queue = directory / 'queue.json'
    smoke_queue = directory / 'smoke_queue.json'
    previous = json.loads(queue.read_text()) if queue.exists() else []
    smokes = json.loads(smoke_queue.read_text()) if smoke_queue.exists() else []
    records = {row['mode']: row for row in previous}
    smoke_records = {row['mode']: row for row in smokes}
    completed = {name for name, row in records.items() if row.get('status') == 'completed'}
    pending = [name for name in modes if name not in completed]
    print('Completed groups reused:', sorted(completed), '; pending order:', pending, flush=True)

    def save():
        queue.write_text(json.dumps(list(records.values()), indent=2) + '\n')
        summarize_safely(directory, machine, list(records.values()))

    if mode in ('run', 'smoke', 'dry-run'):
        for name in pending:
            if smoke_records.get(name, {}).get('status') == 'completed':
                continue
            if not dry_run and time.monotonic() - started >= hours * 3600:
                for candidate in pending:
                    if candidate not in completed:
                        records[candidate] = dict(mode=candidate, status='time_budget_pending',
                                                  reason='smoke_budget_incomplete')
                smoke_queue.write_text(json.dumps(list(smoke_records.values()), indent=2) + '\n')
                save()
                return 0
            record = engine.run(machine, name, directory / ('smoke_' + name), revision, True, dry_run)
            smoke_records[name] = record
            if not dry_run:
                smoke_queue.write_text(json.dumps(list(smoke_records.values()), indent=2) + '\n')
            if record['exit_code']:
                print('Smoke failed; no formal training starts.', flush=True)
                return record['exit_code']
        if mode == 'smoke':
            return 0
    for index, name in enumerate(pending):
        if not dry_run and time.monotonic() - started >= hours * 3600:
            records[name] = dict(mode=name, status='time_budget_pending')
            print('Time budget: not starting', name, flush=True)
        else:
            records[name] = engine.run(machine, name, directory / name, revision, dry_run=dry_run)
            minutes = [row.get('minutes', 0) for row in records.values() if row.get('status') == 'completed']
            if minutes and not dry_run:
                print('ETA for remaining planned groups (min):', round(sum(minutes) / len(minutes)
                    * (len(pending) - index - 1), 1), '; active T10 may finish beyond deadline.', flush=True)
        if not dry_run:
            save()
        if records[name].get('exit_code', 0):
            print('Training failed; preserving evidence and pausing this machine.', flush=True)
            return records[name]['exit_code']
    print('Queue finished:', directory, flush=True)
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--machine', choices=('3090', '5090'), required=True)
    parser.add_argument('--mode', choices=('run', 'smoke', 't10', 'dry-run'), default='run')
    parser.add_argument('--hours', type=float, default=9)
    parser.add_argument('--modes', nargs='+')
    parser.add_argument('--resume', type=Path, help='Reuse the same queue and skip completed formal runs.')
    args = parser.parse_args()
    available = [row['name'] for row in variants(args.machine)]
    modes = args.modes or available
    if len(set(modes)) != len(modes) or any(name not in available for name in modes):
        parser.error('--modes must be unique groups for the selected machine')
    if args.hours <= 0:
        parser.error('--hours must be positive')
    revision = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    directory = args.resume or ROOT / ('logs/shell_logs/imgr10_compact_structure_' + args.machine) / datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    if args.resume:
        try:
            check_resume(directory, args.machine, revision)
        except (ValueError, OSError) as error:
            parser.error(str(error))
    print('Code revision:', revision, '\nOutputs:', directory, '\nQueue PID:', os.getpid(), flush=True)
    print('Order:', modes, '; full T10 seed1993 anchor2.5 CA5 math-SDPA, no checkpoints.', flush=True)
    if args.mode != 'dry-run':
        directory.mkdir(parents=True, exist_ok=True)
        manifest = dict(revision=revision, machine=args.machine, queue_pid=os.getpid(), modes=modes,
                        options={**vars(args), 'resume': str(args.resume) if args.resume else None})
        (directory / ('resume_manifest.json' if args.resume else 'manifest.json')).write_text(json.dumps(manifest, indent=2) + '\n')
    engine.SPEC = SPEC
    engine.settings_for = settings_for
    engine.command_for = command_for
    engine.EXTRA_SOURCE_PATHS = ['scripts/run_compact_structure_night.py', 'scripts/analyze_compact_structure.py']
    return execute_queue(args.machine, modes, directory, revision, args.mode, args.hours)


if __name__ == '__main__':
    raise SystemExit(main())
