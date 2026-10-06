"""Run C10/C00 on 3090 or O on 5090 after verifying the designated old evidence."""
import argparse
import datetime
import json
import os
from pathlib import Path
import subprocess
import sys

import analyze_compact_core_confirmation as analysis
import run_compact_structure_night as queue


ROOT = analysis.ROOT
SPEC = analysis.SPEC
settings_for = analysis.settings_for


def variants(machine):
    return json.loads(SPEC.read_text())[machine]


def command_for(machine, name, directory, smoke=False):
    settings = settings_for(machine, name, smoke)
    settings['prefix'] = 'imgr10_compact_core_' + machine + '_' + name + '_' + directory.parent.name
    command = [sys.executable, 'main.py', '--config', 'exps/dlora/imgr10.json']
    for key, value in settings.items():
        command += ['--set', key + '=' + (value if isinstance(value, str) else json.dumps(value))]
    return command, settings


def check_resume(directory, machine, revision, identity):
    manifest = json.loads((directory / 'manifest.json').read_text())
    if manifest['machine'] != machine or manifest['revision'] != revision:
        raise ValueError('Resume requires the original confirmation machine and revision.')
    for filename in ('queue.json', 'smoke_queue.json'):
        path = directory / filename
        for record in json.loads(path.read_text()) if path.exists() else []:
            if record['status'] == 'failed':
                raise ValueError('Preserve failed evidence; fix the cause and use a new directory.')
            if record['status'] != 'completed':
                continue
            smoke = filename == 'smoke_queue.json'
            summary, snapshot, _ = analysis.read_run(directory,
                dict(record, mode=('smoke_' if smoke else '') + record['mode']))
            issues = analysis.completed_run_issues(summary, snapshot, machine, record['mode'], identity,
                                                  smoke=smoke)
            if snapshot.get('code_revision') != revision:
                issues.append('code_revision')
            if issues:
                raise ValueError('Cannot reuse completed confirmation: ' + str(issues))


def start(machine, modes, directory, revision, mode='run', resume=False):
    if mode != 'dry-run':
        identity = analysis.current_identity()
        summaries, snapshots, _ = analysis.load_references(machine)
        issues = analysis.reference_issues(summaries, snapshots, machine, identity)
        directory.mkdir(parents=True, exist_ok=True)
        (directory / 'reference_validation.json').write_text(json.dumps(dict(
            reference_revision=analysis.REFERENCE_REVISION, execution_revision=revision,
            matching_issues=issues, current_identity=identity, reused_runs=summaries), indent=2) + '\n')
        if issues:
            raise ValueError('Reference mismatch; no training starts: ' + str(issues))
        if resume:
            check_resume(directory, machine, revision, identity)
        manifest = dict(machine=machine, revision=revision, queue_pid=os.getpid(), modes=modes,
                        reference_revision=analysis.REFERENCE_REVISION, spec=str(SPEC), mode=mode)
        (directory / ('resume_manifest.json' if resume else 'manifest.json')).write_text(
            json.dumps(manifest, indent=2) + '\n')
        print('Verified references:', [row['mode'] for row in summaries], flush=True)
    print('Code revision:', revision, '\nQueue PID:', os.getpid(), '\nOutputs:', directory, flush=True)
    print('Fixed new-run order:', modes, '; T10 seed1993 anchor2.5 CA5; no checkpoints.', flush=True)
    original = (queue.analysis, queue.engine.SPEC, queue.engine.settings_for,
                queue.engine.command_for, queue.engine.EXTRA_SOURCE_PATHS)
    try:
        queue.engine.SPEC = SPEC
        queue.engine.settings_for = settings_for
        queue.engine.command_for = command_for
        queue.engine.EXTRA_SOURCE_PATHS = ['scripts/run_compact_core_confirmation.py',
            'scripts/analyze_compact_core_confirmation.py', 'scripts/run_compact_structure_night.py',
            'scripts/analyze_compact_structure.py', 'scripts/sweeps/imgr10_compact_structure_night.json']
        queue.analysis = analysis
        # Only three predeclared runs; the timer does not truncate the finite queue.
        return queue.execute_queue(machine, modes, directory, revision, mode=mode, hours=24)
    finally:
        (queue.analysis, queue.engine.SPEC, queue.engine.settings_for,
         queue.engine.command_for, queue.engine.EXTRA_SOURCE_PATHS) = original


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--machine', choices=('3090', '5090'), required=True)
    parser.add_argument('--mode', choices=('run', 'smoke', 't10', 'dry-run'), default='run')
    parser.add_argument('--modes', nargs='+')
    parser.add_argument('--resume', type=Path)
    args = parser.parse_args()
    available = [row['name'] for row in variants(args.machine)]
    modes = args.modes or available
    if len(set(modes)) != len(modes) or any(name not in available for name in modes):
        parser.error('--modes must be unique new groups for this machine')
    revision = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    directory = args.resume or ROOT / ('logs/shell_logs/imgr10_compact_core_confirmation_' + args.machine) / datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    try:
        return start(args.machine, modes, directory, revision, args.mode, resume=bool(args.resume))
    except (ValueError, OSError, KeyError) as error:
        parser.error(str(error))


if __name__ == '__main__':
    raise SystemExit(main())
