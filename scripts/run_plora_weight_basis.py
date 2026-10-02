"""Task0-2 or full T10 controls for frozen P-A directions; no activation memory."""
import argparse
import csv
import datetime
import hashlib
import json
from pathlib import Path
import re
import shlex
import subprocess
import sys
import time

from run_composed_conflict import read_metrics


ROOT = Path(__file__).resolve().parents[1]
SPEC = ROOT / 'scripts/sweeps/imgr10_plora_weight_basis_t3_3090.json'
MODES = ('random', 'gradient', 'weight_prior')


def settings_for(mode, smoke=False, tasks=3, spec_path=SPEC):
    spec = json.loads(spec_path.read_text())
    settings = dict(spec['common_overrides'])
    settings.update(next(variant['overrides'] for variant in spec['variants'] if variant['name'] == mode))
    settings['max_tasks'] = tasks
    settings['wandb_group'] = settings['wandb_group'].replace('_t3_', '_t' + str(tasks) + '_')
    if smoke:
        settings.update(max_tasks=2, init_epoch=1, epochs=1, ca_epochs=1, wandb_mode='offline')
    return settings


def command_for(mode, directory, smoke=False, tasks=3, spec_path=SPEC):
    settings = settings_for(mode, smoke, tasks, spec_path)
    settings['prefix'] = 'imgr10_weight_basis_' + mode + '_' + directory.parent.name + '_' + directory.name
    command = [sys.executable, 'main.py', '--config', 'exps/dlora/imgr10.json']
    for key, value in settings.items():
        command += ['--set', key + '=' + (value if isinstance(value, str) else json.dumps(value))]
    return command, settings


def read_basis_rows(path, tag):
    return [json.loads(row) for row in re.findall(tag + r' (\{[^\n]+\})', path.read_text())]


def summarize(directory, records, tasks=3):
    summaries, initializations, updates = [], [], []
    for record in records:
        log = directory / record['mode'] / 'training.log'
        metrics, _ = read_metrics(log)
        initial = read_basis_rows(log, 'PGradientAInit')
        actual = read_basis_rows(log, 'PGradientAUpdate')
        summary = dict(**record, **metrics)
        summary['expected_tasks'] = tasks
        summary['full_t10_completed'] = tasks == 10 and metrics['tasks_reported'] == 10 and record['exit_code'] == 0
        summary['max_a_gram_relative_error'] = max((row['a_gram_relative_error'] for row in initial), default=0.)
        for name in ('gradient_energy_fraction', 'wpre_energy_fraction', 'history_energy_fraction'):
            summary['mean_' + name] = sum(row[name] for row in initial) / max(len(initial), 1)
        for name in ('probe_accuracy', 'probe_loss'):
            values = [row[name] for row in initial if name in row]
            summary['mean_' + name] = sum(values) / len(values) if values else None
        for name in ('raw_space_escape', 'effective_space_escape', 'effective_norm',
                     'wpre_overlap_effective', 'history_overlap_effective'):
            summary['mean_' + name] = sum(row[name] for row in actual) / max(len(actual), 1)
        summaries.append(summary)
        initializations.extend(initial)
        updates.extend(actual)
    (directory / 'results.json').write_text(json.dumps(summaries, indent=2) + '\n')
    for name, rows in (('results', summaries), ('basis_initializations', initializations), ('basis_actual_updates', updates)):
        if rows:
            keys = list(dict.fromkeys(key for row in rows for key in row))
            with (directory / (name + '.csv')).open('w', newline='') as stream:
                writer = csv.DictWriter(stream, fieldnames=keys)
                writer.writeheader()
                writer.writerows(rows)
    print('Measured full T10 results:' if tasks == 10 else 'Measured Task0-2 results (not full T10):',
          json.dumps(summaries), flush=True)


def run(mode, directory, revision, smoke=False, dry_run=False, tasks=3, spec_path=SPEC):
    command, settings = command_for(mode, directory, smoke, tasks, spec_path)
    label = 'REAL FULL T10' if tasks == 10 else 'REAL TASK0-2'
    print('Starting:', mode, 'GPU SMOKE, NOT PERFORMANCE' if smoke else label, flush=True)
    print('Command:', shlex.join(command), flush=True)
    if dry_run:
        return 0, 0.
    directory.mkdir(parents=True)
    effective = json.loads((ROOT / 'exps/dlora/imgr10.json').read_text())
    effective.update(settings)
    paths = ('models/attention.py', 'models/network.py', 'methods/dlora.py', 'utils/plora_gradient_init.py',
             'scripts/run_plora_weight_basis.py', str(spec_path.relative_to(ROOT)))
    (directory / 'run.json').write_text(json.dumps(dict(code_revision=revision,
        effective_config=effective, command=command,
        source_sha256={path: hashlib.sha256((ROOT / path).read_bytes()).hexdigest() for path in paths}), indent=2) + '\n')
    started = time.monotonic()
    with (directory / 'training.log').open('w') as stream:
        with subprocess.Popen(command, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                              text=True, bufsize=1) as process:
            for line in process.stdout:
                stream.write(line)
                stream.flush()
                print(line, end='', flush=True)
            code = process.wait()
    minutes = (time.monotonic() - started) / 60
    print('Finished:', mode, 'exit_code=', code, 'minutes=', round(minutes, 1), flush=True)
    return code, minutes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=('run', 'smoke', 't3', 't10', 'dry-run'), default='run')
    parser.add_argument('--tasks', type=int, choices=(3, 10), default=3)
    parser.add_argument('--spec', type=Path, default=SPEC)
    parser.add_argument('--only', nargs='+')
    args = parser.parse_args()
    tasks = 10 if args.mode == 't10' else args.tasks
    spec_path = args.spec.resolve()
    specification = json.loads(spec_path.read_text())
    modes = args.only or [variant['name'] for variant in specification['variants']]
    log_dir = specification['log_dir'].replace('_t3_', '_t' + str(tasks) + '_')
    directory = ROOT / log_dir / datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    revision = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    print('Code revision:', revision, '\nOutputs:', directory, flush=True)
    print('3090; ImageNet-R seed1993; official test Task0-' + str(tasks - 1) + '; full training data; anchor2.5; '
          '20 epochs; CA5; math SDPA. Only Task1+ P-A directions change; A frozen, B starts at zero. '
          'S/gates/CA/inference unchanged. No replay, saved features or checkpoints.', flush=True)
    dry_run = args.mode == 'dry-run'
    if not dry_run:
        directory.mkdir(parents=True)
        options = {**vars(args), 'spec': str(spec_path)}
        (directory / 'manifest.json').write_text(json.dumps(dict(revision=revision, options=options,
            modes=modes, tasks=tasks), indent=2) + '\n')
    if args.mode in ('run', 'smoke', 'dry-run'):
        for mode in modes:
            code, _ = run(mode, directory / ('smoke_' + mode), revision, True, dry_run,
                          tasks=tasks, spec_path=spec_path)
            if code:
                print('Smoke failed; formal queue not started.', flush=True)
                return code
        if args.mode == 'smoke':
            return 0
    records = []
    for mode in modes:
        code, minutes = run(mode, directory / mode, revision, dry_run=dry_run,
                            tasks=tasks, spec_path=spec_path)
        records.append(dict(mode=mode, status='completed' if code == 0 else 'failed', exit_code=code, minutes=minutes))
        if not dry_run:
            (directory / 'queue.json').write_text(json.dumps(records, indent=2) + '\n')
            summarize(directory, records, tasks)
        if code:
            return code
    print('Queue finished:', directory, flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
