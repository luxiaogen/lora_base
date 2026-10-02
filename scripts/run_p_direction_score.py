"""Three full-T10 P scoring controls; no replay, teacher, checkpoint or task-ID inference."""
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
SPEC = ROOT / 'scripts/sweeps/imgr10_p_direction_score_3090.json'
MODES = ('coordinate', 'spectral', 'signed')


def settings_for(mode, smoke=False):
    spec = json.loads(SPEC.read_text())
    settings = dict(spec['common_overrides'])
    settings.update(next(v['overrides'] for v in spec['variants'] if v['name'] == mode))
    if smoke:
        settings.update(max_tasks=2, init_epoch=1, epochs=1, ca_epochs=1, wandb_mode='offline')
    return settings


def command_for(mode, directory, smoke=False):
    settings = settings_for(mode, smoke)
    settings['prefix'] = 'imgr10_p_direction_' + mode + '_' + directory.parent.name + '_' + directory.name
    command = [sys.executable, 'main.py', '--config', 'exps/dlora/imgr10.json']
    for key, value in settings.items():
        command += ['--set', key + '=' + (value if isinstance(value, str) else json.dumps(value))]
    return command, settings


def read_direction_rows(path):
    return [json.loads(row) for row in re.findall(r'PDirectionScore (\{[^\n]+\})', path.read_text())]


def summarize(directory, records):
    summaries, diagnostics = [], []
    for record in records:
        log = directory / record['mode'] / 'training.log'
        metrics, _ = read_metrics(log)
        rows = read_direction_rows(log)
        errors = [abs(actual - target) / max(target, 1e-12)
                  for row in rows for actual, target in zip(row['actual_removed_norm'], row['reference_removed_norm'])]
        summaries.append(dict(**record, **metrics, max_relative_norm_error=max(errors, default=0)))
        for row in rows:
            for index, projection in enumerate(('Q', 'K', 'V')):
                point = dict(mode=row['mode'], task=row['task'], layer=row['layer'], projection=projection)
                for key in ('selected_k', 'reference_k', 'strength', 'reference_removed_norm',
                            'actual_removed_norm', 'base_norm'):
                    point[key] = row[key][index]
                point['merge_error'] = row['merge_error']
                if row.get('signed_absolute_jaccard') is not None:
                    point['signed_absolute_jaccard'] = row['signed_absolute_jaccard'][index]
                if row['spectral_fractions'] is not None:
                    for key, value in zip(('enhancement', 'attenuation', 'mixing'), row['spectral_fractions'][index]):
                        point[key] = value
                diagnostics.append(point)
    (directory / 'results.json').write_text(json.dumps(summaries, indent=2) + '\n')
    for name, rows in (('results', summaries), ('direction_diagnostics', diagnostics)):
        if rows:
            keys = list(dict.fromkeys(key for row in rows for key in row))
            with (directory / (name + '.csv')).open('w', newline='') as stream:
                writer = csv.DictWriter(stream, fieldnames=keys)
                writer.writeheader()
                writer.writerows(rows)
    print('Measured results:', json.dumps(summaries), flush=True)


def run(mode, directory, revision, smoke=False, dry_run=False):
    command, settings = command_for(mode, directory, smoke)
    print('Starting:', mode, 'GPU SMOKE, NOT PERFORMANCE' if smoke else 'REAL FULL T10', flush=True)
    print('Command:', shlex.join(command), flush=True)
    if dry_run:
        return 0, 0.
    directory.mkdir(parents=True)
    effective = json.loads((ROOT / 'exps/dlora/imgr10.json').read_text())
    effective.update(settings)
    paths = ('models/attention.py', 'utils/p_direction_score.py', 'methods/dlora.py',
             'scripts/run_p_direction_score.py', 'scripts/sweeps/imgr10_p_direction_score_3090.json')
    (directory / 'run.json').write_text(json.dumps(dict(code_revision=revision,
        effective_config=effective, command=command,
        source_sha256={p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in paths}), indent=2) + '\n')
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
    parser.add_argument('--mode', choices=('run', 'smoke', 't10', 'dry-run'), default='run')
    parser.add_argument('--only', nargs='+', choices=MODES)
    args = parser.parse_args()
    directory = ROOT / 'logs/shell_logs/imgr10_p_direction_score_3090' / datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    revision = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    print('Code revision:', revision, '\nOutputs:', directory, flush=True)
    print('P ONLY; seed1993; anchor2.5; full T10; 20 epochs; CA5; math SDPA. '
          'S unchanged, no teacher/replay/checkpoints/task-ID inference. No baseline rerun.', flush=True)
    dry_run = args.mode == 'dry-run'
    if not dry_run:
        directory.mkdir(parents=True)
        (directory / 'manifest.json').write_text(json.dumps(dict(revision=revision, options=vars(args)), indent=2) + '\n')
    if args.mode in ('run', 'smoke', 'dry-run'):
        for mode in args.only or MODES:
            code, _ = run(mode, directory / ('smoke_' + mode), revision, True, dry_run)
            if code:
                print('Smoke failed; full queue not started.', flush=True)
                return code
        if args.mode == 'smoke':
            return 0
    records = []
    for mode in args.only or MODES:
        code, minutes = run(mode, directory / mode, revision, dry_run=dry_run)
        records.append(dict(mode=mode, status='completed' if code == 0 else 'failed', exit_code=code, minutes=minutes))
        if not dry_run:
            (directory / 'queue.json').write_text(json.dumps(records, indent=2) + '\n')
            summarize(directory, records)
        if code:
            return code
    print('Queue finished:', directory, flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
