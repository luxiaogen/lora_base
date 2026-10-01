"""Six predeclared real T10 trials, with an eight-hour start-next budget."""
import argparse
import datetime
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import sys
import time
from zoneinfo import ZoneInfo

from run_wpre_distill import settings_for as base_settings


VARIANTS = [('complement_s', 'complement', 's'), ('random_s', 'random_matched', 's'),
            ('teacher_correct_s', 'teacher_correct', 's'), ('complement_p', 'complement', 'p'),
            ('random_p', 'random_matched', 'p'), ('complement_sp', 'complement', 'sp')]


def settings_for(name, selection, scope, stage, directory):
    settings = base_settings('5090', stage, directory)
    settings.update(prefix=f'imgr10_wpre_complement_5090_{name}_{stage}_{directory.name}',
                    wandb_group='imgr10_wpre_complement_5090',
                    wpre_distill_selection=selection, wpre_distill_scope=scope,
                    wpre_distill_normalization='batch', dual_mask_composed_conflict='off')
    return settings


def training_command(settings):
    command = [sys.executable, 'main.py', '--config', 'exps/dlora/imgr10.json']
    for key, value in settings.items():
        command += ['--set', key + '=' + (value if isinstance(value, str) else json.dumps(value))]
    return command


def fits_budget(elapsed_minutes, budget_minutes, completed_minutes):
    # Reserve 10% above the slowest completed T10; never terminate a running trial.
    estimate = max(completed_minutes or [72.]) * 1.1
    return elapsed_minutes + estimate <= budget_minutes


def run_one(name, selection, scope, stage, directory, revision):
    settings = settings_for(name, selection, scope, stage, directory)
    command = training_command(settings)
    print('Stage:', stage, 'Variant:', name, flush=True)
    print('Command:', shlex.join(command), flush=True)
    stage_dir = directory / name / stage
    stage_dir.mkdir(parents=True, exist_ok=True)
    effective = json.loads(Path('exps/dlora/imgr10.json').read_text())
    effective.update(settings)
    paths = ('methods/dlora.py', 'models/attention.py', 'models/network.py',
             'utils/wpre_distill.py', 'utils/frozen_readout.py', 'utils/data_manager.py',
             'scripts/run_wpre_complement.py', 'scripts/run_wpre_distill.py',
             'scripts/sweeps/imgr10_two_expert_oracle_3090.json')
    metadata = dict(code_revision=revision, effective_config=effective, command=command,
                    source_sha256={path: hashlib.sha256(Path(path).read_bytes()).hexdigest() for path in paths},
                    started_at=datetime.datetime.now(ZoneInfo('Asia/Shanghai')).isoformat())
    metadata_path = stage_dir / 'run.json'
    metadata_path.write_text(json.dumps(metadata, indent=2) + '\n')
    started = time.monotonic()
    with (stage_dir / 'training.log').open('w') as stream:
        with subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                              text=True, bufsize=1) as process:
            for line in process.stdout:
                stream.write(line)
                stream.flush()
                print(line, end='', flush=True)
            status = process.wait()
    minutes = (time.monotonic() - started) / 60.
    metadata.update(exit_code=status, elapsed_minutes=minutes)
    metadata_path.write_text(json.dumps(metadata, indent=2) + '\n')
    print('Stage:', stage, 'Variant:', name, 'Exit code:', status, 'Minutes:', round(minutes, 2), flush=True)
    return dict(variant=name, stage=stage, exit_code=status, elapsed_minutes=minutes)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=('run', 'smoke', 't10', 'dry-run'), default='run')
    parser.add_argument('--hours', type=float, default=8.)
    parser.add_argument('--only', nargs='+', choices=[name for name, _, _ in VARIANTS])
    args = parser.parse_args()
    variants = [v for v in VARIANTS if args.only is None or v[0] in args.only]
    stamp = datetime.datetime.now(ZoneInfo('Asia/Shanghai')).strftime('%Y%m%d_%H%M%S_%f')
    directory = Path('logs/shell_logs/imgr10_wpre_complement_5090') / stamp
    revision = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()
    print('Code revision:', revision, flush=True)
    print('REAL full-training ImageNet-R T10, seed1993, anchor2.5, CA5, math-SDPA.', flush=True)
    print('Teacher-correct/student-wrong selection uses ALL seen classes on CURRENT training images.', flush=True)
    print('Weight=1, batch denominator; only the requested S/P B gradients change. No checkpoints or replay.', flush=True)
    print('Start-next budget (hours):', args.hours, '; do not kill a running experiment.', flush=True)
    stages = ['smoke'] if args.mode == 'smoke' else ['t10'] if args.mode == 't10' else ['smoke', 't10']
    if args.mode == 'dry-run':
        for stage in stages:
            for name, selection, scope in variants:
                print(stage, name, shlex.join(training_command(settings_for(
                    name, selection, scope, stage, directory))), flush=True)
        return 0
    directory.mkdir(parents=True, exist_ok=True)
    queue = dict(code_revision=revision, hours=args.hours, variants=variants, results=[], finished=False)
    queue_path = directory / 'queue.json'
    queue_path.write_text(json.dumps(queue, indent=2) + '\n')
    started, completed = time.monotonic(), []
    for stage in stages:
        for name, selection, scope in variants:
            elapsed = (time.monotonic() - started) / 60.
            if stage == 't10' and not fits_budget(elapsed, args.hours * 60., completed):
                queue['results'].append(dict(variant=name, stage=stage, skipped='time_budget'))
                queue_path.write_text(json.dumps(queue, indent=2) + '\n')
                print('Budget skip (not a performance decision):', name, flush=True)
                continue
            result = run_one(name, selection, scope, stage, directory, revision)
            queue['results'].append(result)
            queue_path.write_text(json.dumps(queue, indent=2) + '\n')
            if result['exit_code']:
                print('Queue stopped on training failure:', name, stage, flush=True)
                return result['exit_code']
            if stage == 't10':
                completed.append(result['elapsed_minutes'])
    queue.update(finished=True, elapsed_minutes=(time.monotonic() - started) / 60.)
    queue_path.write_text(json.dumps(queue, indent=2) + '\n')
    print('Finished. Queue and per-run logs:', directory, flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
