"""One T10 per machine, many cheap cache-only legal-selector comparisons."""
import argparse
import datetime
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import sys

from evaluate_expert_legal import evaluate


BASE_SPEC = Path('scripts/sweeps/imgr10_two_expert_oracle_3090.json')
OLD_3090_CACHE = Path('logs/shell_logs/imgr10_two_expert_oracle_3090/20261001_094251_360566/t10/diagnostics')


def resolved_spec(machine):
    spec = json.loads(BASE_SPEC.read_text())
    spec['name'] = f'imgr10_expert_legal_{machine}'
    spec['log_dir'] = f'logs/shell_logs/{spec["name"]}'
    spec['common_overrides'].update(two_expert_calibration_holdout_mod=20 if machine == '3090' else 10,
                                    wandb_group=spec['name'], save_task_weights=False)
    spec['variants'][0].update(name='calibrated_selectors',
        description='One T10; current-only withheld calibration; all predeclared label-free selectors')
    return spec


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--machine', required=True, choices=('3090', '5090'))
    parser.add_argument('--cache', type=Path)
    parser.add_argument('--mode', choices=('--run', '--smoke', '--t10', '--dry-run', '--cache-only'),
                        default='--run')
    args = parser.parse_args()
    mode = args.mode
    spec = resolved_spec(args.machine)
    stamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    directory = Path(spec['log_dir']) / stamp
    revision = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()
    print('Code revision:', revision, flush=True)
    print('No old-image replay, task ID, test-fitting, checkpoints or saved dense features.', flush=True)
    print('Task0 unchanged; Task1+ current-only calibration mod:',
          spec['common_overrides']['two_expert_calibration_holdout_mod'], flush=True)
    print('41 predeclared selectors share ONE T10; primary comparison is train_selected vs same-run base.', flush=True)
    cache = args.cache or (OLD_3090_CACHE if args.machine == '3090' else None)
    cache_complete = cache is not None and all(
        (cache / f'task_{task:02d}_{source}.{suffix}').is_file()
        for task in range(10) for source in ('current_train_seen_probe', 'test_report_only')
        for suffix in ('csv', 'json'))
    if mode in ('--run', '--cache-only') and cache_complete:
        print('Reuse existing cache (seen-train exploratory only):', cache, flush=True)
        evaluate(cache, directory / 'reused_seen_cache', 'current_train_seen_probe')
    elif mode == '--cache-only':
        print('No complete existing cache. Supply --cache PATH; no training was started.', flush=True)
        return 1
    elif mode == '--run' and cache is not None:
        print('Old cache unavailable/incomplete; skip optional reuse, continue with new holdout run.', flush=True)
    if mode == '--cache-only':
        return 0
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
        if mode == '--dry-run':
            continue
        stage_dir = directory / stage
        stage_dir.mkdir(parents=True, exist_ok=True)
        (stage_dir / 'resolved_sweep.json').write_text(json.dumps(spec, indent=2) + '\n')
        effective = json.loads(Path(spec['datasets'][0]['config']).read_text())
        effective.update(settings)
        paths = ('methods/dlora.py', 'utils/data_manager.py', 'utils/two_expert_oracle.py',
                 'utils/expert_legal.py', 'scripts/evaluate_expert_legal.py', 'scripts/run_expert_legal.py')
        metadata = dict(code_revision=revision, effective_config=effective, command=command,
                        base_spec_sha256=hashlib.sha256(BASE_SPEC.read_bytes()).hexdigest(),
                        source_sha256={path: hashlib.sha256(Path(path).read_bytes()).hexdigest() for path in paths})
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
            return status
        # File completeness only. No assertions or gates on accuracy / net benefit.
        evaluate(settings['two_expert_oracle_dir'], stage_dir / 'legal_selectors',
                 'current_train_holdout', expected_tasks=settings['max_tasks'])
    print('Finished. All legal comparisons:', directory, flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
