"""Run real DualMask T10 training with a sealed, legal Ridge fusion predictor."""
import argparse
import datetime
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import sys

from run_expert_legal import resolved_spec


def settings_for(machine, stage, directory):
    spec = resolved_spec(machine)
    settings = dict(spec['common_overrides'])
    settings.update(spec['variants'][0]['overrides'])
    settings.update(seed=[1993], prefix=f'imgr10_ridge_fusion_{machine}_{stage}_{directory.name}',
                    ridge_fusion_enabled=True, ridge_fusion_dir=str(directory / stage / 'fusion'),
                    two_expert_oracle_diagnostic=False, stage_audit=False,
                    wandb_group=f'imgr10_ridge_fusion_{machine}', save_task_weights=False)
    if stage == 'smoke':
        settings.update(max_tasks=2, init_epoch=1, epochs=1, ca_epochs=1, wandb_mode='offline')
    return settings


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--machine', required=True, choices=('3090', '5090'))
    parser.add_argument('--mode', choices=('run', 'smoke', 't10', 'dry-run'), default='run')
    args = parser.parse_args()
    directory = Path(f'logs/shell_logs/imgr10_ridge_fusion_{args.machine}') / datetime.datetime.now().strftime(
        '%Y%m%d_%H%M%S_%f')
    revision = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()
    print('Code revision:', revision, flush=True)
    print('REAL DualMask training + legal Ridge fusion. CNN metrics are the candidate; '
          'RidgeFusionResult contains the SAME-RUN base/candidate pair.', flush=True)
    print('No checkpoints, old-image replay, test-fitted weights or true-task inference.', flush=True)
    stages = ['smoke'] if args.mode == 'smoke' else ['t10'] if args.mode == 't10' else ['smoke', 't10']
    for stage in stages:
        settings = settings_for(args.machine, stage, directory)
        command = [sys.executable, 'main.py', '--config', 'exps/dlora/imgr10.json']
        for key, value in settings.items():
            command += ['--set', key + '=' + (value if isinstance(value, str) else json.dumps(value))]
        print('Stage:', stage, flush=True)
        print('Command:', shlex.join(command), flush=True)
        if args.mode == 'dry-run':
            continue
        stage_dir = directory / stage
        stage_dir.mkdir(parents=True, exist_ok=True)
        effective = json.loads(Path('exps/dlora/imgr10.json').read_text())
        effective.update(settings)
        paths = ('methods/dlora.py', 'utils/ridge_fusion.py', 'utils/frozen_readout.py',
                 'utils/data_manager.py', 'scripts/run_ridge_fusion.py',
                 'scripts/sweeps/imgr10_two_expert_oracle_3090.json')
        (stage_dir / 'run.json').write_text(json.dumps(dict(code_revision=revision,
            effective_config=effective, command=command,
            source_sha256={path: hashlib.sha256(Path(path).read_bytes()).hexdigest() for path in paths}),
            indent=2) + '\n')
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
    print('Finished. Real candidate and paired base:', directory, flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
