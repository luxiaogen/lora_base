"""双机固定预算验证；复用现有运行引擎，两个迁移数据集共享截止时间。"""
import argparse
import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import traceback

import run_baseline_suite as baseline
import run_tail_update as tail
from analyze_core_evidence import METRICS, read_run
from dualmask_config import normalize_dualmask_config

ROOT = baseline.ROOT
SPEC = ROOT / 'scripts/sweeps/budget_verification.json'


def spec():
    return json.loads(SPEC.read_text())


def modes(machine):
    data = spec()[machine]
    return [dataset['name'] + '_' + variant['name'] + '_seed1993'
            for dataset in data['datasets'] for variant in data['variants']]


def identity(machine, name):
    dataset_name, variant_name = name.removesuffix('_seed1993').split('_', 1)
    data = spec()[machine]
    return (next(row for row in data['datasets'] if row['name'] == dataset_name),
            next(row for row in data['variants'] if row['name'] == variant_name))


def settings_for(machine, name, smoke=False):
    dataset, variant = identity(machine, name)
    settings = baseline.settings_for(machine, dataset['name'] + '_seed1993')
    settings.update(spec()['common_overrides'])
    if variant['recipe'] == 'F':
        settings.update(spec()['compact_overrides'], dual_mask_private_rank=dataset['rank'])
    settings.update(variant['overrides'], wandb_group='budget_verification_' + machine)
    if smoke:
        settings.update(max_tasks=2, init_epoch=1, epochs=1, ca_epochs=1,
                        wandb_mode='offline', stage_audit=True)
    return settings


def effective_config(machine, name, smoke=False):
    dataset, _ = identity(machine, name)
    config = dict(json.loads((ROOT / dataset['config']).read_text()), **settings_for(machine, name, smoke))
    return normalize_dualmask_config(config)


def command_for(machine, name, directory, smoke=False):
    dataset, _ = identity(machine, name)
    settings = settings_for(machine, name, smoke)
    settings['prefix'] = 'budget_' + machine + '_' + name + '_' + directory.parent.name
    command = [tail.sys.executable, 'main.py', '--config', dataset['config']]
    for key, value in settings.items():
        command += ['--set', key + '=' + (value if isinstance(value, str) else json.dumps(value))]
    return command, settings


def validate_settings(machine):
    for name in modes(machine):
        dataset, variant = identity(machine, name)
        config = effective_config(machine, name)
        expected = dict(seed=[1993], rank=dataset['rank'], init_epoch=20, epochs=20, ca=True,
            ca_epochs=5, dual_mask_anchor_reg_weight=2.5, dual_mask_task0_gate_mode='unmasked',
            dual_mask_permission_mode='asymmetric', dual_mask_branch_layout='dual',
            dual_mask_conflict_granularity='layer', dual_mask_private_conflict_mode='global',
            dual_mask_gradient_route='off', dual_mask_prototype_position_probe=False,
            dual_mask_uniform_norm_matched=False, p_permission_position='wpre',
            p_permission_release='off', p_direction_score='off', plora_train_a=False,
            wpre_distill_weight=0, old_model_distill_weight=0, old_competition_weight=0,
            save_task_weights=False, disable_fused_sdpa=True)
        if any(config.get(key) != value for key, value in expected.items()):
            raise ValueError('Budget verification recipe changed: ' + name)
        if variant['recipe'] == 'F' and any(config.get(key) != value for key, value in
                dict(spec()['compact_overrides'], dual_mask_private_rank=dataset['rank']).items()):
            raise ValueError('Compact factors changed: ' + name)


def check_resume(directory, machine, revision, selected):
    manifest = json.loads((directory / 'manifest.json').read_text())
    if (manifest['machine'] != machine or manifest['revision'] != revision
            or any(name not in manifest['order'] for name in selected)):
        raise ValueError('Resume requires the original machine, revision and planned groups')
    active = directory / 'active.json'
    if active.exists():
        state = json.loads(active.read_text())
        pids = [state['queue_pid']] + ([state['training_pid']] if state.get('status') not in ('completed', 'failed') else [])
        for pid in pids:
            try:
                os.kill(pid, 0)
            except ProcessLookupError:
                continue
            raise ValueError('Queue or training process still exists: ' + str(pid))
    for filename, smoke in (('queue.json', False), ('smoke_queue.json', True)):
        path = directory / filename
        for record in json.loads(path.read_text()) if path.exists() else []:
            if record['status'] == 'failed':
                raise ValueError('Failure preserved; do not automatically retry')
            if record['status'] != 'completed':
                continue
            row = dict(record, mode=('smoke_' if smoke else '') + record['mode'])
            measured, snapshot, _ = read_run(directory, row)
            expected = effective_config(machine, record['mode'], smoke)
            if (snapshot['code_revision'] != revision or snapshot['machine'] != machine
                    or snapshot['phase'] != ('smoke' if smoke else 'formal')
                    or measured['runtime_error'] or measured['tasks_reported'] != (2 if smoke else 10)
                    or record.get('exit_code') != 0
                    or any(snapshot['effective_config'].get(k) != v for k, v in expected.items() if k != 'prefix')
                    or any(hashlib.sha256((ROOT / p).read_bytes()).hexdigest() != digest
                           for p, digest in snapshot['source_sha256'].items())
                    or any(measured[k] is None or not math.isfinite(measured[k]) for k in METRICS)):
                raise ValueError('Cannot reuse changed or unhealthy result: ' + record['mode'])


def summarize_safely(directory, machine, records):
    try:
        from analyze_budget_verification import summarize_saved
        summarize_saved(directory)
    except Exception:
        with (directory / 'analysis_errors.jsonl').open('a') as stream:
            stream.write(json.dumps(dict(time=datetime.datetime.now().isoformat(), traceback=traceback.format_exc())) + '\n')
        print('Analysis error saved; planned training continues.', flush=True)


def execute(machine, selected, directory, revision, mode='run', hours=10, source_paths=()):
    previous = tail.SPEC, tail.command_for, tail.settings_for, tail.summarize_safely
    tail.SPEC, tail.command_for, tail.settings_for, tail.summarize_safely = SPEC, command_for, settings_for, summarize_safely
    try:
        return tail.execute(machine, selected, directory, revision, mode, hours,
            source_paths=('scripts/run_budget_verification.py', 'scripts/analyze_budget_verification.py') + source_paths)
    finally:
        tail.SPEC, tail.command_for, tail.settings_for, tail.summarize_safely = previous


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--machine', choices=('3090', '5090'), required=True)
    parser.add_argument('--mode', choices=('run', 'smoke', 'dry-run'), default='run')
    parser.add_argument('--modes', nargs='+')
    parser.add_argument('--resume', type=Path)
    parser.add_argument('--hours', type=float, default=spec().get('hours', 10))
    args = parser.parse_args()
    selected = args.modes or modes(args.machine)
    if (not math.isfinite(args.hours) or args.hours <= 0 or len(selected) != len(set(selected))
            or any(name not in modes(args.machine) for name in selected)):
        parser.error('Use positive hours and unique predeclared modes')
    validate_settings(args.machine)
    revision = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    directory = args.resume or ROOT / ('logs/shell_logs/' + spec().get('name', 'budget_verification') + '_' + args.machine) / datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    if args.mode != 'dry-run':
        if args.resume:
            check_resume(directory, args.machine, revision, selected)
        datasets = {identity(args.machine, name)[0]['name']: identity(args.machine, name)[0] for name in selected}
        checked = [baseline.check_data(json.loads((ROOT / item['config']).read_text())) for item in datasets.values()]
        directory.mkdir(parents=True, exist_ok=True)
        (directory / ('resume_manifest.json' if args.resume else 'manifest.json')).write_text(json.dumps(dict(
            machine=args.machine, revision=revision, queue_pid=os.getpid(), order=selected,
            hours=args.hours, datasets=checked, sweep_spec=str(SPEC.relative_to(ROOT)),
            start=datetime.datetime.now().isoformat()), indent=2) + '\n')
    print('Code revision:', revision, '\nQueue PID:', os.getpid(), '\nOutputs:', directory,
          '\nOrder:', selected, flush=True)
    return execute(args.machine, selected, directory, revision, args.mode, args.hours)


if __name__ == '__main__':
    raise SystemExit(main())
