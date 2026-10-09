"""Fixed ten-hour ImageNet-R/A queues; smoke each recipe once, preserve partial evidence."""
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

import run_baseline_suite as baseline
import run_core_evidence_night as engine
from analyze_core_evidence import METRICS, read_run


ROOT = engine.ROOT
SPEC = ROOT / 'scripts/sweeps/tail_update.json'


def spec():
    return json.loads(SPEC.read_text())


def modes(machine):
    data = spec()
    return [variant + '_seed' + str(seed) for stage in ('primary', 'extended')
            for seed in data['seeds'] for variant in data[machine][stage]]


def identity(name):
    variant, seed = name.rsplit('_seed', 1)
    return variant, int(seed)


def settings_for(machine, name, smoke=False):
    data = spec()
    variant, seed = identity(name)
    settings = baseline.settings_for(machine, data[machine]['dataset'] + '_seed' + str(seed))
    settings.update(data['common_overrides'])
    if variant != 'O':
        settings.update(data['compact_overrides'])
        settings['dual_mask_private_rank'] = data[machine]['private_rank']
    settings.update(data['variants'][variant])
    settings['wandb_group'] = ('prototype_gradient_route_' if data.get('analysis') == 'prototype_gradient_route'
                               else 'tail_update_') + machine
    if smoke:
        settings.update(max_tasks=2, init_epoch=1, epochs=1, ca_epochs=1,
                        wandb_mode='offline', stage_audit=True)
    return settings


def command_for(machine, name, directory, smoke=False):
    settings = settings_for(machine, name, smoke)
    settings['prefix'] = 'tail_' + machine + '_' + name + '_' + directory.parent.name
    command = [sys.executable, 'main.py', '--config', spec()[machine]['config']]
    for key, value in settings.items():
        command += ['--set', key + '=' + (value if isinstance(value, str) else json.dumps(value))]
    return command, settings


def validate_settings(machine):
    data = spec()
    for name in modes(machine):
        variant, _ = identity(name)
        config = dict(json.loads((ROOT / data[machine]['config']).read_text()), **settings_for(machine, name))
        expected = dict(init_epoch=20, epochs=40 if variant.endswith('40') else 20,
            ca=True, ca_epochs=5, rank={'imgr10':64,'imga10':32}[data[machine]['dataset']],
            dual_mask_anchor_reg_weight=2.5, dual_mask_task0_gate_mode='unmasked',
            dual_mask_permission_mode='asymmetric', dual_mask_branch_layout='dual',
            dual_mask_conflict_granularity='layer', dual_mask_private_conflict_mode='global',
            dual_mask_position_norm_match='off', dual_mask_uniform_norm_matched=False,
            p_permission_release='off', p_direction_score='off', p_hard_zero_mode='off',
            plora_train_a=False, old_competition_weight=0, old_model_distill_weight=0,
            wpre_distill_weight=0, ridge_fusion_enabled=False, save_task_weights=False)
        if any(config.get(key) != value for key, value in expected.items()):
            raise ValueError('Recipe mismatch: ' + name)
        if variant != 'O' and (config['dual_mask_conflict_score_mode'] != 'magnitude'
                or not config['dual_mask_conflict_exact_topk'] or config['dual_mask_conflict_ratio'] != .1
                or config['dual_mask_conflict_budget_multiplier'] != 1):
            raise ValueError('Soft-tail experiment needs full-layer exact10 magnitude selection.')
        if data.get('analysis') == 'prototype_gradient_route':
            route = {'A': 'all', 'B': 'prototype', 'C': 'random_matched'}[variant]
            required = dict(dual_mask_gradient_route=route, dual_mask_fixed_coverage=.9,
                dual_mask_fixed_protect_strength=.5, dual_mask_fixed_conflict_strength=.5,
                dual_mask_private_rank=64, dual_mask_reg_weight=0, slora_gamma=.5,
                plora_gamma=.75, dual_mask_update_rule='step', sp_staged_s_epochs=0,
                pair_separation_weight=0, head_balance_weight=0,
                dual_mask_anchor_reg_task0_only=True)
            if any(config.get(key, 0) != value for key, value in required.items()):
                raise ValueError('Prototype routing requires the fixed M90/rank64 recipe.')


def check_resume(directory, machine, revision):
    manifest = json.loads((directory / 'manifest.json').read_text())
    if (manifest['machine'] != machine or manifest['revision'] != revision
            or manifest.get('sweep_spec','scripts/sweeps/tail_update.json') != str(SPEC.relative_to(ROOT))):
        raise ValueError('Resume requires the original machine, sweep and revision.')
    active_path = directory / 'active.json'
    if active_path.exists():
        active = json.loads(active_path.read_text())
        pids = [active['queue_pid']]
        if active.get('status') not in ('completed', 'failed'):
            pids.append(active['training_pid'])
        for pid in pids:
            try:
                os.kill(pid, 0)
            except ProcessLookupError:
                continue
            raise ValueError('This queue still has a live process: ' + str(pid))
    for filename, smoke in (('queue.json', False), ('smoke_queue.json', True)):
        path = directory / filename
        for row in json.loads(path.read_text()) if path.exists() else []:
            if row['status'] == 'failed':
                raise ValueError('Failure preserved; fix and start remaining modes in a new queue.')
            if row['status'] != 'completed':
                continue
            name = ('smoke_' if smoke else '') + row['mode']
            measured, snapshot, _ = read_run(directory, dict(row, mode=name))
            config = dict(json.loads((ROOT / spec()[machine]['config']).read_text()),
                          **settings_for(machine, row['mode'], smoke))
            matches = all(snapshot['effective_config'].get(k) == v for k,v in config.items() if k != 'prefix')
            hashes = all(hashlib.sha256((ROOT / p).read_bytes()).hexdigest() == digest
                         for p,digest in snapshot['source_sha256'].items())
            if (not matches or not hashes or snapshot['code_revision'] != revision
                    or snapshot['machine'] != machine or snapshot['phase'] != ('smoke' if smoke else 'formal')
                    or measured['runtime_error'] or measured['tasks_reported'] != (2 if smoke else 10)
                    or row.get('exit_code') != 0
                    or not all(measured.get(k) is not None and math.isfinite(measured[k]) for k in METRICS)):
                raise ValueError('Changed or incomplete evidence cannot be reused: ' + name)


def summarize_safely(directory, machine, records):
    try:
        if spec().get('analysis') == 'prototype_gradient_route':
            import analyze_prototype_gradient_route as analyzer
        else:
            import analyze_tail_update as analyzer
        analyzer.summarize_saved(directory)
    except Exception:
        with (directory / 'analysis_errors.jsonl').open('a') as stream:
            stream.write(json.dumps(dict(time=datetime.datetime.now().isoformat(), traceback=traceback.format_exc())) + '\n')
        print('Analysis error saved; scheduled training continues.', flush=True)


def execute(machine, selected, directory, revision, mode='run', hours=10):
    old = engine.command_for, engine.SPEC, engine.EXTRA_SOURCE_PATHS
    engine.command_for, engine.SPEC = command_for, SPEC
    engine.EXTRA_SOURCE_PATHS = ['scripts/run_tail_update.py', 'scripts/analyze_tail_update.py', 'scripts/run_baseline_suite.py']
    if spec().get('analysis') == 'prototype_gradient_route':
        engine.EXTRA_SOURCE_PATHS += ['scripts/analyze_prototype_gradient_route.py',
                                     'scripts/10_09_imgr10_prototype_gradient_route_3090.sh']
    dry = mode == 'dry-run'
    started = time.monotonic()
    seconds_left = hours * 3600
    if not dry:
        budget_path = directory / 'budget.json'
        if budget_path.exists():
            budget = json.loads(budget_path.read_text())
            seconds_left = max(0, budget['deadline_unix'] - time.time())
        else:
            first_start = time.time()
            budget_path.write_text(json.dumps(dict(first_gpu_smoke_start_unix=first_start,
                deadline_unix=first_start + seconds_left, hours=hours), indent=2) + '\n')
    records = {r['mode']:r for r in json.loads((directory / 'queue.json').read_text())} if (directory / 'queue.json').exists() else {}
    smokes = {r['mode']:r for r in json.loads((directory / 'smoke_queue.json').read_text())} if (directory / 'smoke_queue.json').exists() else {}
    pending = [name for name in selected if records.get(name, {}).get('status') != 'completed']
    for name in selected:
        records.setdefault(name, dict(mode=name, status='pending'))
    def save():
        if not dry:
            (directory / 'queue.json').write_text(json.dumps(list(records.values()), indent=2) + '\n')
            summarize_safely(directory, machine, list(records.values()))
    def expired():
        return not dry and time.monotonic() - started >= seconds_left
    def defer(reason):
        for name in pending:
            if records[name]['status'] != 'completed':
                records[name] = dict(mode=name, status='time_budget_pending', reason=reason)
        save()
        print('Time budget reached; unfinished groups preserved.', flush=True)
    try:
        save()
        # The same recipe needs one smoke, not three duplicate seed smokes.
        recipes = set(identity(r['mode'])[0] for r in smokes.values() if r['status'] == 'completed')
        for name in pending:
            recipe = identity(name)[0]
            if recipe in recipes:
                continue
            if expired():
                defer('smoke_budget_incomplete')
                return 0
            record = engine.run(machine, name, directory / ('smoke_' + name), revision, True, dry)
            smokes[name] = record
            if not dry:
                (directory / 'smoke_queue.json').write_text(json.dumps(list(smokes.values()), indent=2) + '\n')
            if record['exit_code']:
                print('Smoke failed; no formal training starts.', flush=True)
                return record['exit_code']
            recipes.add(recipe)
        if mode == 'smoke':
            return 0
        for index, name in enumerate(pending):
            if expired():
                defer('formal_start_deadline')
                return 0
            records[name] = engine.run(machine, name, directory / name, revision, dry_run=dry)
            save()
            if records[name]['exit_code']:
                print('Training failed; this machine is paused.', flush=True)
                return records[name]['exit_code']
            completed = [r for r in records.values() if r['status'] == 'completed']
            if completed:
                # Scale the remaining 40-epoch jobs using measured cost per training epoch.
                per_epoch = sum(r['minutes'] for r in completed) / sum(20 + 9 * settings_for(machine, r['mode'])['epochs'] for r in completed)
                remaining = sum(20 + 9 * settings_for(machine, n)['epochs'] for n in pending[index + 1:])
                print('Measured remaining ETA (minutes):', round(per_epoch * remaining, 1), flush=True)
        print('All requested formal groups completed:', directory, flush=True)
        return 0
    finally:
        engine.command_for, engine.SPEC, engine.EXTRA_SOURCE_PATHS = old


def main():
    global SPEC
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--machine', choices=('3090', '5090'), required=True)
    parser.add_argument('--spec', type=Path, default=SPEC)
    parser.add_argument('--mode', choices=('run', 'smoke', 'dry-run'), default='run')
    parser.add_argument('--modes', nargs='+')
    parser.add_argument('--hours', type=float, default=10)
    parser.add_argument('--resume', type=Path)
    args = parser.parse_args()
    SPEC = args.spec if args.spec.is_absolute() else ROOT / args.spec
    selected = args.modes or modes(args.machine)
    if not math.isfinite(args.hours) or args.hours <= 0 or len(set(selected)) != len(selected) or any(n not in modes(args.machine) for n in selected):
        parser.error('Use positive hours and unique planned modes.')
    validate_settings(args.machine)
    revision = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    queue_name = 'tail_update_' + args.machine if SPEC.name == 'tail_update.json' else SPEC.stem
    directory = args.resume or ROOT / ('logs/shell_logs/' + queue_name) / datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    if args.mode != 'dry-run':
        if args.resume:
            check_resume(directory, args.machine, revision)
        data = baseline.check_data(json.loads((ROOT / spec()[args.machine]['config']).read_text()))
        directory.mkdir(parents=True, exist_ok=True)
        (directory / ('resume_manifest.json' if args.resume else 'manifest.json')).write_text(json.dumps(dict(
            machine=args.machine, revision=revision, queue_pid=os.getpid(), order=selected,
            hours=args.hours, dataset=data, sweep_spec=str(SPEC.relative_to(ROOT)),
            start=datetime.datetime.now().isoformat()), indent=2) + '\n')
    print('Code revision:', revision, '\nQueue PID:', os.getpid(), '\nOutputs:', directory, '\nOrder:', selected, flush=True)
    return execute(args.machine, selected, directory, revision, args.mode, args.hours)


if __name__ == '__main__':
    raise SystemExit(main())
