"""3090分层覆盖×S保护强度；默认先完成全部短测，再串行T10。"""
from contextlib import contextmanager
import datetime
import json
import math
import sys
import traceback

import run_budget_verification as budget
from dualmask_config import normalize_dualmask_config

ROOT = budget.ROOT
SPEC = ROOT / 'scripts/sweeps/depth_permissions_3090.json'
_execute, _check_resume = budget.execute, budget.check_resume
REQUIRED_SOURCE = {'models/attention.py', 'methods/dlora.py', 'utils/depth_permissions.py',
    'scripts/dualmask_config.py', 'scripts/run_depth_permissions.py',
    'scripts/analyze_depth_permissions.py', 'scripts/sweeps/depth_permissions_3090.json'}


def spec():
    plan = json.loads(SPEC.read_text())
    base = json.loads((SPEC.parent / plan['base_spec']).read_text())
    base['common_overrides'].update(plan['common_overrides'])
    return dict(base, **{k: v for k, v in plan.items() if k != 'common_overrides'})


def modes(machine):
    return ['imgr10_' + row['name'] + '_seed1993' for row in spec()[machine]['variants']]


def identity(machine, name):
    variant = name.removeprefix('imgr10_').removesuffix('_seed1993')
    return spec()[machine]['datasets'][0], next(
        row for row in spec()[machine]['variants'] if row['name'] == variant)


def settings_for(machine, name, smoke=False):
    _, variant = identity(machine, name)
    settings = budget.baseline.settings_for(machine, 'imgr10_seed1993')
    settings.update(spec()['common_overrides'])
    settings.update(spec()['compact_overrides'], dual_mask_private_rank=64)
    settings.update(variant['overrides'], wandb_group='depth_permissions_' + machine)
    if smoke:
        settings.update(max_tasks=2, init_epoch=1, epochs=1, ca_epochs=1,
                        wandb_mode='offline', stage_audit=True)
    return settings


def effective_config(machine, name, smoke=False):
    return normalize_dualmask_config(dict(json.loads((ROOT / 'exps/dlora/imgr10.json').read_text()),
                                         **settings_for(machine, name, smoke)))


def validate_settings(machine):
    if machine != '3090':
        raise ValueError('This queue is authorized for 3090 only')
    expected = dict(seed=[1993], rank=64, dual_mask_private_rank=64, slora_gamma=.5,
        plora_gamma=.75, init_epoch=20, epochs=20, ca=True, ca_epochs=5,
        dual_mask_anchor_reg_weight=2.5, dual_mask_task0_gate_mode='unmasked',
        dual_mask_conflict_granularity='layer', dual_mask_private_conflict_mode='global',
        dual_mask_conflict_budget_multiplier=1., dual_mask_conflict_old_overlap_adaptive=True,
        dual_mask_conflict_merge_mode='suppress', dual_mask_composed_conflict='off',
        dual_mask_gradient_route='off', plora_train_a=False, sp_staged_s_epochs=0,
        dual_mask_depth_audit=True, save_task_weights=False, disable_fused_sdpa=True)
    for name in modes(machine):
        config = effective_config(machine, name)
        if any(config.get(k) != v for k, v in dict(expected, **spec()['compact_overrides']).items()):
            raise ValueError('Depth permission recipe changed: ' + name)


def command_for(machine, name, directory, smoke=False):
    settings = settings_for(machine, name, smoke)
    settings['prefix'] = 'depth_' + machine + '_' + name + '_' + directory.parent.name
    command = [sys.executable, 'main.py', '--config', 'exps/dlora/imgr10.json']
    for key, value in settings.items():
        command += ['--set', key + '=' + (value if isinstance(value, str) else json.dumps(value))]
    return command, settings


def summarize_safely(directory, machine, records):
    try:
        from analyze_depth_permissions import summarize_saved
        summarize_saved(directory)
    except Exception:
        with (directory / 'analysis_errors.jsonl').open('a') as stream:
            stream.write(json.dumps(dict(time=datetime.datetime.now().isoformat(),
                                        traceback=traceback.format_exc())) + '\n')
        print('Analysis error saved; planned training continues.', flush=True)


def smoke_issues(directory, name):
    from analyze_depth_permissions import permission_issues
    from analyze_protect_position import json_rows
    try:
        snapshot = json.loads((directory / 'run.json').read_text())
        config = effective_config('3090', name, True)
        issues = ['smoke_config'] if any(snapshot['effective_config'].get(k) != v
            for k, v in config.items() if k != 'prefix') else []
        if not REQUIRED_SOURCE <= snapshot.get('source_sha256', {}).keys():
            issues.append('missing_source_fingerprint')
        content = (directory / 'training.log').read_text()
        issues += permission_issues(json_rows(content, 'DepthPermission'),
                                   json_rows(content, 'DepthPermissionBudget'), config, tasks=(1,))
        merges = json_rows(content, 'AppliedConflictBudget')
        expected = {(task, layer, branch) for task in (0, 1) for layer in range(12)
                    for branch in (('S',) if task == 0 else ('S', 'P'))}
        if (len(merges) != len(expected)
                or {(r['task'], r['layer'], r['branch']) for r in merges} != expected
                or any(not math.isfinite(r['merge_error']) or r['merge_error'] > 1e-6 for r in merges)):
            issues.append('merge_evidence')
        return sorted(set(issues))
    except (OSError, ValueError, KeyError, TypeError):
        return ['unreadable_smoke_evidence']


def execute(machine, selected, directory, revision, mode='run', hours=10):
    engine = budget.tail.engine
    original = engine.run
    def checked_run(machine, name, folder, revision, smoke=False, dry_run=False):
        record = original(machine, name, folder, revision, smoke, dry_run)
        if smoke and not dry_run and record['status'] == 'completed':
            issues = smoke_issues(folder, name)
            if issues:
                record.update(status='failed', exit_code=2, reason='depth_smoke_evidence', issues=issues)
                active_path = folder.parent / 'active.json'
                if active_path.exists():
                    active = json.loads(active_path.read_text())
                    active.update(status='failed', exit_code=2, issues=issues)
                    active_path.write_text(json.dumps(active, indent=2) + '\n')
                print('Depth smoke evidence failed:', issues, flush=True)
        return record
    engine.run = checked_run
    try:
        return _execute(machine, selected, directory, revision, mode, hours,
            source_paths=tuple(sorted(REQUIRED_SOURCE)) + (
                'scripts/sweeps/budget_verification.json', 'scripts/10_13_imgr10_depth_permissions_3090.sh'))
    finally:
        engine.run = original


def check_resume(directory, machine, revision, selected):
    for filename, smoke in (('queue.json', False), ('smoke_queue.json', True)):
        path = directory / filename
        for row in json.loads(path.read_text()) if path.exists() else []:
            if row['status'] == 'completed':
                name = ('smoke_' if smoke else '') + row['mode']
                snapshot = json.loads((directory / name / 'run.json').read_text())
                if not REQUIRED_SOURCE <= snapshot.get('source_sha256', {}).keys():
                    raise ValueError('Missing depth source fingerprint: ' + name)
    return _check_resume(directory, machine, revision, selected)


@contextmanager
def configured_runner():
    names = ('SPEC', 'spec', 'modes', 'identity', 'settings_for', 'effective_config',
             'validate_settings', 'command_for', 'summarize_safely', 'execute', 'check_resume')
    previous = {name: getattr(budget, name) for name in names}
    for name in names:
        setattr(budget, name, globals()[name])
    try:
        yield
    finally:
        for name, value in previous.items():
            setattr(budget, name, value)


if __name__ == '__main__':
    with configured_runner():
        raise SystemExit(budget.main())
