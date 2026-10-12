"""同提交七组任务起点实验；复用预算队列的短测、截止与续跑机制。"""
from contextlib import contextmanager
import datetime
import json
import sys
import traceback

import run_budget_verification as budget
from dualmask_config import normalize_dualmask_config

ROOT = budget.ROOT
SPEC = ROOT / 'scripts/sweeps/task_start_prior.json'
_execute = budget.execute
_check_resume = budget.check_resume
REQUIRED_SOURCE = {'models/attention.py', 'utils/task_start_prior.py', 'utils/protection_strength.py',
    'scripts/dualmask_config.py', 'scripts/run_task_start_prior.py',
    'scripts/sweeps/task_start_prior.json', 'scripts/sweeps/budget_verification.json'}


def spec():
    plan = json.loads(SPEC.read_text())
    base = json.loads((SPEC.parent / plan['base_spec']).read_text())
    base['common_overrides'].update(plan['common_overrides'])
    return dict(base, **{k: v for k, v in plan.items() if k != 'common_overrides'})


def modes(machine):
    return ['imgr10_' + v['name'] + '_seed1993' for v in spec()[machine]['variants']]


def identity(machine, name):
    variant = name.removeprefix('imgr10_').removesuffix('_seed1993')
    data = spec()[machine]
    return data['datasets'][0], next(row for row in data['variants'] if row['name'] == variant)


def settings_for(machine, name, smoke=False):
    dataset, variant = identity(machine, name)
    data = spec()
    settings = budget.baseline.settings_for(machine, 'imgr10_seed1993')
    settings.update(data['common_overrides'])
    settings.update(data['compact_overrides'], dual_mask_private_rank=dataset['rank'])
    settings.update(variant['overrides'], wandb_group='task_start_prior_' + machine)
    if smoke:
        settings.update(max_tasks=2, init_epoch=1, epochs=1, ca_epochs=1,
                        wandb_mode='offline', stage_audit=True)
    return settings


def effective_config(machine, name, smoke=False):
    return normalize_dualmask_config(dict(json.loads((ROOT / 'exps/dlora/imgr10.json').read_text()),
                                         **settings_for(machine, name, smoke)))


def validate_settings(machine):
    expected = dict(seed=[1993], rank=64, dual_mask_private_rank=64,
        slora_gamma=.5, plora_gamma=.75, init_epoch=20, epochs=20, ca=True, ca_epochs=5,
        dual_mask_anchor_reg_weight=2.5, dual_mask_task0_gate_mode='unmasked',
        dual_mask_fixed_coverage=.9, dual_mask_fixed_protect_strength=.5,
        dual_mask_fixed_conflict_strength=None, dual_mask_conflict_strength=.5,
        dual_mask_conflict_old_overlap_adaptive=True, dual_mask_reg_weight=0,
        dual_mask_protection_rule='energy_equal', dual_mask_update_rule='step',
        dual_mask_permission_mode='asymmetric', dual_mask_branch_layout='dual',
        dual_mask_conflict_exact_topk=True, dual_mask_conflict_ratio=.1,
        dual_mask_conflict_budget_multiplier=1., dual_mask_private_conflict_mode='global',
        dual_mask_conflict_granularity='layer', dual_mask_position_norm_match='off',
        dual_mask_task_start_probe=True, dual_mask_gradient_route='off',
        dual_mask_prototype_position_probe=False, p_permission_release='off',
        p_permission_position='wpre', p_direction_score='off', plora_train_a=False,
        wpre_distill_weight=0, old_model_distill_weight=0, old_competition_weight=0,
        save_task_weights=False, disable_fused_sdpa=True)
    for name in modes(machine):
        config = effective_config(machine, name)
        if any(config.get(k) != v for k, v in expected.items()):
            raise ValueError('Task-start recipe changed: ' + name)


def command_for(machine, name, directory, smoke=False):
    settings = settings_for(machine, name, smoke)
    settings['prefix'] = 'task_start_' + machine + '_' + name + '_' + directory.parent.name
    command = [sys.executable, 'main.py', '--config', 'exps/dlora/imgr10.json']
    for key, value in settings.items():
        command += ['--set', key + '=' + (value if isinstance(value, str) else json.dumps(value))]
    return command, settings


def summarize_safely(directory, machine, records):
    try:
        from analyze_task_start_prior import summarize_saved
        summarize_saved(directory)
    except Exception:
        with (directory / 'analysis_errors.jsonl').open('a') as stream:
            stream.write(json.dumps(dict(time=datetime.datetime.now().isoformat(),
                                        traceback=traceback.format_exc())) + '\n')
        print('Analysis error saved; planned training continues.', flush=True)


def execute(machine, selected, directory, revision, mode='run', hours=8):
    return _execute(machine, selected, directory, revision, mode, hours,
        source_paths=('scripts/run_task_start_prior.py', 'scripts/analyze_task_start_prior.py',
                      'scripts/sweeps/budget_verification.json'))


def check_resume(directory, machine, revision, selected):
    for filename, smoke in (('queue.json', False), ('smoke_queue.json', True)):
        path = directory / filename
        for row in json.loads(path.read_text()) if path.exists() else []:
            if row['status'] != 'completed':
                continue
            name = ('smoke_' if smoke else '') + row['mode']
            snapshot = json.loads((directory / name / 'run.json').read_text())
            if (not REQUIRED_SOURCE <= snapshot.get('source_sha256', {}).keys()
                    or not snapshot.get('hardware') or not snapshot.get('software')):
                raise ValueError('Missing task-start evidence fingerprint: ' + name)
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
