"""3090同提交四组固定P支持；沿用短测先行、截止与续跑引擎。"""
from contextlib import contextmanager
import datetime
import json
import sys
import traceback

import run_budget_verification as budget
from dualmask_config import normalize_dualmask_config

ROOT = budget.ROOT
SPEC = ROOT / 'scripts/sweeps/p_fixed_support_3090.json'
_execute, _check_resume = budget.execute, budget.check_resume
REQUIRED_SOURCE = {'models/attention.py', 'methods/dlora.py', 'utils/p_fixed_support.py',
    'scripts/dualmask_config.py', 'scripts/run_p_fixed_support.py',
    'scripts/analyze_p_fixed_support.py', 'scripts/sweeps/p_fixed_support_3090.json'}


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
    settings.update(variant['overrides'], wandb_group='p_fixed_support_' + machine)
    if smoke:
        settings.update(max_tasks=2, init_epoch=1, epochs=2, ca_epochs=1,
                        p_fixed_support_warmup_epochs=1, wandb_mode='offline', stage_audit=True)
    return settings


def effective_config(machine, name, smoke=False):
    return normalize_dualmask_config(dict(json.loads((ROOT / 'exps/dlora/imgr10.json').read_text()),
                                         **settings_for(machine, name, smoke)))


def validate_settings(machine):
    if machine != '3090':
        raise ValueError('This queue is authorized for 3090 only')
    expected = dict(seed=[1993], rank=64, dual_mask_private_rank=64,
        slora_gamma=.5, plora_gamma=.75, init_epoch=20, epochs=20, ca=True, ca_epochs=5,
        dual_mask_anchor_reg_weight=2.5, dual_mask_task0_gate_mode='unmasked',
        dual_mask_fixed_coverage=.9, dual_mask_fixed_protect_strength=.5,
        dual_mask_fixed_conflict_strength=None, dual_mask_conflict_strength=.5,
        dual_mask_conflict_old_overlap_adaptive=True, dual_mask_ncm_conflict_mode='scaled',
        dual_mask_reg_weight=0, dual_mask_protection_rule='static', dual_mask_update_rule='step',
        dual_mask_conflict_exact_topk=True, dual_mask_conflict_ratio=.1,
        dual_mask_conflict_budget_multiplier=1., p_fixed_support_keep_ratio=.5,
        p_fixed_support_warmup_epochs=2, p_fixed_support_audit=True,
        dual_mask_conflict_score_mode='magnitude', dual_mask_position_audit=False,
        dual_mask_mechanism_audit=True, save_task_weights=False, disable_fused_sdpa=True)
    for name in modes(machine):
        config = effective_config(machine, name)
        if any(config.get(k) != v for k, v in expected.items()):
            raise ValueError('Fixed P-support recipe changed: ' + name)


def command_for(machine, name, directory, smoke=False):
    settings = settings_for(machine, name, smoke)
    settings['prefix'] = 'p_support_' + machine + '_' + name + '_' + directory.parent.name
    command = [sys.executable, 'main.py', '--config', 'exps/dlora/imgr10.json']
    for key, value in settings.items():
        command += ['--set', key + '=' + (value if isinstance(value, str) else json.dumps(value))]
    return command, settings


def summarize_safely(directory, machine, records):
    try:
        from analyze_p_fixed_support import summarize_saved
        summarize_saved(directory)
    except Exception:
        with (directory / 'analysis_errors.jsonl').open('a') as stream:
            stream.write(json.dumps(dict(time=datetime.datetime.now().isoformat(),
                                        traceback=traceback.format_exc())) + '\n')
        print('Analysis error saved; planned training continues.', flush=True)


def execute(machine, selected, directory, revision, mode='run', hours=10):
    return _execute(machine, selected, directory, revision, mode, hours,
        source_paths=('scripts/run_p_fixed_support.py', 'scripts/analyze_p_fixed_support.py',
                      'scripts/sweeps/budget_verification.json',
                      'scripts/10_12_imgr10_p_fixed_support_3090.sh'))


def check_resume(directory, machine, revision, selected):
    for filename, smoke in (('queue.json', False), ('smoke_queue.json', True)):
        path = directory / filename
        for row in json.loads(path.read_text()) if path.exists() else []:
            if row['status'] == 'completed':
                name = ('smoke_' if smoke else '') + row['mode']
                snapshot = json.loads((directory / name / 'run.json').read_text())
                if not REQUIRED_SOURCE <= snapshot.get('source_sha256', {}).keys():
                    raise ValueError('Missing fixed-support source fingerprint: ' + name)
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
