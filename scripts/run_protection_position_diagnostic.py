"""3090单组 F 搭载只读位置诊断；短测证据完整后才正式训练。"""
from contextlib import contextmanager
import datetime
import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
import traceback

import run_budget_verification as budget
import run_p_fixed_support as reference
from dualmask_config import normalize_dualmask_config

ROOT = budget.ROOT
SPEC = ROOT / 'scripts/sweeps/protection_position_diagnostic_3090.json'
_execute, _check_resume = budget.execute, budget.check_resume
REQUIRED_SOURCE = {'models/attention.py', 'methods/dlora.py', 'utils/protection_position_diagnostic.py',
    'scripts/run_protection_position_diagnostic.py', 'scripts/analyze_protection_position_diagnostic.py',
    'scripts/sweeps/protection_position_diagnostic_3090.json'}


def spec():
    return json.loads(SPEC.read_text())


def modes(machine):
    return ['imgr10_F_seed1993'] if machine == '3090' else []


def identity(machine, name):
    return spec()[machine]['datasets'][0], spec()[machine]['variants'][0]


def settings_for(machine, name, smoke=False):
    settings = reference.settings_for(machine, 'imgr10_A_seed1993')
    settings.update(spec()['common_overrides'], wandb_group='protection_position_diagnostic_3090')
    if smoke:
        settings.update(max_tasks=2, init_epoch=1, epochs=3, ca_epochs=1,
                        position_signal_tasks=[1], position_signal_epochs=[2, 3],
                        wandb_mode='offline', stage_audit=True)
    return settings


def effective_config(machine, name, smoke=False):
    return normalize_dualmask_config(dict(json.loads((ROOT / 'exps/dlora/imgr10.json').read_text()),
                                         **settings_for(machine, name, smoke)))


def validate_settings(machine):
    if machine != '3090':
        raise ValueError('Only the single 3090 diagnostic run is authorized')
    candidate = effective_config(machine, modes(machine)[0])
    original = reference.effective_config(machine, 'imgr10_A_seed1993')
    allowed = set(spec()['common_overrides']) | {'wandb_group'}
    if any(candidate.get(k) != original.get(k) for k in candidate.keys() | original.keys() if k not in allowed):
        raise ValueError('The diagnostic must preserve the reference F recipe')
    if (candidate['dual_mask_fixed_protect_strength'] != .5 or candidate['p_fixed_support'] != 'off'
            or candidate['dual_mask_position_norm_match'] != 'off'
            or candidate['position_signal_tasks'] != [1, 5, 9] or candidate['position_signal_epochs'] != [2, 20]
            or not candidate['dual_mask_position_signal_audit']):
        raise ValueError('The predeclared S-only diagnostic recipe changed')


def command_for(machine, name, directory, smoke=False):
    settings = settings_for(machine, name, smoke)
    settings['prefix'] = 'position_signal_3090_' + name + '_' + directory.parent.name
    command = [sys.executable, 'main.py', '--config', 'exps/dlora/imgr10.json']
    for key, value in settings.items():
        command += ['--set', key + '=' + (value if isinstance(value, str) else json.dumps(value))]
    return command, settings


def verify_diagnostics(directory, smoke):
    from analyze_protection_position_diagnostic import extract_rows, evidence_issues
    return evidence_issues(extract_rows((directory / 'training.log').read_text()),
                           tasks=(1,) if smoke else (1, 5, 9), epochs=(2, 3) if smoke else (2, 20))


def smoke_fingerprint_issues(directory, revision):
    """正式进程创建前重新读取实际源码和环境，与本次短测配对。"""
    from analyze_protection_position_diagnostic import fingerprint_issues
    path = directory.parent / ('smoke_' + directory.name) / 'run.json'
    try:
        smoke = json.loads(path.read_text())
        current = dict(code_revision=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
            machine='3090', phase='formal',
            source_sha256={p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in smoke['source_sha256']},
            hardware=dict(hostname=platform.node(), cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),
                gpus=subprocess.check_output(['nvidia-smi', '--query-gpu=index,uuid,name,pci.bus_id,driver_version',
                    '--format=csv,noheader'], text=True).strip().splitlines()),
            software=dict(python=sys.version, platform=platform.platform(),
                packages={p: importlib.metadata.version(p) for p in ('torch', 'torchvision', 'timm', 'numpy')}))
    except (OSError, ValueError, KeyError, subprocess.CalledProcessError) as error:
        return ['preflight_error: ' + str(error)]
    return fingerprint_issues(current, smoke, revision)


def summarize_safely(directory, machine, records):
    try:
        from analyze_protection_position_diagnostic import summarize_saved
        summarize_saved(directory)
    except Exception:
        with (directory / 'analysis_errors.jsonl').open('a') as stream:
            stream.write(json.dumps(dict(time=datetime.datetime.now().isoformat(), traceback=traceback.format_exc())) + '\n')
        print('Analysis error saved separately.', flush=True)


def execute(machine, selected, directory, revision, mode='run', hours=10):
    engine = budget.tail.engine
    original = engine.run
    def checked_run(machine, name, folder, revision, smoke=False, dry_run=False):
        if not smoke and not dry_run:
            issues = smoke_fingerprint_issues(folder, revision)
            if issues:
                print('Smoke fingerprint changed; no formal process starts:', issues, flush=True)
                return dict(mode=name, status='failed', exit_code=3, minutes=0,
                            reason='smoke_fingerprint_mismatch', issues=issues)
        record = original(machine, name, folder, revision, smoke, dry_run)
        if not dry_run and record.get('exit_code') == 0:
            issues = verify_diagnostics(folder, smoke)
            if issues:
                record.update(status='failed', exit_code=3, reason='incomplete_diagnostic_evidence', issues=issues)
                active_path = folder.parent / 'active.json'
                if active_path.exists():
                    active = json.loads(active_path.read_text())
                    active.update(status='failed', exit_code=3, reason=record['reason'])
                    active_path.write_text(json.dumps(active, indent=2) + '\n')
        return record
    engine.run = checked_run
    try:
        return _execute(machine, selected, directory, revision, mode, hours,
            source_paths=tuple(REQUIRED_SOURCE - {'models/attention.py', 'methods/dlora.py'}) +
                ('scripts/run_p_fixed_support.py', 'scripts/sweeps/p_fixed_support_3090.json',
                 'scripts/sweeps/budget_verification.json', 'scripts/10_12_imgr10_protection_position_diagnostic_3090.sh'))
    finally:
        engine.run = original


def check_resume(directory, machine, revision, selected):
    _check_resume(directory, machine, revision, selected)
    for filename, smoke in (('queue.json', False), ('smoke_queue.json', True)):
        path = directory / filename
        for row in json.loads(path.read_text()) if path.exists() else []:
            if row['status'] == 'completed':
                folder = directory / (('smoke_' if smoke else '') + row['mode'])
                snapshot = json.loads((folder / 'run.json').read_text())
                if not REQUIRED_SOURCE <= snapshot['source_sha256'].keys() or verify_diagnostics(folder, smoke):
                    raise ValueError('Missing or changed diagnostic evidence: ' + str(folder))


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
