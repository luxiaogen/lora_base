"""3090单组普通LoRA对照，Task0保留F，增量AB训练参数量匹配F。"""
from contextlib import contextmanager
import datetime
import json
import sys
import traceback

import run_final_f_validation as final_f
from analyze_core_evidence import read_run, write_csv
from dualmask_config import normalize_dualmask_config

ROOT = final_f.ROOT
SPEC = ROOT / 'scripts/sweeps/plain_lora_control.json'
MODE = 'imgr10_PlainLoRA_seed1993'
SOURCE_PATHS = ('scripts/run_plain_lora_control.py', 'scripts/sweeps/plain_lora_control.json',
    'scripts/10_13_imgr10_plain_lora_3090.sh', 'scripts/run_final_f_validation.py',
    'scripts/analyze_final_f_validation.py', 'scripts/sweeps/final_f_validation.json')


def spec():
    return json.loads(SPEC.read_text())


def modes(machine):
    return [MODE] if machine == '3090' else []


def identity(machine, name):
    return final_f.identity(machine, 'imgr10_C11_seed1993')


def settings_for(machine, name, smoke=False):
    settings = final_f.settings_for(machine, 'imgr10_C11_seed1993', smoke)
    settings.update(spec()['overrides'], wandb_group='plain_lora_control_3090')
    return settings


def effective_config(machine, name, smoke=False):
    return normalize_dualmask_config(dict(json.loads((ROOT / 'exps/dlora/imgr10.json').read_text()),
                                         **settings_for(machine, name, smoke)))


def validate_settings(machine):
    if machine != '3090':
        raise ValueError('Only the authorized 3090 single-seed control is available')
    final_f.validate_settings(machine)
    config = effective_config(machine, MODE)
    if config['rank'] != 64 or config['dual_mask_private_rank'] != 64:
        raise ValueError('This control matches F64/64 with trainable AB rank96')


def command_for(machine, name, directory, smoke=False):
    settings = settings_for(machine, name, smoke)
    settings['prefix'] = 'plain_lora_3090_' + name + '_' + directory.parent.name
    command = [sys.executable, 'main.py', '--config', 'exps/dlora/imgr10.json']
    for key, value in settings.items():
        command += ['--set', key + '=' + (value if isinstance(value, str) else json.dumps(value))]
    return command, settings


def summarize_safely(directory, machine, records):
    try:
        measured, tables = [], {}
        for record in records:
            path = directory / record['mode']
            if not (path / 'run.json').exists():
                measured.append(record)
                continue
            summary, _, rows = read_run(directory, record)
            summary.update(seed=1993, std=None, complete_three_seeds=False)
            measured.append(summary)
            for key, values in rows.items():
                tables.setdefault(key, []).extend(values)
        (directory / 'results.json').write_text(json.dumps(measured, indent=2) + '\n')
        write_csv(directory / 'results.csv', measured)
        for key, rows in tables.items():
            write_csv(directory / (key + '.csv'), rows)
    except Exception:
        with (directory / 'analysis_errors.jsonl').open('a') as stream:
            stream.write(json.dumps(dict(time=datetime.datetime.now().isoformat(),
                                        traceback=traceback.format_exc())) + '\n')
        print('Analysis error saved; planned training continues.', flush=True)


def execute(machine, selected, directory, revision, mode='run', hours=2):
    return final_f._execute(machine, selected, directory, revision, mode, hours,
                            source_paths=SOURCE_PATHS)


def check_resume(directory, machine, revision, selected):
    return final_f.check_resume(directory, machine, revision, selected)


@contextmanager
def configured_runner():
    names = ('SPEC', 'spec', 'modes', 'identity', 'settings_for', 'effective_config',
             'validate_settings', 'command_for', 'summarize_safely', 'execute', 'check_resume')
    previous = {name: getattr(final_f.budget, name) for name in names}
    for name in names:
        setattr(final_f.budget, name, globals()[name])
    try:
        yield
    finally:
        for name, value in previous.items():
            setattr(final_f.budget, name, value)


if __name__ == '__main__':
    with configured_runner():
        raise SystemExit(final_f.budget.main())
