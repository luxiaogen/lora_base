"""Head-start / Task0-anchor screening and conditional T10 queue, seed1993."""
import argparse
import csv
import datetime
import json
import math
from pathlib import Path
import shlex
import subprocess
import sys


def read_tasks(text):
    tasks = {}
    for line in text.splitlines():
        if 'IncrementalHoldout ' in line:
            row = json.loads(line.split('IncrementalHoldout ', 1)[1])
            tasks[int(row['task'])] = row
    return tasks


def screen_result(reference, candidate, anchor=False):
    """Predeclared validation policy, not a runtime training assertion."""
    reasons, deltas = [], {}
    if set(reference) != {0, 1, 2} or set(candidate) != {0, 1, 2}:
        return {'pass': False, 'reasons': ['incomplete Task0–2 holdout'], 'delta': {}}
    for task in (0, 1, 2):
        a, b = reference[task], candidate[task]
        if (a['sample_sha256'] != b['sample_sha256'] or a['class_count'] != b['class_count']
                or a['source'] != 'train_holdout' or b['source'] != 'train_holdout'):
            reasons.append(f'Task{task} validation identity mismatch')
    if not anchor and abs(reference[0]['metrics']['total']['accuracy'] -
                          candidate[0]['metrics']['total']['accuracy']) > 1e-5:
        reasons.append('Task0 differs although head-start is disabled there')
    for group in ('total', 'old', 'new'):
        differences = [candidate[t]['metrics'][group]['accuracy'] -
                       reference[t]['metrics'][group]['accuracy'] for t in (1, 2)]
        deltas[group] = sum(differences) / 2
        if not all(math.isfinite(x) for x in differences):
            reasons.append(f'{group}: nonfinite validation metric')
        if group != 'total' and min(differences) < -.30 - 1e-6:
            reasons.append(f'{group}: a task drops over 0.30pp')
    if deltas['total'] < .20 - 1e-6:
        reasons.append('mean Total gain below 0.20pp')
    if deltas['old'] < -1e-6 or deltas['new'] < -1e-6:
        reasons.append('mean Old or New declines')
    return {'pass': not reasons, 'reasons': reasons, 'delta': deltas}


def full_candidates(spec, decisions):
    selected = [v for v in spec['variants'] if v['name'] in ('prototype', 'warm', 'proto_warm')
                and decisions.get(v['name'], {}).get('pass', False)]
    anchors = [v for v in spec['variants'] if v['name'].startswith('anchor')
               and decisions.get(v['name'], {}).get('pass', False)]
    if anchors:
        selected.append(max(anchors, key=lambda v: decisions[v['name']]['delta']['total']))
    for warm in ('warm', 'proto_warm'):
        if not any(v['name'] == warm for v in selected):
            continue
        # Match initialization as well as the extra head update count.
        control = dict(spec['training_volume_control'])
        control['overrides'] = dict(control['overrides'])
        if warm == 'proto_warm':
            control['name'] = 'proto_head_post'
            control['overrides']['head_start_init'] = 'prototype'
        selected.append(control)
    return selected


def run_settings(spec, variant, phase, smoke=False):
    settings = dict(spec['common_overrides'])
    settings.update(variant['overrides'])
    settings.update(seed=[1993], max_tasks=10 if phase == 'full' else 3,
                    incremental_holdout=phase != 'full')
    if smoke:
        settings.update(max_tasks=2, init_epoch=1, epochs=1, ca_epochs=1,
                        wandb_mode='offline')
        if settings['head_start_epochs']:
            settings['head_start_epochs'] = 1
    return settings


def run(spec, variant, phase, directory, options):
    settings = run_settings(spec, variant, phase, options.smoke)
    settings['prefix'] = f"{spec['name']}_{phase}_{variant['name']}_{directory.name}"
    command = [sys.executable, 'main.py', '--config', spec['datasets'][0]['config']]
    for key, value in settings.items():
        command += ['--set', key + '=' + (value if isinstance(value, str) else json.dumps(value))]
    print(f"Starting: {variant['name']} phase={phase} {variant['description']}", flush=True)
    print(shlex.join(command), flush=True)
    if options.dry_run:
        return 0, {}
    path = directory / f"{phase}_{variant['name']}.log"
    with path.open('w') as stream:
        stream.write('Command: ' + shlex.join(command) + '\n')
        stream.write('Settings: ' + json.dumps(settings) + '\n')
        with subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                              text=True, bufsize=1) as process:
            for line in process.stdout:
                stream.write(line)
                stream.flush()
                print(line, end='', flush=True)
            code = process.wait()
        stream.write(f'ExitCode: {code}\n')
    print(f"Finished: {variant['name']} phase={phase} exit_code={code} log={path}", flush=True)
    return code, read_tasks(path.read_text())


def save_summary(directory, records, decisions):
    with (directory / 'holdout.csv').open('w', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['variant', 'task', 'total', 'old', 'new', 'sample_sha256'])
        for name, tasks in records.items():
            for task, row in sorted(tasks.items()):
                writer.writerow([name, task] + [row['metrics'][g]['accuracy']
                                                for g in ('total', 'old', 'new')] + [row['sample_sha256']])
    (directory / 'decisions.json').write_text(json.dumps(decisions, indent=2, allow_nan=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('machine', choices=('3090', '5090'))
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--screen-only', action='store_true')
    parser.add_argument('--fresh-control', action='store_true',
                        help='Rerun short control instead of using archived 3090 holdout')
    parser.add_argument('--full', nargs='+', help='Explicit full-run names for a manual continuation')
    args = parser.parse_args()
    spec = json.loads(Path(f'scripts/sweeps/imgr10_head_start_anchor_{args.machine}.json').read_text())
    stamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    directory = Path(spec['log_dir']) / stamp
    revision = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()
    status = subprocess.check_output(['git', 'status', '--short'], text=True)
    print(f'Code revision: {revision}\n{status}\nOutputs: {directory}', flush=True)
    if not args.dry_run:
        directory.mkdir(parents=True)
        (directory / 'manifest.json').write_text(json.dumps({'revision': revision, 'status': status,
                                                          'spec': spec, 'options': vars(args)}, indent=2))
    failed = 0
    if args.full:
        variants = {v['name']: v for v in spec['variants']}
        variants['head_post'] = spec['training_volume_control']
        variants['proto_head_post'] = dict(spec['training_volume_control'], name='proto_head_post',
                                         overrides={'head_start_init': 'prototype', 'head_start_epochs': 5,
                                                    'head_start_stage': 'post'})
        for name in args.full:
            code, _ = run(spec, variants[name], 'full', directory, args)
            failed = failed or code
        return failed
    records, decisions = {}, {}
    control_failed = False
    if spec['reference'] and not (args.smoke or args.fresh_control):
        archived = json.loads(Path(spec['reference']).read_text())
        # Only reuse the declared recipe. Candidate sample hashes/Task0 are checked below.
        reference_settings = dict(archived['settings'])
        current_settings = dict(spec['common_overrides'])
        reference_settings.pop('wandb_group', None)
        current_settings.pop('wandb_group', None)
        if reference_settings == current_settings:
            records['control'] = {int(k): v for k, v in archived['run']['tasks'].items()}
            print('Reusing short control:', spec['reference'], 'raw_sha256=', archived['sha256'], flush=True)
    for variant in spec['variants']:
        name = variant['name']
        if name == 'control' and name in records:
            continue
        code, tasks = run(spec, variant, 'screen', directory, args)
        failed = failed or code
        records[name] = tasks
        if name == 'control':
            control_failed = bool(code)
        if name != 'control' and not (args.smoke or args.dry_run):
            decision = screen_result(records['control'], tasks, anchor=name.startswith('anchor'))
            if code:
                decision.update({'pass': False, 'reasons': decision['reasons'] + ['training process failed']})
            if control_failed:
                decision.update({'pass': False, 'reasons': decision['reasons'] + ['control process failed']})
            decisions[name] = decision
            print('ScreenDecision', name, json.dumps(decision), flush=True)
        if not args.dry_run:
            save_summary(directory, records, decisions)
    if args.smoke:
        code, _ = run(spec, spec['training_volume_control'], 'screen', directory, args)
        return failed or code  # one-epoch route tests are never ranked
    if args.screen_only:
        return failed
    if args.dry_run:
        decisions = {v['name']: {'pass': True, 'delta': {'total': .2}}
                     for v in spec['variants'] if v['name'] != 'control'}
        print('Dry-run only: full commands below are CONDITIONAL examples; anchor tied -> first.', flush=True)
    selected = full_candidates(spec, decisions)
    print('Full T10 queue:', [v['name'] for v in selected], flush=True)
    for variant in selected:
        code, _ = run(spec, variant, 'full', directory, args)
        failed = failed or code
    return failed


if __name__ == '__main__':
    sys.exit(main())
