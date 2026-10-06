"""Original DualMask: four datasets x three seeds, serial 3090 runs without checkpoints."""
import argparse
import datetime
import hashlib
import json
import math
from pathlib import Path
import statistics
import subprocess
import sys
import traceback

from analyze_core_evidence import METRICS, read_run
from analyze_protect_position import write_csv
import run_core_evidence_night as engine


ROOT = engine.ROOT
SPEC = ROOT / 'scripts/sweeps/baseline_4datasets_3090.json'
DATASET_FIELDS = ('dataset', 'init_cls', 'increment', 'total_sessions', 'batch_size', 'optim',
    'init_lr', 'init_weight_decay', 'lrate', 'weight_decay', 'rank', 'scale', 'margin',
    'num_workers', 'ca', 'ca_epochs', 'ca_lrate', 'logit_norm', 'slora_gamma', 'plora_gamma',
    'dual_mask_general_ratio')


def modes():
    spec = json.loads(SPEC.read_text())
    return [item['name'] + '_seed' + str(seed) for item in spec['datasets'] for seed in spec['seeds']]


def run_identity(name):
    dataset, seed = name.rsplit('_seed', 1)
    return next(item for item in json.loads(SPEC.read_text())['datasets'] if item['name'] == dataset), int(seed)


def settings_for(machine, name, smoke=False):
    item, seed = run_identity(name)
    config = json.loads((ROOT / item['config']).read_text())
    # Keep the original dataset recipes, not ImageNet-R's learning rate/rank/CA/margin.
    settings = dict(json.loads(engine.BASE.read_text())['common_overrides'])
    for field in DATASET_FIELDS:
        settings.pop(field, None)
    settings.update(json.loads(SPEC.read_text())['common_overrides'])
    settings.update(item.get('overrides', {}))
    settings.update(seed=[seed], task0_margin=config['margin'])
    if smoke:
        settings.update(max_tasks=2, init_epoch=1, epochs=1, ca_epochs=1, wandb_mode='offline')
    return settings


def command_for(machine, name, directory, smoke=False):
    item, _ = run_identity(name)
    settings = settings_for(machine, name, smoke)
    settings['prefix'] = 'baseline_' + name + '_' + directory.parent.name
    command = [sys.executable, 'main.py', '--config', item['config']]
    for key, value in settings.items():
        command += ['--set', key + '=' + (value if isinstance(value, str) else json.dumps(value))]
    return command, settings


def validate_settings():
    original = dict(dual_mask_conflict_score_mode='conflict', dual_mask_conflict_exact_topk=False,
        dual_mask_private_rank=0, dual_mask_reg_weight=.01, dual_mask_branch_layout='dual',
        dual_mask_permission_mode='asymmetric', dual_mask_fixed_coverage=None,
        dual_mask_fixed_protect_strength=None, dual_mask_fixed_conflict_strength=None,
        dual_mask_anchor_reg_weight=2.5, save_task_weights=False)
    for name in modes():
        settings = settings_for('3090', name)
        if any(settings.get(key) != value for key, value in original.items()):
            raise ValueError('This queue requires original DualMask, not M or a scoring candidate.')


def check_data(config):
    root = Path(config['data_path'])
    dataset = config['dataset'].lower()
    if dataset == 'cifar100':
        if not all((root / 'cifar-100-python' / name).is_file() for name in ('train', 'test', 'meta')):
            raise FileNotFoundError('CIFAR-100 files missing; no automatic download: ' + str(root))
        from torchvision.datasets import CIFAR100
        train, test = CIFAR100(str(root), train=True, download=False), CIFAR100(str(root), train=False, download=False)
    else:
        if not all((root / name).is_dir() for name in ('train', 'test')):
            raise FileNotFoundError('Existing train/test splits required; no automatic split: ' + str(root))
        from torchvision.datasets import ImageFolder
        train, test = ImageFolder(str(root / 'train')), ImageFolder(str(root / 'test'))
        if train.class_to_idx != test.class_to_idx or len(train.classes) != 200:
            raise ValueError('Expected identical 200-class train/test mappings: ' + str(root))
    return dict(dataset=config['dataset'], data_path=str(root), train_count=len(train), test_count=len(test),
                class_count=len(train.classes))


def summarize(directory, records):
    summaries, tasks = [], []
    for record in records:
        if record['status'] not in ('completed', 'failed'):
            continue
        measured, snapshot, rows = read_run(directory, record)
        measured.update(dataset=snapshot['effective_config']['dataset'], seed=snapshot['effective_config']['seed'][0])
        summaries.append(measured)
        tasks.extend(rows['tasks'])
    aggregate = []
    for item in json.loads(SPEC.read_text())['datasets']:
        selected = [row for row in summaries if row['mode'].startswith(item['name'] + '_seed') and row['valid_performance']]
        expected = set(json.loads(SPEC.read_text())['seeds'])
        complete = {row['seed'] for row in selected} == expected
        row = dict(dataset=item['name'], completed_seeds=sorted(r['seed'] for r in selected), complete_three_seeds=complete)
        if complete:
            for key in METRICS:
                values = [r[key] for r in selected]
                row.update({key + '_mean': statistics.fmean(values), key + '_std': statistics.stdev(values)})
        aggregate.append(row)
    for filename, rows in (('results', summaries), ('aggregate', aggregate)):
        (directory / (filename + '.json')).write_text(json.dumps(rows, indent=2) + '\n')
        write_csv(directory / (filename + '.csv'), rows)
    write_csv(directory / 'tasks.csv', tasks)


def check_resume(directory, revision):
    manifest = json.loads((directory / 'manifest.json').read_text())
    if manifest['revision'] != revision:
        raise ValueError('Resume needs the original code revision.')
    for filename, smoke in (('queue.json', False), ('smoke_queue.json', True)):
        path = directory / filename
        for record in json.loads(path.read_text()) if path.exists() else []:
            if record['status'] == 'failed':
                raise ValueError('Failed evidence preserved; fix first, do not automatically retry.')
            if record['status'] != 'completed':
                continue
            name = ('smoke_' if smoke else '') + record['mode']
            measured, snapshot, _ = read_run(directory, dict(record, mode=name))
            item, _ = run_identity(record['mode'])
            expected = dict(json.loads((ROOT / item['config']).read_text()), **settings_for('3090', record['mode'], smoke))
            config_match = all(snapshot['effective_config'].get(k) == v for k,v in expected.items() if k != 'prefix')
            source_match = all(hashlib.sha256((ROOT / p).read_bytes()).hexdigest() == digest
                               for p,digest in snapshot['source_sha256'].items())
            if (snapshot['code_revision'] != revision or snapshot['machine'] != '3090'
                    or snapshot['phase'] != ('smoke' if smoke else 'formal') or not config_match or not source_match
                    or record.get('exit_code') != 0 or measured['runtime_error']
                    or measured['tasks_reported'] != (2 if smoke else 10)
                    or not all(measured.get(k) is not None and math.isfinite(measured[k]) for k in METRICS)):
                raise ValueError('Cannot reuse changed or incomplete group: ' + name)


def execute(directory, revision, mode='run'):
    previous = engine.command_for, engine.SPEC, engine.EXTRA_SOURCE_PATHS
    engine.command_for, engine.SPEC = command_for, SPEC
    engine.EXTRA_SOURCE_PATHS = ['scripts/run_baseline_suite.py']
    try:
        dry = mode == 'dry-run'
        records = {row['mode']:row for row in json.loads((directory / 'queue.json').read_text())} if (directory / 'queue.json').exists() else {}
        smokes = {row['mode']:row for row in json.loads((directory / 'smoke_queue.json').read_text())} if (directory / 'smoke_queue.json').exists() else {}
        for name in modes():
            records.setdefault(name, dict(mode=name, status='pending'))
        if not dry:
            (directory / 'queue.json').write_text(json.dumps(list(records.values()), indent=2) + '\n')
        # One two-task smoke per dataset; seeds do not need duplicate path smokes.
        for name in modes()[::3]:
            if smokes.get(name, {}).get('status') == 'completed':
                continue
            smokes[name] = engine.run('3090', name, directory / ('smoke_' + name), revision, True, dry)
            if not dry:
                (directory / 'smoke_queue.json').write_text(json.dumps(list(smokes.values()), indent=2) + '\n')
            if smokes[name]['exit_code']:
                return smokes[name]['exit_code']
        if mode == 'smoke':
            return 0
        for name in modes():
            if records[name]['status'] == 'completed':
                continue
            records[name] = engine.run('3090', name, directory / name, revision, dry_run=dry)
            if not dry:
                (directory / 'queue.json').write_text(json.dumps(list(records.values()), indent=2) + '\n')
                try:
                    summarize(directory, list(records.values()))
                except Exception:
                    with (directory / 'analysis_errors.jsonl').open('a') as stream:
                        stream.write(json.dumps(dict(traceback=traceback.format_exc())) + '\n')
                    print('Analysis error saved; scheduled training continues.', flush=True)
            if records[name]['exit_code']:
                return records[name]['exit_code']
        print('All 12 formal T10 runs finished:', directory, flush=True)
        return 0
    finally:
        engine.command_for, engine.SPEC, engine.EXTRA_SOURCE_PATHS = previous


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=('run', 'smoke', 'dry-run'), default='run')
    parser.add_argument('--resume', type=Path)
    args = parser.parse_args()
    validate_settings()
    revision = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    directory = args.resume or ROOT / 'logs/shell_logs/baseline_4datasets_3090' / datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    if args.mode != 'dry-run':
        if args.resume:
            check_resume(directory, revision)
        data = [check_data(json.loads((ROOT / item['config']).read_text())) for item in json.loads(SPEC.read_text())['datasets']]
        directory.mkdir(parents=True, exist_ok=True)
        (directory / ('resume_manifest.json' if args.resume else 'manifest.json')).write_text(json.dumps(dict(
            revision=revision, machine='3090', queue_pid=engine.os.getpid(), modes=modes(), datasets=data), indent=2) + '\n')
    print('Code revision:', revision, '\nQueue PID:', engine.os.getpid(), '\nOutputs:', directory, flush=True)
    print('Original DualMask, anchor2.5, 12 T10 runs; no checkpoints, no time cutoff.', flush=True)
    return execute(directory, revision, args.mode)


if __name__ == '__main__':
    raise SystemExit(main())
