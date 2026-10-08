"""Original O: three protection positions x three seeds, serial T10 without checkpoints."""
import argparse
import datetime
import hashlib
import json
import math
from pathlib import Path
import statistics
import subprocess
import traceback
from urllib.parse import urlsplit

from analyze_core_evidence import METRICS, read_run
from analyze_protect_position import write_csv
import run_baseline_suite as baseline


engine = baseline.engine
ROOT = engine.ROOT
SPEC = ROOT / 'scripts/sweeps/imgr10_region_prior_3090.json'
REQUIRED_SOURCE = {'models/attention.py', 'methods/dlora.py', 'utils/protect_position.py'}


def complete_fingerprint(snapshot):
    return (REQUIRED_SOURCE <= snapshot.get('source_sha256', {}).keys()
            and bool(snapshot.get('hardware')) and bool(snapshot.get('software')))


def pretrained_fingerprint():
    """Record the cached artifact used by the unchanged model loader; never download it here."""
    engine.sys.path.insert(0, str(ROOT))
    import torch
    from models.vit import resolve_pretrained_cfg
    config = resolve_pretrained_cfg('vit_base_patch16_224_in21k')
    path = Path(torch.hub.get_dir()) / 'checkpoints' / Path(urlsplit(config['url']).path).name
    checksum = hashlib.sha256()
    if path.is_file():
        with path.open('rb') as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b''):
                checksum.update(block)
    return dict(variant='vit_base_patch16_224_in21k', config=config, cached_path=str(path),
                exists=path.is_file(), sha256=checksum.hexdigest() if path.is_file() else None)


def modes():
    spec = json.loads(SPEC.read_text())
    return [row['name'] + '_seed' + str(seed) for seed in spec['seeds'] for row in spec['variants']]


def settings_for(machine, name, smoke=False):
    variant, seed = name.rsplit('_seed', 1)
    spec = json.loads(SPEC.read_text())
    settings = baseline.settings_for(machine, 'imgr10_seed' + seed, smoke)
    settings.update(spec['common_overrides'])
    settings['dual_mask_protect_position'] = next(row['position'] for row in spec['variants'] if row['name'] == variant)
    return settings


def command_for(machine, name, directory, smoke=False):
    settings = settings_for(machine, name, smoke)
    settings['prefix'] = 'region_prior_' + name + '_' + directory.parent.name
    command = [engine.sys.executable, 'main.py', '--config', 'exps/dlora/imgr10.json']
    for key, value in settings.items():
        command += ['--set', key + '=' + (value if isinstance(value, str) else json.dumps(value))]
    return command, settings


def summarize(directory, records):
    results, snapshots, tasks, updates, masks = [], {}, [], [], []
    expected_updates = {(task, layer, branch, projection) for task in range(10) for layer in range(12)
                        for branch in (('S',) if task == 0 else ('S', 'P')) for projection in ('Q', 'K', 'V')}
    expected_masks = {(task, layer) for task in range(10) for layer in range(12)}
    for record in records:
        if record['status'] not in ('completed', 'failed'):
            continue
        measured, snapshot, rows = read_run(directory, record)
        variant, seed = record['mode'].rsplit('_seed', 1)
        measured.update(mode=record['mode'], variant=variant, seed=int(seed),
                        diagnostics_complete={(r['task'], r['layer'], r['branch'], r['projection'])
                            for r in rows['updates']} == expected_updates
                        and {(r['task'], r['layer']) for r in rows['masks']} == expected_masks)
        results.append(measured)
        snapshots[record['mode']] = snapshot
        for target, key in ((tasks, 'tasks'), (updates, 'updates'), (masks, 'masks')):
            target.extend(rows[key])
    for name, rows in (('results', results), ('tasks', tasks), ('updates', updates), ('masks', masks)):
        write_csv(directory / (name + '.csv'), rows)
    (directory / 'results.json').write_text(json.dumps(results, indent=2) + '\n')
    good = {r['mode']: r for r in results if r['valid_performance'] and r['diagnostics_complete']}
    issues, pairs, means = [], [], []
    grid_reference = next(iter(snapshots.values()), None)
    for name, snapshot in snapshots.items():
        seed = int(name.rsplit('_seed', 1)[1])
        if not complete_fingerprint(snapshot):
            issues.append(dict(seed=seed, mode=name, field='missing_fingerprint'))
        expected = dict(json.loads((ROOT / 'exps/dlora/imgr10.json').read_text()),
                        **settings_for('3090', name))
        for key, value in expected.items():
            if key != 'prefix' and snapshot['effective_config'].get(key) != value:
                issues.append(dict(seed=seed, mode=name, field='expected_config.' + key))
        for key, value in (('machine', '3090'), ('phase', 'formal')):
            if snapshot.get(key) != value:
                issues.append(dict(seed=seed, mode=name, field='expected_' + key))
        for key in ('code_revision', 'source_sha256', 'hardware', 'software'):
            if snapshot.get(key) != grid_reference.get(key):
                issues.append(dict(seed=seed, mode=name, field='grid.' + key))
    for seed in json.loads(SPEC.read_text())['seeds']:
        names = [row['name'] + '_seed' + str(seed) for row in json.loads(SPEC.read_text())['variants']]
        if not all(name in good for name in names):
            continue
        first = snapshots[names[0]]
        for name in names[1:]:
            other = snapshots[name]
            for key in ('code_revision', 'source_sha256', 'hardware', 'software', 'machine', 'phase'):
                if first[key] != other[key]:
                    issues.append(dict(seed=seed, mode=name, field=key))
            for key in first['effective_config'].keys() | other['effective_config'].keys():
                if key not in ('prefix', 'dual_mask_protect_position') and first['effective_config'].get(key) != other['effective_config'].get(key):
                    issues.append(dict(seed=seed, mode=name, field='config.' + key))
            if good[name]['Task0'] != good[names[0]]['Task0']:
                issues.append(dict(seed=seed, mode=name, field='Task0_startpoint'))
            left = [r for r in masks if r['mode'] == names[0]]
            right = [r for r in masks if r['mode'] == name]
            counts = lambda rows: {(r['task'], r['layer']): r['qkv_protect_counts'] for r in rows}
            if counts(left) != counts(right):
                issues.append(dict(seed=seed, mode=name, field='per_projection_protection_budget'))
        if not any(row['seed'] == seed for row in issues):
            for name in names[1:]:
                pairs.append(dict(seed=seed, contrast='A_minus_' + name.rsplit('_seed', 1)[0],
                                  **{key: good[names[0]][key] - good[name][key] for key in METRICS}))
    complete = set(good) == set(modes()) and not issues
    if complete:
        for variant in json.loads(SPEC.read_text())['variants']:
            selected = [r for r in results if r['variant'] == variant['name']]
            row = dict(variant=variant['name'], n=3)
            for key in METRICS:
                values = [r[key] for r in selected]
                row.update({key + '_mean': statistics.fmean(values), key + '_std': statistics.stdev(values)})
            means.append(row)
    report = dict(complete_nine_units=complete, matching_issues=issues, paired_differences=pairs,
                  group_means=means, uncertainty='three paired seeds; no significance or equivalence claim')
    (directory / 'aggregate.json').write_text(json.dumps(report, indent=2) + '\n')
    for filename, rows, empty_header in (('pairs.csv', pairs, 'seed,contrast'),
                                         ('aggregate.csv', means, 'variant,n')):
        if rows:
            write_csv(directory / filename, rows)
        else:
            # These are derived reports, never raw evidence: avoid stale successful aggregates.
            (directory / filename).write_text(empty_header + '\n')


def execute(directory, revision, mode='run'):
    previous = engine.command_for, engine.SPEC, engine.EXTRA_SOURCE_PATHS
    engine.command_for, engine.SPEC = command_for, SPEC
    engine.EXTRA_SOURCE_PATHS = ['scripts/run_region_prior.py', 'scripts/run_baseline_suite.py',
                                 str(baseline.SPEC.relative_to(ROOT))]
    dry = mode == 'dry-run'
    try:
        read = lambda name: {r['mode']: r for r in json.loads((directory / name).read_text())} if (directory / name).exists() else {}
        records, smokes = read('queue.json'), read('smoke_queue.json')
        for name in modes():
            records.setdefault(name, dict(mode=name, status='pending'))
        if not dry:
            (directory / 'queue.json').write_text(json.dumps(list(records.values()), indent=2) + '\n')
        for name in modes()[:3]:
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
                    print('Analysis error saved; training queue continues.', flush=True)
            if records[name]['exit_code']:
                return records[name]['exit_code']
        print('Nine formal T10 units finished:', directory, flush=True)
        return 0
    finally:
        engine.command_for, engine.SPEC, engine.EXTRA_SOURCE_PATHS = previous


def check_resume(directory, revision):
    if json.loads((directory / 'manifest.json').read_text())['revision'] != revision:
        raise ValueError('Resume requires the original revision; previous evidence is preserved.')
    for filename, smoke in (('queue.json', False), ('smoke_queue.json', True)):
        path = directory / filename
        for record in json.loads(path.read_text()) if path.exists() else []:
            if record['status'] == 'failed':
                raise ValueError('Failed run preserved; diagnose before starting another queue.')
            if record['status'] != 'completed':
                unit = directory / (('smoke_' if smoke else '') + record['mode'])
                if unit.exists():
                    raise ValueError('Interrupted or still-running unit preserved: ' + str(unit)
                        + '. Confirm its process has stopped and rename only this unit before resuming; no checkpoint resume is available.')
                continue
            name = ('smoke_' if smoke else '') + record['mode']
            measured, snapshot, rows = read_run(directory, dict(record, mode=name))
            expected = dict(json.loads((ROOT / 'exps/dlora/imgr10.json').read_text()),
                            **settings_for('3090', record['mode'], smoke))
            same_config = all(snapshot['effective_config'].get(k) == v for k, v in expected.items() if k != 'prefix')
            same_source = all(hashlib.sha256((ROOT / p).read_bytes()).hexdigest() == digest
                              for p, digest in snapshot['source_sha256'].items())
            task_count = 2 if smoke else 10
            expected_masks = {(task, layer) for task in range(task_count) for layer in range(12)}
            expected_updates = {(task, layer, branch, projection) for task in range(task_count) for layer in range(12)
                                for branch in (('S',) if task == 0 else ('S', 'P')) for projection in ('Q', 'K', 'V')}
            complete_telemetry = ({(r['task'], r['layer']) for r in rows['masks']} == expected_masks
                and {(r['task'], r['layer'], r['branch'], r['projection']) for r in rows['updates']} == expected_updates)
            finite_metrics = all(measured.get(k) is not None and math.isfinite(measured[k]) for k in METRICS)
            if (not same_config or not same_source or not complete_fingerprint(snapshot) or snapshot['code_revision'] != revision
                    or snapshot.get('machine') != '3090' or snapshot.get('phase') != ('smoke' if smoke else 'formal')
                    or record.get('exit_code') != 0 or not finite_metrics or not complete_telemetry
                    or measured['runtime_error'] or measured['tasks_reported'] != task_count):
                raise ValueError('Completed unit no longer strictly matches: ' + name)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=('run', 'smoke', 'dry-run', 'analyze'), default='run')
    parser.add_argument('--resume', type=Path)
    args = parser.parse_args()
    if args.mode == 'analyze':
        summarize(args.resume, json.loads((args.resume / 'queue.json').read_text()))
        return 0
    revision = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    directory = args.resume or ROOT / 'logs/shell_logs/imgr10_region_prior_3090' / datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    if args.mode != 'dry-run':
        if args.resume:
            check_resume(directory, revision)
        config = json.loads((ROOT / 'exps/dlora/imgr10.json').read_text())
        data = baseline.check_data(config)
        directory.mkdir(parents=True, exist_ok=True)
        (directory / ('resume_manifest.json' if args.resume else 'manifest.json')).write_text(json.dumps(dict(
            revision=revision, machine='3090', queue_pid=engine.os.getpid(), modes=modes(), dataset=data,
            pretrained=pretrained_fingerprint(),
            reuse_decision='Older scores reused in the manuscript, not in this same-source intervention-logged pairing.'), indent=2) + '\n')
    print('Code revision:', revision, '\nQueue PID:', engine.os.getpid(), '\nOutputs:', directory, flush=True)
    print('Original O, anchor2.5, 9 T10 units; no checkpoints, no time cutoff.', flush=True)
    return execute(directory, revision, args.mode)


if __name__ == '__main__':
    raise SystemExit(main())
