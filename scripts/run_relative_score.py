"""Two finite M score trials per GPU; reuse verified M, never select on smoke accuracy."""
import argparse
import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys

import analyze_compact_core_confirmation as evidence
import analyze_compact_structure as plotter
from analyze_core_evidence import METRICS, read_run
from analyze_protect_position import write_csv
import run_compact_structure_night as queue


ROOT = queue.ROOT
SPEC = ROOT / 'scripts/sweeps/imgr10_relative_score.json'


def settings_for(machine, mode, smoke=False):
    spec = json.loads(SPEC.read_text())
    settings = queue.settings_for(machine, spec['reference_modes'][machine], smoke)
    settings['dual_mask_conflict_score_mode'] = 'magnitude' if mode == 'M' else mode
    settings['wandb_group'] = 'imgr10_relative_score_' + machine
    return settings


def command_for(machine, mode, directory, smoke=False):
    settings = settings_for(machine, mode, smoke)
    settings['prefix'] = 'imgr10_relative_' + machine + '_' + mode + '_' + directory.parent.name
    command = [sys.executable, 'main.py', '--config', 'exps/dlora/imgr10.json']
    for key, value in settings.items():
        command += ['--set', key + '=' + (value if isinstance(value, str) else json.dumps(value))]
    return command, settings


def load_reference(machine):
    spec = json.loads(SPEC.read_text())
    directory, mode = ROOT / spec['reference_queues'][machine], spec['reference_modes'][machine]
    record = next(row for row in json.loads((directory / 'queue.json').read_text()) if row['mode'] == mode)
    summary, snapshot, rows = read_run(directory, record)
    summary.update(mode='M', reused=True, source_mode=mode, source_directory=str(directory / mode),
                   code_revision=snapshot['code_revision'],
                   training_log_sha256=hashlib.sha256((directory / mode / 'training.log').read_bytes()).hexdigest(),
                   snapshot_sha256=hashlib.sha256((directory / mode / 'run.json').read_bytes()).hexdigest())
    rows = {name: [dict(row, mode='M') for row in values] for name, values in rows.items()}
    return summary, snapshot, rows


def matching_issues(summary, snapshot, machine, mode, identity):
    spec = json.loads(SPEC.read_text())
    issues = []
    if not summary.get('valid_performance') or not summary.get('epoch_records_complete') or not summary.get('position_diagnostics_complete'):
        issues.append('incomplete_or_unhealthy_T10')
    for field, expected in (('machine', machine), ('phase', 'formal'),
                            ('software', identity['software']), ('hardware', identity['hardware'])):
        if snapshot.get(field) != expected:
            issues.append(field)
    expected = dict(identity['effective_config'], **settings_for(machine, mode))
    actual = snapshot['effective_config']
    issues += ['config.' + key for key in expected.keys() | actual.keys()
               if key not in evidence.LOGGING_KEYS and expected.get(key) != actual.get(key)]
    hashes = evidence.training_hashes(snapshot)
    current = identity['training_source_sha256']
    for path in hashes.keys() | current.keys():
        if mode == 'M' and path in spec['changed_training_files']:
            original = subprocess.check_output(['git', 'show', spec['reference_revision'] + ':' + path], cwd=ROOT)
            expected_hash = hashlib.sha256(original).hexdigest()
        else:
            expected_hash = current.get(path)
        if hashes.get(path) != expected_hash:
            issues.append('source.' + path)
    if mode == 'M' and (snapshot['code_revision'] != spec['reference_revision']
                         or snapshot['mode'] != spec['reference_modes'][machine]):
        issues.append('reference_identity')
    if mode != 'M' and snapshot['mode'] != mode:
        issues.append('mode')
    return issues


def summarize(directory, machine, records):
    reference, snapshot, rows = load_reference(machine)
    identity = json.loads((directory / 'reference_validation.json').read_text())['current_identity']
    summaries = [reference]
    issues = [dict(mode='M', field=field) for field in matching_issues(reference, snapshot, machine, 'M', identity)]
    for record in records:
        if record['status'] not in ('completed', 'failed'):
            summaries.append(record)
            continue
        summary, snapshot, measured = read_run(directory, record)
        summary.update(reused=False, code_revision=snapshot['code_revision'])
        summaries.append(summary)
        issues += [dict(mode=record['mode'], field=field)
                   for field in matching_issues(summary, snapshot, machine, record['mode'], identity)]
        if summary.get('Task0') != reference['Task0']:
            issues.append(dict(mode=record['mode'], field='Task0_accuracy'))
        for name in rows:
            rows[name].extend(measured[name])
    complete = {row['mode']: row for row in summaries if row.get('valid_performance')}
    comparisons = {name + '_minus_M': {key: row[key] - reference[key] for key in METRICS}
                   for name, row in complete.items() if name != 'M'} if not issues else {}
    limitation = ('Cross-commit M reuse; attention score and read-only telemetry changed, default path regression-tested. '
                  'Fixed coordinate count and beta do NOT match removed norms. Single seed, no significance claim.')
    (directory / 'results.json').write_text(json.dumps(summaries, indent=2) + '\n')
    write_csv(directory / 'results.csv', summaries)
    for name, values in rows.items():
        write_csv(directory / (name + '.csv'), values)
    (directory / 'contrasts.json').write_text(json.dumps(dict(machine=machine, matching_issues=issues,
        completed_pairs=comparisons, limitation=limitation), indent=2) + '\n')
    lines = ['# ' + machine + ' relative-score trials', '', limitation, '',
             '| Mode | Reused | Status | ' + ' | '.join(METRICS) + ' |',
             '|---|---|---|' + '---:|' * len(METRICS)]
    for row in summaries:
        lines.append('| ' + ' | '.join([row['mode'], str(row.get('reused', False)), row['status']]
            + [str(row.get(key, '')) for key in METRICS]) + ' |')
    lines += ['', 'Matching issues: ' + json.dumps(issues), '', json.dumps(comparisons, indent=2)]
    (directory / 'report.md').write_text('\n'.join(lines) + '\n')
    plotter.draw(directory, complete, rows, 'relative_score')


def validate_settings(machine):
    fixed = dict(dual_mask_conflict_granularity='layer', dual_mask_conflict_exact_topk=True,
        dual_mask_conflict_budget_multiplier=1.0, dual_mask_private_conflict_mode='global',
        dual_mask_reg_weight=0, dual_mask_branch_layout='dual', dual_mask_uniform_norm_matched=False)
    for mode in json.loads(SPEC.read_text())['modes']:
        settings = settings_for(machine, mode)
        if any(settings.get(key) != value for key, value in fixed.items()):
            raise ValueError('Relative-score queue requires the designated fixed-budget M protocol.')


def check_resume(directory, machine, revision):
    manifest = json.loads((directory / 'manifest.json').read_text())
    if manifest['revision'] != revision or manifest['machine'] != machine:
        raise ValueError('Resume needs the original machine and revision.')
    for filename, smoke in (('queue.json', False), ('smoke_queue.json', True)):
        path = directory / filename
        for record in json.loads(path.read_text()) if path.exists() else []:
            if record['status'] == 'failed':
                raise ValueError('Failed evidence preserved; fix then start unfinished modes in a new queue.')
            if record['status'] != 'completed':
                continue
            summary, snapshot, _ = read_run(directory, dict(record,
                mode=('smoke_' if smoke else '') + record['mode']))
            expected = dict(json.loads((ROOT / 'exps/dlora/imgr10.json').read_text()),
                            **settings_for(machine, record['mode'], smoke))
            hashes_match = all(hashlib.sha256((ROOT / p).read_bytes()).hexdigest() == digest
                               for p, digest in snapshot['source_sha256'].items())
            config_match = all(snapshot['effective_config'].get(k) == v for k, v in expected.items() if k != 'prefix')
            if (summary['tasks_reported'] != (2 if smoke else 10) or summary['runtime_error']
                    or record.get('exit_code') != 0 or snapshot['code_revision'] != revision
                    or snapshot['machine'] != machine or snapshot['phase'] != ('smoke' if smoke else 'formal')
                    or not all(summary.get(k) is not None and math.isfinite(summary[k]) for k in METRICS)
                    or not hashes_match or not config_match):
                raise ValueError('Cannot resume unverified completed group: ' + record['mode'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--machine', choices=('3090', '5090'), required=True)
    parser.add_argument('--mode', choices=('run', 'smoke', 'dry-run'), default='run')
    parser.add_argument('--modes', nargs='+', choices=json.loads(SPEC.read_text())['modes'])
    parser.add_argument('--resume', type=Path)
    args = parser.parse_args()
    modes = args.modes or json.loads(SPEC.read_text())['modes']
    if len(set(modes)) != len(modes):
        parser.error('duplicate modes')
    validate_settings(args.machine)
    revision = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    directory = args.resume or ROOT / ('logs/shell_logs/imgr10_relative_score_' + args.machine) / datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    if args.mode != 'dry-run':
        identity = evidence.current_identity()
        summary, snapshot, _ = load_reference(args.machine)
        issues = matching_issues(summary, snapshot, args.machine, 'M', identity)
        if issues:
            parser.error('M reference mismatch; no training started: ' + str(issues))
        directory.mkdir(parents=True, exist_ok=True)
        if args.resume:
            check_resume(directory, args.machine, revision)
        (directory / 'reference_validation.json').write_text(json.dumps(dict(matching_issues=issues,
            current_identity=identity, reused_run=summary), indent=2) + '\n')
        (directory / ('resume_manifest.json' if args.resume else 'manifest.json')).write_text(json.dumps(dict(
            machine=args.machine, revision=revision, queue_pid=os.getpid(), modes=modes), indent=2) + '\n')
    print('Code revision:', revision, '\nQueue PID:', os.getpid(), '\nOutputs:', directory, flush=True)
    queue.engine.SPEC = SPEC
    queue.engine.settings_for = settings_for
    queue.engine.command_for = command_for
    queue.engine.EXTRA_SOURCE_PATHS = ['scripts/run_relative_score.py', 'scripts/run_compact_structure_night.py',
        'scripts/analyze_compact_structure.py', 'scripts/analyze_compact_core_confirmation.py',
        'scripts/sweeps/imgr10_compact_structure_night.json', 'scripts/sweeps/imgr10_compact_core_confirmation.json']
    queue.analysis = sys.modules[__name__]
    return queue.execute_queue(args.machine, modes, directory, revision, args.mode, hours=24)


if __name__ == '__main__':
    raise SystemExit(main())
