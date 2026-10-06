"""Three-run confirmation with source-verified reuse of completed same-machine runs."""
import argparse
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import sys

import run_compact_structure_night as previous
import analyze_compact_structure as plotter
from analyze_core_evidence import METRICS, read_run
from analyze_protect_position import write_csv


ROOT = Path(__file__).resolve().parents[1]
SPEC = ROOT / 'scripts/sweeps/imgr10_compact_core_confirmation.json'
REFERENCE_REVISION = json.loads(SPEC.read_text())['reference_revision']
LOGGING_KEYS = {'prefix', 'wandb_group'}


def settings_for(machine, mode, smoke=False):
    spec = json.loads(SPEC.read_text())
    source = spec['references'][machine].get(mode, 'R0' if machine == '3090' else 'S2')
    settings = previous.settings_for(machine, source, smoke)
    if mode not in spec['references'][machine]:
        settings.update(next(row['overrides'] for row in spec[machine] if row['name'] == mode))
    settings['wandb_group'] = 'imgr10_compact_core_confirmation_' + machine
    return settings


def training_hashes(snapshot):
    return {path: digest for path, digest in snapshot['source_sha256'].items()
            if path in ('main.py', 'trainer.py') or path.startswith(('models/', 'methods/', 'utils/'))}


def current_identity():
    paths = [ROOT / 'main.py', ROOT / 'trainer.py']
    paths += [path for folder in ('models', 'methods', 'utils') for path in (ROOT / folder).rglob('*.py')]
    return dict(
        training_source_sha256={str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
                               for path in paths},
        effective_config=json.loads((ROOT / 'exps/dlora/imgr10.json').read_text()),
        software=dict(python=sys.version, platform=platform.platform(),
                      packages={name: importlib.metadata.version(name)
                                for name in ('torch', 'torchvision', 'timm', 'numpy')}),
        hardware=dict(hostname=platform.node(), cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),
            gpus=subprocess.check_output(['nvidia-smi',
                '--query-gpu=index,uuid,name,pci.bus_id,driver_version', '--format=csv,noheader'],
                text=True).strip().splitlines()))


def load_references(machine):
    spec = json.loads(SPEC.read_text())
    directory = ROOT / spec['reference_queues'][machine]
    records = {row['mode']: row for row in json.loads((directory / 'queue.json').read_text())}
    summaries, snapshots = [], {}
    rows = {name: [] for name in ('epochs', 'updates', 'masks', 'diagnostics', 'costs', 'storage', 'tasks')}
    for alias, original in spec['references'][machine].items():
        summary, snapshot, measured = read_run(directory, records[original])
        path = directory / original
        summary.update(mode=alias, reused=True, source_mode=original, source_directory=str(path),
                       code_revision=snapshot['code_revision'],
                       training_log_sha256=hashlib.sha256((path / 'training.log').read_bytes()).hexdigest(),
                       snapshot_sha256=hashlib.sha256((path / 'run.json').read_bytes()).hexdigest())
        summaries.append(summary)
        snapshots[alias] = snapshot
        for name in rows:
            rows[name].extend(dict(row, mode=alias) for row in measured[name])
    return summaries, snapshots, rows


def completed_run_issues(summary, snapshot, machine, mode, identity, smoke=False):
    issues = []
    healthy = (summary.get('tasks_reported') == 2 and summary.get('exit_code') == 0
               and not summary.get('runtime_error')
               and all(summary.get(key) is not None and math.isfinite(summary[key]) for key in METRICS)) \
              if smoke else summary.get('valid_performance')
    if not healthy:
        issues.append('incomplete_or_unhealthy_T10')
    for key in (() if smoke else ('epoch_records_complete', 'position_diagnostics_complete')):
        if not summary.get(key):
            issues.append(key)
    if snapshot.get('machine') != machine or snapshot.get('phase') != ('smoke' if smoke else 'formal'):
        issues.append('machine_or_phase')
    expected_mode = json.loads(SPEC.read_text())['references'][machine].get(mode, mode)
    if snapshot.get('mode') != expected_mode:
        issues.append('mode')
    if training_hashes(snapshot) != identity['training_source_sha256']:
        issues.append('training_source_sha256')
    for field in ('software', 'hardware'):
        if snapshot.get(field) != identity[field]:
            issues.append(field)
    expected = dict(identity['effective_config'])
    expected.update(settings_for(machine, mode, smoke))
    actual = snapshot['effective_config']
    issues += ['config.' + key for key in expected.keys() | actual.keys()
               if key not in LOGGING_KEYS and expected.get(key) != actual.get(key)]
    return issues


def reference_issues(summaries, snapshots, machine, identity):
    issues = []
    references = json.loads(SPEC.read_text())['references'][machine]
    if set(snapshots) != set(references):
        issues.append(dict(mode='references', field='missing_reference'))
    for summary in summaries:
        mode, snapshot = summary['mode'], snapshots[summary['mode']]
        issues += [dict(mode=mode, field=field)
                   for field in completed_run_issues(summary, snapshot, machine, mode, identity)]
        if snapshot.get('code_revision') != REFERENCE_REVISION or snapshot.get('mode') != references[mode]:
            issues.append(dict(mode=mode, field='reference_revision_or_mode'))
    return issues


def matching_issues(snapshots, machine):
    if not snapshots:
        return []
    reference = snapshots['C11' if machine == '3090' else 'M']
    spec = json.loads(SPEC.read_text())
    factors = {key for variant in spec[machine] for key in variant['overrides']}
    issues = []
    for mode, snapshot in snapshots.items():
        for field in ('machine', 'phase', 'software', 'hardware'):
            if snapshot.get(field) != reference.get(field):
                issues.append(dict(mode=mode, field=field))
        if training_hashes(snapshot) != training_hashes(reference):
            issues.append(dict(mode=mode, field='training_source_sha256'))
        config = snapshot['effective_config']
        for key in config.keys() | reference['effective_config'].keys():
            if key not in factors | LOGGING_KEYS and config.get(key) != reference['effective_config'].get(key):
                issues.append(dict(mode=mode, field='config.' + key))
        for key, value in settings_for(machine, mode).items():
            if key not in LOGGING_KEYS and config.get(key) != value:
                issues.append(dict(mode=mode, field='wrong_factor.' + key))
    return issues


def contrasts(complete, machine):
    pairs = [('C11', 'C10'), ('C11', 'C01'), ('C10', 'C00'), ('C01', 'C00')]
    if machine == '5090':
        pairs = [('M', 'O')]
    values = {a + '_minus_' + b: {key: complete[a][key] - complete[b][key] for key in METRICS}
              for a, b in pairs if a in complete and b in complete}
    if machine == '3090' and all(name in complete for name in ('C00', 'C01', 'C10', 'C11')):
        values['permissions_suppression_2x2'] = {}
        for key in METRICS:
            a, b, c, d = [complete[name][key] for name in ('C00', 'C01', 'C10', 'C11')]
            values['permissions_suppression_2x2'][key] = dict(
                permission_effect=((c + d) - (a + b)) / 2,
                suppression_effect=((b + d) - (a + c)) / 2,
                interaction=(d - c) - (b - a))
    return values


def summarize(directory, machine, records):
    summaries, snapshots, rows = load_references(machine)
    reference = snapshots['C11' if machine == '3090' else 'M']
    validation = directory / 'reference_validation.json'
    identity = json.loads(validation.read_text())['current_identity'] if validation.exists() else dict(
        training_source_sha256=training_hashes(reference),
        effective_config=reference['effective_config'],
        software=reference['software'], hardware=reference['hardware'])
    issues = reference_issues(summaries, snapshots, machine, identity)
    for record in records:
        if record['status'] not in ('completed', 'failed'):
            summaries.append(record)
            continue
        summary, snapshot, measured = read_run(directory, record)
        summary.update(reused=False, source_directory=str(directory / record['mode']),
                       code_revision=snapshot['code_revision'])
        summaries.append(summary)
        snapshots[record['mode']] = snapshot
        if record['status'] == 'completed':
            issues += [dict(mode=record['mode'], field=field)
                       for field in completed_run_issues(summary, snapshot, machine, record['mode'], identity)]
        for name in rows:
            rows[name].extend(measured[name])
    issues += matching_issues(snapshots, machine)
    complete = {row['mode']: row for row in summaries if row.get('valid_performance')}
    reference_result = complete.get('C11' if machine == '3090' else 'M')
    if reference_result:
        issues += [dict(mode=mode, field='Task0_startpoint_accuracy')
                   for mode, row in complete.items() if row['Task0'] != reference_result['Task0']]
    result = contrasts(complete, machine) if not issues else {}
    write_csv(directory / 'results.csv', summaries)
    (directory / 'results.json').write_text(json.dumps(summaries, indent=2) + '\n')
    for name, data in rows.items():
        write_csv(directory / (name + '.csv'), data)
    (directory / 'contrasts.json').write_text(json.dumps(dict(machine=machine,
        matching_issues=issues, completed_pairs=result,
        limitation='Cross commits with identical training sources; single seed; no equivalence claim.'), indent=2) + '\n')
    lines = ['# ' + machine + ' compact-core confirmation', '',
        '| Run | Reused | Status | Average | Last | Old | New | StageOld | StageNew | Forgetting |',
        '|---|---|---|---:|---:|---:|---:|---:|---:|---:|']
    for row in summaries:
        values = [row['mode'], str(row.get('reused', False)), row['status']]
        values += [str(row.get(key, '')) for key in METRICS]
        lines.append('| ' + ' | '.join(values) + ' |')
    lines += ['', '## Matched differences (first minus second, pp)', '',
              '| Pair | ' + ' | '.join(METRICS) + ' |', '|---|' + '---:|' * len(METRICS)]
    for name, differences in result.items():
        if name != 'permissions_suppression_2x2':
            lines.append('| ' + name + ' | ' + ' | '.join(format(differences[key], '.4f')
                                                        for key in METRICS) + ' |')
    effects = result.get('permissions_suppression_2x2', {})
    if effects:
        lines += ['', '## Descriptive 2x2 effects', '',
                  '| Metric | Permission | Suppression | Interaction |', '|---|---:|---:|---:|']
        for key, values in effects.items():
            lines.append('| ' + key + ' | ' + ' | '.join(format(value, '.4f')
                                                        for value in values.values()) + ' |')
    lines += ['', 'Matching issues: ' + json.dumps(issues),
        'C11/C01 reuse R0/R3; M reuses S2. Reused runs are not independent repeats.',
        'Permission is the whole asymmetric policy, not an individual S/P component.',
        'O restores the complete adaptive recipe and mask penalties; it is not F.',
        'Task0 anchor/CA are retained. CA covariance and competence evaluation costs remain.',
        'Joint performance does not prove synergy or precise old-sensitive position selection.']
    (directory / 'report.md').write_text('\n'.join(lines) + '\n')
    plotter.draw(directory, complete, rows, 'confirmation')
    if effects:
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 2, figsize=(9, 4), constrained_layout=True)
        for axis, key in zip(axes, ('Average', 'Last')):
            values = [[complete[name][key] for name in pair]
                      for pair in (('C00', 'C01'), ('C10', 'C11'))]
            axis.imshow(values, cmap='viridis')
            for i in range(2):
                for j in range(2):
                    axis.text(j, i, format(values[i][j], '.3f'), ha='center', va='center', color='white')
            axis.set(xticks=[0, 1], yticks=[0, 1], xticklabels=['Suppression off', 'Suppression on'],
                     yticklabels=['Permissions off', 'Permissions on'], title=key + ' (%)')
        for suffix in ('png', 'pdf'):
            fig.savefig(directory / ('permissions_suppression_2x2.' + suffix), dpi=160)
        plt.close(fig)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    args = parser.parse_args()
    manifest = json.loads((args.directory / 'manifest.json').read_text())
    summarize(args.directory, manifest['machine'], json.loads((args.directory / 'queue.json').read_text()))
