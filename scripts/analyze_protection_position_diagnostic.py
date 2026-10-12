"""单组正式成绩与只读诊断分别核验；不产生代理优胜训练结论。"""
import argparse
import json
import math
from pathlib import Path
import re

from analyze_core_evidence import read_run
from analyze_protect_position import json_rows, write_csv
import run_protection_position_diagnostic as runner

POSITIONS = ('W', 'WR', 'WRperm', 'MC', 'Random')
MARKERS = ('PositionSignalPrepared', 'PositionSignalFisher', 'PositionSignalMask',
           'PositionSignalScore', 'PositionSignalUpdate', 'PositionSignalIntervention', 'PositionSignalCost')


def extract_rows(content):
    return {marker: json_rows(content, marker) for marker in MARKERS}


def evidence_issues(rows, tasks=(1, 5, 9), epochs=(2, 20)):
    issues = []
    expected = {
        'PositionSignalPrepared': ({(t,) for t in tasks}, ('task',)),
        'PositionSignalFisher': ({(t, l) for t in tasks for l in range(12)}, ('task', 'layer')),
        'PositionSignalMask': ({(t, l, p) for t in tasks for l in range(12) for p in POSITIONS},
                               ('task', 'layer', 'position')),
        'PositionSignalScore': ({(t, l, s) for t in tasks for l in range(12) for s in ('spectrum', 'residual')},
                                ('task', 'layer', 'source')),
        'PositionSignalCost': ({(t, e) for t in tasks for e in epochs}, ('task', 'epoch')),
        'PositionSignalUpdate': ({(t, e, l, p) for t in tasks for e in epochs for l in range(12) for p in POSITIONS},
                                 ('task', 'epoch', 'layer', 'position')),
        'PositionSignalIntervention': ({(t, e, a, p) for t in tasks for e in epochs for a in ('old', 'new')
                                       for p in POSITIONS + ('ActualF',)}, ('task', 'epoch', 'partition', 'position'))
    }
    for marker, (keys, fields) in expected.items():
        values = rows.get(marker, [])
        if len(values) != len(keys) or {tuple(r.get(k) for k in fields) for r in values} != keys:
            issues.append(marker)
    for row in rows.get('PositionSignalPrepared', []):
        if not row.get('score_old_disjoint'):
            issues.append('overlapping_samples')
        if any(row.get('sample_counts', {}).get(k, 0) <= 0 for k in ('score', 'old', 'new')):
            issues.append('empty_sample_pool')
    def finite(row, keys):
        return all(isinstance(row.get(k), (int, float)) and math.isfinite(row[k]) for k in keys)
    for row in rows.get('PositionSignalFisher', []):
        fields = ('samples', 'draws_per_image', 'scale', 's_b_norm', 'p_b_norm',
                  'draw_correlation', 'draw_mask_jaccard', 'fisher_sum', 'residual_norm')
        if (not finite(row, fields) or row.get('samples', 0) <= 0 or row.get('draws_per_image') != 2
                or row.get('scale') != 20 or row.get('s_b_norm') != 0 or row.get('p_b_norm') != 0
                or row.get('classifier_scope') != 'previous_seen_classes'
                or row.get('source') != 'old_train_score_only'
                or not -1.000001 <= row.get('draw_correlation', 2) <= 1.000001
                or not 0 <= row.get('draw_mask_jaccard', -1) <= 1):
            issues.append('invalid_fisher_evidence')
    for row in rows.get('PositionSignalScore', []):
        quantiles = row.get('quantiles', [])
        if (not finite(row, ('mass', 'fisher_correlation')) or len(quantiles) != 6
                or any(not isinstance(v, (int, float)) or not math.isfinite(v) for v in quantiles)
                or quantiles != sorted(quantiles)):
            issues.append('invalid_score_evidence')
    for row in rows.get('PositionSignalCost', []):
        if not finite(row, ('seconds',)) or row['seconds'] < 0:
            issues.append('invalid_cost_evidence')
    for row in rows.get('PositionSignalMask', []):
        if row.get('qkv_counts') != row.get('reference_counts'):
            issues.append('changed_protection_count')
    groups = {}
    for row in rows.get('PositionSignalUpdate', []):
        groups.setdefault((row['task'], row['epoch'], row['layer']), []).append(row)
        if not 0 <= row.get('scale', -1) <= 1:
            issues.append('update_amplified')
    for values in groups.values():
        norms = [row['s_effective_norm'] for row in values]
        if max(norms) - min(norms) > 1e-6 + 5e-5 * max(norms):
            issues.append('s_norm_mismatch')
        if len({row['p_effective_norm'] for row in values}) != 1:
            issues.append('p_changed')
    for row in rows.get('PositionSignalIntervention', []):
        if (row.get('source') != 'independent_train_diagnostic' or row.get('count', 0) <= 0
                or row.get('prediction_reference') != 'W_matched'
                or any(not isinstance(row.get(k), (int, float)) or not math.isfinite(row[k])
                       for k in ('accuracy', 'loss', 'margin'))):
            issues.append('invalid_intervention')
    for values in rows.values():
        if any(isinstance(v, float) and not math.isfinite(v) for row in values for v in row.values()):
            issues.append('nonfinite_diagnostic')
    return sorted(set(issues))


def fingerprint_issues(snapshot, smoke_snapshot, revision):
    issues = []
    if (snapshot.get('code_revision') != revision or snapshot.get('phase') != 'formal'
            or snapshot.get('machine') != '3090'):
        issues.append('revision_or_phase')
    if smoke_snapshot is None:
        return issues + ['missing_smoke_snapshot']
    if (smoke_snapshot.get('code_revision') != revision or smoke_snapshot.get('phase') != 'smoke'
            or smoke_snapshot.get('machine') != '3090'):
        issues.append('smoke_identity')
    for field in ('source_sha256', 'hardware', 'software'):
        if not snapshot.get(field) or snapshot.get(field) != smoke_snapshot.get(field):
            issues.append(field)
    if not runner.REQUIRED_SOURCE <= snapshot.get('source_sha256', {}).keys():
        issues.append('missing_source_fingerprint')
    return issues


def summarize_saved(directory):
    manifest = json.loads((directory / 'manifest.json').read_text())
    records = json.loads((directory / 'queue.json').read_text()) if (directory / 'queue.json').exists() else []
    results, issues, all_rows = [], [], {marker: [] for marker in MARKERS}
    tasks = []
    for record in records:
        if record['status'] not in ('completed', 'failed'):
            results.append(dict(record, valid_performance=False, diagnostic_evidence_complete=False))
            continue
        try:
            measured, snapshot, telemetry = read_run(directory, record)
            content = (directory / record['mode'] / 'training.log').read_text()
        except (OSError, ValueError, KeyError, SyntaxError) as error:
            results.append(dict(record, valid_performance=False, diagnostic_evidence_complete=False,
                                evidence_error=str(error)))
            continue
        local = []
        expected = runner.effective_config(manifest['machine'], record['mode'])
        if any(expected.get(k) != snapshot['effective_config'].get(k)
               for k in expected.keys() | snapshot['effective_config'].keys() if k != 'prefix'):
            local.append('config')
        smoke_path = directory / ('smoke_' + record['mode']) / 'run.json'
        smoke_snapshot = json.loads(smoke_path.read_text()) if smoke_path.exists() else None
        local.extend(fingerprint_issues(snapshot, smoke_snapshot, manifest['revision']))
        epochs = {(int(t), int(e)) for t, e in re.findall(r'LoRA learning rates: task=(\d+), epoch=(\d+)', content)}
        measured['training_epochs_complete'] = epochs == {(t, e) for t in range(10) for e in range(1, 21)}
        measured['valid_performance'] &= measured['training_epochs_complete'] and not local
        rows = extract_rows(content)
        local.extend(evidence_issues(rows))
        for marker, values in rows.items():
            all_rows[marker].extend(dict(row, run=record['mode']) for row in values)
        tasks.extend(telemetry['tasks'])
        measured.update(diagnostic_evidence_complete=not local, seed=1993, std=None,
                        complete_three_seeds=False, code_revision=snapshot['code_revision'])
        results.append(measured)
        issues.extend(dict(mode=record['mode'], field=field) for field in local)
    for name, values in dict(results=results, matching_issues=issues, tasks=tasks, **all_rows).items():
        (directory / (name + '.json')).write_text(json.dumps(values, indent=2) + '\n')
        if values:
            write_csv(directory / (name + '.csv'), values)
        else:
            (directory / (name + '.csv')).write_text('')
    report = ['# 3090 F：保护位置只读诊断', '',
        '| 组 | 状态 | Average | Last | Old | New | 阶段Old | 阶段New | Forgetting | 诊断完整 |',
        '|---|---|---:|---:|---:|---:|---:|---:|---:|---|']
    for row in results:
        report.append('| ' + ' | '.join([row['mode'], row['status']] +
            [str(row.get(k, '')) for k in ('Average', 'Last', 'Old', 'New', 'StageOld', 'StageNew', 'Forgetting',
                                         'diagnostic_evidence_complete')]) + ' |')
    report += ['', '诊断／身份问题：' + json.dumps(issues, ensure_ascii=False),
        '只有一条正式训练轨迹；五位置仅S-only干预，P固定。MC位置使用旧训练来源特权信息，不是正式方法或性能上界。',
        '同状态匹配S范数，不保证S+P范数或函数扰动相同。报告六个相关状态，不把它们当作六个seed。',
        '正式训练始终使用W_pre谱位置；旧测试集不参与评分或挑选。无自动追加训练或基线替换。']
    (directory / 'report.md').write_text('\n'.join(report) + '\n')
    if any(r.get('valid_performance') and r.get('diagnostic_evidence_complete') for r in results):
        draw(directory, all_rows, tasks)


def draw(directory, rows, tasks):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    interventions = rows['PositionSignalIntervention']
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    for position in POSITIONS:
        for ax, partition in zip(axes, ('old', 'new')):
            values = [r for r in interventions if r['position'] == position and r['partition'] == partition]
            ax.plot(range(len(values)), [r['margin'] for r in values], '.-', label=position)
            ax.set(xlabel='Task1/5/9 epoch2/20 states', ylabel='Cosine margin', title=partition)
    axes[0].legend()
    fig.savefig(directory / 's_only_position_margin.pdf')
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(6, 4), constrained_layout=True)
    for position in POSITIONS:
        old = [r for r in interventions if r['position'] == position and r['partition'] == 'old']
        new = [r for r in interventions if r['position'] == position and r['partition'] == 'new']
        ax.plot([r['margin'] for r in new], [r['margin'] for r in old], '.', label=position)
    ax.set(xlabel='New cosine margin', ylabel='Old cosine margin')
    ax.legend()
    fig.savefig(directory / 'diagnostic_old_new.pdf')
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    for position in POSITIONS:
        values = [r for r in rows['PositionSignalUpdate'] if r['position'] == position]
        axes[0].plot([r['risk_mass'] for r in values], label=position)
        axes[1].plot([r['net_effective_norm'] for r in values], label=position)
    axes[0].set(xlabel='Diagnostic state / layer', ylabel='Risk mass in protected coordinates')
    axes[1].set(xlabel='Diagnostic state / layer', ylabel='Combined S+P norm')
    axes[0].legend()
    fig.savefig(directory / 'risk_and_combined_norm.pdf')
    plt.close(fig)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    summarize_saved(parser.parse_args().directory)
