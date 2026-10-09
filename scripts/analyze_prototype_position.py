"""两机独立原型位置证据；缺失组、异指纹不产生配对结论。"""
import argparse
import json
import math
from pathlib import Path
import re

from analyze_core_evidence import METRICS, read_run
from analyze_protect_position import json_rows
from analyze_tail_update import write_csv


def position_evidence_complete(telemetry):
    positions = ('wpre', 'prototype_high', 'prototype_shuffled', 'permuted')
    expected = dict(
        norms={(t, l, b, p, pos) for t in (1, 5, 9) for l in range(12)
               for b in ('S', 'P') for p in ('Q', 'K', 'V') for pos in positions},
        holdout={(t, l, pos) for t in range(1, 10) for l in range(12)
                 for pos in positions + ('prototype_low',)},
        diagnostics={(t, part, pos) for t in (1, 5, 9) for part in ('old', 'new') for pos in positions})
    keys = dict(norms=('task', 'layer', 'branch', 'projection', 'position'),
                holdout=('task', 'layer', 'position'), diagnostics=('task', 'partition', 'position'))
    for section, fields in keys.items():
        rows = telemetry.get(section, [])
        if len(rows) != len(expected[section]) or {tuple(r.get(k) for k in fields) for r in rows} != expected[section]:
            return False
    def finite(row, fields):
        return all(isinstance(row.get(k), (int, float)) and math.isfinite(row[k]) for k in fields)
    return (all(finite(r, ('raw_norm', 'effective_norm', 'removed_norm', 'same_state_norm_residual'))
                and 0 <= r['same_state_norm_residual'] <= 5e-5 * max(1., r['effective_norm'])
                for r in telemetry['norms'])
        and all(finite(r, ('samples', 'true_sensitive_mass', 'shuffled_sensitive_mass'))
                and 0 < r['samples'] <= 64 for r in telemetry['holdout'])
        and all(finite(r, ('mean_margin', 'count'))
                and 0 < r['count'] <= (512 if r['partition'] == 'old' else 128)
                and r.get('norm_control') == 'same_state_four_position_min' for r in telemetry['diagnostics']))


def paired_results(records, issues, machine):
    complete = {r['mode'].split('_seed')[0]: r for r in records
                if r.get('valid_performance')} if not issues else {}
    first = 'R1' if machine == '3090' else 'S1'
    others = ('R0', 'R2', 'R3') if machine == '3090' else ('S0', 'S2', 'S3')
    return [dict(comparison=first + '_minus_' + second,
                 **{k: complete[first][k] - complete[second][k] for k in METRICS})
            for second in others if first in complete and second in complete]


def summarize_saved(directory):
    import run_tail_update as runner
    manifest = json.loads((directory / 'manifest.json').read_text())
    machine = manifest['machine']
    runner.SPEC = runner.ROOT / manifest['sweep_spec']
    results, snapshots, issues = [], [], []
    rows = {name: [] for name in ('tasks', 'epochs', 'costs', 'storage', 'probes',
                                  'masks', 'holdout', 'diagnostics', 'norms')}
    markers = dict(probes='PrototypePositionProbe', masks='PrototypePositionMask',
        holdout='PrototypePositionHoldout', diagnostics='PrototypePositionDiagnostic',
        norms='PrototypePositionNorm')
    for record in json.loads((directory / 'queue.json').read_text()):
        if record['status'] not in ('completed', 'failed'):
            results.append(dict(record, valid_performance=False))
            continue
        row, snapshot, measured = read_run(directory, record)
        text = (directory / record['mode'] / 'training.log').read_text()
        expected = dict(json.loads((runner.ROOT / runner.spec()[machine]['config']).read_text()),
                        **runner.settings_for(machine, record['mode']))
        difference = [k for k, v in expected.items() if k != 'prefix'
                      and snapshot['effective_config'].get(k) != v]
        if difference or snapshot['machine'] != machine or snapshot['phase'] != 'formal':
            issues.append(dict(mode=record['mode'], fields=difference, kind='identity_or_config'))
        telemetry = {name: json_rows(text, marker) for name, marker in markers.items()}
        epochs = {(int(t), int(e)) for t, e in re.findall(
            r'LoRA learning rates: task=(\d+), epoch=(\d+)', text)}
        sampled = {(t, e, l, b, p) for t in range(10) for e in (1, 5, 10, 20)
                   for l in range(12) for b in (('S',) if t == 0 else ('S', 'P'))
                   for p in ('Q', 'K', 'V')}
        row.update(training_epochs_complete=epochs == {(t, e) for t in range(10) for e in range(1, 21)},
            probe_records_complete={r['task'] for r in telemetry['probes']} == set(range(1, 10))
                and len(telemetry['probes']) == 9,
            mask_records_complete=len(telemetry['masks']) == 9 * 12 * 5
                and all(r['qkv_counts'] == r['reference_counts'] for r in telemetry['masks']),
            update_records_complete={(r['task'], r['epoch'], r['layer'], r['branch'], r['projection'])
                for r in measured['epochs']} == sampled,
            prototype_diagnostics_complete={(r['task'], r['partition'], r['position'])
                for r in telemetry['diagnostics']} == {(t, part, pos) for t in (1, 5, 9)
                for part in ('old', 'new') for pos in ('wpre', 'prototype_high', 'prototype_shuffled', 'permuted')},
            code_revision=snapshot['code_revision'], std=None, complete_three_seeds=False)
        row['valid_performance'] &= row['training_epochs_complete']
        row['mechanism_records_complete'] = all(row[key] for key in ('probe_records_complete',
            'mask_records_complete', 'update_records_complete', 'prototype_diagnostics_complete')) and position_evidence_complete(telemetry)
        row['probe_seconds'] = sum(r['seconds'] for r in telemetry['probes'])
        row['peak_allocated_bytes'] = max((r['cuda_peak_allocated_bytes'] for r in measured['costs']), default=0)
        row['position'] = expected['dual_mask_protect_position']
        row['norm_mode'] = expected['dual_mask_position_norm_match']
        results.append(row)
        snapshots.append(snapshot)
        for name in ('tasks', 'epochs', 'costs', 'storage'):
            rows[name].extend(measured[name])
        for name, values in telemetry.items():
            rows[name].extend(dict(r, mode=record['mode']) for r in values)
    if snapshots:
        reference = snapshots[0]
        for snapshot in snapshots[1:]:
            for field in ('code_revision', 'source_sha256', 'software', 'hardware'):
                if snapshot[field] != reference[field]:
                    issues.append(dict(mode=snapshot['mode'], field=field))
            for key in reference['effective_config'].keys() | snapshot['effective_config'].keys():
                if key not in ('prefix', 'dual_mask_protect_position') and (
                        reference['effective_config'].get(key) != snapshot['effective_config'].get(key)):
                    issues.append(dict(mode=snapshot['mode'], field='config.' + key))
    complete = [r for r in results if r.get('valid_performance')]
    if len({r['Task0'] for r in complete}) > 1:
        issues.append(dict(field='Task0_startpoint_accuracy'))
    pairs = paired_results(results, issues, machine)
    for name, values in (('results', results), ('pairs', pairs), ('matching_issues', issues)):
        (directory / (name + '.json')).write_text(json.dumps(values, indent=2) + '\n')
        write_csv(directory / (name + '.csv'), values)
    for name, values in rows.items():
        write_csv(directory / (name + '.csv'), values)
    lines = ['# ' + machine + ' 原型保护位置实验', '',
        '| 组 | 状态 | Average | Last | Old | New | 阶段Old | 阶段New | Forgetting |',
        '|---|---|---:|---:|---:|---:|---:|---:|---:|']
    for row in results:
        lines.append('| ' + ' | '.join([row['mode'], row['status']] + [
            str(row.get(k, '')) for k in METRICS]) + ' |')
    lines += ['', '## 完整配对差值（前者减后者，百分点）', '', json.dumps(pairs, ensure_ascii=False),
        '', '指纹差异：' + json.dumps(issues, ensure_ascii=False), '',
        '仅单seed、同机比较；不据此宣布显著、等效或替换正式基线。',
        '同状态四位置等范数不保证不同训练轨迹等量。',
        '原型敏感度来自当前训练图片；测试标签只用于诊断，不参与掩码构造。',
        '谱参照同样执行评分探针；成本包含本轮验证开销，CA协方差仍保留。',
        '机制记录完整性独立于性能完整性；未启动组为待完成，不是负结果。']
    (directory / 'report.md').write_text('\n'.join(lines) + '\n')
    if complete:
        draw(directory, complete, rows)


def draw(directory, complete, rows):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    for row in complete:
        name = row['mode']
        tasks = [r for r in rows['tasks'] if r['mode'] == name]
        axes[0].plot([r['task'] for r in tasks], [r['total'] for r in tasks], '.-', label=name)
        axes[1].scatter(row['StageOld'], row['StageNew'])
        axes[1].annotate(name, (row['StageOld'], row['StageNew']), fontsize=8)
    axes[0].set(xlabel='Task', ylabel='All-seen accuracy (%)')
    axes[0].legend(fontsize=8)
    axes[1].set(xlabel='Task1-9 mean Old (%)', ylabel='Task1-9 mean New (%)')
    for suffix in ('png', 'pdf'):
        fig.savefig(directory / ('old_new_and_tasks.' + suffix), dpi=180)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    for row in complete:
        for ax, branch in zip(axes, ('S', 'P')):
            selected = [r for r in rows['epochs'] if r['mode'] == row['mode'] and r['branch'] == branch and r['task']]
            points = sorted({(r['task'], r['epoch']) for r in selected})
            means = [sum(r['effective_norm'] for r in selected if (r['task'], r['epoch']) == p) /
                     sum((r['task'], r['epoch']) == p for r in selected) for p in points]
            ax.plot(range(len(points)), means, label=row['mode'])
            ax.set(xlabel='Task1-9 sampled epochs', ylabel='Mean effective update norm', title=branch)
    axes[0].legend(fontsize=8)
    fig.savefig(directory / 'actual_norm_trajectories.png', dpi=180)
    plt.close(fig)
    for row in complete:
        fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
        for ax, partition in zip(axes, ('old', 'new')):
            for position in ('wpre', 'prototype_high', 'prototype_shuffled', 'permuted'):
                selected = [r for r in rows['diagnostics'] if r['mode'] == row['mode']
                            and r['partition'] == partition and r['position'] == position]
                ax.plot([r['task'] for r in selected], [r['mean_margin'] for r in selected], '.-', label=position)
            ax.set(xlabel='Task, last epoch before merge', ylabel='All-seen margin', title=partition)
        axes[0].legend(fontsize=7)
        fig.savefig(directory / (row['mode'] + '_equal_norm_margin.png'), dpi=180)
        plt.close(fig)
    fig, ax = plt.subplots(figsize=(9, 4), constrained_layout=True)
    names = [r['mode'] for r in complete]
    bottom = [0.] * len(names)
    for stage in sorted({r['stage'] for r in rows['costs']}):
        values = [sum(r['seconds'] for r in rows['costs'] if r['mode'] == n and r['stage'] == stage) / 60 for n in names]
        ax.bar(names, values, bottom=bottom, label=stage)
        bottom = [a + b for a, b in zip(bottom, values)]
    ax.set(ylabel='Measured minutes', title='Costs include validation-only probes/diagnostics')
    ax.legend(fontsize=7)
    fig.savefig(directory / 'stage_costs.png', dpi=180)
    plt.close(fig)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    summarize_saved(parser.parse_args().directory)
