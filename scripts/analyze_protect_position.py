"""Report the four measured cells; incomplete/smoke data never become full-T10 evidence."""
import argparse
import ast
import csv
import json
import math
from pathlib import Path
import re


LABELS = {'A': 'Wpre / product', 'B': 'Wpre / magnitude',
          'C': 'Permuted / product', 'D': 'Permuted / magnitude'}
METRICS = ('Average', 'Last', 'Old', 'New', 'StageOld', 'StageNew', 'Forgetting')
ALLOWED_CONFIG_DIFFS = {'prefix', 'dual_mask_protect_position', 'dual_mask_conflict_score_mode'}


def json_rows(content, tag):
    return [json.loads(row) for row in re.findall(tag + r' (\{[^\n]+\})', content)]


def write_csv(path, rows):
    if rows:
        with path.open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(dict.fromkeys(k for row in rows for k in row)))
            writer.writeheader()
            writer.writerows(rows)


def read_run(directory, record):
    content = (directory / record['mode'] / 'training.log').read_text()
    snapshot = json.loads((directory / record['mode'] / 'run.json').read_text())
    tasks = [ast.literal_eval(row) for row in re.findall(r'\[trainer.py\] => CNN: (\{[^\n]+\})', content)]
    averages = re.findall(r'\[trainer.py\] => Average Accuracy: ([\d.]+)', content)
    forgetting = re.findall(r'\[trainer.py\] => Forgetting:\s*([\d.-]+)', content)
    updates = json_rows(content, 'ProtectionPositionUpdate')
    masks = json_rows(content, 'ProtectionPositionMask')
    for row in updates + masks:
        row['mode'] = record['mode']
    incremental = [row for row in updates if row['task'] > 0]
    expected_masks = {(task, layer) for task in range(10) for layer in range(12)}
    expected_updates = {(task, layer, branch, projection) for task in range(10)
        for layer in range(12) for branch in (('S',) if task == 0 else ('S', 'P'))
        for projection in ('Q', 'K', 'V')}
    summary = dict(record, label=LABELS[record['mode']], tasks_reported=len(tasks),
        full_t10_completed=len(tasks) == 10 and record['exit_code'] == 0 and snapshot.get('phase') == 'formal',
        traceback_present='Traceback (most recent call last)' in content,
        mask_records=len(masks), update_records=len(updates),
        diagnostics_complete={(r['task'], r['layer']) for r in masks} == expected_masks
            and {(r['task'], r['layer'], r['branch'], r['projection']) for r in updates} == expected_updates)
    summary.update({key: None for key in METRICS})
    if tasks:
        summary.update(Task0=tasks[0]['total'], Last=tasks[-1]['total'],
                       Old=tasks[-1]['old'], New=tasks[-1]['new'])
    if len(tasks) > 1:
        summary.update(StageOld=sum(t['old'] for t in tasks[1:]) / (len(tasks) - 1),
                       StageNew=sum(t['new'] for t in tasks[1:]) / (len(tasks) - 1))
    if averages:
        summary['Average'] = float(averages[-1])
    if forgetting:
        summary['Forgetting'] = float(forgetting[-1])
    summary['metrics_complete'] = summary['full_t10_completed'] and all(
        summary[key] is not None and math.isfinite(summary[key]) for key in METRICS)
    if incremental:
        for field in ('raw_norm', 'effective_norm', 'total_removed_norm', 'total_removed_ratio',
                      'conflict_removed_norm', 'conflict_removed_ratio', 'protect_density',
                      'effective_conflict_density'):
            summary['mean_' + field] = sum(row[field] for row in incremental) / len(incremental)
        summary['max_merge_error'] = max(row['merge_error'] for row in updates)
    return summary, masks, updates, [dict(mode=record['mode'], task=i, **t) for i, t in enumerate(tasks)]


def comparison_differences(directory, modes):
    snapshots = {mode: json.loads((directory / mode / 'run.json').read_text()) for mode in modes}
    reference = snapshots[modes[0]]
    differences = []
    for mode, snapshot in snapshots.items():
        for field in ('machine', 'phase', 'source_sha256', 'software', 'hardware'):
            if snapshot.get(field) != reference.get(field):
                differences.append(dict(mode=mode, field=field))
        if not snapshot.get('hardware'):
            differences.append(dict(mode=mode, field='missing_hardware'))
        for key in reference['effective_config'].keys() | snapshot['effective_config'].keys():
            if key not in ALLOWED_CONFIG_DIFFS and reference['effective_config'].get(key) != snapshot['effective_config'].get(key):
                differences.append(dict(mode=mode, field='config.' + key))
        expected_position = 'wpre' if mode in ('A', 'B') else 'permuted'
        expected_score = 'conflict' if mode in ('A', 'C') else 'magnitude'
        for key, expected in (('dual_mask_protect_position', expected_position),
                              ('dual_mask_conflict_score_mode', expected_score)):
            if snapshot['effective_config'].get(key) != expected:
                differences.append(dict(mode=mode, field='wrong_factor.' + key))
        if snapshot.get('phase') != 'formal' or snapshot.get('machine') != '3090':
            differences.append(dict(mode=mode, field='not_formal_3090'))
    return differences


def factor_effects(cells):
    result = {}
    for metric in METRICS:
        a, b, c, d = (cells[mode][metric] for mode in ('A', 'B', 'C', 'D'))
        if None not in (a, b, c, d):
            result[metric] = dict(wpre_vs_permuted=((a + b) - (c + d)) / 2,
                product_vs_magnitude=((a + c) - (b + d)) / 2,
                interaction=(a - c) - (b - d),
                product_position_pair=a - c, magnitude_position_pair=b - d,
                magnitude_vs_product_at_wpre=b - a)
    return result


def draw(directory, summaries):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    complete = {row['mode']: row for row in summaries if row['full_t10_completed']}
    if not complete:
        return
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), constrained_layout=True)
    for ax, old, new, title in ((axes[0], 'Old', 'New', 'Final task'),
                               (axes[1], 'StageOld', 'StageNew', 'Task1-9 stage mean')):
        for mode, row in complete.items():
            ax.scatter(row[old], row[new], s=65,
                       color='#2865a8' if mode in ('A', 'B') else '#dd8547',
                       marker='o' if mode in ('A', 'C') else '^')
            ax.annotate(mode + ': ' + LABELS[mode], (row[old], row[new]),
                        xytext=(5, 5), textcoords='offset points', fontsize=8)
        for target, source in (('A', 'C'), ('B', 'D')):
            if target in complete and source in complete:
                ax.annotate('', xy=(complete[target][old], complete[target][new]),
                            xytext=(complete[source][old], complete[source][new]),
                            arrowprops=dict(arrowstyle='->', color='#888888', alpha=.65))
        ax.set(xlabel='Old accuracy (%)', ylabel='New accuracy (%)', title=title)
        ax.grid(alpha=.2)
    for extension in ('png', 'pdf'):
        fig.savefig(directory / ('old_new.' + extension), dpi=200)
    plt.close(fig)


def summarize(directory, records):
    directory = Path(directory)
    summaries, masks, updates, tasks = [], [], [], []
    for record in records:
        summary, m, u, t = read_run(directory, record)
        summaries.append(summary)
        masks.extend(m)
        updates.extend(u)
        tasks.extend(t)
    for name, rows in (('results', summaries), ('mask_diagnostics', masks),
                       ('update_diagnostics', updates), ('per_task', tasks)):
        write_csv(directory / (name + '.csv'), rows)
    (directory / 'results.json').write_text(json.dumps(summaries, indent=2) + '\n')
    differences = comparison_differences(directory, [r['mode'] for r in records])
    ready = len(summaries) == 4 and {r['mode'] for r in summaries} == set(LABELS)
    if summaries:
        for row in summaries[1:]:
            if row.get('Task0') != summaries[0].get('Task0'):
                differences.append(dict(mode=row['mode'], field='Task0'))
    metrics_complete = all(r['metrics_complete'] and not r['traceback_present'] for r in summaries)
    diagnostics_complete = all(r['diagnostics_complete'] for r in summaries)
    ready = ready and metrics_complete and diagnostics_complete and not differences
    effects = factor_effects({r['mode']: r for r in summaries}) if ready else {}
    (directory / 'factor_effects.json').write_text(json.dumps(dict(ready=ready,
        metrics_complete=metrics_complete, diagnostics_complete=diagnostics_complete,
        unexpected_differences=differences, effects=effects), indent=2) + '\n')
    lines = ['# 保护位置与冲突排名归因', '',
             '四组归因资料完整且指纹匹配：' + str(ready),
             '指标完整且无Traceback：' + str(metrics_complete) + '；实际更新诊断完整：' + str(diagnostics_complete), '',
             '| 组 | Average | Last | Old | New | 阶段Old | 阶段New | Forgetting | 完成任务 |',
             '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for row in summaries:
        values = [str(round(row[k], 4)) if row[k] is not None else '待完成' for k in METRICS]
        lines.append('| ' + row['mode'] + ' | ' + ' | '.join(values) + ' | ' + str(row['tasks_reported']) + ' |')
    lines += ['', '位置效应为 (A+B−C−D)/2；排名效应为 (A+C−B−D)/2；交互为 (A−C)−(B−D)。',
              '单seed用于初步归因。相同保护数量不保证相同抑制量；原始/有效/移除范数见 update_diagnostics.csv。',
              '仍保留 W_pre 重要性、正则和任务控制；本轮不是整个 W_pre 的去除实验。']
    if ready:
        lines += ['', '## 已测量效应（pp）', '', '```json', json.dumps(effects, indent=2), '```']
        draw(directory, summaries)
        lines += ['', '![Old–New](old_new.png)']
    if differences:
        lines += ['', '意外指纹差异：' + json.dumps(differences, ensure_ascii=False)]
    (directory / 'report.md').write_text('\n'.join(lines) + '\n')
    print('Measured formal results:', json.dumps(summaries), flush=True)
    return summaries


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    args = parser.parse_args()
    summarize(args.directory, json.loads((args.directory / 'queue.json').read_text()))


if __name__ == '__main__':
    main()
