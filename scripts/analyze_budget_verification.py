"""仅汇总健康完整T10，分数据集核对有效配置与源码，不跨机器归因。"""
import argparse
import json
import math
from pathlib import Path
import re

from analyze_core_evidence import METRICS, read_run
from analyze_protect_position import json_rows, write_csv
import run_budget_verification as runner


def pairs_for(results, issues, machine):
    invalid = {row['dataset'] for row in issues}
    complete = {row['mode']: row for row in results if row.get('valid_performance')
                and row['dataset'] not in invalid}
    comparisons = [('R1', 'R0'), ('R1', 'R2'), ('R3', 'R4'), ('R5', 'R0'),
                   ('R5', 'R6'), ('R7', 'R8')] if machine == '3090' else [
                       ('S1', 'S0'), ('S2', 'S1'), ('S2', 'S3'), ('S4', 'S1'), ('S4', 'S5')]
    pairs = []
    for dataset in runner.spec()[machine]['datasets']:
        for first, second in comparisons:
            names = [dataset['name'] + '_' + name + '_seed1993' for name in (first, second)]
            if all(name in complete for name in names):
                delta = {key: complete[names[0]][key] - complete[names[1]][key] for key in METRICS}
                pairs.append(dict(dataset=dataset['name'], first=names[0], second=names[1], **delta,
                    practical_threshold_passed=(delta['Average'] >= .2 and delta['Last'] >= 0
                                                and delta['StageOld'] >= -.1)))
    return pairs


def summarize_saved(directory):
    manifest = json.loads((directory / 'manifest.json').read_text())
    machine = manifest['machine']
    queue_path = directory / 'queue.json'
    records = json.loads(queue_path.read_text()) if queue_path.exists() else []
    results, issues, snapshots = [], [], {}
    rows = {name: [] for name in ('tasks', 'epochs', 'updates', 'masks', 'diagnostics', 'costs', 'storage', 'budgets')}
    for record in records:
        dataset, _ = runner.identity(machine, record['mode'])
        item = dict(record, dataset=dataset['name'], valid_performance=False)
        if record['status'] not in ('completed', 'failed'):
            results.append(item)
            continue
        measured, snapshot, telemetry = read_run(directory, record)
        content = (directory / record['mode'] / 'training.log').read_text()
        measured.update(dataset=dataset['name'], seed=1993, std=None, complete_three_seeds=False,
                        code_revision=snapshot['code_revision'])
        expected = runner.effective_config(machine, record['mode'])
        problem = lambda field: issues.append(dict(dataset=dataset['name'], mode=record['mode'], field=field))
        for key in expected.keys() | snapshot['effective_config'].keys():
            if key != 'prefix' and expected.get(key) != snapshot['effective_config'].get(key):
                problem('config.' + key)
        for key, value in (('machine', machine), ('phase', 'formal'), ('code_revision', manifest['revision'])):
            if snapshot.get(key) != value:
                problem(key)
        for key in ('source_sha256', 'software', 'hardware'):
            if dataset['name'] in snapshots and snapshot.get(key) != snapshots[dataset['name']].get(key):
                problem(key)
        snapshots.setdefault(dataset['name'], snapshot)
        learned = {(int(t), int(e)) for t, e in re.findall(r'LoRA learning rates: task=(\d+), epoch=(\d+)', content)}
        measured['training_epochs_complete'] = learned == {(t, e) for t in range(10) for e in range(1, 21)}
        measured['valid_performance'] = measured['valid_performance'] and measured['training_epochs_complete']
        if not measured['training_epochs_complete']:
            problem('training_epochs')
        budget = [dict(row, mode=record['mode']) for row in json_rows(content, 'BudgetUpdateAudit')]
        expected_budget = {(t, e, l, b) for t in range(1, 10) for e in (1, 5, 10, 20)
                           for l in range(12) for b in ('S', 'P')}
        measured['budget_records_complete'] = (len(budget) == len(expected_budget) and
            {(r['task'], r['epoch'], r['layer'], r['branch']) for r in budget} == expected_budget)
        if not measured['budget_records_complete']:
            problem('budget_records')
        for row in budget:
            for key, value in row.items():
                if isinstance(value, float) and not math.isfinite(value):
                    problem('nonfinite_budget.' + key)
            for key in ('uniform_match_residual', 'pair_matching_residual'):
                if row.get(key) is not None and row[key] > 1e-5:
                    problem(key)
            if row['region_budget_applies'] and row['budget_violation'] > 1e-5:
                problem('budget_violation')
        diagnostic = telemetry['diagnostics']
        measured['budget_position_diagnostics_complete'] = (len(diagnostic) == 6 and
            {(r['task'], r['epoch'], r['partition']) for r in diagnostic}
            == {(t, 20, p) for t in (1, 5, 9) for p in ('old', 'new')} and
            all(r['scope'] == 'all_seen' and r['norm_control'] == 'same_state_paired_layer_min'
                and 0 < r['count'] <= (512 if r['partition'] == 'old' else 128) for r in diagnostic))
        if not measured['budget_position_diagnostics_complete']:
            problem('position_diagnostics')
        expected_updates = {(t, e, l, b, p) for t in range(10) for e in (1, 5, 10, 20)
            for l in range(12) for b in (('S',) if t == 0 else ('S', 'P')) for p in ('Q', 'K', 'V')}
        measured['budget_epoch_records_complete'] = (len(telemetry['epochs']) == len(expected_updates) and
            {(r['task'], r['epoch'], r['layer'], r['branch'], r['projection']) for r in telemetry['epochs']}
            == expected_updates)
        if not measured['budget_epoch_records_complete']:
            problem('epoch_update_records')
        for name in ('epochs', 'updates', 'diagnostics', 'costs', 'storage'):
            for row in telemetry[name]:
                if any(isinstance(v, float) and not math.isfinite(v) for v in row.values()):
                    problem('nonfinite_' + name)
        measured['peak_allocated_bytes'] = max((r['cuda_peak_allocated_bytes'] for r in telemetry['costs']), default=0)
        results.append(measured)
        for name in telemetry:
            rows[name].extend(telemetry[name])
        rows['budgets'].extend(budget)
    for dataset in snapshots:
        starts = {row['Task0'] for row in results if row['dataset'] == dataset and row.get('valid_performance')}
        if len(starts) > 1:
            issues.append(dict(dataset=dataset, field='Task0_startpoint'))
    pairs = pairs_for(results, issues, machine)
    for name, values in (('results', results), ('matching_issues', issues), ('pairs', pairs)):
        (directory / (name + '.json')).write_text(json.dumps(values, indent=2) + '\n')
        write_csv(directory / (name + '.csv'), values)
    for name, values in rows.items():
        write_csv(directory / (name + '.csv'), values)
    lines = ['# ' + machine + ' 更新预算验证', '',
             '| 数据集 / 组 | 状态 | Average | Last | 阶段Old | 阶段New | Forgetting |',
             '|---|---|---:|---:|---:|---:|---:|']
    for row in results:
        lines.append('| ' + ' | '.join([row['mode'], row['status']] + [str(row.get(k, ''))
            for k in ('Average', 'Last', 'StageOld', 'StageNew', 'Forgetting')]) + ' |')
    lines += ['', '完整配对（前者减后者）：', json.dumps(pairs, ensure_ascii=False), '',
              '身份与机制问题：' + json.dumps(issues, ensure_ascii=False), '',
              '全部seed1993、同机配对。预算只约束S；P硬权限与幅度门仍保留。',
              '均匀对照匹配额外移除范数，位置组匹配同状态整层有效范数，训练轨迹不保证等量。',
              'O–F为组合精简比较。测试标签仅用于只读诊断。CA仍保存类别协方差。',
              '时间待完成不等于负结果；单seed不证明显著、等效、新颖性或自动替换基线。']
    (directory / 'report.md').write_text('\n'.join(lines) + '\n')
    if any(row.get('valid_performance') for row in results):
        draw(directory, results, rows)


def draw(directory, results, rows, diagnostic_label='Wpre minus permuted margin'):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    complete = [r for r in results if r.get('valid_performance')]
    for dataset in sorted({r['dataset'] for r in complete}):
        selected = [r for r in complete if r['dataset'] == dataset]
        fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
        for row in selected:
            name = row['mode']
            stages = [r for r in rows['tasks'] if r['mode'] == name]
            axes[0].plot([r['task'] for r in stages], [r['total'] for r in stages], '.-', label=name)
            axes[1].scatter(row['StageOld'], row['StageNew'])
            axes[1].annotate(name, (row['StageOld'], row['StageNew']), fontsize=7)
        axes[0].set(xlabel='Task', ylabel='All-seen accuracy (%)')
        axes[0].legend(fontsize=6)
        axes[1].set(xlabel='Task1-9 mean Old (%)', ylabel='Task1-9 mean New (%)')
        fig.savefig(directory / (dataset + '_accuracy_old_new.pdf'))
        plt.close(fig)
        fig, axes = plt.subplots(1, 3, figsize=(14, 4), constrained_layout=True)
        for row in selected:
            budgets = [r for r in rows['budgets'] if r['mode'] == row['mode'] and r['branch'] == 'S']
            points = sorted({(r['task'], r['epoch']) for r in budgets})
            for ax, field in zip(axes, ('reference_alpha', 'effective_protected_norm', 'effective_plastic_norm')):
                means = [sum(r[field] for r in budgets if (r['task'], r['epoch']) == p) /
                         sum((r['task'], r['epoch']) == p for r in budgets) for p in points]
                ax.plot(range(len(points)), means, label=row['mode'])
                ax.set(xlabel='Sampled task/epoch', ylabel=field)
        axes[0].legend(fontsize=6)
        fig.savefig(directory / (dataset + '_budget_trajectories.pdf'))
        plt.close(fig)
        fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
        for row in selected:
            for ax, branch in zip(axes, ('S', 'P')):
                updates = [r for r in rows['epochs'] if r['mode'] == row['mode']
                           and r['branch'] == branch and r['task'] > 0]
                points = sorted({(r['task'], r['epoch']) for r in updates})
                for field, style in (('raw_norm', '--'), ('effective_norm', '-')):
                    means = [sum(r[field] for r in updates if (r['task'], r['epoch']) == p) /
                             sum((r['task'], r['epoch']) == p for r in updates) for p in points]
                    ax.plot(range(len(points)), means, style, label=row['mode'] + ' ' + field)
                ax.set(xlabel='Sampled task/epoch', ylabel='Mean Q/K/V update norm', title=branch)
        axes[0].legend(fontsize=5)
        fig.savefig(directory / (dataset + '_update_norms.pdf'))
        plt.close(fig)
        fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
        for row in selected:
            values = [r for r in rows['diagnostics'] if r['mode'] == row['mode']]
            for ax, partition in zip(axes, ('old', 'new')):
                current = [r for r in values if r['partition'] == partition]
                ax.plot([r['task'] for r in current], [r['wpre_margin'] - r['permuted_margin'] for r in current],
                        '.-', label=row['mode'])
                ax.set(xlabel='Task', ylabel=diagnostic_label, title=partition)
                ax.axhline(0, color='gray', linewidth=.5)
        axes[0].legend(fontsize=6)
        fig.savefig(directory / (dataset + '_position_diagnostics.pdf'))
        plt.close(fig)
        fig, ax = plt.subplots(figsize=(10, 4), constrained_layout=True)
        names = [r['mode'] for r in selected]
        bottom = [0.] * len(names)
        for stage in sorted({r['stage'] for r in rows['costs']}):
            values = [sum(r['seconds'] for r in rows['costs'] if r['mode'] == name and r['stage'] == stage) / 60
                      for name in names]
            ax.bar(names, values, bottom=bottom, label=stage)
            bottom = [a + b for a, b in zip(bottom, values)]
        ax.set(ylabel='Measured minutes')
        ax.legend(fontsize=6)
        fig.savefig(directory / (dataset + '_costs.pdf'))
        plt.close(fig)
        fig, ax = plt.subplots(figsize=(10, 4), constrained_layout=True)
        final = {name: next((r for r in reversed(rows['storage']) if r['mode'] == name
                            and r['stage'] == 'post_ca_merged'), {}) for name in names}
        bottom = [0.] * len(names)
        for group in sorted({g for r in final.values() for g in r.get('groups_bytes', {})}):
            values = [final[name].get('groups_bytes', {}).get(group, 0) / 2**20 for name in names]
            ax.bar(names, values, bottom=bottom, label=group)
            bottom = [a + b for a, b in zip(bottom, values)]
        ax.set(ylabel='Live persistent tensor MiB; excludes optimizer')
        ax.legend(fontsize=6)
        fig.savefig(directory / (dataset + '_storage.pdf'))
        plt.close(fig)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    summarize_saved(parser.parse_args().directory)
