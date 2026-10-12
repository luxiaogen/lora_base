"""分层权限四格；完整训练、配置及实际权限证据齐全才计算效应。"""
import argparse
import json
import math
from pathlib import Path
import re

from analyze_core_evidence import METRICS, read_run
from analyze_protect_position import json_rows, write_csv
import run_depth_permissions as runner


def permission_issues(depth, budgets, config, tasks=tuple(range(1, 10))):
    issues = []
    if (len(depth) != 12 * len(tasks) or {(r['task'], r['layer']) for r in depth}
            != {(t, layer) for t in tasks for layer in range(12)}
            or len(budgets) != len(tasks) or {r['task'] for r in budgets} != set(tasks)):
        issues.append('missing_depth_records')
    if len({r['center'] for r in budgets}) != 1:
        issues.append('coverage_center_changed')
    for row in depth:
        alpha = .5 if config['dual_mask_depth_strength'] == 'off' else .75 - .5 * row['layer'] / 11
        center = next((r['center'] for r in budgets if r['task'] == row['task']), .9)
        target = .9 if config['dual_mask_depth_coverage'] == 'off' else center + .05 * (1 - 2 * row['layer'] / 11)
        if (not all(math.isfinite(row[k]) for k in ('alpha', 'coverage_target', 'protected_importance_fraction'))
                or abs(row['alpha'] - alpha) > 1e-10 or abs(row['coverage_target'] - target) > 1e-10
                or row['coverage_mode'] != config['dual_mask_depth_coverage']
                or row['strength_mode'] != config['dual_mask_depth_strength']
                or row['s_rank'] != 64 or row['p_rank'] != 64
                or sum(row['qkv_protect_counts']) != row['protect_count']
                or row['protected_importance_fraction'] + 1e-6 < target):
            issues.append('wrong_effective_permissions')
    for row in budgets:
        current = [r for r in depth if r['task'] == row['task']]
        if (not math.isfinite(row['center']) or not .05 <= row['center'] <= .95
                or sum(r['protect_count'] for r in current) != row['protect_count']
                or sum(r['reference_count'] for r in current) != row['reference_count']
                or row['count_residual'] != row['protect_count'] - row['reference_count']):
            issues.append('wrong_count_budget')
    return sorted(set(issues))


def contrasts(results, issues):
    bad = {row['mode'] for row in issues}
    good = {r['mode']: r for r in results if r.get('valid_performance')
            and r.get('depth_evidence_complete') and r['mode'] not in bad}
    groups = {name: good.get('imgr10_' + name + '_seed1993') for name in 'ABCD'}
    values = {}
    for first, second in [('B', 'A'), ('C', 'A'), ('D', 'A'), ('D', 'B'), ('D', 'C')]:
        if groups[first] and groups[second]:
            values[first + '_minus_' + second] = {
                key: groups[first][key] - groups[second][key] for key in METRICS}
    if all(groups.values()):
        a, b, c, d = [groups[name] for name in 'ABCD']
        values.update(coverage_effect={k: (b[k] + d[k] - a[k] - c[k]) / 2 for k in METRICS},
            strength_effect={k: (c[k] + d[k] - a[k] - b[k]) / 2 for k in METRICS},
            interaction={k: d[k] - b[k] - c[k] + a[k] for k in METRICS})
    return values


def summarize_saved(directory):
    manifest = json.loads((directory / 'manifest.json').read_text())
    records = json.loads((directory / 'queue.json').read_text()) if (directory / 'queue.json').exists() else []
    results, issues, reference = [], [], None
    rows = {key: [] for key in ('tasks', 'epochs', 'updates', 'costs', 'storage', 'depth', 'budgets')}
    for record in records:
        name = record['mode']
        if record['status'] not in ('completed', 'failed'):
            results.append(dict(record, valid_performance=False, depth_evidence_complete=False))
            continue
        def problem(field):
            issues.append(dict(mode=name, field=field))
        try:
            measured, snapshot, telemetry = read_run(directory, record)
            content = (directory / name / 'training.log').read_text()
        except (OSError, ValueError, KeyError, SyntaxError, TypeError) as error:
            problem('unreadable_run_evidence')
            results.append(dict(record, valid_performance=False, depth_evidence_complete=False,
                                evidence_error=str(error)))
            continue
        expected = runner.effective_config(manifest['machine'], name)
        for key in expected.keys() | snapshot['effective_config'].keys():
            if key != 'prefix' and expected.get(key) != snapshot['effective_config'].get(key):
                problem('config.' + key)
        if (snapshot.get('code_revision') != manifest['revision']
                or snapshot.get('phase') != 'formal' or snapshot.get('machine') != manifest['machine']):
            problem('revision_phase_machine')
        if not runner.REQUIRED_SOURCE <= snapshot.get('source_sha256', {}).keys():
            problem('missing_source_fingerprint')
        for key in ('source_sha256', 'hardware', 'software'):
            if not snapshot.get(key) or reference is not None and snapshot[key] != reference[key]:
                problem(key)
        reference = reference or snapshot
        learned = {(int(t), int(e)) for t, e in re.findall(r'LoRA learning rates: task=(\d+), epoch=(\d+)', content)}
        measured['training_epochs_complete'] = learned == {(t, e) for t in range(10) for e in range(1, 21)}
        measured['valid_performance'] &= measured['training_epochs_complete']
        depth = [dict(row, mode=name) for row in json_rows(content, 'DepthPermission')]
        budgets = [dict(row, mode=name) for row in json_rows(content, 'DepthPermissionBudget')]
        for field in permission_issues(depth, budgets, expected):
            problem(field)
        if not measured['epoch_records_complete'] or not telemetry['costs']:
            problem('missing_update_or_cost_records')
        measured.update(seed=1993, std=None, complete_three_seeds=False,
            depth_evidence_complete=not any(r['mode'] == name for r in issues),
            code_revision=snapshot['code_revision'],
            peak_allocated_bytes=max((r['cuda_peak_allocated_bytes'] for r in telemetry['costs']), default=None))
        results.append(measured)
        rows['depth'].extend(depth)
        rows['budgets'].extend(budgets)
        for key in ('tasks', 'epochs', 'updates', 'costs', 'storage'):
            rows[key].extend(telemetry[key])
    effects = contrasts(results, issues)
    for name, values in dict(results=results, matching_issues=issues, **rows).items():
        (directory / (name + '.json')).write_text(json.dumps(values, indent=2) + '\n')
        write_csv(directory / (name + '.csv'), values) if values else (directory / (name + '.csv')).write_text('')
    (directory / 'effects.json').write_text(json.dumps(effects, indent=2) + '\n')
    report = ['# 3090分层权限结果', '',
        '| 组 | 状态 | Average | Last | Old | New | 阶段Old | 阶段New | Forgetting |',
        '|---|---|---:|---:|---:|---:|---:|---:|---:|']
    for row in results:
        report.append('| ' + ' | '.join([row['mode'], row['status']] + [str(row.get(k, ''))
            for k in ('Average', 'Last', 'Old', 'New', 'StageOld', 'StageNew', 'Forgetting')]) + ' |')
    report += ['', '完整且健康的配对效应：' + json.dumps(effects, ensure_ascii=False),
        '配置／机制问题：' + json.dumps(issues, ensure_ascii=False), '',
        '覆盖指重要性质量；分层组仅校准全模型保护坐标总数，实际端点见depth.csv。',
        'α只改变S软保护，P仍为保护区补集。β仍由R_old调节，并非固定0.5。',
        '数量接近不保证各层或有效更新范数相同；单seed不证明等效、显著或方向特异性。']
    (directory / 'report.md').write_text('\n'.join(report) + '\n')
    valid = [r for r in results if r.get('valid_performance') and r.get('depth_evidence_complete')]
    if valid:
        draw(directory, valid, rows)


def draw(directory, results, rows):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    for result in results:
        current = [r for r in rows['tasks'] if r['mode'] == result['mode']]
        axes[0].plot([r['task'] for r in current], [r['total'] for r in current], '.-', label=result['mode'])
        axes[1].scatter(result['StageNew'], result['StageOld'], label=result['mode'])
    axes[0].set(xlabel='Task', ylabel='All-seen accuracy (%)')
    axes[1].set(xlabel='Mean stage New (%)', ylabel='Mean stage Old (%)')
    axes[0].legend(fontsize=7)
    fig.savefig(directory / 'accuracy_old_new.pdf')
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    for result in results:
        current = [r for r in rows['depth'] if r['mode'] == result['mode'] and r['task'] == 1]
        for ax, field in zip(axes, ('coverage_target', 'alpha')):
            ax.plot([r['layer'] for r in current], [r[field] for r in current], '.-', label=result['mode'])
            ax.set(xlabel='Layer (shallow to deep)', ylabel=field)
    axes[0].legend(fontsize=7)
    fig.savefig(directory / 'depth_profiles.pdf')
    plt.close(fig)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    summarize_saved(parser.parse_args().directory)
