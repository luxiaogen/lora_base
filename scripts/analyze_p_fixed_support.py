"""固定支持与随机/均匀对照；缺失健康、配置或机制证据不作配对。"""
import argparse
import json
import math
from pathlib import Path
import re

from analyze_core_evidence import METRICS, read_run
from analyze_protect_position import json_rows, write_csv
import run_p_fixed_support as runner


def uniform_match_valid(row):
    keys = ('extra_removed_norm', 'reference_removed_norm', 'matching_residual',
            'uniform_coefficient', 'p_plastic_norm')
    if any(not isinstance(row.get(k), (int, float)) or not math.isfinite(row[k]) for k in keys):
        return False
    actual, target = row['extra_removed_norm'], row['reference_removed_norm']
    tolerance = 1e-6 + 5e-5 * max(actual, target)
    reporting_tolerance = 1e-8 + 5e-7 * max(actual, target)
    return (0 <= row['uniform_coefficient'] <= 1 and actual >= 0 and target >= 0
            and 0 <= row['matching_residual'] <= tolerance
            and abs(actual - target) <= tolerance
            and abs(row['matching_residual'] - abs(actual - target)) <= reporting_tolerance
            and abs(actual - row['uniform_coefficient'] * row['p_plastic_norm']) <= tolerance)


def pairs_for(results, issues):
    bad = {r['mode'] for r in issues}
    good = {r['mode']: r for r in results if r.get('valid_performance')
            and r.get('support_evidence_complete') and r['mode'] not in bad}
    return [dict(first=a, second=b, **{k: good[a][k] - good[b][k] for k in METRICS})
            for first, second in [('B', 'A'), ('B', 'C'), ('B', 'D'), ('C', 'A'), ('D', 'A')]
            for a, b in [('imgr10_' + first + '_seed1993', 'imgr10_' + second + '_seed1993')]
            if a in good and b in good]


def summarize_saved(directory):
    manifest = json.loads((directory / 'manifest.json').read_text())
    records = json.loads((directory / 'queue.json').read_text()) if (directory / 'queue.json').exists() else []
    results, issues, reference = [], [], None
    rows = {k: [] for k in ('tasks', 'epochs', 'updates', 'costs', 'storage', 'supports', 'support_updates')}
    for record in records:
        name = record['mode']
        if record['status'] not in ('completed', 'failed'):
            results.append(dict(record, valid_performance=False, support_evidence_complete=False))
            continue
        def problem(field):
            issues.append(dict(mode=name, field=field))
        try:
            measured, snapshot, telemetry = read_run(directory, record)
            content = (directory / name / 'training.log').read_text()
        except (OSError, ValueError, KeyError, SyntaxError, TypeError) as error:
            problem('unreadable_run_evidence')
            results.append(dict(record, valid_performance=False, support_evidence_complete=False,
                                evidence_error=str(error)))
            continue
        expected = runner.effective_config(manifest['machine'], name)
        for k in expected.keys() | snapshot['effective_config'].keys():
            if k != 'prefix' and expected.get(k) != snapshot['effective_config'].get(k):
                problem('config.' + k)
        if snapshot.get('code_revision') != manifest['revision'] or snapshot.get('phase') != 'formal':
            problem('revision_or_phase')
        if not runner.REQUIRED_SOURCE <= snapshot.get('source_sha256', {}).keys():
            problem('missing_source_fingerprint')
        for key in ('source_sha256', 'hardware', 'software'):
            if not snapshot.get(key) or reference is not None and snapshot[key] != reference[key]:
                problem(key)
        reference = reference or snapshot
        learned = {(int(t), int(e)) for t, e in re.findall(r'LoRA learning rates: task=(\d+), epoch=(\d+)', content)}
        measured['training_epochs_complete'] = learned == {(t, e) for t in range(10) for e in range(1, 21)}
        measured['valid_performance'] &= measured['training_epochs_complete']
        if not measured['training_epochs_complete']:
            problem('training_epochs')
        selected = [dict(r, run=name) for r in json_rows(content, 'PFixedSupportSelection')]
        updates = [dict(r, run=name) for r in json_rows(content, 'PFixedSupportUpdate')]
        if (len(selected) != 108 or {(r['task'], r['layer']) for r in selected}
                != {(t, l) for t in range(1, 10) for l in range(12)}):
            problem('selection_records')
        mode = expected['p_fixed_support']
        for row in selected:
            if (row['capture_epoch'] != 2 or row['first_training_epoch'] != 3
                    or row['mode'] != mode or row['audit_only'] != (mode == 'off')
                    or row['retained_coordinates'] != int(.5 * row['plastic_coordinates'])
                    or len(row['mask_sha256']) != 64):
                problem('support_selection')
        expected_updates = {(t, l, e, 'epoch') for t in range(1, 10) for l in range(12)
                            for e in (1, 2, 3, 5, 10, 20)}
        expected_updates |= {(t, l, None, 'merge') for t in range(1, 10) for l in range(12)}
        if len(updates) != len(expected_updates) or {
                (r['task'], r['layer'], r['epoch'], r['event']) for r in updates} != expected_updates:
            problem('update_records')
        for row in updates:
            if row['mode'] != mode or any(isinstance(v, float) and not math.isfinite(v) for v in row.values()):
                problem('nonfinite_or_wrong_mode')
            active = mode != 'off' and (row['event'] == 'merge' or row['epoch'] >= 2)
            if row['fixed_active'] != active or row['effective_nonzero_coordinates'] > row['plastic_coordinates']:
                problem('active_support')
            if mode in ('magnitude', 'random') and active and row['effective_nonzero_coordinates'] > row['support_coordinates']:
                problem('outside_fixed_support')
            if mode == 'uniform_matched' and active and not uniform_match_valid(row):
                problem('uniform_matching_residual')
        costs_complete = bool(telemetry['costs'])
        if not costs_complete:
            problem('missing_cost_records')
        measured.update(seed=1993, std=None, complete_three_seeds=False,
            support_evidence_complete=not any(r['mode'] == name for r in issues),
            code_revision=snapshot['code_revision'],
            cost_records_complete=costs_complete,
            peak_allocated_bytes=max((r['cuda_peak_allocated_bytes'] for r in telemetry['costs']), default=None))
        results.append(measured)
        rows['supports'].extend(selected)
        rows['support_updates'].extend(updates)
        for key in ('tasks', 'epochs', 'updates', 'costs', 'storage'):
            rows[key].extend(telemetry[key])
    pairs = pairs_for(results, issues)
    for name, values in dict(results=results, matching_issues=issues, pairs=pairs, **rows).items():
        (directory / (name + '.json')).write_text(json.dumps(values, indent=2) + '\n')
        if values:
            write_csv(directory / (name + '.csv'), values)
        else:
            (directory / (name + '.csv')).write_text('')
    report = ['# 3090固定P支持结果', '',
        '| 组 | 状态 | Average | Last | Old | New | 阶段Old | 阶段New | Forgetting |',
        '|---|---|---:|---:|---:|---:|---:|---:|---:|']
    for row in results:
        report.append('| ' + ' | '.join([row['mode'], row['status']] + [str(row.get(k, ''))
            for k in ('Average', 'Last', 'Old', 'New', 'StageOld', 'StageNew', 'Forgetting')]) + ' |')
    report += ['', '健康且证据完整的配对：' + json.dumps(pairs, ensure_ascii=False),
        '配置／机制问题：' + json.dumps(issues, ensure_ascii=False), '',
        '分母：固定支持为可塑区坐标数的50%；原幅度门为完整QKV的10%；W_pre为90%重要性质量。',
        '三候选第2轮结束固定Q，第3轮开始使用。D只匹配同状态额外移除范数，独立轨迹仍可能不同。',
        '单seed不报告标准差、不证明等效或创新，不自动替换正式基线。']
    (directory / 'report.md').write_text('\n'.join(report) + '\n')
    valid = [r for r in results if r.get('valid_performance') and r.get('support_evidence_complete')]
    # 原始日志保留；清掉本汇总器生成的旧图，避免分析失败后呈现旧配对。
    for name in ('accuracy_old_new.pdf', 'support_update_norms.pdf'):
        (directory / name).unlink(missing_ok=True)
    if valid:
        draw(directory, valid, rows)


def draw(directory, results, rows):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    for result in results:
        task = [r for r in rows['tasks'] if r['mode'] == result['mode']]
        axes[0].plot([r['task'] for r in task], [r['total'] for r in task], '.-', label=result['mode'])
        axes[1].scatter(result['StageNew'], result['StageOld'], label=result['mode'])
    axes[0].set(xlabel='Task', ylabel='All-seen accuracy (%)')
    axes[1].set(xlabel='Mean stage New (%)', ylabel='Mean stage Old (%)')
    axes[0].legend(fontsize=6)
    fig.savefig(directory / 'accuracy_old_new.pdf')
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    for result in results:
        current = [r for r in rows['support_updates'] if r['run'] == result['mode'] and r['event'] == 'merge']
        tasks = sorted({r['task'] for r in current})
        for ax, field in zip(axes, ('p_effective_norm', 'net_effective_norm')):
            ax.plot(tasks, [sum(r[field] for r in current if r['task'] == t) / 12 for t in tasks],
                    '.-', label=result['mode'])
            ax.set(xlabel='Task', ylabel=field)
    axes[0].legend(fontsize=6)
    fig.savefig(directory / 'support_update_norms.pdf')
    plt.close(fig)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    summarize_saved(parser.parse_args().directory)
