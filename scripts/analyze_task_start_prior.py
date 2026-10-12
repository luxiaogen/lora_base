"""任务起点位置和冲突来源的同机证据；缺失配对不归因。"""
import argparse
import json
import math
from pathlib import Path
from unittest.mock import patch

from analyze_core_evidence import METRICS
from analyze_protect_position import json_rows, write_csv
import analyze_budget_verification as budget
import run_task_start_prior as runner

_draw = budget.draw


def pairs_for(results, issues, machine):
    invalid = {r['dataset'] for r in issues}
    good = {r['mode']: r for r in results if r.get('valid_performance') and r['dataset'] not in invalid}
    comparisons = [('R1', 'R0'), ('R1', 'R2')] if machine == '3090' else [
        ('S2', 'S0'), ('S2', 'S1'), ('S2', 'S3'), ('S1', 'S0')]
    return [dict(dataset='imgr10', first=a, second=b,
                 **{k: good[a][k] - good[b][k] for k in METRICS})
            for first, second in comparisons
            for a, b in [('imgr10_' + first + '_seed1993', 'imgr10_' + second + '_seed1993')]
            if a in good and b in good]


def draw(directory, results, rows):
    _draw(directory, results, rows, diagnostic_label='Task-start minus permuted task-start margin')


def summarize_saved(directory):
    captured = {}
    def capture_rows(directory, results, rows):
        captured['rows'] = rows
    # 先提取事实；任务起点证据核验完成前不发布配对或绘图。
    with runner.configured_runner(), patch.object(budget, 'pairs_for', lambda *args: []), \
            patch.object(budget, 'draw', capture_rows):
        budget.summarize_saved(directory)
    manifest = json.loads((directory / 'manifest.json').read_text())
    results = json.loads((directory / 'results.json').read_text())
    issues = json.loads((directory / 'matching_issues.json').read_text())
    rows = {'priors': [], 'conflict_overlaps': []}
    for result in results:
        if result['status'] not in ('completed', 'failed'):
            continue
        content = (directory / result['mode'] / 'training.log').read_text()
        def problem(field):
            issues.append(dict(dataset='imgr10', mode=result['mode'], field=field))
        snapshot_path = directory / result['mode'] / 'run.json'
        snapshot = json.loads(snapshot_path.read_text()) if snapshot_path.exists() else {}
        if (not runner.REQUIRED_SOURCE <= snapshot.get('source_sha256', {}).keys()
                or not snapshot.get('hardware') or not snapshot.get('software')):
            problem('missing_task_start_fingerprint')
        for key, marker, expected in (
                ('priors', 'TaskStartPrior', {(t, l) for t in range(1, 10) for l in range(12)}),
                ('conflict_overlaps', 'TaskStartConflictAudit',
                 {(t, e, l, b) for t in range(1, 10) for e in (1, 5, 10, 20)
                  for l in range(12) for b in ('S', 'P')})):
            current = [dict(r, mode=result['mode']) for r in json_rows(content, marker)]
            fields = ('task', 'layer') if key == 'priors' else ('task', 'epoch', 'layer', 'branch')
            if len(current) != len(expected) or {tuple(r[k] for k in fields) for r in current} != expected:
                problem(key + '_incomplete')
            rows[key].extend(current)
        for row in [r for r in rows['priors'] if r['mode'] == result['mode']]:
            if row['qkv_reference_counts'] != row['qkv_task_start_counts']:
                problem('protection_counts')
            if any(not math.isfinite(row[k]) for k in ('relative_weight_drift', 'score_seconds')):
                problem('nonfinite_prior')
        diagnostics = json_rows(content, 'CorePositionDiagnostic')
        if any((r.get('reference_position'), r.get('permuted_position')) !=
               ('task_start_svd', 'task_start_svd_permuted') for r in diagnostics):
            problem('diagnostic_positions')
        result['task_start_evidence_complete'] = not any(i.get('mode') == result['mode'] for i in issues)
    pairs = pairs_for(results, issues, manifest['machine'])
    for name, values in dict(results=results, matching_issues=issues, pairs=pairs, **rows).items():
        (directory / (name + '.json')).write_text(json.dumps(values, indent=2) + '\n')
        write_csv(directory / (name + '.csv'), values)
    # 通用预算图的字段名保留，但图与报告均显式标出真实的任务起点对照。
    report = ['# ' + manifest['machine'] + ' 任务起点权重验证', '',
        '| 组 | 状态 | Average | Last | Old | New | 阶段Old | 阶段New | Forgetting |',
        '|---|---|---:|---:|---:|---:|---:|---:|---:|']
    for row in results:
        report.append('| ' + ' | '.join([row['mode'], row['status']] + [str(row.get(k, ''))
            for k in ('Average', 'Last', 'Old', 'New', 'StageOld', 'StageNew', 'Forgetting')]) + ' |')
    report += ['', '完整配对差值：' + json.dumps(pairs, ensure_ascii=False),
        '身份及机制问题：' + json.dumps(issues, ensure_ascii=False), '',
        '全部seed1993；α规则为ρ=1，实际α及抑制范数可能随位置/评分变化。',
        '正式训练不匹配范数；只读诊断比较任务起点掩码与其置换掩码，同状态整层有效范数匹配。',
        '旧字段wpre_margin对应reference_position标签，不表示W_pre位置。',
        '单seed结果不证明显著性、等效或新颖性；缺失配对不归因，不自动替换基线。']
    (directory / 'report.md').write_text('\n'.join(report) + '\n')
    validated = [r for r in results if r.get('valid_performance') and r.get('task_start_evidence_complete')]
    if validated and captured:
        draw(directory, validated, captured['rows'])
    if rows['priors']:
        # 通用图已生成，只补任务起点变化图。
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(9, 4), constrained_layout=True)
        for result in results:
            current = [r for r in rows['priors'] if r['mode'] == result['mode']]
            tasks = sorted({r['task'] for r in current})
            values = [sum(sum(r['qkv_reference_jaccards']) / 3 for r in current if r['task'] == t)
                      / sum(r['task'] == t for r in current) for t in tasks]
            ax.plot(tasks, values, '.-', label=result['mode'])
        ax.set(xlabel='Task', ylabel='Mean task-start vs Wpre protection Jaccard')
        ax.legend(fontsize=6)
        fig.savefig(directory / 'task_start_prior_changes.pdf')
        plt.close(fig)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    summarize_saved(parser.parse_args().directory)
