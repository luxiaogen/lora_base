"""单seed保护强度验证；正式参照同提交，缺失配对不生成结论。"""
import json
import math

from analyze_core_evidence import METRICS
from analyze_protect_position import json_rows, write_csv
import analyze_prototype_position as position


def compare_complete(results, issues):
    complete = {row['mode']: row for row in results if row.get('valid_performance')}
    reference = complete.get('REF_seed1993')
    if reference is None or issues:
        return []
    return [dict(mode=name, reference='REF_seed1993',
                 **{key: row[key] - reference[key] for key in METRICS})
            for name, row in complete.items() if name != 'REF_seed1993']


def summarize_saved(directory):
    position.summarize_saved(directory)
    results = json.loads((directory / 'results.json').read_text())
    issues = json.loads((directory / 'matching_issues.json').read_text())
    rows = []
    expected = {(task, epoch, layer) for task in range(1, 10)
                for epoch in (1, 5, 10, 20) for layer in range(12)}
    for row in results:
        log = directory / row['mode'] / 'training.log'
        if not log.exists() or row['status'] not in ('completed', 'failed'):
            continue
        audit = json_rows(log.read_text(), 'ProtectionStrengthAudit')
        valid = (len(audit) == len(expected) and
                 {(r['task'], r['epoch'], r['layer']) for r in audit} == expected and
                 all(math.isfinite(r['alpha']) and 0 <= r['alpha'] <= 1 and
                     r['pre_conflict_reconstruction_error'] <= 1e-6 for r in audit))
        row['protection_audit_complete'] = valid
        if not valid:
            issues.append(dict(mode=row['mode'], field='protection_audit'))
        rows.extend(dict(r, mode=row['mode']) for r in audit)
    pairs = compare_complete(results, issues)
    for name, values in (('results', results), ('matching_issues', issues), ('pairs', pairs)):
        (directory / (name + '.json')).write_text(json.dumps(values, indent=2) + '\n')
        write_csv(directory / (name + '.csv'), values)
    write_csv(directory / 'protection_strengths.csv', rows)
    lines = ['# 3090 保护强度：seed1993', '',
        '| 配置 | 状态 | Average | Last | 阶段Old | 阶段New | Forgetting |',
        '|---|---|---:|---:|---:|---:|---:|']
    for row in results:
        lines.append('| ' + ' | '.join([row['mode'], row['status']] +
            [str(row.get(key, '')) for key in ('Average', 'Last', 'StageOld', 'StageNew', 'Forgetting')]) + ' |')
    lines += ['', '完整同提交配对差值：', json.dumps(pairs, ensure_ascii=False), '',
        '指纹与记录问题：' + json.dumps(issues, ensure_ascii=False), '',
        '全部为seed1993候选筛查。固定0.25/0.75和原NCM结果是历史参考；',
        '本轮新训练源码含保护强度接口，因此以本轮REF作严格参照。',
        'Average优先，同时报告Last、Old/New和遗忘；不自动替换正式基线。',
        'R_old来自当前训练图片的跨类混淆，不能称旧类遗忘真值。',
        '更新量规则仅限制S保护区内的更新；P硬权限和冲突门保持原配方。']
    (directory / 'protection_report.md').write_text('\n'.join(lines) + '\n')
