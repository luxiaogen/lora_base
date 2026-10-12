"""最终 F 九组确认：缺失或不匹配的证据不进入组件归因。"""
import argparse
import json
import math
from pathlib import Path
import re

from analyze_core_evidence import METRICS, read_run
from analyze_protect_position import json_rows, write_csv
import run_final_f_validation as runner


def factor_effects(cells):
    if not all(name in cells for name in ('C11', 'C01', 'C10', 'C00')):
        return {}
    result = {}
    for key in METRICS:
        a, b, c, d = [cells[name][key] for name in ('C11', 'C01', 'C10', 'C00')]
        result[key] = dict(permission_effect=((a + c) - (b + d)) / 2,
                           gate_effect=((a + b) - (c + d)) / 2,
                           interaction=a - b - c + d)
    return result


def update_evidence_issues(updates, budgets, config):
    issues = []
    expected_updates = {(t, l, b, p) for t in range(10) for l in range(12)
                        for b in (('S',) if t == 0 else ('S', 'P')) for p in ('Q', 'K', 'V')}
    if (len(updates) != len(expected_updates) or
            {(r['task'], r['layer'], r['branch'], r['projection']) for r in updates} != expected_updates):
        issues.append('merge_update_records')
    for r in updates:
        expected = dict(position=config['dual_mask_protect_position'] if r['task'] else 'wpre',
            permission_mode=config['dual_mask_permission_mode'], score_mode='magnitude',
            position_norm_match=config['dual_mask_position_norm_match'] if r['task'] else 'off',
            merge_error_definition='same_factor_cached_gate_arithmetic')
        if (any(r.get(k) != v for k, v in expected.items())
                or not math.isfinite(r['merge_error']) or r['merge_error'] > 1e-5):
            issues.append('merge_update_policy')
            break
    budgets = [r for r in budgets if r['task'] > 0]
    expected_budgets = {(t, l, b) for t in range(1, 10) for l in range(12) for b in ('S', 'P')}
    if (len(budgets) != len(expected_budgets) or
            {(r['task'], r['layer'], r['branch']) for r in budgets} != expected_budgets):
        issues.append('delivered_gate_records')
    definition = ('post_permission_conflict_and_norm_control'
                  if config['dual_mask_position_norm_match'] != 'off' else 'conflict_only')
    reference_k = int(3 * config['embd_dim'] ** 2 * config['dual_mask_conflict_ratio'])
    for r in budgets:
        enabled = config['dual_mask_s_conflict_enabled' if r['branch'] == 'S' else 'dual_mask_p_conflict_enabled']
        if (r.get('gate_enabled') != enabled or r.get('granularity') != 'layer'
                or r.get('reference_k') != reference_k
                or r.get('applied_k') != (reference_k if enabled else 0)
                or r.get('removed_norm_definition') != definition
                or r.get('merge_error_definition') != 'same_factor_cached_gate_arithmetic'
                or not math.isfinite(r['merge_error']) or r['merge_error'] > 1e-5):
            issues.append('delivered_gate_policy')
            break
    return issues


def epoch_selection_issues(epochs, config):
    totals = {}
    for row in epochs:
        key = (row['task'], row['epoch'], row['layer'], row['branch'])
        totals[key] = totals.get(key, 0) + row['selected_coordinates']
    reference_k = int(3 * config['embd_dim'] ** 2 * config['dual_mask_conflict_ratio'])
    for (task, _, _, branch), count in totals.items():
        enabled = bool(task) and config['dual_mask_s_conflict_enabled'
            if branch == 'S' else 'dual_mask_p_conflict_enabled']
        if count != (reference_k if enabled else 0):
            return ['epoch_selected_count']
    return []


def pairs_for(results, issues, machine):
    bad = {row['mode'] for row in issues}
    complete = {runner.identity(machine, row['mode'])[1]['name']: row for row in results
                if row.get('valid_performance') and row.get('evidence_complete') and row['mode'] not in bad}
    wanted = [('C11', 'C01'), ('C11', 'C10'), ('C11', 'C00'), ('Soft', 'C11'), ('Bfixed', 'C11')]
    if machine == '5090':
        wanted = [('Lspectrum', 'Labs'), ('Lspectrum', 'Lperm'), ('Labs', 'Lperm')]
    pairs = [dict(first=complete[a]['mode'], second=complete[b]['mode'],
                  **{k: complete[a][k] - complete[b][k] for k in METRICS})
             for a, b in wanted if a in complete and b in complete]
    return pairs, factor_effects(complete) if machine == '3090' else {}


def summarize_saved(directory):
    manifest = json.loads((directory / 'manifest.json').read_text())
    machine = manifest['machine']
    path = directory / 'queue.json'
    records = json.loads(path.read_text()) if path.exists() else []
    results, issues, snapshots = [], [], {}
    rows = {k: [] for k in ('tasks', 'epochs', 'updates', 'masks', 'diagnostics', 'costs', 'storage', 'norm_matches', 'gate_budgets')}
    for record in records:
        name = record['mode']
        if record['status'] not in ('completed', 'failed'):
            results.append(dict(record, valid_performance=False, evidence_complete=False))
            continue
        def problem(field):
            issues.append(dict(mode=name, field=field))
        try:
            measured, snapshot, telemetry = read_run(directory, record)
            content = (directory / name / 'training.log').read_text()
        except (OSError, ValueError, KeyError, TypeError, SyntaxError) as error:
            problem('unreadable_evidence')
            results.append(dict(record, valid_performance=False, evidence_complete=False, error=str(error)))
            continue
        expected = runner.effective_config(machine, name)
        for k in expected.keys() | snapshot['effective_config'].keys():
            if k != 'prefix' and expected.get(k) != snapshot['effective_config'].get(k):
                problem('config.' + k)
        for k, v in (('machine', machine), ('phase', 'formal'), ('code_revision', manifest['revision'])):
            if snapshot.get(k) != v:
                problem(k)
        if not runner.REQUIRED_SOURCE <= snapshot.get('source_sha256', {}).keys():
            problem('source_fingerprint')
        reference = next(iter(snapshots.values()), None)
        for k in ('source_sha256', 'software', 'hardware'):
            if not snapshot.get(k) or reference is not None and snapshot[k] != reference[k]:
                problem(k)
        snapshots[name] = snapshot
        learned = {(int(t), int(e)) for t, e in re.findall(r'LoRA learning rates: task=(\d+), epoch=(\d+)', content)}
        measured['training_epochs_complete'] = learned == {(t, e) for t in range(10) for e in range(1, 21)}
        measured['valid_performance'] &= measured['training_epochs_complete']
        if not measured['training_epochs_complete']:
            problem('training_epochs')
        if not measured['epoch_records_complete']:
            problem('epoch_update_records')
        for field in epoch_selection_issues(telemetry['epochs'], expected):
            problem(field)
        for r in telemetry['epochs']:
            wanted = dict(permission_mode=expected['dual_mask_permission_mode'],
                position=expected['dual_mask_protect_position'] if r['task'] else 'wpre',
                position_norm_match=expected['dual_mask_position_norm_match'] if r['task'] else 'off')
            enabled = bool(r['task']) and expected['dual_mask_s_conflict_enabled'
                if r['branch'] == 'S' else 'dual_mask_p_conflict_enabled']
            if (any(r.get(k) != v for k, v in wanted.items())
                    or not enabled and r['selected_coordinates'] != 0
                    or r['effective_norm'] > r['raw_norm'] + 1e-5):
                problem('epoch_update_policy')
                break
            if r['task'] and expected['dual_mask_position_norm_match'] == 'prior_layer_min' and (
                    r.get('conflict_removed_norm') is not None or r.get('post_permission_removed_norm') is None):
                problem('epoch_removed_definition')
                break
        gate_budgets = [dict(r, mode=name) for r in json_rows(content, 'AppliedConflictBudget')]
        for field in update_evidence_issues(telemetry['updates'], gate_budgets, expected):
            problem(field)
        expected_masks = {(t, l) for t in range(10) for l in range(12)}
        masks = telemetry['masks']
        if len(masks) != 120 or {(r['task'], r['layer']) for r in masks} != expected_masks:
            problem('mask_records')
        if machine == '5090' and any(r['qkv_protect_counts'] != r['qkv_reference_counts'] for r in masks):
            problem('protection_count')
        for r in masks:
            if (r['seed'] != 1993 or r['position'] != (expected['dual_mask_protect_position'] if r['task'] else 'wpre')
                    or r['task'] and (r['private_rank'] != 64 or r['protect_strength'] != expected['dual_mask_fixed_protect_strength'])
                    or len(r['qkv_protect_counts']) != 3
                    or any(not 0 <= x <= expected['embd_dim'] ** 2 for x in r['qkv_protect_counts'])):
                problem('mask_policy')
                break
        norm_matches = [dict(r, mode=name) for r in json_rows(content, 'PriorLayerNormMatch')]
        expected_norm = {(t, e, l, b) for t in range(1, 10) for e in range(1, 21)
                         for l in range(12) for b in ('S', 'P')}
        if machine == '5090':
            if (len(norm_matches) != len(expected_norm) or
                    {(r['task'], r['epoch'], r['layer'], r['branch']) for r in norm_matches} != expected_norm):
                problem('norm_match_records')
            for r in norm_matches:
                if (not 0 <= r['actual_scale'] <= 1 or r['matching_residual'] > 1e-5
                        or not all(math.isfinite(x) for x in r['candidate_norms']
                                   + [r['target_norm'], r['actual_norm'], r['matching_residual']])
                        or abs(r['target_norm'] - min(r['candidate_norms'])) > 1e-5
                        or abs(r['actual_norm'] - r['target_norm']) > 1e-5):
                    problem('norm_match_residual')
        diagnostics = telemetry['diagnostics']
        expected_diagnostics = {(t, e, p) for t in (1, 5, 9) for e in (1, 5, 10, 20) for p in ('old', 'new')}
        norm_label = 'same_state_prior_layer_min' if machine == '5090' else 'same_state_paired_min'
        if (len(diagnostics) != 24 or {(r['task'], r['epoch'], r['partition']) for r in diagnostics} != expected_diagnostics
                or any(r['scope'] != 'all_seen' or r['norm_control'] != norm_label
                       or not 0 < r['count'] <= (512 if r['partition'] == 'old' else 128) for r in diagnostics)):
            problem('position_diagnostics')
        for key in ('epochs', 'updates', 'diagnostics', 'costs', 'storage'):
            if any(any(isinstance(v, float) and not math.isfinite(v) for v in r.values()) for r in telemetry[key]):
                problem('nonfinite_' + key)
        if not telemetry['costs'] or not telemetry['storage']:
            problem('missing_cost_or_storage')
        measured.update(seed=1993, std=None, complete_three_seeds=False,
            code_revision=snapshot['code_revision'], evidence_complete=not any(r['mode'] == name for r in issues),
            peak_allocated_bytes=max((r['cuda_peak_allocated_bytes'] for r in telemetry['costs']), default=None))
        results.append(measured)
        for k in telemetry:
            rows[k].extend(telemetry[k])
        rows['norm_matches'].extend(norm_matches)
        rows['gate_budgets'].extend(gate_budgets)
    if len({r['Task0'] for r in results if r.get('valid_performance')}) > 1:
        issues.extend(dict(mode=r['mode'], field='Task0_startpoint') for r in results if r.get('valid_performance'))
    if machine == '5090':
        counts = {r['mode']: {(m['task'], m['layer']): m['qkv_reference_counts'] for m in rows['masks']
                            if m['mode'] == r['mode']} for r in results if r.get('valid_performance')}
        reference = next(iter(counts.values()), None)
        for name, value in counts.items():
            if value != reference:
                issues.append(dict(mode=name, field='cross_run_protection_count'))
    pairs, effects = pairs_for(results, issues, machine)
    for name, values in dict(results=results, matching_issues=issues, pairs=pairs, effects=effects, **rows).items():
        (directory / (name + '.json')).write_text(json.dumps(values, indent=2) + '\n')
        if isinstance(values, list):
            if values:
                write_csv(directory / (name + '.csv'), values)
            else:
                (directory / (name + '.csv')).write_text('')
    report = ['# ' + machine + ' 最终 F 单 seed 确认', '',
        '| 组 | 状态 | Average | Last | Old | New | 阶段Old | 阶段New | Forgetting |',
        '|---|---|---:|---:|---:|---:|---:|---:|---:|']
    for row in results:
        report.append('| ' + ' | '.join([row['mode'], row['status']] + [str(row.get(k, ''))
            for k in METRICS]) + ' |')
    report += ['', '已完成且匹配的配对：' + json.dumps(pairs, ensure_ascii=False),
               '2×2描述性效应：' + json.dumps(effects, ensure_ascii=False),
               '证据问题：' + json.dumps(issues, ensure_ascii=False), '',
               '只用seed1993，不报告跨seed标准差或等效性。两台结果不跨机器归因。',
               '位置匹配同状态每层每分支完整QKV的共同最小范数；独立训练轨迹仍可能不同。',
               '同状态切换保留训练适应偏向，P硬权限尤其如此；不是位置精准性的独立证明。',
               '关闭权限仍保留保护图/NCM流程；固定β仍执行NCM，不声称已减少该成本。',
               '未启动项保留待完成；不自动更换基线。']
    (directory / 'report.md').write_text('\n'.join(report) + '\n')
    valid = [r for r in results if r.get('valid_performance')]
    if valid:
        from analyze_core_evidence import draw
        draw(directory, {r['mode']: r for r in valid}, rows)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    summarize_saved(parser.parse_args().directory)
