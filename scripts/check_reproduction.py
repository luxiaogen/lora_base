"""校验完整运行记录；准确率差异只报告，不据此重跑或挑选实验。"""
import argparse
from collections import Counter
import json
import math
import os
from pathlib import Path
import re
import sys


ROOT = Path(__file__).resolve().parents[1]
EXPECTED = {key: value for key, value in json.loads((ROOT / 'exps/dlora/imgr10.json').read_text()).items()
            if key not in ('device', 'data_path', 'init_epoch', 'epochs', 'ca_epochs', 'max_tasks')}


def check_record(directory, tasks, epochs):
    result = json.loads((directory / 'result.json').read_text())
    config = json.loads((directory / 'effective_config.json').read_text())
    info = json.loads((directory / 'run_info.json').read_text())
    rows = result['tasks']
    if (result['status'] != 'completed' or result['tasks_completed'] != tasks
            or result['planned_tasks'] != tasks or [row['task'] for row in rows] != list(range(tasks))):
        raise ValueError('运行未完整成功，不启动下一阶段')
    for key, expected in EXPECTED.items():
        if config.get(key) != expected:
            raise ValueError('配方不匹配：{}={}，应为{}'.format(key, config.get(key), expected))
    if config['init_epoch'] != epochs or config['epochs'] != epochs or config['max_tasks'] != tasks:
        raise ValueError('训练轮数或任务数不匹配')
    if config['ca_epochs'] != (1 if epochs == 1 else 5):
        raise ValueError('CA轮数不匹配')
    if not info.get('source_revision') or info.get('source_dirty') is not False:
        raise ValueError('源码不是已提交的干净版本')
    if os.environ.get('EXPECTED_REVISION') and info['source_revision'] != os.environ['EXPECTED_REVISION']:
        raise ValueError('实际训练源码与指定提交不匹配')
    center = rows[1]['controls']['coverage_center']
    if not math.isfinite(center) or not .05 <= center <= .95:
        raise ValueError('分层覆盖中心无效')
    for row in rows:
        for key in ('top1', 'old', 'new'):
            if not math.isfinite(row[key]) or not 0 <= row[key] <= 100:
                raise ValueError('存在无效准确率')
        controls = row['controls']
        task = row['task']
        if (controls['S_rank'] != [64] * 12 or controls['P_init_rank'] != [39 if task == 0 else 64] * 12
                or controls['P_active'] != (task > 0) or controls['protection_active'] != (task > 0)
                or controls['alpha'] != [.5] * 12):
            raise ValueError('实际秩、权限或保护强度不匹配')
        beta = min(.5 * (1. + controls['R_old']), 1.)
        if len(controls['beta']) != 12 or not all(math.isclose(x, beta, abs_tol=1e-12) for x in controls['beta']):
            raise ValueError('实际beta不符合R_old规则')
        expected_coverage = [.9] * 12 if task == 0 else [center + .05 * (1. - 2. * i / 11.) for i in range(12)]
        if (controls['coverage_center'] != (None if task == 0 else center)
                or len(controls['layer_coverage']) != 12
                or not all(math.isclose(x, y, abs_tol=1e-12) for x, y in zip(controls['layer_coverage'], expected_coverage))):
            raise ValueError('校准中心未保持或分层覆盖公式不匹配')
    for value in result['summary'].values():
        if value is not None and not math.isfinite(value):
            raise ValueError('存在非有限汇总指标')
    curve = [row['top1'] for row in rows]
    if not math.isclose(result['summary']['Average'], sum(curve) / tasks, abs_tol=1e-10):
        raise ValueError('Average与逐任务结果不一致')
    if result['summary']['Last'] != curve[-1]:
        raise ValueError('Last与最后阶段不一致')
    log = (directory / 'train.log').read_text()
    if re.search(r'Traceback|out of memory|CUDA error|\b(?:nan|inf)\b', log, re.I):
        raise ValueError('日志含训练异常')
    records = re.findall(r'\] => Task (\d+), Epoch (\d+)/(\d+) => Loss (\S+), CE (\S+), '
                         r'anchor_reg (\S+), anchor_reg_weighted (\S+),', log)
    if len(records) != tasks * epochs or Counter(int(row[0]) for row in records) != Counter({t: epochs for t in range(tasks)}):
        raise ValueError('缺失训练轮记录')
    for task in range(tasks):
        selected = [row for row in records if int(row[0]) == task]
        if [int(row[1]) for row in selected] != list(range(1, epochs + 1)) or any(int(row[2]) != epochs for row in selected):
            raise ValueError('训练轮顺序错误')
        for row in selected:
            if not all(math.isfinite(float(value)) for value in row[3:]):
                raise ValueError('损失分项非有限')
    ca = re.findall(r'\] => CA Task (\d+), Epoch (\d+)/(\d+) => Loss (\S+)', log)
    expected_ca = [(task, epoch, config['ca_epochs']) for task in range(1, tasks)
                   for epoch in range(1, config['ca_epochs'] + 1)]
    if ([tuple(map(int, row[:3])) for row in ca] != expected_ca
            or any(not math.isfinite(float(row[3])) for row in ca)):
        raise ValueError('CA记录不完整或损失非有限')
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('directory', type=Path)
    parser.add_argument('--tasks', type=int, choices=(2, 10), default=10)
    parser.add_argument('--epochs', type=int, choices=(1, 20), default=20)
    args = parser.parse_args()
    try:
        result = check_record(args.directory, args.tasks, args.epochs)
    except (OSError, ValueError, KeyError, TypeError) as error:
        print('验证失败：{}'.format(error), file=sys.stderr)
        return 1
    print('健康检查通过：{}任务，每任务{}轮，源码{}'.format(args.tasks, args.epochs,
          json.loads((args.directory / 'run_info.json').read_text())['source_revision']))
    if args.tasks == 10:
        reference = json.loads((ROOT / 'test/fixtures/old_b_seed1993.json').read_text())
        differences = {key: result['summary'][key] - value for key, value in reference['summary'].items()}
        curve = [row['top1'] for row in result['tasks']]
        comparison = dict(reference=reference['reference_revision'], actual=result['summary'],
                          differences=differences, curve_differences=[a - b for a, b in zip(curve, reference['curve'])],
                          coverage_center=result['tasks'][1]['controls']['coverage_center'],
                          coverage_center_difference=result['tasks'][1]['controls']['coverage_center'] - reference['coverage_center'],
                          identical_reported_curve=curve == reference['curve'],
                          identical_reported_metrics=all(abs(x) < 1e-9 for x in differences.values()))
        (args.directory / 'comparison.json').write_text(json.dumps(comparison, indent=2) + '\n')
        print(json.dumps(comparison, ensure_ascii=False, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
