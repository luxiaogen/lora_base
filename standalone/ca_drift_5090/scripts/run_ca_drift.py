#!/usr/bin/env python3
"""5090 上先四组短测、再按固定顺序串行完整训练；失败保留现场。"""
import argparse
import csv
from datetime import datetime
import fcntl
import hashlib
import io
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time


BUNDLE = Path(__file__).resolve().parents[1]
MODES = {'baseline': 'A_baseline', 'full': 'B_full',
         'isotropic': 'C_isotropic', 'permuted': 'D_permuted'}
METRICS = ('Average', 'Last', 'Old', 'New', 'StageOld', 'StageNew', 'Forgetting')


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    temporary.replace(path)


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8'))


def sha(value):
    return hashlib.sha256(value).hexdigest()


def config_sha(config):
    return sha(json.dumps(config, sort_keys=True, separators=(',', ':')).encode())


def sources(root):
    # 只收集独立包源码，绝不把动态结果重新纳入自己的指纹。
    files = [BUNDLE / 'main.py', BUNDLE / 'trainer.py', *BUNDLE.glob('*.json')]
    for name in ('methods', 'models', 'utils', 'exps', 'scripts', 'test'):
        files.extend((BUNDLE / name).rglob('*'))
    return {str(p.relative_to(BUNDLE)): sha(p.read_bytes()) for p in sorted(set(files))
            if p.is_file() and p.suffix in ('.py', '.json', '.sh')
            and root not in p.parents and '__pycache__' not in p.parts}


def fingerprint(cli, root):
    commit = subprocess.run(['git', '-C', str(BUNDLE), 'rev-parse', 'HEAD'],
                            text=True, capture_output=True)
    return {'git_commit': commit.stdout.strip() if commit.returncode == 0 else None,
            'source_sha256': sources(root), 'config_sha256': sha(cli.config.read_bytes()),
            'entrypoint': str(cli.entrypoint), 'entrypoint_sha256': sha(cli.entrypoint.read_bytes())}


def normalized_config(config):
    return {k: v for k, v in config.items() if k not in ('output', 'logdir', 'device')}


def expected_config(base, cli, phase, mode):
    epochs, ca_epochs, tasks = (1, 1, 2) if phase == 'smoke' else (20, 5, 10)
    return dict(base, data_path=str(cli.data_path), seed=1993, device=['cuda:0'],
                init_epoch=epochs, epochs=epochs, ca_epochs=ca_epochs,
                max_tasks=tasks, ca_drift_augmentation=mode)


def verify_group(group, expected, snapshot):
    output = Path(group['output'])
    actual = read_json(output / 'effective_config.json')
    if normalized_config(actual) != normalized_config(expected):
        raise RuntimeError('{} effective configuration changed'.format(group['name']))
    if actual.get('device') not in ('0', 'cuda:0', ['cuda:0']):
        raise RuntimeError('{} was not run on CUDA device 0'.format(group['name']))
    if actual.get('output') != str(output):
        raise RuntimeError('{} output identity changed'.format(group['name']))
    info = read_json(output / 'run_info.json')
    recorded = info.get('source_sha256', {})
    if any(recorded.get(k) != v for k, v in snapshot['source_sha256'].items()):
        raise RuntimeError('{} training source SHA does not match queue'.format(group['name']))
    if not info.get('gpu_names') or any('5090' not in name for name in info['gpu_names']):
        raise RuntimeError('{} training metadata does not confirm RTX 5090'.format(group['name']))
    # 完成身份必须独立核对；只有此后纯指标汇总错误才可继续下一组。
    training = read_json(output / 'result.json')
    if not isinstance(training, dict) or training.get('status') != 'completed' \
            or training.get('tasks_completed') != expected['max_tasks'] \
            or training.get('planned_tasks') != expected['max_tasks']:
        raise RuntimeError('{} did not complete all planned tasks'.format(group['name']))
    tasks = training.get('tasks')
    if not isinstance(tasks, list) or len(tasks) != expected['max_tasks'] \
            or any(not isinstance(row, dict) for row in tasks) \
            or [row.get('task') for row in tasks] != list(range(expected['max_tasks'])):
        raise RuntimeError('{} completed task IDs do not match the protocol'.format(group['name']))
    return actual


def summarize_group(group, snapshot, actual):
    result = {'name': group['name'], 'mode': group['mode'], 'phase': group['phase'],
              'status': 'completed', 'returncode': 0, 'output': group['output'],
              'log_path': group['log_path'], 'training_result': str(Path(group['output']) / 'result.json'),
              'fingerprint': snapshot, 'config_sha256': config_sha(normalized_config(actual)),
              'effective_config': actual, 'analysis_errors': []}
    try:
        training = read_json(Path(group['output']) / 'result.json')
        if training.get('status') != 'completed' or training.get('tasks_completed') != actual['max_tasks']:
            raise ValueError('training result does not report all planned tasks completed')
        result.update({key: training['summary'][key] for key in METRICS})
        result['tasks'] = training['tasks']
        result['ca_drift'] = [row.get('ca_drift', {}) for row in training['tasks']]
        result['elapsed_seconds'] = training.get('elapsed_seconds')
    except (OSError, ValueError, KeyError, TypeError) as error:
        result['analysis_errors'].append(str(error))
    write_json(Path(group['directory']) / 'result.json', result)
    return result


def summarize_queue(root, queue):
    rows = []
    for group in queue['groups']:
        if group['phase'] != 'full' or group['status'] != 'completed':
            continue
        try:
            rows.append(read_json(Path(group['directory']) / 'result.json'))
        except (OSError, ValueError) as error:
            queue['analysis_errors'].append({'group': group['name'], 'error': str(error)})
    complete = len(rows) == 4 and not any(row.get('analysis_errors') for row in rows)
    summary = {'groups': rows, 'selected_modes': queue['modes'], 'all_four_complete': complete,
               'analysis_errors': queue['analysis_errors'],
               'interpretation': 'No attribution is made by this runner; missing groups are not evidence.'}
    write_json(root / 'summary.json', summary)
    buffer = io.StringIO()
    fields = ['name', 'mode', *METRICS, 'tasks', 'ca_drift', 'analysis_errors', 'log_path', 'training_result']
    writer = csv.DictWriter(buffer, fieldnames=fields, extrasaction='ignore')
    writer.writeheader()
    for row in rows:
        writer.writerow({key: json.dumps(value, ensure_ascii=False) if isinstance(value, (dict, list))
                         else value for key, value in row.items()})
    (root / 'summary.csv').write_text(buffer.getvalue(), encoding='utf-8')


def analyze_group(group, queue, actual):
    try:
        summary = summarize_group(group, queue['fingerprint'], actual)
        queue['analysis_errors'].extend({'group': group['name'], 'phase': group['phase'], 'error': error}
                                        for error in summary['analysis_errors'])
    except Exception as error:
        queue['analysis_errors'].append({'group': group['name'], 'phase': group['phase'], 'error': str(error)})


def pid_alive(pid):
    if not pid:
        return False
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, help='全新运行总目录；已有目录必须显式 --resume。')
    parser.add_argument('--resume', action='store_true', help='核对完成记录后继续；失败训练绝不自动重试。')
    parser.add_argument('--config', type=Path, default=BUNDLE / 'exps/dlora/imgr10.json')
    parser.add_argument('--data-path', type=Path, help='含 train/test 的 ImageNet-R 根目录。')
    parser.add_argument('--entrypoint', type=Path, default=BUNDLE / 'main.py', help='测试/诊断入口；默认独立包 main.py。')
    parser.add_argument('--modes', nargs='+', choices=list(MODES), default=None,
                        help='显式选组；仍按 baseline/full/isotropic/permuted 顺序运行。')
    parser.add_argument('--hours', type=float, default=None, help='首个 GPU 短测起算预算，默认10小时。')
    cli = parser.parse_args(argv)
    if cli.resume and cli.root is None:
        parser.error('--resume requires --root')
    if cli.hours is not None and cli.hours < 0:
        parser.error('--hours must be nonnegative')
    cli.root = (cli.root or BUNDLE / 'runs' / datetime.now().strftime('ca_drift_5090_%Y%m%d_%H%M%S_%f')).expanduser().resolve()
    cli.config = cli.config.expanduser().resolve()
    cli.entrypoint = cli.entrypoint.expanduser().resolve()
    return cli


def run(cli):
    root = cli.root
    base = read_json(cli.config)
    saved_queue = read_json(root / 'queue.json') if cli.resume else None
    stored_path = saved_queue['data_path'] if saved_queue else base['data_path']
    cli.data_path = (cli.data_path or Path(stored_path)).expanduser().resolve()
    for split in ('train', 'test'):
        if not (cli.data_path / split).is_dir() or not any((cli.data_path / split).iterdir()):
            raise RuntimeError('ImageNet-R requires nonempty train/test: {}'.format(cli.data_path))
    identity = {'dataset': 'ImageNet_R', 'init_cls': 20, 'increment': 20, 'total_sessions': 10, 'rank': 64}
    if any(base.get(key) != value for key, value in identity.items()):
        raise RuntimeError('configuration must use ImageNet-R T10 rank64 base')
    modes = [mode for mode in MODES if cli.modes is None or mode in cli.modes]
    snapshot = fingerprint(cli, root)
    if cli.resume:
        queue = saved_queue
        active = read_json(root / 'active.json') if (root / 'active.json').exists() else {}
        if any(pid_alive(active.get(key)) for key in ('queue_pid', 'child_pid')):
            raise RuntimeError('queue or training child is still alive; refusing a second runner')
        if snapshot != queue['fingerprint'] or str(cli.data_path) != queue['data_path']:
            raise RuntimeError('source/config/entrypoint/data fingerprint changed; resume refused')
        if read_json(root / 'base_config.json') != base:
            raise RuntimeError('frozen base configuration changed; resume refused')
        if cli.modes is not None and modes != queue['modes']:
            raise RuntimeError('--modes differs from original queue')
        modes = queue['modes']
        if cli.hours is not None:
            queue['hours'] = cli.hours
    else:
        queue = {'status': 'ready', 'queue_pid': os.getpid(), 'modes': modes,
                 'hours': 10. if cli.hours is None else cli.hours, 'first_gpu_started_at': None,
                 'data_path': str(cli.data_path), 'fingerprint': snapshot, 'analysis_errors': [], 'groups': []}
        for phase in ('smoke', 'full'):
            for mode in modes:
                directory = root / phase / MODES[mode]
                queue['groups'].append({'name': MODES[mode], 'mode': mode, 'phase': phase,
                    'status': 'pending', 'directory': str(directory),
                    'output': str(directory / 'training'), 'log_path': str(directory / 'console.log')})
        write_json(root / 'base_config.json', base)
    queue['queue_pid'] = os.getpid()
    queue['status'] = 'running'
    write_json(root / 'queue.json', queue)
    active = {'queue_pid': os.getpid(), 'child_pid': None, 'status': 'running'}
    write_json(root / 'active.json', active)
    child = None
    try:
        # 恢复时先验证全部完成组，防止在旧结果被改动后启动任何新训练。
        for group in queue['groups']:
            if group['status'] == 'completed':
                actual = verify_group(group, expected_config(base, cli, group['phase'], group['mode']), snapshot)
                try:
                    read_json(Path(group['directory']) / 'result.json')
                except (OSError, ValueError) as error:
                    queue['analysis_errors'].append({'group': group['name'], 'phase': group['phase'],
                                                     'error': 'rebuilding summary: {}'.format(error)})
                analyze_group(group, queue, actual)
            elif group['status'] != 'pending':
                raise RuntimeError('{} {} previously {}; no automatic retry'.format(
                    group['phase'], group['name'], group['status']))
        gpu_id = os.environ.get('CUDA_VISIBLE_DEVICES', '0').split(',')[0]
        gpu = subprocess.run(['nvidia-smi', '-i', gpu_id, '--query-gpu=name', '--format=csv,noheader'],
                             text=True, capture_output=True, check=True)
        if '5090' not in gpu.stdout or len(gpu.stdout.strip().splitlines()) != 1:
            raise RuntimeError('selected CUDA device must be a single RTX 5090: {}'.format(gpu.stdout.strip()))
        queue['gpu_name'] = gpu.stdout.strip()
        for group in queue['groups']:
            if group['status'] == 'completed':
                continue
            if group['phase'] == 'full' and time.time() - queue['first_gpu_started_at'] >= queue['hours'] * 3600:
                queue['status'] = 'budget_exhausted'
                break
            if fingerprint(cli, root) != snapshot:
                raise RuntimeError('source/config changed during queue; refusing next group')
            directory = Path(group['directory'])
            directory.mkdir(parents=True, exist_ok=False)
            expected = expected_config(base, cli, group['phase'], group['mode'])
            command = [sys.executable, str(cli.entrypoint), '--config', str(root / 'base_config.json'),
                '--data-path', str(cli.data_path), '--device', '0', '--seed', '1993',
                '--epochs', str(expected['epochs']), '--ca-epochs', str(expected['ca_epochs']),
                '--max-tasks', str(expected['max_tasks']), '--set',
                'ca_drift_augmentation=' + json.dumps(group['mode']), '--output', group['output']]
            group.update(status='running', command=command, started_at=time.time(),
                         config_sha256=config_sha(normalized_config(expected)))
            if queue['first_gpu_started_at'] is None:
                queue['first_gpu_started_at'] = time.time()
            write_json(root / 'queue.json', queue)
            print('Starting {} {} -> {}'.format(group['phase'], group['name'], group['log_path']), flush=True)
            with Path(group['log_path']).open('w', encoding='utf-8') as log:
                child = subprocess.Popen(command, cwd=str(BUNDLE), stdout=log, stderr=subprocess.STDOUT)
                active = dict(queue_pid=os.getpid(), child_pid=child.pid, status='running',
                              phase=group['phase'], group=group['name'], command=command,
                              started_at=group['started_at'], log_path=group['log_path'])
                write_json(root / 'active.json', active)
                returncode = child.wait()
            group.update(returncode=returncode, finished_at=time.time())
            active.update(child_pid=None, status='completed' if returncode == 0 else 'failed')
            write_json(root / 'active.json', active)
            if returncode:
                group['status'] = 'failed'
                write_json(directory / 'result.json', dict(group, fingerprint=snapshot))
                raise RuntimeError('{} {} failed with exit {}; queue paused'.format(
                    group['phase'], group['name'], returncode))
            if fingerprint(cli, root) != snapshot:
                group['status'] = 'verification_failed'
                raise RuntimeError('source changed while training; completion cannot be trusted')
            try:
                actual = verify_group(group, expected, snapshot)
            except (OSError, ValueError, RuntimeError) as error:
                group['status'] = 'verification_failed'
                raise RuntimeError('completion verification failed: {}'.format(error)) from error
            group['status'] = 'completed'
            analyze_group(group, queue, actual)
            try:
                summarize_queue(root, queue)
            except (OSError, ValueError, TypeError) as error:
                queue['analysis_errors'].append({'group': group['name'], 'error': str(error)})
            write_json(root / 'queue.json', queue)
        else:
            queue['status'] = 'completed'
        return 0
    except (Exception, KeyboardInterrupt) as error:
        if child is not None and child.poll() is None:
            child.terminate()
            child.wait()
        for group in queue['groups']:
            if group['status'] == 'running':
                group['status'] = 'interrupted'
        queue.update(status='paused', error=str(error))
        print(str(error), file=sys.stderr, flush=True)
        return 1
    finally:
        active.update(child_pid=None, status=queue['status'])
        write_json(root / 'active.json', active)
        try:
            summarize_queue(root, queue)
        except (OSError, ValueError, TypeError) as error:
            queue['analysis_errors'].append({'error': str(error)})
        write_json(root / 'queue.json', queue)


def main(argv=None):
    cli = parse_args(argv)
    # SIGTERM 与 Ctrl+C 都落盘当前状态；恢复不自动重试中断的训练。
    signal.signal(signal.SIGTERM, lambda signum, frame: (_ for _ in ()).throw(KeyboardInterrupt('SIGTERM')))
    try:
        if not cli.resume:
            cli.root.mkdir(parents=True, exist_ok=False)
        with (cli.root / 'queue.lock').open('a') as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as error:
                raise RuntimeError('another runner owns this queue; no state was changed') from error
            return run(cli)
    except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as error:
        print(str(error), file=sys.stderr)
        return 1


if __name__ == '__main__':
    sys.exit(main())
