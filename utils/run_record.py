"""Small, best-effort run records; no experiment validation or scheduling."""
from datetime import datetime, timezone
import json
import logging
import os
from pathlib import Path
import platform
import subprocess
import sys

import torch


def _git(*args):
    try:
        result = subprocess.run(['git', *args], cwd=Path(__file__).resolve().parents[1],
                                capture_output=True, text=True, timeout=5)
        return result.stdout.strip() if result.returncode == 0 else None
    except (OSError, subprocess.TimeoutExpired):
        return None


def _write(path, payload):
    try:
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str) + '\n', encoding='utf-8')
    except OSError as exc:
        logging.warning('Run record not saved: %s (%s)', path, exc)


def start_run_record(args, train_log):
    now = datetime.now(timezone.utc)
    directory = Path(args['logdir']) / 'records' / '{}_seed{}_pid{}'.format(
        now.strftime('%Y%m%dT%H%M%S.%fZ'), args['seed'], os.getpid())
    try:
        directory.mkdir(parents=True, exist_ok=False)
    except OSError as exc:
        logging.warning('Run record directory unavailable: %s', exc)
        return None
    _write(directory / 'effective_config.json', args)
    devices = [str(d) for d in args['device']]
    _write(directory / 'run_info.json', {
        'started_at_utc': now.isoformat(), 'pid': os.getpid(),
        'git_commit': _git('rev-parse', 'HEAD'),
        'git_branch': _git('branch', '--show-current'),
        'git_status': _git('status', '--porcelain', '--untracked-files=normal'),
        'python': platform.python_version(), 'torch': str(torch.__version__),
        'cuda': torch.version.cuda, 'devices': devices,
        'gpu_names': [torch.cuda.get_device_name(torch.device(d)) for d in devices if d.startswith('cuda')],
        'argv': sys.argv, 'working_directory': str(Path.cwd()),
        'train_log': str(Path(train_log).resolve()),
        'pretrained_sha256': None, 'dataset_split_sha256': None,
    })
    update_run_record(directory, [], args.get('max_tasks', args['total_sessions']), 0.)
    logging.info('Run records: %s', directory)
    return directory


def update_run_record(directory, tasks, planned_tasks, elapsed_seconds, completed=False):
    if directory is None:
        return
    curve = [row['top1'] for row in tasks]
    final = tasks[-1] if tasks else {}
    incremental = [row for row in tasks if row['task'] > 0]
    means = {}
    for key in ('old', 'new'):
        values = [row[key] for row in incremental if row.get(key) is not None]
        means[key] = sum(values) / len(values) if values else None
    _write(directory / 'result.json', {
        'status': 'completed' if completed else 'running',
        'updated_at_utc': datetime.now(timezone.utc).isoformat(),
        'tasks_completed': len(tasks), 'planned_tasks': planned_tasks,
        'elapsed_seconds': elapsed_seconds, 'tasks': tasks,
        'average_accuracy': sum(curve) / len(curve) if curve else None,
        'last_accuracy': curve[-1] if curve else None,
        'final_old': final.get('old'), 'final_new': final.get('new'),
        'stage_mean_old': means['old'], 'stage_mean_new': means['new'],
        'forgetting': final.get('forgetting'),
    })
