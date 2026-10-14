"""固定 B 的任务调度、简洁日志和可核对的运行记录。"""
from datetime import datetime, timezone
from importlib import metadata
import hashlib
import json
import logging
import os
from pathlib import Path
import platform
import random
import sys
import time

import numpy as np
import torch

from methods.dlora import Learner
from utils.data_manager import DataManager


ROOT = Path(__file__).resolve().parent
LOG_FORMAT = '%(asctime)s [%(filename)s] => %(message)s'
REFERENCE_REVISION = 'fa53a839d4a8df7edd28afdf230b577b548179b6'


def _write_json(path, payload):
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str) + '\n',
                    encoding='utf-8')


def _set_random(args):
    # 与参考 _train 保持相同的调用位置及随机初始化顺序。
    random.seed(args['seed'])
    np.random.seed(args['seed'])
    torch.manual_seed(args['seed'])
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args['seed'])
        torch.cuda.manual_seed_all(args['seed'])
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)


def _set_device(args):
    device = str(args['device'])
    args['device'] = [torch.device('cpu' if device in ('cpu', '-1')
                                 else device if device.startswith('cuda:')
                                 else 'cuda:{}'.format(device))]


def _metadata(args):
    layerwise_alpha = args.get('layerwise_alpha', False)
    source_hashes = {
        str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(ROOT.rglob('*'))
        if path.is_file() and path.suffix in ('.py', '.json', '.sh')
        and (len(path.relative_to(ROOT).parts) == 1 or path.relative_to(ROOT).parts[0]
             in {'methods', 'models', 'utils', 'exps', 'scripts', 'test'})
        and not {'runs', 'logs', '.git'} & set(path.relative_to(ROOT).parts)
        and '__pycache__' not in path.parts
    }
    packages = {}
    for name in ('torch', 'torchvision', 'timm', 'numpy', 'scipy', 'Pillow'):
        try:
            packages[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            packages[name] = None
    return {
        'started_at_utc': datetime.now(timezone.utc).isoformat(),
        'pid': os.getpid(), 'reference_revision': REFERENCE_REVISION,
        'python': platform.python_version(), 'packages': packages,
        'cuda': torch.version.cuda, 'devices': [str(d) for d in args['device']],
        'gpu_names': [torch.cuda.get_device_name(d) for d in args['device'] if d.type == 'cuda'],
        'argv': sys.argv, 'working_directory': str(Path.cwd()),
        'bundle_directory': str(ROOT), 'source_sha256': source_hashes,
        'effective_config': args,
        'recipe': {
            'name': ('calibrated-depth-coverage-layerwise-alpha' if layerwise_alpha
                     else 'frozen-debug-calibrated-depth-B'), 'S_rank': args['rank'], 'P_rank': args['rank'],
            'task0_p_init_rank': args['task0_p_init_rank'], 'svd_importance': 'full spectrum',
            'ca_drift_augmentation': args.get('ca_drift_augmentation', 'off'),
            'protect_alpha': ('0.50 + 0.25 * (1 - 2 * layer / 11)' if layerwise_alpha else 0.5),
            'uniform_reference_coverage': 0.9,
            'depth_offset': '0.05 * (1 - 2 * layer / 11)',
            'depth_center': 'calibrate once at Task1 to match uniform reference protected count',
            'conflict_score': 'magnitude', 'conflict_scope': 'each branch full QKV',
            'conflict_exact_top_fraction': 0.1, 'beta': 'min(0.5 * (1 + R_old), 1)',
            'mask_regularization': 0, 'task0_anchor': 2.5, 'sdpa': 'math',
        },
    }


def _result(tasks, planned_tasks, elapsed_seconds, status, acc_matrix=None):
    curve = [row['top1'] for row in tasks]
    last = tasks[-1] if tasks else {}
    incremental = tasks[1:]
    summary = {
        'Average': float(np.mean(curve)) if curve else None,
        'Last': last.get('top1'), 'Old': last.get('old'), 'New': last.get('new'),
        'StageOld': float(np.mean([row['old'] for row in incremental])) if incremental else None,
        'StageNew': float(np.mean([row['new'] for row in incremental])) if incremental else None,
        'Forgetting': last.get('forgetting'),
    }
    return {
        'status': status, 'updated_at_utc': datetime.now(timezone.utc).isoformat(),
        'tasks_completed': len(tasks), 'planned_tasks': planned_tasks,
        'elapsed_seconds': elapsed_seconds, 'tasks': tasks, 'summary': summary,
        'accuracy_matrix': acc_matrix.tolist() if acc_matrix is not None else None,
        'average_accuracy': summary['Average'], 'last_accuracy': summary['Last'],
        'final_old': summary['Old'], 'final_new': summary['New'],
        'stage_mean_old': summary['StageOld'], 'stage_mean_new': summary['StageNew'],
        'forgetting': summary['Forgetting'],
    }


def train(args):
    return _train(dict(args))


def _train(args):
    run_start_time = time.time()
    output = Path(args['output'])
    output.mkdir(parents=True, exist_ok=False)
    args['logdir'] = str(output)
    logging.basicConfig(level=logging.INFO, format=LOG_FORMAT, force=True,
                        handlers=[logging.FileHandler(output / 'train.log', encoding='utf-8'),
                                  logging.StreamHandler(sys.stdout)])
    _set_random(args)
    _set_device(args)
    # metadata读取GPU属性会初始化CUDA；之后才能重置显存统计。
    run_info = _metadata(args)
    if args['device'][0].type == 'cuda':
        torch.cuda.reset_peak_memory_stats(args['device'][0])
    _write_json(output / 'run_info.json', run_info)
    _write_json(output / 'effective_config.json', args)
    tasks = []
    max_tasks = args['max_tasks']
    _write_json(output / 'result.json', _result(tasks, max_tasks, 0., 'running'))
    logging.info('Start DualMask %s | seed=%s device=%s tasks=%s/10 epochs=%s CA=%s '
                 '| calibrated depth coverage alpha=%s S/P=64/64 | output=%s',
                 'layerwise-alpha' if args.get('layerwise_alpha', False) else 'B',
                 args['seed'], args['device'][0], max_tasks, args['epochs'], args['ca_epochs'],
                 '0.75->0.25' if args.get('layerwise_alpha', False) else '0.5', output)
    model = None
    try:
        data_manager = DataManager(args['dataset'], args['shuffle'], args['seed'], args['init_cls'], args['increment'], args)
        run_info['class_order'] = data_manager._class_order
        _write_json(output / 'run_info.json', run_info)
        model = Learner(args)
        for task_id in range(max_tasks):
            started = time.time()
            model.incremental_train(data_manager)
            train_seconds = time.time() - started
            started = time.time()
            cnn, with_task, nme, task_prediction = model.eval_task()
            # 此测试集遍历会推进 DataLoader 随机状态，不能只保留记录而删掉调用。
            w0_accuracy = model.eval_w0_task()
            eval_seconds = time.time() - started
            model.after_task()
            forgetting = backward = None
            if task_id > 0:
                diagonal = np.diag(model.acc_matrix)
                forgetting = float(np.mean((np.max(model.acc_matrix, axis=1)
                                            - model.acc_matrix[:, task_id])[:task_id]))
                backward = float(np.mean((model.acc_matrix[:, task_id] - diagonal)[:task_id]))
            row = {
                'task': task_id, 'top1': float(cnn['top1']),
                'old': float(cnn['grouped']['old']), 'new': float(cnn['grouped']['new']),
                'grouped': {key: float(value) for key, value in cnn['grouped'].items()},
                'with_task_top1': float(with_task['top1']),
                'nme_top1': float(nme['top1']) if nme is not None else None,
                'task_prediction_accuracy': float(task_prediction),
                'w0_top1': float(w0_accuracy) if w0_accuracy is not None else None,
                'forgetting': forgetting, 'backward': backward,
                'train_seconds': train_seconds, 'eval_seconds': eval_seconds,
                'ca_drift': model._ca_drift_audit,
                'controls': {
                    'coverage_center': model._depth_coverage_center,
                    'layer_coverage': [m.effective_energy_coverage for m in model._iter_lora_modules()],
                    'layer_protected_counts': [int(m.general_mask.sum()) for m in model._iter_lora_modules()],
                    'alpha': [m.effective_protect_strength for m in model._iter_lora_modules()],
                    'beta': [float(m._conflict_parameters()[1]) for m in model._iter_lora_modules()],
                    'C_new': model._w0_competence_new, 'C_all': model._w0_competence_all_seen,
                    'R_old': model._w0_old_overlap_risk,
                },
            }
            tasks.append(row)
            _write_json(output / 'result.json', _result(
                tasks, max_tasks, time.time() - run_start_time, 'running', model.acc_matrix))
            logging.info('CNN: %s', cnn['grouped'])
            logging.info('CNN top1 curve: %s', [task['top1'] for task in tasks])
            if forgetting is not None:
                logging.info('Forgetting: %.4f\tBackward: %.4f', forgetting, backward)
    except BaseException:
        _write_json(output / 'result.json', _result(
            tasks, max_tasks, time.time() - run_start_time, 'failed',
            model.acc_matrix if model is not None else None))
        raise
    result = _result(tasks, max_tasks, time.time() - run_start_time, 'completed', model.acc_matrix)
    result['gpu_peak_allocated_bytes'] = (torch.cuda.max_memory_allocated(args['device'][0])
                                          if args['device'][0].type == 'cuda' else None)
    _write_json(output / 'result.json', result)
    summary = result['summary']
    logging.info('Accuracy Matrix: \n %s', model.acc_matrix.T.round(2))
    logging.info('Average Accuracy: %s', summary['Average'])
    logging.info('Last Accuracy: %s', summary['Last'])
    logging.info('Total experiment time: %.2fs (%.2fmin)', result['elapsed_seconds'], result['elapsed_seconds'] / 60.)
    return result
