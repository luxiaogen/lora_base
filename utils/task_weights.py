"""Post-merge/post-CA model weights for evaluation, not training resumption."""
from datetime import datetime
import json
import logging
from pathlib import Path

import torch


def checkpoint_directory(args):
    # Each invocation gets its own folder, even when a manual prefix is reused.
    return Path(args['logdir']) / 'checkpoints' / datetime.now().strftime('%Y%m%d_%H%M%S_%f')


def save_task_weights(directory, learner, data_manager, args, task_results):
    network = learner._network
    if isinstance(network, torch.nn.DataParallel):
        network = network.module
    # Copy tensors only; never move the live model or change its mode/grad flags.
    state = network.state_dict()
    for name, value in state.items():
        state[name] = value.detach().cpu().clone()
    results = json.loads(json.dumps(task_results, default=float))
    payload = {
        'format_version': 1,
        'stage': 'post_merge_post_ca_evaluated',
        'task': int(learner._cur_task),
        'numtask': int(network.numtask),
        'known_classes': int(learner._known_classes),
        'total_classes': int(learner._total_classes),
        'class_order': [int(v) for v in data_manager._class_order],
        'increments': [int(v) for v in data_manager._increments],
        'config': json.loads(json.dumps(args, default=str)),
        'tasks': results,
        'average_accuracy': sum(row['top1'] for row in results) / len(results),
        'last_accuracy': results[-1]['top1'],
        'model_state_dict': state,
    }
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f'task_{learner._cur_task:02d}.pt'
    temporary = path.with_suffix('.pt.tmp')
    torch.save(payload, temporary)
    temporary.replace(path)
    logging.info('Saved task weights: %s (task=%s, average=%.6f, last=%.6f)',
                 path.resolve(), learner._cur_task, payload['average_accuracy'], payload['last_accuracy'])
    return path


def load_task_weights(network, path):
    """Load into a fresh matching network with merged (empty) LoRA slots.

    Class order/config are returned for the caller's evaluation data pipeline.
    This does not restore optimizers, RNG, prototypes or a continual learner.
    """
    payload = torch.load(path, map_location='cpu', weights_only=True)
    if isinstance(network, torch.nn.DataParallel):
        network = network.module
    network.load_state_dict(payload['model_state_dict'], strict=True)
    network.numtask = payload['numtask']
    network.eval()
    return payload
