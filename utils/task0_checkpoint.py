"""Same-checkout, same-device Task0 snapshots for paired experiments.

These contain a pickled learner, including non-buffer DualMask state. Load only
checkpoints you generated yourself, not downloaded/untrusted files.
"""
import copy
import hashlib
import json
import logging
from pathlib import Path
import random

import numpy as np
import torch
from torch.utils.data import DataLoader


def dataset_signature(data_manager):
    # Fingerprint the loaded split, labels and order, not image file contents.
    digest = hashlib.sha256()
    for name in ('_class_order', '_train_data', '_train_targets', '_test_data', '_test_targets'):
        value = np.asarray(getattr(data_manager, name))
        digest.update(json.dumps([name, list(value.shape), str(value.dtype)]).encode())
        if value.dtype.kind in 'OUS':
            digest.update(json.dumps(value.tolist(), ensure_ascii=True).encode())
        else:
            digest.update(value.tobytes())
    return digest.hexdigest()


def _protocol(args):
    ignored = {'config', 'overrides', 'prefix', 'logdir', 'experiment_tracker',
               'classification_training_mode', 'dual_mask_task_bias_calibration',
               'dual_mask_previous_function_enabled', 'dual_mask_boundary_calibration',
               'dual_mask_boundary_real_current',
               'task0_checkpoint_save', 'task0_checkpoint_resume', 'max_tasks'}
    return {key: str(value) for key, value in args.items() if key not in ignored and not key.startswith('wandb_')}


def _rng_state():
    return {'python': random.getstate(), 'numpy': np.random.get_state(), 'torch': torch.get_rng_state(),
            'cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}


def save_task0_checkpoint(path, model, curves, signature):
    if model._cur_task != 0 or model._known_classes != model._total_classes:
        raise ValueError('Save a Task0 snapshot only after evaluation and after_task().')
    # Do not pickle dataset objects/workers; incremental_train rebuilds loaders.
    snapshot = copy.copy(model)
    snapshot.__dict__ = {key: value for key, value in vars(model).items() if not isinstance(value, DataLoader)}
    payload = {'model': snapshot, 'curves': curves, 'protocol': _protocol(model.args),
               'dataset': signature, 'rng': _rng_state(), 'torch_version': str(torch.__version__)}
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('xb') as stream:
        torch.save(payload, stream)
    logging.info('Task0 checkpoint saved: %s; dataset signature=%s', path, signature)


def load_task0_checkpoint(path, args, signature):
    # Full learner is necessary: state_dict alone omits plain tensor histories.
    payload = torch.load(path, weights_only=False)
    if payload['protocol'] != _protocol(args):
        changed = sorted(key for key in payload['protocol'].keys() | _protocol(args).keys()
                         if payload['protocol'].get(key) != _protocol(args).get(key))
        raise ValueError('Task0 checkpoint protocol differs: ' + ', '.join(changed))
    if payload['dataset'] != signature or payload['torch_version'] != str(torch.__version__):
        raise ValueError('Task0 checkpoint dataset/order or PyTorch version differs.')
    model = payload['model']
    if model._cur_task != 0 or model._known_classes != model._total_classes:
        raise ValueError('Expected a completed Task0 checkpoint.')
    model.args.clear()
    model.args.update(args)
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    logging.info('Task0 checkpoint resumed: %s; SHA256=%s; start_task=1', path, digest.hexdigest())
    state = payload['rng']
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch'])
    if state['cuda']:
        torch.cuda.set_rng_state_all(state['cuda'])
    return model, payload['curves']
