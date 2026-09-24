"""Read-only fingerprints for repeated Task0 runs; no additional loader passes."""

import hashlib
import json
import logging
import platform
from pathlib import Path

import torch


def tensor_hash(named_tensors):
    digest = hashlib.sha256()
    for name, tensor in named_tensors:
        value = tensor.detach().cpu().contiguous()
        digest.update(json.dumps([name, str(value.dtype), list(value.shape)]).encode())
        digest.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def model_fingerprint(model):
    cuda_devices = sorted({p.device.index for p in model.parameters() if p.is_cuda})
    return {
        'model_sha256': tensor_hash(model.state_dict().items()),
        'lora_sha256': tensor_hash((n, p) for n, p in model.named_parameters() if 'lora' in n.lower()),
        'head_sha256': tensor_hash((n, p) for n, p in model.named_parameters() if 'classifier_pool' in n),
        'cpu_rng_sha256': tensor_hash([('rng', torch.get_rng_state())]),
        'cuda_rng_sha256': {str(d): tensor_hash([('rng', torch.cuda.get_rng_state(d))]) for d in cuda_devices},
    }


def log_record(stage, **values):
    logging.info('Task0Repro %s', json.dumps(dict(stage=stage, **values), sort_keys=True))


def log_environment(args, train_dataset, test_dataset):
    root = Path(__file__).resolve().parents[1]
    files = ['trainer.py', 'methods/dlora.py', 'models/attention.py',
             'models/network.py', 'utils/data.py', 'utils/data_manager.py',
             'utils/task0_repro.py', 'exps/dlora/imgr10.json']
    device = args['device'][0]
    log_record(
        'environment', python=platform.python_version(), torch=str(torch.__version__),
        cuda=torch.version.cuda, cudnn=torch.backends.cudnn.version(),
        gpu=torch.cuda.get_device_name(device) if device.type == 'cuda' else str(device),
        deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),
        cudnn_deterministic=torch.backends.cudnn.deterministic,
        cudnn_benchmark=torch.backends.cudnn.benchmark,
        matmul_tf32=torch.backends.cuda.matmul.allow_tf32,
        flash_sdp=torch.backends.cuda.flash_sdp_enabled(),
        memory_efficient_sdp=torch.backends.cuda.mem_efficient_sdp_enabled(),
        source_sha256={name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in files},
    )
    for split, dataset in [('train', train_dataset), ('test', test_dataset)]:
        # Hash the ordered path/label manifest, not image bytes. No augmentation is called.
        manifest = list(zip(map(str, dataset.images), map(int, dataset.labels)))
        log_record('dataset', split=split, count=len(manifest),
                   manifest_sha256=hashlib.sha256(json.dumps(manifest).encode()).hexdigest())
