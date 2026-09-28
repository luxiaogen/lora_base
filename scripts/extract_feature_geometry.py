#!/usr/bin/env python3
"""Extract once from a merged checkpoint; no training, optimizer or W&B."""
import argparse
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
from torch.utils.data import DataLoader

from utils.feature_geometry import extract_features, remap_labels


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def load_split(data_path, split, class_order, total_classes):
    from torchvision.datasets import ImageFolder
    from torchvision import transforms
    from utils.data import iIMAGENET_R
    from utils.data_manager import DummyDataset
    # No download_data(): its missing-split fallback can move source images.
    folder = ImageFolder(Path(data_path) / split)
    targets = remap_labels(folder.targets, class_order)
    order = np.concatenate([np.flatnonzero(targets == c) for c in range(total_classes)])
    paths = np.asarray([path for path, _ in folder.samples])
    transform = transforms.Compose([*iIMAGENET_R.test_trsf, *iIMAGENET_R.common_trsf])
    return DummyDataset(paths[order], targets[order], transform, use_path=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--config', default='exps/dlora/imgr10.json', help='Only machine-local data_path/device are read')
    parser.add_argument('--out', required=True)
    parser.add_argument('--device', default=None)
    parser.add_argument('--batch-size', type=int, default=48)
    parser.add_argument('--workers', type=int, default=4)
    cli = parser.parse_args()
    # Lazy imports: cached plotting never imports the model/timm/easydict or needs CUDA.
    from models.network import MANet
    from models.attention import Attention_LoRA
    from trainer import _set_random

    started = time.perf_counter()
    payload = torch.load(cli.checkpoint, map_location='cpu', weights_only=True)
    args = dict(payload['config'])
    local = json.loads(Path(cli.config).read_text())
    args['data_path'] = local['data_path']
    value = local.get('device', '0')
    value = value[0] if isinstance(value, list) else str(value).split(',')[0]
    device = torch.device(cli.device or ('cpu' if str(value) in ('-1', 'cpu') else f'cuda:{value}'))
    _set_random(args)
    network = MANet(args)
    for module in network.modules():
        if isinstance(module, Attention_LoRA):
            module._init_params(args)
    network.load_state_dict(payload['model_state_dict'], strict=True)
    network.numtask = payload['numtask']
    network.eval().requires_grad_(False).to(device)
    # Read existing ImageNet-R splits directly, solely for offline diagnosis.
    arrays, sample_manifest = {}, {}
    for split in ('train', 'test'):
        dataset = load_split(args['data_path'], split, payload['class_order'], payload['total_classes'])
        loader = DataLoader(dataset, batch_size=cli.batch_size, shuffle=False,
                            num_workers=cli.workers, generator=torch.Generator().manual_seed(args['seed']))
        x, y, indices = extract_features(network, loader, device, split)
        arrays.update({f'{split}_features': x, f'{split}_labels': y, f'{split}_ids': indices})
        sample_manifest[split] = [str(path) for path in dataset.images]
    arrays['heads'] = torch.cat([head.weight for head in network.classifier_pool[:network.numtask]]).detach().cpu().numpy()
    out = Path(cli.out)
    out.mkdir(parents=True, exist_ok=True)
    # Explicit extract command overwrites this output's cache; use a fresh output per checkpoint.
    np.savez_compressed(out / 'features.npz', **arrays)
    (out / 'sample_paths.json').write_text(json.dumps(sample_manifest, ensure_ascii=False))
    metadata = dict(checkpoint=str(Path(cli.checkpoint).resolve()), checkpoint_sha256=file_hash(cli.checkpoint),
                    task=payload['task'], stage=payload['stage'], class_order=payload['class_order'],
                    increments=payload['increments'], total_classes=payload['total_classes'],
                    checkpoint_last_accuracy=payload['last_accuracy'], config=args,
                    device=str(device), batch_size=cli.batch_size, transform=repr(dataset.trsf),
                    python=platform.python_version(), torch=torch.__version__, numpy=np.__version__,
                    git_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
                    git_status=subprocess.check_output(['git', 'status', '--short'], text=True).strip(),
                    seconds=time.perf_counter() - started, usage='offline_diagnostic_only_no_training')
    (out / 'metadata.json').write_text(json.dumps(metadata, indent=2, ensure_ascii=False))
    print(f'Cached features: {out / "features.npz"}; seconds={metadata["seconds"]:.1f}', flush=True)


if __name__ == '__main__':
    main()
