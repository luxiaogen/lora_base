#!/usr/bin/env python3
"""NCM vs incremental ridge on immutable original AugReg ViT features."""
import argparse
import csv
import datetime
import hashlib
import json
from pathlib import Path
import platform
import random
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from utils.frozen_readout import FrozenReadout, load_reference, readout_report, select_alpha, training_partition


REFERENCES = {
    '3090': 'logs/shell_logs/imgr10_expert_legal_3090/20261001_150130_524490/t10/diagnostics',
    '5090': 'logs/shell_logs/imgr10_expert_legal_5090/20261001_150145_686685/t10/diagnostics',
}


def configuration(local, reference, machine):
    source = reference.parent / 'run.json'
    if source.is_file():
        args = dict(json.loads(source.read_text())['effective_config'])
        seed = args['seed'][0] if isinstance(args['seed'], list) else args['seed']
        args['_reference_protocol_compatible'] = (
            seed == 1993 and args.get('dataset') == 'ImageNet_R'
            and args.get('shuffle') is True and args.get('init_cls') == 20
            and args.get('increment') == 20 and args.get('total_sessions') == 10
            and args.get('disable_fused_sdpa') is True)
        args.setdefault('two_expert_calibration_holdout_mod', 0)
    else:
        args = dict(local)
        spec = json.loads(Path('scripts/sweeps/imgr10_two_expert_oracle_3090.json').read_text())
        args.update(spec['common_overrides'])
        args['two_expert_calibration_holdout_mod'] = 20 if machine == '3090' else 10
        args['_reference_protocol_compatible'] = False
    args.update(data_path=local['data_path'], device=local.get('device', '0'), seed=1993,
                disable_fused_sdpa=True)
    return args, str(source) if source.is_file() else 'fixed_readout_protocol_without_reference_metadata'


def freeze_original_encoder(network, args, device):
    """Only called on a freshly constructed, original pretrained network."""
    from models.attention import Attention_LoRA
    for module in network.modules():
        if isinstance(module, Attention_LoRA):
            module._init_params(args)
            module.set_pretrained_anchor_mode(True)
    network.numtask = 1
    network.eval().requires_grad_(False).to(device)


@torch.inference_mode()
def collect(network, dataset, positions, args, device):
    loader = DataLoader(Subset(dataset, positions.tolist()), batch_size=args['batch_size'], shuffle=False,
                        num_workers=args['num_workers'],
                        generator=torch.Generator().manual_seed(args['seed']))
    features, targets = [], []
    for step, (_, inputs, labels) in enumerate(loader):
        features.append(network.extract_vector(inputs.to(device)).cpu().float())
        targets.append(labels.long())
        if step % 25 == 0 or step + 1 == len(loader):
            print(f'  features batch {step + 1}/{len(loader)}', flush=True)
    return torch.cat(features), torch.cat(targets)


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def run_readouts(network, train, test, args, device, out, reference, tasks):
    out.mkdir(parents=True, exist_ok=True)
    classes = args['init_cls'] + (args['total_sessions'] - 1) * args['increment']
    stats = FrozenReadout(args['embd_dim'], classes)
    reports, test_parts, target_parts = [], [], []
    known, alpha = 0, None
    for task in range(tasks):
        seen = known + (args['init_cls'] if task == 0 else args['increment'])
        fit, holdout = training_partition(train.labels, known, seen,
            args['two_expert_calibration_holdout_mod'], args['dual_mask_competence_holdout_mod'])
        positions = np.sort(np.concatenate((fit, holdout)))
        print(f'Task{task}: current train only; fit={len(fit)}, holdout={len(holdout)}, classes={seen}', flush=True)
        features, labels = collect(network, train, positions, args, device)
        fit_mask = torch.from_numpy(np.isin(positions, fit))
        stats.update(features[fit_mask], labels[fit_mask])
        if task == 0:
            alpha, candidates = select_alpha(stats, features[~fit_mask], labels[~fit_mask], seen)
            write_json(out / 'task0_alpha.json', dict(source='task0_train_holdout_only',
                selected_alpha=alpha, candidates=candidates, tie_rule='first_predeclared_alpha',
                selected_before_test_access=True, refit_with_holdout=False))
            print(f'Task0 train-holdout selected alpha={alpha}; fixed for every later task.', flush=True)
        weight, regularizer = stats.ridge_weight(alpha, seen)
        # Current training features are discarded; old train images are never loaded again.
        del features, labels
        current_test = np.flatnonzero((test.labels >= known) & (test.labels < seen))
        x, y = collect(network, test, current_test, args, device)
        test_parts.append(x)
        target_parts.append(y)
        all_x, all_y = torch.cat(test_parts), torch.cat(target_parts)
        ncm = stats.ncm_scores(all_x, seen).argmax(1)
        ridge = (all_x @ weight).argmax(1)
        cache_file = reference / f'task_{task:02d}_test_report_only.csv'
        base, cached_ncm, status = load_reference(cache_file, all_y)
        if base is not None and not args.get('_reference_protocol_compatible', True):
            base, cached_ncm, status = None, None, 'reference_metadata_missing_or_protocol_mismatch'
        agreement = None
        if base is not None:
            agreement = 100. * float(ncm.eq(cached_ncm).double().mean())
            # Report discrepancies, do not abort feature extraction or invent a tolerance.
            if not torch.equal(ncm, cached_ncm):
                base, status = None, 'ncm_reproduction_mismatch'
        report = dict(task=task, classes=seen, alpha=alpha, regularizer=regularizer,
                      fit_samples=len(fit), holdout_samples=len(holdout), cumulative_fit=stats.samples,
                      statistics_bytes=stats.storage_bytes, reference_status=status,
                      ncm_vs_ridge=readout_report(ncm, ridge, all_y, known),
                      base_vs_ncm=None, base_vs_ridge=None, reference_ncm_agreement_percent=agreement,
                      reference_sha256=hashlib.sha256(cache_file.read_bytes()).hexdigest() if cache_file.is_file() else None,
                      oracle_uses_true_labels=True,
                      readout_uses_test_labels_or_task_id=False)
        if base is not None:
            report.update(base_vs_ncm=readout_report(base, ncm, all_y, known),
                          base_vs_ridge=readout_report(base, ridge, all_y, known))
        write_json(out / f'task_{task:02d}.json', report)
        with (out / f'task_{task:02d}_predictions.csv').open('w', newline='') as stream:
            writer = csv.writer(stream)
            columns = ['index', 'target', 'ncm_prediction', 'ridge_prediction']
            arrays = [range(len(all_y)), all_y.tolist(), ncm.tolist(), ridge.tolist()]
            if base is not None:
                columns.append('base_prediction')
                arrays.append(base.tolist())
            writer.writerow(columns)
            writer.writerows(zip(*arrays))
        reports.append(report)
        print('FrozenReadoutTask ' + json.dumps(report, allow_nan=False), flush=True)
        known = seen
    mean = lambda values: sum(values) / len(values)
    summary = dict(tasks=len(reports), selected_alpha=alpha, statistics_bytes=stats.storage_bytes,
                   ncm_average=mean([r['ncm_vs_ridge']['total']['base_accuracy'] for r in reports]),
                   ridge_average=mean([r['ncm_vs_ridge']['total']['readout_accuracy'] for r in reports]),
                   ncm_last=reports[-1]['ncm_vs_ridge']['total']['base_accuracy'],
                   ridge_last=reports[-1]['ncm_vs_ridge']['total']['readout_accuracy'],
                   paired_reference_tasks=sum(r['base_vs_ridge'] is not None for r in reports),
                   ridge_complement_oracle_average=None, ncm_complement_oracle_average=None,
                   base_average=None, last_report=reports[-1])
    if summary['paired_reference_tasks'] == len(reports):
        summary.update(base_average=mean([r['base_vs_ridge']['total']['base_accuracy'] for r in reports]),
                       ridge_complement_oracle_average=mean([r['base_vs_ridge']['total']['oracle_accuracy'] for r in reports]),
                       ncm_complement_oracle_average=mean([r['base_vs_ncm']['total']['oracle_accuracy'] for r in reports]))
    write_json(out / 'summary.json', summary)
    print('FrozenReadoutSummary ' + json.dumps(summary, allow_nan=False), flush=True)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--machine', required=True, choices=('3090', '5090'))
    parser.add_argument('--config', type=Path, default=Path('exps/dlora/imgr10.json'))
    parser.add_argument('--reference-dir', type=Path)
    parser.add_argument('--out', type=Path)
    parser.add_argument('--smoke', action='store_true', help='Only Task0-1; no separate warmup training')
    parser.add_argument('--dry-run', action='store_true')
    cli = parser.parse_args()
    reference = cli.reference_dir or Path(REFERENCES[cli.machine])
    args, source = configuration(json.loads(cli.config.read_text()), reference, cli.machine)
    tasks = 2 if cli.smoke else 10
    out = cli.out or Path(f'logs/frozen_readout/{cli.machine}') / datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    plan = dict(machine=cli.machine, seed=1993, tasks=tasks, config_source=source,
                reference_dir=str(reference), data_path=args['data_path'], out=str(out),
                outer_holdout_mod=args['two_expert_calibration_holdout_mod'],
                reference_protocol_compatible=args['_reference_protocol_compatible'],
                ncm_and_ridge_fit_same_samples=True, train_dualmask=False, save_weights=False,
                save_dense_features=False, old_train_image_replay=False,
                alpha_selection='task0_train_holdout_only_then_fixed', disable_fused_sdpa=True)
    if cli.dry_run:
        print(json.dumps(plan, indent=2))
        return 0
    print(json.dumps(plan, indent=2), flush=True)
    print('No new DualMask run. Missing reference cache => standalone scores only.', flush=True)
    started = time.perf_counter()
    random.seed(args['seed'])
    np.random.seed(args['seed'])
    torch.manual_seed(args['seed'])
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)
    value = args['device']
    value = str(value[0] if isinstance(value, list) else value).split(',')[0]
    device = torch.device('cpu' if value in ('cpu', '-1') else f'cuda:{value}')
    from models.network import MANet
    from scripts.extract_feature_geometry import load_split
    network = MANet(args)
    freeze_original_encoder(network, args, device)
    digest = hashlib.sha256()
    for name, parameter in network.image_encoder.named_parameters():
        if 'lora' not in name.lower():
            digest.update(name.encode())
            digest.update(parameter.detach().cpu().contiguous().numpy().tobytes())
    class_order = np.random.RandomState(args['seed']).permutation(200) if args['shuffle'] else np.arange(200)
    # Enumerate paths read-only. Load only current-task train images inside the loop.
    train = load_split(args['data_path'], 'train', class_order.tolist(), 200)
    test = load_split(args['data_path'], 'test', class_order.tolist(), 200)
    out.mkdir(parents=True, exist_ok=True)
    source_files = ('scripts/run_frozen_readout.py', 'utils/frozen_readout.py',
                    'scripts/extract_feature_geometry.py', 'models/network.py', 'models/attention.py', 'models/vit.py')
    paths_hash = lambda dataset: hashlib.sha256('\n'.join(map(str, dataset.images)).encode()).hexdigest()
    manifest = dict(plan, effective_config=args, class_order=class_order.tolist(),
                    backbone_sha256=digest.hexdigest(), transform=repr(train.trsf),
                    train_paths_sha256=paths_hash(train), test_paths_sha256=paths_hash(test),
                    reference_config_sha256=hashlib.sha256(Path(source).read_bytes()).hexdigest() if Path(source).is_file() else None,
                    source_sha256={p: hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in source_files},
                    code_revision=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
                    python=platform.python_version(), torch=torch.__version__, numpy=np.__version__, device=str(device),
                    statistics_dtype='float32; temporary solve=float64',
                    transient_test_cache='RAM only; never used for fitting; each test image encoded once',
                    reference_alignment_limit='index/label/config plus NCM agreement; old cache has no image-content hash')
    write_json(out / 'manifest.json', manifest)
    print('Code revision:', manifest['code_revision'], flush=True)
    print('Fixed backbone SHA256:', manifest['backbone_sha256'], flush=True)
    run_readouts(network, train, test, args, device, out, reference, tasks)
    print(f'Finished: {out}; seconds={time.perf_counter() - started:.1f}', flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
