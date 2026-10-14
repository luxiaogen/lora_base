"""当前任务漂移驱动的CA特征增广；不估计旧类真实分布、不更新均值。"""
import random

import numpy as np
import torch


def collect_current_features(network, loader, device):
    """取当前编码器的确定性视图，恢复训练模式和所有训练随机状态。"""
    modes = [(module, module.training) for module in network.modules()]
    py_state, np_state = random.getstate(), np.random.get_state()
    generator = loader.generator
    generator_state = generator.get_state() if generator is not None else None
    devices = sorted({p.device.index for p in network.parameters() if p.is_cuda})
    features = []
    try:
        with torch.random.fork_rng(devices=devices), torch.no_grad():
            network.eval()
            for _, inputs, _ in loader:
                features.append(network.extract_vector(inputs.to(device)).detach().cpu())
    finally:
        random.setstate(py_state)
        np.random.set_state(np_state)
        if generator is not None:
            generator.set_state(generator_state)
        for module, training in modes:
            module.training = training
    return torch.cat(features)


@torch.no_grad()
def drift_covariance(before, after):
    """先逐图片求位移，再中心化；不是均值位移外积。"""
    delta = after.double() - before.double()
    centered = delta - delta.mean(0)
    return centered.T @ centered / max(delta.shape[0] - 1, 1)


@torch.no_grad()
def covariance_increment(covariance, mode, seed):
    if mode == 'baseline':
        return torch.zeros_like(covariance)
    if mode == 'full':
        return covariance
    if mode == 'isotropic':
        return torch.eye(covariance.shape[0], dtype=covariance.dtype,
                         device=covariance.device) * (covariance.trace() / covariance.shape[0])
    # 入口只允许permuted落入此分支；置换跨任务固定，不消费训练RNG。
    generator = torch.Generator().manual_seed(int(seed) + 91403)
    permutation = torch.randperm(covariance.shape[0], generator=generator).to(covariance.device)
    return covariance[permutation][:, permutation]


@torch.no_grad()
def augment_old_covariances(old_covariances, before, after, mode, seed):
    covariance = drift_covariance(before, after)
    increment = covariance_increment(covariance, mode, seed).to(old_covariances)
    # baseline不作加零运算，保证旧统计逐位不变。
    updated = old_covariances if mode == 'baseline' else old_covariances + increment
    delta = after.double() - before.double()
    audit = {
        'mode': mode, 'samples': int(delta.shape[0]),
        'drift_rms': float(delta.square().mean().sqrt()),
        'mean_drift_norm': float(delta.mean(0).norm()),
        'drift_cov_trace': float(covariance.trace()),
        'increment_trace': float(increment.double().trace()),
        'old_cov_trace_before': float(old_covariances.diagonal(dim1=-2, dim2=-1).sum(-1).mean()),
        'old_cov_trace_after': float(updated.diagonal(dim1=-2, dim2=-1).sum(-1).mean()),
    }
    return updated, audit
