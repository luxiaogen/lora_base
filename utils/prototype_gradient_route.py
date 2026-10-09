"""冻结 W_pre 原型分配当前 B 的分类梯度；不增加损失或推理路由。"""
import json
import logging
import time

import torch
from torch.nn import functional as F

from utils.wpre_distill import teacher_features


@torch.no_grad()
def assign_samples(features, prototypes, class_ids, labels, mode, generator=None):
    prediction = class_ids[(F.normalize(features, dim=1) @
                            F.normalize(prototypes, dim=1).t()).argmax(1)]
    correct = prediction.eq(labels)
    if mode == 'random_matched':
        order = torch.randperm(len(correct), generator=generator).to(correct.device)
        route = correct[order]
    else:
        route = torch.ones_like(correct) if mode == 'all' else correct
    return route, correct


def prepare_batch(learner, inputs, targets):
    mode = learner.args.get('dual_mask_gradient_route', 'off')
    if learner._cur_task == 0 or mode == 'off':
        return {}
    if getattr(learner, '_prototype_route_task', None) != learner._cur_task:
        # 原型由对应类别到达时的训练图片建立；缓存中只包含已见类别。
        learner._prototype_route_task = learner._cur_task
        ids = list(range(learner._total_classes))
        learner._prototype_route_ids = torch.tensor(ids, device=inputs.device)
        learner._prototype_route_means = torch.stack(
            [learner._w0_class_means[i] for i in ids]).to(inputs.device).detach()
        learner._prototype_route_generator = torch.Generator().manual_seed(
            int(learner.args['seed']) + 10007 * learner._cur_task)
    if inputs.is_cuda:
        torch.cuda.synchronize(inputs.device)
    started = time.perf_counter()
    features = teacher_features(learner._network, inputs, learner._pretrained_anchor_context())
    route, correct = assign_samples(features, learner._prototype_route_means,
        learner._prototype_route_ids, targets + learner._known_classes, mode,
        learner._prototype_route_generator)
    if inputs.is_cuda:
        torch.cuda.synchronize(inputs.device)
    return dict(gradient_route_mask=route, gradient_route_correct=correct,
                gradient_route_probe_ms=(time.perf_counter() - started) * 1000)


def prepare_gradients(learner, output):
    """在同一学生前向图上，按整批分母计算两支各自的分类梯度。"""
    named = [(module.layer_idx, branch, unit.B_weight)
             for module in learner._iter_lora_modules()
             for branch, units in (('S', module.S_lora), ('P', module.P_lora))
             for unit in (units[learner._cur_task],)
             if unit is not None and unit.B_weight.requires_grad]
    mode = learner.args['dual_mask_gradient_route']
    if mode == 'all':
        return named, {}
    losses, route = output['gradient_route_per_sample'], output['gradient_route_mask']
    routed = {}
    for branch in ('S', 'P'):
        params = [param for _, b, param in named if b == branch]
        selected = route if branch == 'S' else ~route
        gradients = torch.autograd.grad((losses * selected.to(losses.dtype)).mean(),
                                        params, retain_graph=True)
        routed.update((id(param), grad.detach()) for param, grad in zip(params, gradients))
    return named, routed


@torch.no_grad()
def apply_gradients(learner, output, named, routed):
    """普通反传之后仅替换当前 B；分类头及其他参数保持普通梯度。"""
    mode = learner.args['dual_mask_gradient_route']
    epoch, batch = output['gradient_route_location']
    count = len(output['gradient_route_mask'])
    s_count = int(output['gradient_route_mask'].sum())
    norms = {}
    for branch in ('S', 'P'):
        params = [param for _, b, param in named if b == branch]
        full_sq = sum(param.grad.detach().float().square().sum() for param in params)
        for param in params:
            if mode != 'all':
                param.grad = routed[id(param)]
        assigned_sq = sum(param.grad.detach().float().square().sum() for param in params)
        norms[branch + '_full_grad_norm'] = float(full_sq.sqrt())
        norms[branch + '_assigned_grad_norm'] = float(assigned_sq.sqrt())
    logging.info('PrototypeRouteBatch %s', json.dumps(dict(task=learner._cur_task,
        epoch=epoch, batch=batch, mode=mode, samples=count, s_samples=s_count,
        p_samples=count if mode == 'all' else count - s_count,
        prototype_correct=int(output['gradient_route_correct'].sum()),
        probe_ms=output['gradient_route_probe_ms'], **norms)))
