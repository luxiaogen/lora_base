"""训练入口与运行快照共用的边界配置规范。"""
import math
import warnings


def normalize_dualmask_config(config: dict) -> dict:
    """在启动前规范边界选项，不改变合法配方。"""
    config.setdefault('dual_mask_conflict_old_overlap_adaptive', False)
    route = config.get('dual_mask_gradient_route', 'off')
    if route not in ('off', 'all', 'prototype', 'random_matched'):
        raise ValueError('Unknown dual_mask_gradient_route: ' + str(route))
    if route != 'off' and config.get('dual_mask_branch_layout', 'dual') != 'dual':
        raise ValueError('Prototype gradient routing requires dual S/P branches')
    if route != 'off':
        if (not config.get('use_slora', True) or not config.get('use_plora', True)
                or config.get('plora_train_a', False) or config.get('sp_staged_s_epochs', 0)):
            raise ValueError('Gradient routing requires both current B branches active and A frozen')
    if route in ('prototype', 'random_matched'):
        extra_weights = ('dual_mask_reg_weight', 'wpre_distill_weight', 'pair_separation_weight',
                         'old_competition_weight', 'old_model_distill_weight', 'head_balance_weight')
        if any(config.get(key, 0) != 0 for key in extra_weights) or (
                config.get('dual_mask_anchor_reg_weight', 0) > 0
                and not config.get('dual_mask_anchor_reg_task0_only', False)):
            raise ValueError('This gradient-routing experiment supports incremental classification loss only')
    strength = config.get('dual_mask_fixed_protect_strength')
    if strength is not None:
        if not math.isfinite(strength):
            raise ValueError('dual_mask_fixed_protect_strength must be finite')
        bounded = min(max(strength, 0.0), 1.0)
        if bounded != strength:
            warnings.warn('dual_mask_fixed_protect_strength clipped: {} -> {}'.format(
                strength, bounded), RuntimeWarning, stacklevel=2)
            config['dual_mask_fixed_protect_strength'] = bounded
    task_count = config.get('max_tasks', config.get('total_sessions', 2))
    if (int(task_count) > 1 and config.get('dual_mask_branch_layout', 'dual') == 'single'
            and int(config.get('sp_staged_s_epochs', 0)) > 0):
        raise ValueError('single branch layout does not support S/P staged training')
    return config
