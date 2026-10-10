"""训练入口与运行快照共用的边界配置规范。"""
import math
import warnings


def normalize_dualmask_config(config: dict) -> dict:
    """在启动前规范边界选项，不改变合法配方。"""
    config.setdefault('dual_mask_conflict_old_overlap_adaptive', False)
    protection_rule = config.get('dual_mask_protection_rule', 'static')
    if protection_rule != 'static':
        if protection_rule not in ('ncm_all', 'ncm_risk', 'warmup', 'cooldown', 'energy_equal', 'energy_half'):
            raise ValueError('Unknown dual_mask_protection_rule')
        required = dict(dual_mask_permission_mode='asymmetric', dual_mask_update_rule='step',
            dual_mask_position_norm_match='off', dual_mask_uniform_norm_matched=False,
            dual_mask_composed_conflict='off', dual_mask_conflict_merge_mode='suppress',
            dual_mask_conflict_score_mode='magnitude', dual_mask_conflict_exact_topk=True,
            dual_mask_private_conflict_mode='global', dual_mask_gradient_route='off',
            p_permission_position='wpre', p_permission_release='off', p_direction_score='off',
            dual_mask_reg_weight=0, late_weight_average_epochs=0)
        if any(config.get(key, value) != value for key, value in required.items()):
            raise ValueError('Protection rules require M step updates without extra permissions or losses')
    if config.get('dual_mask_ncm_conflict_mode', 'direct') not in ('direct', 'scaled'):
        raise ValueError('Unknown dual_mask_ncm_conflict_mode')
    route = config.get('dual_mask_gradient_route', 'off')
    position = config.get('dual_mask_protect_position', 'wpre')
    prototype_probe = config.get('dual_mask_prototype_position_probe', False)
    norm_mode = config.get('dual_mask_position_norm_match', 'off')
    if position in ('prototype_high', 'prototype_shuffled', 'prototype_low') or norm_mode == 'prototype_min':
        if not prototype_probe:
            raise ValueError('Prototype positions require the task-start prototype probe')
    if prototype_probe:
        required = dict(dual_mask_gradient_route='off', dual_mask_branch_layout='dual',
            dual_mask_permission_mode='asymmetric', dual_mask_update_rule='step',
            dual_mask_competence_holdout_mod=5, dual_mask_conflict_score_mode='magnitude',
            dual_mask_conflict_exact_topk=True, dual_mask_conflict_granularity='layer',
            dual_mask_private_conflict_mode='global', p_permission_release='off',
            p_permission_position='wpre', p_direction_score='off')
        if any(config.get(key, value) != value for key, value in required.items()) or (
                norm_mode not in ('off', 'prototype_min')
                or position not in ('wpre', 'permuted', 'prototype_high', 'prototype_shuffled', 'prototype_low')
                or norm_mode == 'prototype_min' and position == 'prototype_low'):
            raise ValueError('Prototype protection supports joint S/P M step updates and off/prototype_min only')
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
