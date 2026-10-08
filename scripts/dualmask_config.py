"""训练入口与运行快照共用的边界配置规范。"""
import math
import warnings


def normalize_dualmask_config(config: dict) -> dict:
    """在启动前规范边界选项，不改变合法配方。"""
    config.setdefault('dual_mask_conflict_old_overlap_adaptive', False)
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
