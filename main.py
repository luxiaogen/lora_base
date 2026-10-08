import json
import argparse
import math
import warnings
from trainer import train

def apply_overrides(config, overrides):
    for item in overrides or []:
        if "=" not in item:
            raise ValueError("Override must use KEY=VALUE: {}".format(item))
        key, value = item.split("=", 1)
        key = key.strip()
        if not key:
            raise ValueError("Override key cannot be empty")
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            pass
        config[key] = value
    return config

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

def main():
    # args = setup_parser().parse_args()

    parser = setup_parser()
    cli_args = parser.parse_args()

    config = load_json(cli_args.config)
    apply_overrides(config, cli_args.overrides)
    normalize_dualmask_config(config)

    args = vars(cli_args)
    args.update(config)


    # args = vars(args)  # Converting argparse Namespace to a dict.
    # args.update(param)  # Add parameters from json
    train(args)


def load_json(settings_path):
    with open(settings_path) as data_file:
        param = json.load(data_file)

    return param


def setup_parser():
    parser = argparse.ArgumentParser(description='Reproduce of multiple continual learning algorthms.')
    parser.add_argument('--config', type=str, default='exps/dlora/cub10.json',
                       help='Json file of settings.')
    # parser.add_argument('--config', type=str, default='exps/dlora/imga10.json',
    #                     help='Json file of settings.')
    # parser.add_argument('--config', type=str, default='exps/dlora/cub10.json',
    #                     help='Json file of settings.')
    # parser.add_argument('--config', type=str, default='exps/dlora/cifar10.json',
    #                     help='Json file of settings.')
    # parser.add_argument('--config', type=str, default='exps/dlora/domainnet.json',
    #                     help='Json file of settings.')

    # parser.add_argument('--device', type=str, default='2')

    parser.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Override a JSON setting; may be repeated.",
    )

    return parser


if __name__ == '__main__':
    main()
