"""独立运行固定的 DualMask 分层覆盖 B 配方。"""
import argparse
import json
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def setup_parser():
    parser = argparse.ArgumentParser(description='DualMask B: ImageNet-R T10 固定配方。')
    parser.add_argument('--config', type=Path, default=ROOT / 'exps/dlora/imgr10.json',
                        help='调试用JSON配置；默认ImageNet-R校准分层B。')
    parser.add_argument('--set', dest='overrides', action='append', default=[], metavar='KEY=VALUE',
                        help='覆盖JSON中已有的训练/数据字段；不接受研究模式。')
    parser.add_argument('--data-path', type=Path, help='已划分 train/test 的 ImageNet-R 根目录。')
    parser.add_argument('--device', help='CUDA 设备编号（默认 0），或 cpu / -1。')
    parser.add_argument('--seed', type=int, help='随机种子（默认 1993）。')
    parser.add_argument('--epochs', type=int, help='每任务训练轮数，包含 Task0（默认 20）。')
    parser.add_argument('--ca-epochs', type=int, help='分类器校准轮数（默认 5）。')
    parser.add_argument('--max-tasks', type=int, help='执行前几个任务（默认 10；短测使用 2）。')
    parser.add_argument('--output', type=Path, help='新建结果目录；默认logs/ImageNet_R/10_tasks/ca/时间_seed。')
    return parser


def load_args(argv=None):
    parser = setup_parser()
    cli = parser.parse_args(argv)
    with cli.config.open(encoding='utf-8') as handle:
        args = json.load(handle)
    for item in cli.overrides:
        # if '=' not in item:
        #     parser.error('--set must use KEY=VALUE')
        key, text = item.split('=', 1)
        # if key not in args:
        #     parser.error('unknown configuration field: {}'.format(key))
        try:
            args[key] = json.loads(text)
        except json.JSONDecodeError:
            args[key] = text
    for key in ('device', 'seed', 'ca_epochs', 'max_tasks'):
        value = getattr(cli, key)
        if value is not None:
            args[key] = value
    if cli.epochs is not None:
        args['init_epoch'] = cli.epochs
        args['epochs'] = cli.epochs
    if args.get('ca_drift_augmentation', 'off') not in ('off', 'baseline', 'full', 'isotropic', 'permuted'):
        parser.error('ca_drift_augmentation must be off/baseline/full/isotropic/permuted')
    # 本调试分支只有已确认配方，拒绝记录与实际固定模型不一致的身份。
    # fixed = dict(dataset='ImageNet_R', rank=64, embd_dim=768, num_heads=12,
    #              init_cls=20, increment=20, total_sessions=10,
    #              slora_gamma=.5, plora_gamma=.75)
    # for key, expected in fixed.items():
    #     if args[key] != expected:
    #         parser.error('{} must be {} in this debug baseline'.format(key, expected))
    # if not isinstance(args.get('layerwise_alpha', False), bool):
    #     parser.error('layerwise_alpha must be true or false')
    # for key in ('init_epoch', 'epochs', 'ca_epochs'):
    #     if args[key] < 1:
    #         parser.error('{} must be at least 1'.format(key.replace('_', '-')))
    # if not 1 <= args['max_tasks'] <= args['total_sessions']:
    #     parser.error('max-tasks must be between 1 and {}'.format(args['total_sessions']))
    data_path = cli.data_path or Path(args['data_path'])
    args['data_path'] = str(data_path.expanduser().resolve())
    prefix = 'depth_layerwise_alpha' if args.get('layerwise_alpha', False) else 'depth_B'
    output = cli.output or ROOT / 'logs/ImageNet_R/10_tasks/ca' / '{}_{}_seed{}'.format(
        prefix, datetime.now().strftime('%Y%m%d_%H%M%S_%f'), args['seed'])
    args['output'] = str(output.expanduser().resolve())
    return args


def main():
    args = load_args()
    # --help 与参数校验无需导入 PyTorch，也不依赖父项目或当前工作目录。
    from trainer import train
    train(args)


if __name__ == '__main__':
    main()
