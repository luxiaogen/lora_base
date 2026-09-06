import json
import argparse
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

def main():
    # args = setup_parser().parse_args()

    parser = setup_parser()
    cli_args = parser.parse_args()

    config = load_json(cli_args.config)
    apply_overrides(config, cli_args.overrides)

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
