import argparse
import json
from trainer import train


def apply_overrides(config, overrides):
    for item in overrides or []:
        key, value = item.split("=", 1)
        if key not in config:
            raise ValueError(f"Unknown baseline setting: {key}")
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            pass
        config[key] = value


def main():
    parser = argparse.ArgumentParser(description="Pure DualMask baseline")
    parser.add_argument("--config", default="exps/dlora/imgr10.json")
    parser.add_argument("--set", dest="overrides", action="append", metavar="KEY=VALUE")
    cli = parser.parse_args()
    with open(cli.config, encoding="utf-8") as handle:
        config = json.load(handle)
    apply_overrides(config, cli.overrides)
    train(config)


if __name__ == "__main__":
    main()
