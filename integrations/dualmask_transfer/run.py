"""Run an unchanged pinned host with optional DualMask runtime adapters."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
HOSTS = {
    "cl": ("https://github.com/JiangpengHe/CL-LoRA.git", "df4e74efe589ca1aa872d3843315ab6851192cf3", "exps/inr.json"),
    "sd": ("https://github.com/WuYichen-97/SD-LoRA-CL.git", "8bacded6eb44786db071f66fb90a87dd660d94ea", "exps/sdlora_inr.json"),
}


def source(host, setup=False):
    url, revision, _ = HOSTS[host]
    path = ROOT / ".external" / ("dualmask_" + host)
    if setup and not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["git", "clone", url, str(path)], check=True)
        subprocess.run(["git", "-C", str(path), "checkout", "--detach", revision], check=True)
    actual = subprocess.check_output(["git", "-C", str(path), "rev-parse", "HEAD"], text=True).strip()
    if actual != revision:
        raise RuntimeError(f"{path}: expected {revision}, found {actual}; do not overwrite an existing checkout")
    if subprocess.check_output(["git", "-C", str(path), "status", "--porcelain", "--untracked-files=no"], text=True).strip():
        raise RuntimeError(f"Official source has local tracked edits: {path}")
    return path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", choices=HOSTS, required=True)
    parser.add_argument("--mode", choices=("off", "on"), default="off")
    parser.add_argument("--data-config", type=Path, default=ROOT / "exps/dlora/imgr10.json")
    parser.add_argument("--seed", type=int, default=1993)
    parser.add_argument("--setup", action="store_true")
    parser.add_argument("--smoke", action="store_true", help="Original T10 partition, first two tasks, one epoch each")
    parser.add_argument("--dry-run", action="store_true")
    cli = parser.parse_args()
    repo = source(cli.host, cli.setup)
    if cli.setup:
        print(f"Pinned {cli.host}: {HOSTS[cli.host][1]} at {repo}")
        return
    dataset = json.loads(cli.data_config.read_text())["data_path"]
    data_path = Path(dataset).expanduser().resolve()
    if not all((data_path / part).is_dir() for part in ("train", "test")):
        raise FileNotFoundError(f"Read {data_path} from {cli.data_config}; existing train/test required; no automatic split")
    args = json.loads((repo / HOSTS[cli.host][2]).read_text())
    # Explicit common protocol adaptation: CL upstream example is 40 tasks.
    args.update(seed=[cli.seed], init_cls=20, increment=20, data_path=str(data_path))
    if cli.smoke:
        for key in ("init_epoch", "epochs", "init_epochs", "later_epochs"):
            if key in args:
                args[key] = 1
    tag = f"{cli.host}_dualmask_{cli.mode}_seed{cli.seed}_{'smoke' if cli.smoke else 't10'}"
    work = ROOT / "logs" / "dualmask_transfer" / f"{tag}_{time.time_ns()}"
    args["prefix"] = tag
    # SD filepath means factor CHECKPOINT output, not its dataset path.
    args["filepath"] = str(work / "factors") + "/"
    fingerprint = {name: hashlib.sha256((Path(__file__).parent / name).read_bytes()).hexdigest()
                   for name in ("core.py", "hosts.py", "run.py")}
    record = {"official_commit": HOSTS[cli.host][1], "mode": cli.mode, "plugin_sha256": fingerprint, "args": args}
    print(json.dumps(record, indent=2), flush=True)
    if cli.dry_run:
        return
    work.mkdir(parents=True)
    (work / "factors").mkdir()
    (work / "effective_config.json").write_text(json.dumps(record, indent=2))
    sys.path.insert(0, str(ROOT / "integrations"))
    sys.path.insert(0, str(repo))
    os.chdir(work)

    import torch
    import timm
    import torchvision
    if cli.host == "cl":
        # timm >=1 moved this module; this is an import alias, not an initializer change.
        import importlib
        try:
            importlib.import_module("timm.models.layers.weight_init")
        except ModuleNotFoundError:
            sys.modules["timm.models.layers.weight_init"] = importlib.import_module("timm.layers.weight_init")
    print(f"Runtime python={sys.version} torch={torch.__version__} torchvision={torchvision.__version__} timm={timm.__version__}", flush=True)
    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA server is required for the official host training")

    # Both arms use the same local ImageFolder loader and original transforms.
    from utils.data import iImageNetR, split_images_labels
    from torchvision.datasets import ImageFolder

    def download_data(self):
        train = ImageFolder(str(data_path / "train"))
        test = ImageFolder(str(data_path / "test"))
        if train.class_to_idx != test.class_to_idx or len(train.classes) != 200:
            raise ValueError("ImageNet-R must have 200 matching train/test class folders")
        self.train_data, self.train_targets = split_images_labels(train.imgs)
        self.test_data, self.test_targets = split_images_labels(test.imgs)
        manifest = "\n".join(f"{Path(p).relative_to(data_path)}\t{y}" for p, y in train.imgs + test.imgs)
        print(f"Dataset train={len(train)} test={len(test)} manifest_sha256={hashlib.sha256(manifest.encode()).hexdigest()}", flush=True)

    iImageNetR.download_data = download_data
    if cli.host == "sd":
        # Upstream saves Python nn.Module objects. Allow only its locally created
        # factor files, not arbitrary downloaded pickle checkpoints.
        original_load = torch.load
        def load(file, *pos, **kw):
            if isinstance(file, (str, Path)) and Path(file).resolve().is_relative_to(work / "factors"):
                kw.setdefault("weights_only", False)
            return original_load(file, *pos, **kw)
        torch.load = load
    if cli.mode == "on":
        from dualmask_transfer.hosts import install_cl, install_sd
        (install_cl if cli.host == "cl" else install_sd)()
    if cli.smoke:
        from utils.data_manager import DataManager
        original_manager = DataManager.__init__
        def init_manager(self, *pos, **kw):
            original_manager(self, *pos, **kw)
            self._increments = self._increments[:2]
        DataManager.__init__ = init_manager

    from utils import factory
    original_factory = factory.get_model
    def get_model(*pos, **kw):
        model = original_factory(*pos, **kw)
        digest = hashlib.sha256()
        backbone = model._network.backbone
        blocks = backbone.blocks if cli.host == "cl" else backbone.base_vit.blocks
        for block in blocks:
            names = ("q_proj", "k_proj", "v_proj") if cli.host == "cl" else ("qkv",)
            for name in names:
                weight = getattr(block.attn, name).weight.detach().cpu().contiguous()
                digest.update(weight.numpy().tobytes())
        print(f"PRETRAINED_ATTENTION_SHA256={digest.hexdigest()}", flush=True)
        return model
    factory.get_model = get_model

    from trainer import train
    started = time.monotonic()
    train(args)
    print(f"TRANSFER_COMPLETE host={cli.host} mode={cli.mode} seconds={time.monotonic()-started:.1f}", flush=True)


if __name__ == "__main__":
    main()
