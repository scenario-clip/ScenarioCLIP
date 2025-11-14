import warnings
warnings.filterwarnings("ignore")

import os
import random
from pathlib import Path
from argparse import ArgumentParser
from datetime import datetime

import numpy as np
import torch
import lightning as L
from lightning.pytorch.loggers import CSVLogger, TensorBoardLogger
from lightning.pytorch.callbacks import DeviceStatsMonitor, ModelCheckpoint

from models.pyramid_clip import PyramidCLIP
from models.scenario_clip_0 import ScenarioCLIP0
from models.scenario_clip_1 import ScenarioCLIP1
from data.datamodule import ActionGenomeDataModule


def set_seeds(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def make_model(arch: str, vision: str, text: str, tok: str, lr: float, classes_json=None):
    if arch == "pyramid_clip":
        return PyramidCLIP(vision, text, tok, lr=lr)
    if arch == "scenario_clip_0":
        return ScenarioCLIP0(vision, text, tok, lr=lr)
    if arch == "scenario_clip_1":
        return ScenarioCLIP1(vision, text, tok, lr=lr, classes_json=classes_json)
    raise NotImplementedError(f"Unknown architecture: {arch}")


def derive_run_identity_from_ckpt(ckpt_path: Path):
    p = ckpt_path.resolve()
    run_dir = p.parents[1]
    version = run_dir.name

    pretrain_root = None
    for anc in p.parents:
        if anc.name == "pretrain":
            pretrain_root = anc
            break

    if pretrain_root is not None:
        name = run_dir.parent.relative_to(pretrain_root).as_posix()
    else:
        name = run_dir.parent.name

    return name, version


def main():
    torch.set_float32_matmul_precision("high")
    set_seeds(42)

    parser = ArgumentParser()
    parser.add_argument("--architecture", type=str, default="scenario_clip",
                        choices=["pyramid_clip", "scenario_clip_0", "scenario_clip_1"])
    parser.add_argument("--save_dir", type=str, default=f"/scratch/{os.environ.get('USER')}/Exps/scenario-clip")
    parser.add_argument("--metadata_dir", type=str, default="./metadata")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=16)
    parser.add_argument("--max_epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--vision_model_name", type=str, default="openai/clip-vit-base-patch32")
    parser.add_argument("--text_model_name", type=str, default="openai/clip-vit-base-patch32")
    parser.add_argument("--tokenizer_name", type=str, default="openai/clip-vit-base-patch32")
    parser.add_argument("--monitor_device_usage_stats", action="store_true")
    parser.add_argument("--checkpoint_path", type=str, default=None)
    parser.add_argument("--every_n_train_steps", type=int, default=500)
    args = parser.parse_args()

    dm = ActionGenomeDataModule(
        metadata_dir=args.metadata_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    if args.architecture == 'scenario_clip_1': model = make_model(args.architecture, args.vision_model_name, args.text_model_name, args.tokenizer_name, lr=args.lr, classes_json=f"{args.metadata_dir}/classes.json")
    else:
        model = make_model(args.architecture, args.vision_model_name, args.text_model_name, args.tokenizer_name, lr=args.lr)

    base_dir = Path(args.save_dir) / "pretrain"

    if args.checkpoint_path:
        name, _ = derive_run_identity_from_ckpt(Path(args.checkpoint_path))
        version = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    else:
        name = f"{args.architecture}/bs_{args.batch_size}_lr_{args.lr}"
        version = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    print(f"Logging experiment to: {base_dir}/{name}/{version}")

    csv_logger = CSVLogger(save_dir=str(base_dir), name=name, version=version)
    tb_logger = TensorBoardLogger(save_dir=str(base_dir), name=name, version=version, default_hp_metric=False)
    loggers = [csv_logger, tb_logger]

    monitor_key = "val_contrastive_loss" if args.architecture in ("pyramid_clip", "scenario_clip_0") else "val_total_loss"
    metric_field = "{%s:.2f}" % monitor_key
    filename_fmt = f"epoch={{epoch}}-step={{step}}-val_loss={metric_field}"
    ckpt_dir = Path(csv_logger.log_dir) / "checkpoints"
    ckpt_callback = ModelCheckpoint(monitor=monitor_key, dirpath=str(ckpt_dir), filename=filename_fmt, save_top_k=50, mode="min", save_last=True, every_n_train_steps=args.every_n_train_steps, save_on_train_epoch_end=False)
    periodic_filename_fmt = f"epoch={{epoch}}-step={{step}}"
    periodic_dir = Path(csv_logger.log_dir) / "periodic_checkpoints"
    periodic_ckpt = ModelCheckpoint(dirpath=str(periodic_dir), filename=periodic_filename_fmt, every_n_train_steps=500, save_top_k=-1, save_last=False)
    callbacks = [ckpt_callback, periodic_ckpt]
    if args.monitor_device_usage_stats:
        callbacks.append(DeviceStatsMonitor())

    trainer = L.Trainer(
        max_epochs=args.max_epochs,
        devices=1,
        accelerator="gpu",
        callbacks=callbacks,
        logger=loggers,
        log_every_n_steps=10,
        gradient_clip_algorithm="norm",
        gradient_clip_val=1.0,
    )

    ckpt_to_resume = None
    if args.checkpoint_path:
        ckpt_to_resume = args.checkpoint_path
    else:
        last_ckpt = ckpt_dir / "last.ckpt"
        if last_ckpt.exists():
            ckpt_to_resume = str(last_ckpt)

    trainer.fit(model, datamodule=dm, ckpt_path=ckpt_to_resume)


if __name__ == "__main__":
    main()
