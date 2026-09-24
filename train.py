import argparse
import gc
import os
import time
import warnings

import torch
from torch.utils.data import DataLoader

import metrics
from callbacks import CSVLogger, Checkpoint, TimeLimit
from losses import losses
from models import models
from optimizers import optimizers
from plugins import dataloaders
from utils import set_random_seed
from utils.settings import base_config

METRIC_NAMES = ["BinaryPrecision", "BinaryRecall", "BinaryAUC", "BinaryF1"]
THRESHOLDS = [0.3, 0.4, 0.5, 0.6, 0.7]


def parse_args():
    parser = argparse.ArgumentParser(description="Train a brain hemorrhage slice classifier.")
    parser.add_argument("--image_dir", required=True)
    parser.add_argument("--train_csv", required=True)
    parser.add_argument("--val_csv", required=True)
    parser.add_argument("--backbone", default="EfficientNetV2B3", choices=["EfficientNetV2B3", "ResNet18", "DINOv2ViTS14"])
    parser.add_argument("--image_size", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--init_lr", type=float, default=0.001)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--seed", type=int, default=51)
    parser.add_argument("--log_dir", default="./run/")
    parser.add_argument("--ckpt_dir", default="./checkpoints/")
    parser.add_argument("--limit_time", type=int, default=1e9)
    return parser.parse_args()


def build_cfg(args):
    cfg = base_config(args.backbone, args.image_size, args.batch_size)
    cfg.update({
        "image_dir": args.image_dir,
        "train_csv_path": args.train_csv,
        "val_csv_path": args.val_csv,
        "init_lr": args.init_lr,
        "epochs": args.epochs,
        "random_seed": args.seed,
    })
    return cfg


def build_metrics():
    metric_objects = []
    for name in METRIC_NAMES:
        for threshold in THRESHOLDS:
            metric_objects.append(getattr(metrics, name)(name=f"{name}@{threshold}", threshold=threshold))
    return metric_objects


def main():
    args = parse_args()
    cfg = build_cfg(args)

    warnings.filterwarnings("ignore", category=UserWarning)
    set_random_seed(cfg["random_seed"])

    task, backbone_name, image_size = cfg["task"], cfg["backbone"], cfg["image_size"][0]
    print(f"Training model: task={task} backbone={backbone_name}")

    loss_cls, default_args = losses[task][cfg["loss"]]
    criterion = loss_cls(**{**default_args, **cfg["loss_args"]})

    model = models[task][backbone_name](
        optimizer=optimizers[cfg["optimizer"]],
        criterion=criterion,
        config=cfg,
        init_lr=cfg["init_lr"],
        metrics=build_metrics(),
    )

    train_dataset = dataloaders[cfg["plugin"]][task](cfg, mode="train")
    val_dataset = dataloaders[cfg["plugin"]][task](cfg, mode="val")
    print(f" --- number of slices in TrainSet: {len(train_dataset)} --- ")
    print(f" --- number of slices in ValSet: {len(val_dataset)} --- ")

    train_loader = DataLoader(train_dataset, batch_size=cfg["batch_size"], shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=cfg["batch_size"], shuffle=False, num_workers=2)

    os.makedirs(args.log_dir, exist_ok=True)
    log_path = os.path.join(args.log_dir, f"{backbone_name}_{image_size}.csv")
    callbacks = [
        Checkpoint(model, f"{args.ckpt_dir}/{task}_{backbone_name}__{image_size}_{cfg['init_lr']}",
                   monitor="val_loss", mode="min", load_best_on_end=True),
        CSVLogger(log_path),
        TimeLimit(args.limit_time),
    ]

    start_time = time.time()
    model.fit(
        train_loader,
        epochs=cfg["epochs"],
        train_step_per_epoch=None,
        valid_datagen=val_loader,
        valid_step_per_epoch=None,
        callbacks=callbacks,
        verbose=1,
        init_epoch=1,
        device=None,
    )
    print(f"Finished training {task}-{backbone_name} in {(time.time() - start_time) / 60:.2f} minutes.")

    del train_loader, val_loader, train_dataset, val_dataset, criterion, model
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
