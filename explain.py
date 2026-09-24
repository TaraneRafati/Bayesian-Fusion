import argparse
import gc
import os

import torch
from torch.utils.data import DataLoader

from plugins import dataloaders
from utils.settings import base_config
from utils.xai_utils.cam import generate_cam_batch, visualize_results
from utils.xai_utils.cam_utils import CAM_METHODS, cam_mask_refine
from utils.xai_utils.gmm_params import load_gmm_parameters
from utils.xai_utils.model_loader import load_model
from utils.xai_utils.opt_cam_thresh import apply_optimal_threshold_and_evaluate
from utils.xai_utils.optimal_fusion_threshold_hu import apply_optimal_fusion_threshold_and_evaluate
from utils.xai_utils.statistics_utils import run_dice_statistics

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"


def parse_args():
    parser = argparse.ArgumentParser(description="CAM / attention localisation with optional GMM fusion.")
    parser.add_argument("--image_dir", required=True)
    parser.add_argument("--masks_dir", required=True)
    parser.add_argument("--test_csv", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--train_csv", default=None)
    parser.add_argument("--output_dir", default="outputs")
    parser.add_argument("--backbone", default="EfficientNetV2B3", choices=["EfficientNetV2B3", "ResNet18", "DINOv2ViTS14"])
    parser.add_argument("--cam_method", default="HiResCAM", choices=list(CAM_METHODS))
    parser.add_argument("--dino_mode", default="cam", choices=["cam", "attention", "rollout"])
    parser.add_argument("--target_layer", type=int, default=-2)
    parser.add_argument("--image_size", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--max_inferences", type=int, default=None)
    parser.add_argument("--evaluate_optimal_threshold", action="store_true")
    parser.add_argument("--eval_opt_thresh_on_train", action="store_true")
    parser.add_argument("--cam_threshold", type=float, default=None)
    parser.add_argument("--fusion", action="store_true")
    parser.add_argument("--gmm_params", default="gmm_parameters.csv")
    parser.add_argument("--fusion_threshold", type=float, default=0.5)
    parser.add_argument("--tau_low", type=float, default=0.35)
    parser.add_argument("--spatial_prior_sigma", type=float, default=15.0)
    parser.add_argument("--find_opt_fu_thresh", action="store_true")
    parser.add_argument("--calc_subtypes_metrics", action="store_true")
    parser.add_argument("--calculate_statistics", action="store_true")
    parser.add_argument("--save_vis_for_paper", action="store_true")
    parser.add_argument("--no_visualizations", action="store_true")
    return parser.parse_args()


def build_cfg(args):
    cfg = base_config(args.backbone, args.image_size, args.batch_size)
    out = args.output_dir
    cfg.update({
        "image_dir": args.image_dir,
        "masks_dir": args.masks_dir,
        "test_csv_path": args.test_csv,
        "checkpoint_path": args.checkpoint,
        "visualize_dir": os.path.join(out, "visualizations"),
        "vis_thresh_dir": os.path.join(out, "cam_thresholds"),
        "vis_fusion_thresh_dir": os.path.join(out, "fusion_thresholds"),
        "vis_statistics_path": os.path.join(out, "dice_statistics"),
        "paper_vis_dir": os.path.join(out, "paper_figures"),
        "cam_method": args.cam_method,
        "cam_threshold_mode": "pixel_dice",
        "target_layers": [args.target_layer],
        "max_inferences": args.max_inferences,
        "evaluate_optimal_threshold": args.evaluate_optimal_threshold,
        "eval_opt_thresh_on_train": args.eval_opt_thresh_on_train,
        "find_opt_fu_thresh": args.find_opt_fu_thresh,
        "calc_subtypes_metrics": args.calc_subtypes_metrics,
        "calculate_statistics": args.calculate_statistics,
        "save_vis_for_paper": args.save_vis_for_paper,
        "save_visualizations": not args.no_visualizations,
        "num_vis_statistics": 10,
        "train_augmentations": {},
        "hu_filter": {
            "enabled": args.fusion,
            "fusion_threshold": args.fusion_threshold,
            "tau_low": args.tau_low,
            "use_hysteresis": True,
            "spatial_prior_k_sigma": args.spatial_prior_sigma,
        },
    })
    if args.train_csv:
        cfg["train_csv_path"] = args.train_csv
    if args.cam_threshold is not None:
        cfg["cam_threshold"] = args.cam_threshold
    if args.backbone == "DINOv2ViTS14":
        cfg["dino"] = {f"use_{args.dino_mode}": True}
    if args.fusion:
        cfg["gmm_parameters"] = load_gmm_parameters(args.gmm_params)
    return cfg


def setup_dataloader(cfg, mode="test"):
    dataset = dataloaders[cfg["plugin"]][cfg["task"]](cfg, mode=mode)
    return DataLoader(dataset, batch_size=cfg["batch_size"], shuffle=False, num_workers=4)


def run_cam(model, dataloader, cfg):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    return generate_cam_batch(cfg, model, dataloader, cfg["target_layers"], cfg["max_inferences"], device=device)


def main():
    cfg = build_cfg(parse_args())

    model = load_model(cfg)

    evaluate_optimal_threshold = cfg["evaluate_optimal_threshold"]
    test_loader = setup_dataloader(cfg, mode="test")
    results = run_cam(model, test_loader, cfg)

    search_results = results
    if evaluate_optimal_threshold and cfg["eval_opt_thresh_on_train"]:
        if not cfg.get("train_csv_path"):
            raise ValueError("--eval_opt_thresh_on_train requires --train_csv.")
        train_loader = setup_dataloader(cfg, mode="train")
        search_results = run_cam(model, train_loader, cfg)
        del train_loader
        gc.collect()

    print("[INFO] Moving classification model to CPU and deleting to free VRAM...")
    model.cpu()
    del model

    gc.collect()
    torch.cuda.empty_cache()

    if evaluate_optimal_threshold:
        results = apply_optimal_threshold_and_evaluate(cfg, search_results, eval_results=results)
    else:
        results = cam_mask_refine(results, cfg)

        if cfg["find_opt_fu_thresh"]:
            if not cfg["hu_filter"]["enabled"]:
                raise ValueError(
                    "--find_opt_fu_thresh requires --fusion (fusion_threshold only binarizes the "
                    "probabilistic posterior, which cam_mask_refine populates)."
                )
            results = apply_optimal_fusion_threshold_and_evaluate(cfg, results)

    if cfg["calculate_statistics"]:
        run_dice_statistics(results, cfg)

    if cfg["save_visualizations"]:
        visualize_results(results, cfg["visualize_dir"], cfg)

    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    os.environ["NO_ALBUMENTATIONS_UPDATE"] = "1"
    main()
