from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from .hu import get_hu_plot_config
from .visualize_for_paper import matches_whitelist, sanitize_scan_id, save_paper_figures

FEATURE_TYPE_LABELS = {
    "cam": ("CAM Overlay", "Baseline vs GT"),
    "attention": ("Attention Overlay", "Attention Mask vs GT"),
    "rollout": ("Rollout Overlay", "Rollout Mask vs GT"),
}


def plot_result(result, save_path, cfg):
    if cfg.get("save_vis_for_paper", False):
        save_path = Path(save_path)
        output_root = Path(cfg["paper_vis_dir"]) if cfg.get("paper_vis_dir") else save_path.parent / "paper_figures"
        scan_id = sanitize_scan_id(save_path.stem)

        whitelist = cfg.get("save_vis_for_paper_only")
        if whitelist and not matches_whitelist(result, scan_id, whitelist):
            return

        save_paper_figures(result, str(output_root), scan_id, cfg)
        return

    gt = result['ground_truth_mask']
    h, w = gt.shape[:2]

    feature_type = result.get('feature_type', 'cam')
    overlay_title, baseline_title = FEATURE_TYPE_LABELS.get(feature_type, FEATURE_TYPE_LABELS["cam"])

    plot_configs = [
        (result['input'], "Original CT"),
        (result['overlay'], overlay_title),
        (create_mask_overlay(gt, result['cam_mask'], h, w), baseline_title)
    ]

    if cfg.get("hu_filter", {}).get("enabled") and result.get('hu_mask') is not None:
        plot_configs.extend(get_hu_plot_config(result, create_mask_overlay))

    num_plots = len(plot_configs)
    plt.figure(figsize=(5 * num_plots, 5))

    for i, (img, title) in enumerate(plot_configs):
        plt.subplot(1, num_plots, i + 1)
        plt.imshow(img)
        plt.title(title)
        plt.axis('off')

    plt.suptitle(f"File: {result['filename']}", fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def create_mask_overlay(gt_mask, cam_mask, height, width):
    overlay_mask = np.zeros((height, width, 3), dtype=np.uint8)
    gt_only = (gt_mask == 1) & (cam_mask == 0)
    cam_only = (cam_mask == 1) & (gt_mask == 0)
    intersection = (gt_mask == 1) & (cam_mask == 1)
    overlay_mask[gt_only] = [0, 255, 0]
    overlay_mask[cam_only] = [255, 0, 0]
    overlay_mask[intersection] = [255, 255, 0]
    return overlay_mask


def plot_metrics(metrics_list, output_dir):
    thresholds = [m['threshold'] for m in metrics_list]
    metric_keys = [k for k in metrics_list[0].keys() if k != 'threshold']
    for metric_name in metric_keys:
        values = [m[metric_name] for m in metrics_list]
        max_value = max(values)
        optimal_index = values.index(max_value)
        optimal_threshold = thresholds[optimal_index]
        plt.figure()
        plt.plot(thresholds, values, marker='o')
        plt.axvline(x=optimal_threshold, color='red', linestyle='--')
        plt.title(f"{metric_name.replace('_', ' ').title()} vs Threshold")
        plt.xlabel("Threshold")
        plt.ylabel(metric_name.replace('_', ' ').title())
        plt.grid(True)
        plt.savefig(Path(output_dir) / f"{metric_name}_vs_threshold.png")
        plt.close()
