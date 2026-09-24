from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from metrics import BboxDice, BboxIoU, BinaryDice, BinaryIoU
from .bbox import get_bounding_box
from .cam_metrics import compute_segmentation_metrics, write_metrics_to_csv
from .cam_utils import binarize_cam_mask
from .visualization import plot_metrics


def apply_optimal_threshold_and_evaluate(cfg, search_results, eval_results=None):
    if eval_results is None:
        eval_results = search_results

    threshold_mode = cfg.get("cam_threshold_mode", "pixel_dice")
    vis_thresh_dir = cfg['vis_thresh_dir']
    thresholds = np.linspace(0.1, 0.9, 50)

    feature_type = search_results[0].get('feature_type', 'cam') if search_results else 'cam'
    threshold_key = "dino_attention_threshold" if feature_type in ("attention", "rollout") else "cam_threshold"

    metrics_list = evaluate_thresholds(search_results, thresholds, output_dir=vis_thresh_dir, return_metrics=True, cfg=cfg)
    best = max(metrics_list, key=lambda x: x[threshold_mode])
    optimal_threshold = round(best['threshold'], 2)

    cfg[threshold_key] = optimal_threshold

    leakage_note = "search set == eval set: leakage-biased" if eval_results is search_results else "search set != eval set: leakage-free"
    print(f"\n[INFO] Optimal {threshold_key} (feature_type='{feature_type}') by maximum "
        f"'{threshold_mode}' = {optimal_threshold}  [{leakage_note}]")

    for r in eval_results:
        r['cam_mask'] = binarize_cam_mask(r['cam'], optimal_threshold)
        r['cam_bboxes'] = get_bounding_box(r['cam_mask'])

    compute_segmentation_metrics(eval_results, cfg)
    return eval_results


def evaluate_thresholds(results, thresholds, output_dir, return_metrics=False, cfg=None):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "threshold_metrics.csv"

    metrics_list = []
    for threshold in tqdm(thresholds, desc="Evaluating optimal threshold"):
        metrics = compute_metrics_for_threshold(results, threshold, cfg)
        metrics_list.append(metrics)

    write_metrics_to_csv(metrics_list, csv_path)
    plot_metrics(metrics_list, output_dir)
    print(f"[INFO] Saved metrics and plots to {output_dir}")

    if return_metrics:
        return metrics_list


def compute_metrics_for_threshold(results, threshold, cfg):
    pixel_iou_metric = BinaryIoU(threshold=threshold, from_logits=False)
    pixel_dice_metric = BinaryDice(threshold=threshold, from_logits=False)
    bbox_iou_metric = BboxIoU()
    bbox_dice_metric = BboxDice()

    for r in results:
        cam_mask = binarize_cam_mask(r['cam'], threshold)

        cam_mask_tensor = torch.tensor(cam_mask).float()
        gt_mask_tensor = torch.tensor(r['ground_truth_mask']).float()

        if cam_mask_tensor.shape != gt_mask_tensor.shape:
            print(f"[WARN] Skipping {r['filename']} due to shape mismatch: CAM={cam_mask_tensor.shape}, GT={gt_mask_tensor.shape}")
            continue

        pixel_iou_metric.update_states(cam_mask_tensor, gt_mask_tensor)
        pixel_dice_metric.update_states(cam_mask_tensor, gt_mask_tensor)

        cam_boxes = get_bounding_box(cam_mask)
        gt_boxes = r['gt_bboxes']
        bbox_iou_metric.update_states(cam_boxes, gt_boxes)
        bbox_dice_metric.update_states(cam_boxes, gt_boxes)

    return {
        'threshold': threshold,
        'pixel_iou': pixel_iou_metric.result(),
        'pixel_dice': pixel_dice_metric.result(),
        'box_iou': bbox_iou_metric.result(),
        'box_dice': bbox_dice_metric.result()
    }
