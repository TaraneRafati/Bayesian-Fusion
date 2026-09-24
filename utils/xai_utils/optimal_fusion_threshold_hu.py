import numpy as np
import torch
from pathlib import Path
from tqdm import tqdm
from metrics import BinaryIoU, BinaryDice, BboxIoU, BboxDice
from .bbox import get_bounding_box
from .hu_prob import hysteresis_threshold_posterior
from .cam_metrics import write_metrics_to_csv, eval_mask_performance
from .visualization import plot_metrics


def binarize_fusion_posterior(posterior, fusion_threshold, cfg, brain_mask=None):
    hu_cfg = cfg.get("hu_filter", {})
    if hu_cfg.get("use_hysteresis", True):
        tau_low = hu_cfg.get("tau_low", 0.35)
        return hysteresis_threshold_posterior(
            posterior, tau_high=fusion_threshold, tau_low=tau_low, domain_mask=brain_mask
        )
    return (posterior > fusion_threshold).astype(np.uint8)


def compute_metrics_for_fusion_threshold(results, fusion_threshold, cfg):
    pixel_iou_metric = BinaryIoU(from_logits=False)
    pixel_dice_metric = BinaryDice(from_logits=False)
    bbox_iou_metric = BboxIoU()
    bbox_dice_metric = BboxDice()

    for r in results:
        posterior = r.get('hu_posterior')
        if posterior is None:
            raise ValueError(
                f"'{r.get('filename', '?')}' has no hu_posterior. find_opt_fu_thresh requires "
                f"cfg['hu_filter']['enabled']=True and cam_mask_refine must have already run "
                f"so the posterior is populated."
            )
        hu_mask = binarize_fusion_posterior(posterior, fusion_threshold, cfg, brain_mask=r.get('hu_brain_mask'))

        hu_mask_tensor = torch.tensor(hu_mask).float()
        gt_mask_tensor = torch.tensor(r['ground_truth_mask']).float()
        if hu_mask_tensor.shape != gt_mask_tensor.shape:
            print(f"[WARN] Skipping {r.get('filename', '?')} due to shape mismatch: HU={hu_mask_tensor.shape}, GT={gt_mask_tensor.shape}")
            continue

        pixel_iou_metric.update_states(hu_mask_tensor, gt_mask_tensor)
        pixel_dice_metric.update_states(hu_mask_tensor, gt_mask_tensor)

        hu_boxes = get_bounding_box(hu_mask)
        gt_boxes = r['gt_bboxes']
        bbox_iou_metric.update_states(hu_boxes, gt_boxes)
        bbox_dice_metric.update_states(hu_boxes, gt_boxes)

    return {
        'threshold': fusion_threshold,
        'pixel_iou': pixel_iou_metric.result(),
        'pixel_dice': pixel_dice_metric.result(),
        'box_iou': bbox_iou_metric.result(),
        'box_dice': bbox_dice_metric.result()
    }


def evaluate_fusion_thresholds(results, thresholds, output_dir, return_metrics=False, cfg=None):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "fusion_threshold_metrics.csv"

    metrics_list = []
    for threshold in tqdm(thresholds, desc="Evaluating optimal fusion threshold"):
        metrics_list.append(compute_metrics_for_fusion_threshold(results, threshold, cfg))

    write_metrics_to_csv(metrics_list, csv_path)
    plot_metrics(metrics_list, output_dir)
    print(f"[INFO] Saved fusion-threshold metrics and plots to {output_dir}")

    if return_metrics:
        return metrics_list


def apply_optimal_fusion_threshold_and_evaluate(cfg, results):
    threshold_mode = cfg.get("cam_threshold_mode", "pixel_dice")
    vis_thresh_dir = cfg['vis_fusion_thresh_dir']
    thresholds = np.linspace(0.1, 0.9, 50)

    metrics_list = evaluate_fusion_thresholds(results, thresholds, output_dir=vis_thresh_dir, return_metrics=True, cfg=cfg)
    best = max(metrics_list, key=lambda x: x[threshold_mode])
    optimal_threshold = round(best['threshold'], 2)

    cfg.setdefault("hu_filter", {})["fusion_threshold"] = optimal_threshold
    print(f"\n[INFO] Optimal fusion_threshold by maximum '{threshold_mode}' = {optimal_threshold}")

    for r in results:
        r['hu_mask'] = binarize_fusion_posterior(r['hu_posterior'], optimal_threshold, cfg, brain_mask=r.get('hu_brain_mask'))
        r['hu_bboxes'] = get_bounding_box(r['hu_mask'])

    eval_mask_performance(results, mask_key='hu_mask', bbox_key='hu_bboxes')
    return results
