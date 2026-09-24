import csv

import numpy as np
import torch

from metrics import BboxDice, BboxIoU, BinaryDice, BinaryIoU, BinaryPrecision, BinaryRecall, LooseHit
from .hu import evaluate_hu_performance


def compute_segmentation_metrics(results, cfg):
    print("\n" + "=" * 45 + "\n[BASELINE METRICS]\n" + "=" * 45)
    eval_mask_performance(results, mask_key='cam_mask', bbox_key='cam_bboxes')

    if cfg.get("hu_filter", {}).get("enabled"):
        evaluate_hu_performance(results, eval_mask_performance)


def eval_mask_performance(results, mask_key, bbox_key):
    global_metrics = {
        'pixel_iou': BinaryIoU(from_logits=False),
        'pixel_dice': BinaryDice(from_logits=False),
        'pixel_precision': BinaryPrecision(from_logits=False),
        'pixel_recall': BinaryRecall(from_logits=False),
        'bbox_iou': BboxIoU(),
        'bbox_dice': BboxDice(),
        'loose_hit': LooseHit()
    }

    accumulators = {k: [] for k in ['pixel_ious', 'pixel_dices', 'pixel_precisions', 'pixel_recalls', 'bbox_ious', 'bbox_dices']}

    for r in results:
        temp_result = r.copy()
        temp_result['cam_mask'] = r[mask_key]
        temp_result['cam_bboxes'] = r[bbox_key]

        update_all_metrics(temp_result, global_metrics, accumulators)

    print_segmentation_metrics(global_metrics, accumulators)


def print_segmentation_metrics(global_metrics, accumulators):
    print("\nAggregate Segmentation Metrics Across All Inferences:")
    print(f"[Global]   Loose Hit Rate:     {global_metrics['loose_hit'].result():.4f}")
    print(f"[Global]   Pixelwise IoU:      {global_metrics['pixel_iou'].result():.4f}")
    print(f"[Global]   Pixelwise Dice:     {global_metrics['pixel_dice'].result():.4f}")
    print(f"[Global]   Pixelwise Precision:{global_metrics['pixel_precision'].result():.4f}")
    print(f"[Global]   Pixelwise Recall:   {global_metrics['pixel_recall'].result():.4f}")
    print(f"[Global]   BBox IoU:           {global_metrics['bbox_iou'].result():.4f}")
    print(f"[Global]   BBox Dice:          {global_metrics['bbox_dice'].result():.4f}")
    print("-" * 50)
    print(f"[Average]  Pixelwise IoU:      {np.mean(accumulators['pixel_ious']):.4f}")
    print(f"[Average]  Pixelwise Dice:     {np.mean(accumulators['pixel_dices']):.4f}")
    print(f"[Average]  Pixelwise Precision:{np.mean(accumulators['pixel_precisions']):.4f}")
    print(f"[Average]  Pixelwise Recall:   {np.mean(accumulators['pixel_recalls']):.4f}")
    print(f"[Average]  BBox IoU:           {np.mean(accumulators['bbox_ious']):.4f}")
    print(f"[Average]  BBox Dice:          {np.mean(accumulators['bbox_dices']):.4f}")


def write_metrics_to_csv(metrics_list, csv_path):
    if not metrics_list:
        return
    metric_keys = [k for k in metrics_list[0].keys() if k != 'threshold']
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['threshold'] + metric_keys)
        for m in metrics_list:
            writer.writerow([m['threshold']] + [m[k] for k in metric_keys])


def update_all_metrics(result, global_metrics, accumulators):
    cam_mask_tensor = torch.tensor(result['cam_mask']).float()
    gt_mask_tensor = torch.tensor(result['ground_truth_mask']).float()

    pixel_iou_calc = BinaryIoU(from_logits=False)
    pixel_dice_calc = BinaryDice(from_logits=False)
    pixel_iou_calc.update_states(cam_mask_tensor, gt_mask_tensor)
    pixel_dice_calc.update_states(cam_mask_tensor, gt_mask_tensor)

    result['pixel_iou'] = pixel_iou_calc.result()
    result['pixel_dice'] = pixel_dice_calc.result()

    accumulators['pixel_ious'].append(result['pixel_iou'])
    accumulators['pixel_dices'].append(result['pixel_dice'])

    for metric in ['pixel_iou', 'pixel_dice', 'pixel_precision', 'pixel_recall']:
        global_metrics[metric].update_states(cam_mask_tensor, gt_mask_tensor)

    single_precision = BinaryPrecision(from_logits=False)
    single_recall = BinaryRecall(from_logits=False)
    single_precision.update_states(cam_mask_tensor, gt_mask_tensor)
    single_recall.update_states(cam_mask_tensor, gt_mask_tensor)
    accumulators['pixel_precisions'].append(single_precision.result())
    accumulators['pixel_recalls'].append(single_recall.result())

    global_metrics['loose_hit'].update_states(result['cam_mask'], result['ground_truth_mask'])

    bbox_iou_single = BboxIoU()
    bbox_iou_single.update_states(result['cam_bboxes'], result['gt_bboxes'])
    accumulators['bbox_ious'].append(bbox_iou_single.result())
    global_metrics['bbox_iou'].update_states(result['cam_bboxes'], result['gt_bboxes'])

    bbox_dice_single = BboxDice()
    bbox_dice_single.update_states(result['cam_bboxes'], result['gt_bboxes'])
    accumulators['bbox_dices'].append(bbox_dice_single.result())
    global_metrics['bbox_dice'].update_states(result['cam_bboxes'], result['gt_bboxes'])
