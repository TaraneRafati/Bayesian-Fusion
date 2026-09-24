import os
import csv
import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict

from metrics import BinaryDice, BinaryIoU, BinaryPrecision, BinaryRecall
from .visualization import create_mask_overlay


def _single_sample_metrics(pred_mask, gt_mask):
    pred_t = torch.tensor(pred_mask).float()
    gt_t = torch.tensor(gt_mask).float()

    dice_c = BinaryDice(from_logits=False)
    iou_c = BinaryIoU(from_logits=False)
    prec_c = BinaryPrecision(from_logits=False)
    rec_c = BinaryRecall(from_logits=False)

    dice_c.update_states(pred_t, gt_t)
    iou_c.update_states(pred_t, gt_t)
    prec_c.update_states(pred_t, gt_t)
    rec_c.update_states(pred_t, gt_t)

    return {
        'dice': float(dice_c.result()),
        'iou': float(iou_c.result()),
        'precision': float(prec_c.result()),
        'recall': float(rec_c.result()),
    }


def _parse_patient_and_slice(filename):
    stem = os.path.splitext(filename)[0]
    if '_' in stem:
        patient_id, _, slice_part = stem.rpartition('_')
    else:
        patient_id, slice_part = stem, '0'
    return patient_id, slice_part


def compute_dice_difference_records(results, baseline_key='cam_mask', comparison_key='hu_mask'):
    records = []
    for r in results:
        filename = r.get('filename', 'unknown')
        baseline_mask = r.get(baseline_key)
        comparison_mask = r.get(comparison_key)
        gt_mask = r.get('ground_truth_mask')

        if baseline_mask is None or comparison_mask is None or gt_mask is None:
            continue

        base_m = _single_sample_metrics(baseline_mask, gt_mask)
        fuse_m = _single_sample_metrics(comparison_mask, gt_mask)

        patient_id, slice_idx = _parse_patient_and_slice(filename)

        record = {
            'filename': filename,
            'patient_id': patient_id,
            'slice_idx': slice_idx,
            'baseline_dice': base_m['dice'],
            'fusion_dice': fuse_m['dice'],
            'dice_diff': base_m['dice'] - fuse_m['dice'],
            'baseline_iou': base_m['iou'],
            'fusion_iou': fuse_m['iou'],
            'iou_diff': base_m['iou'] - fuse_m['iou'],
            'baseline_precision': base_m['precision'],
            'fusion_precision': fuse_m['precision'],
            'precision_diff': base_m['precision'] - fuse_m['precision'],
            'baseline_recall': base_m['recall'],
            'fusion_recall': fuse_m['recall'],
            'recall_diff': base_m['recall'] - fuse_m['recall'],
            '_result_ref': r,
        }
        records.append(record)

    records.sort(key=lambda x: x['dice_diff'], reverse=True)
    return records


def save_records_to_csv(records, csv_path):
    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    if not records:
        return
    fieldnames = [k for k in records[0].keys() if k != '_result_ref']
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for rec in records:
            writer.writerow({k: rec[k] for k in fieldnames})
    print(f"[INFO] Saved dice-difference CSV ({len(records)} rows) to {csv_path}")


def summarize_worse_cases(records):
    worse_records = [r for r in records if r['dice_diff'] > 0]
    per_patient_worse = defaultdict(int)
    for r in worse_records:
        per_patient_worse[r['patient_id']] += 1

    print("\n" + "=" * 45 + "\n[DICE DEGRADATION SUMMARY]\n" + "=" * 45)
    print(f"[INFO] Slices with worse dice after fusion:  {len(worse_records)} / {len(records)}")
    print(f"[INFO] Patients with >=1 worse slice:         {len(per_patient_worse)}")

    return worse_records, per_patient_worse


def plot_worse_slices_per_patient_histogram(per_patient_worse, save_path):
    if not per_patient_worse:
        print("[INFO] No worsened slices found; skipping per-patient histogram.")
        return
    counts = list(per_patient_worse.values())
    max_count = max(counts)
    bins = np.arange(1, max_count + 2) - 0.5

    plt.figure(figsize=(7, 5))
    plt.hist(counts, bins=bins, edgecolor='black', color='#c0392b')
    plt.xlabel("Number of worsened slices per patient")
    plt.ylabel("Number of patients")
    plt.title("Distribution of Worsened-Slice Counts per Patient")
    plt.xticks(range(1, max_count + 1))
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"[INFO] Saved per-patient worsened-slice histogram to {save_path}")


def plot_worse_metric_diff_histograms(worse_records, save_path):
    if not worse_records:
        print("[INFO] No worsened slices found; skipping metric-diff histograms.")
        return

    metric_diffs = {
        'dice_diff': [r['dice_diff'] for r in worse_records],
        'iou_diff': [r['iou_diff'] for r in worse_records],
        'precision_diff': [r['precision_diff'] for r in worse_records],
        'recall_diff': [r['recall_diff'] for r in worse_records],
    }

    fig, axes = plt.subplots(1, 4, figsize=(20, 4.5))
    for ax, (name, values) in zip(axes, metric_diffs.items()):
        ax.hist(values, bins=20, edgecolor='black', color='#2980b9')
        ax.set_title(name.replace('_', ' ').title())
        ax.set_xlabel("Baseline − Fusion")
        ax.set_ylabel("Count")
        ax.grid(True, alpha=0.3)

    plt.suptitle("Metric Degradation Distributions (Worsened-Dice Slices Only)", fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"[INFO] Saved metric-degradation histograms to {save_path}")


def visualize_top_k_worst(records, output_dir, k=10, baseline_key='cam_mask', comparison_key='hu_mask'):
    output_dir = Path(output_dir) / "top_dice_drops"
    output_dir.mkdir(parents=True, exist_ok=True)

    for rank, rec in enumerate(records[:k], start=1):
        r = rec.get('_result_ref')
        if r is None:
            continue

        gt = r['ground_truth_mask']
        h, w = gt.shape[:2]

        baseline_overlay = create_mask_overlay(gt, r[baseline_key], h, w)
        fusion_overlay = create_mask_overlay(gt, r[comparison_key], h, w)

        plt.figure(figsize=(15, 5))

        plt.subplot(1, 3, 1)
        plt.imshow(r['input'])
        plt.title("Original CT")
        plt.axis('off')

        plt.subplot(1, 3, 2)
        plt.imshow(baseline_overlay)
        plt.title(f"Baseline vs GT (Dice={rec['baseline_dice']:.3f})")
        plt.axis('off')

        plt.subplot(1, 3, 3)
        plt.imshow(fusion_overlay)
        plt.title(f"Fusion vs GT (Dice={rec['fusion_dice']:.3f})")
        plt.axis('off')

        plt.suptitle(f"Rank {rank} | {rec['filename']} | Dice drop = {rec['dice_diff']:.3f}", fontsize=13)
        plt.tight_layout()

        safe_name = os.path.splitext(rec['filename'])[0]
        plt.savefig(output_dir / f"rank{rank:02d}_{safe_name}.png")
        plt.close()

    n_saved = min(k, len(records))
    print(f"[INFO] Saved top-{n_saved} worst-case visualizations to {output_dir}")


def run_dice_statistics(results, cfg):
    if not cfg.get("calculate_statistics", False):
        return

    out_dir = cfg.get("vis_statistics_path")
    if not out_dir:
        print("[WARN] calculate_dice_statistics=True but vis_statistics_path is not set; skipping.")
        return
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    baseline_key = cfg.get("dice_statistics_baseline_key", "cam_mask")
    comparison_key = cfg.get("dice_statistics_comparison_key", "hu_mask")
    num_vis = cfg.get("num_vis_statistics", 10)

    print("\n" + "=" * 45 + "\n[RUNNING DICE DIFFERENCE STATISTICS]\n" + "=" * 45)

    records = compute_dice_difference_records(results, baseline_key, comparison_key)
    if not records:
        print("[WARN] No valid records found for dice-difference statistics "
              f"(check that '{baseline_key}' and '{comparison_key}' exist in results).")
        return

    save_records_to_csv(records, out_dir / "dice_differences.csv")

    worse_records, per_patient_worse = summarize_worse_cases(records)

    plot_worse_slices_per_patient_histogram(per_patient_worse, out_dir / "worsened_slices_per_patient_hist.png")
    plot_worse_metric_diff_histograms(worse_records, out_dir / "worsened_metric_diff_hist.png")

    visualize_top_k_worst(records, out_dir, k=num_vis, baseline_key=baseline_key, comparison_key=comparison_key)

    print(f"[INFO] Dice-difference statistics complete. Outputs saved under {out_dir}")
