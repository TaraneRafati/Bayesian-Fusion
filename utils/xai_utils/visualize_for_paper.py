import os
import re
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.colorbar as mcolorbar
import cv2

PALETTE = {
    "precision": "#2a78d6",
    "recall": "#eb6834",
    "f1": "#1baf7a",
}

CAM_CMAP = "inferno"
POSTERIOR_CMAP = "viridis"
LLR_CMAP = "RdBu_r"
UNCERTAINTY_CMAP = "cividis"

DEFAULT_DPI = 300
DEFAULT_SIZE_IN = 3.0
DEFAULT_LINE_THICKNESS = 1
MASK_ALPHA = 0.45
HEATMAP_ALPHA = 0.55
LLR_CLIP = 20.0


def sanitize_scan_id(name):
    name = os.path.splitext(str(name))[0]
    name = re.sub(r"[^\w\-. ]+", "_", name).strip()
    return name or "case"


def matches_whitelist(result, scan_id, whitelist):
    if not whitelist:
        return True
    filename = str(result.get("filename", ""))
    stem = os.path.splitext(os.path.basename(filename))[0]
    candidates = {filename, stem, scan_id, sanitize_scan_id(filename)}
    return any(w in candidates for w in whitelist)


def _hex_to_rgb01(hex_color):
    hex_color = hex_color.lstrip("#")
    return tuple(int(hex_color[i:i + 2], 16) / 255.0 for i in (0, 2, 4))


def _to_gray_rgb(img):
    img = np.asarray(img, dtype=np.float32)
    if img.ndim == 2:
        img = np.stack([img] * 3, axis=-1)
    if img.max() > 1.0 + 1e-6:
        img = img / 255.0
    return np.clip(img, 0.0, 1.0)


def _resize_to(arr, h, w, nearest=False):
    if arr.shape[:2] == (h, w):
        return arr
    interp = cv2.INTER_NEAREST if nearest else cv2.INTER_LINEAR
    return cv2.resize(arr, (w, h), interpolation=interp)


def _blend(base_rgb, region_bool, hex_color, alpha=MASK_ALPHA):
    color = np.array(_hex_to_rgb01(hex_color), dtype=np.float32)
    out = base_rgb.copy()
    out[region_bool] = (1 - alpha) * out[region_bool] + alpha * color
    return out


def _solid_mask_overlay(base_rgb, mask, hex_color):
    h, w = base_rgb.shape[:2]
    mask_bool = _resize_to((mask > 0).astype(np.uint8), h, w, nearest=True).astype(bool)
    return _blend(base_rgb, mask_bool, hex_color)


def _heatmap_overlay(base_rgb, heat, cmap_name, vmin, vmax):
    h, w = base_rgb.shape[:2]
    heat = _resize_to(np.asarray(heat, dtype=np.float32), h, w)
    heat_n = np.clip((heat - vmin) / (vmax - vmin + 1e-9), 0.0, 1.0)
    colored = matplotlib.colormaps[cmap_name](heat_n)[..., :3].astype(np.float32)
    out = (1 - HEATMAP_ALPHA) * base_rgb + HEATMAP_ALPHA * colored
    return np.clip(out, 0.0, 1.0)


def _error_map(base_rgb, pred_mask, gt_mask):
    h, w = base_rgb.shape[:2]
    pred = _resize_to((pred_mask > 0).astype(np.uint8), h, w, nearest=True).astype(bool)
    gt = _resize_to((gt_mask > 0).astype(np.uint8), h, w, nearest=True).astype(bool)

    tp = pred & gt
    fp = pred & ~gt
    fn = ~pred & gt

    out = base_rgb.copy()
    out = _blend(out, fn, PALETTE["recall"])
    out = _blend(out, fp, PALETTE["precision"])
    out = _blend(out, tp, PALETTE["f1"])
    return out


def _binary_entropy(p):
    eps = 1e-7
    p = np.clip(np.asarray(p, dtype=np.float32), eps, 1 - eps)
    return -(p * np.log2(p) + (1 - p) * np.log2(1 - p))


def _contour_overlay(base_rgb, mask_color_pairs, thickness):
    h, w = base_rgb.shape[:2]
    img_u8 = (np.clip(base_rgb, 0.0, 1.0) * 255).astype(np.uint8).copy()
    for mask, hex_color in mask_color_pairs:
        if mask is None:
            continue
        mask_b = _resize_to((mask > 0).astype(np.uint8), h, w, nearest=True)
        contours, _ = cv2.findContours(mask_b, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        color_rgb = tuple(int(round(c * 255)) for c in _hex_to_rgb01(hex_color))
        cv2.drawContours(img_u8, contours, -1, color_rgb, thickness)
    return img_u8.astype(np.float32) / 255.0


def _bbox_overlay(base_rgb, bbox_color_pairs, thickness):
    img_u8 = (np.clip(base_rgb, 0.0, 1.0) * 255).astype(np.uint8).copy()
    for bboxes, hex_color in bbox_color_pairs:
        color_rgb = tuple(int(round(c * 255)) for c in _hex_to_rgb01(hex_color))
        for bbox in (bboxes or []):
            x1, y1, x2, y2 = map(int, bbox)
            cv2.rectangle(img_u8, (x1, y1), (x2, y2), color_rgb, thickness)
    return img_u8.astype(np.float32) / 255.0


def _figsize_for(h, w, size_in):
    aspect = w / max(h, 1)
    if aspect >= 1:
        return (size_in, size_in / aspect)
    return (size_in * aspect, size_in)


def _save_panel(img, path, figsize, dpi):
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    ax.set_axis_off()
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    ax.imshow(img)
    fig.savefig(path, format="pdf", bbox_inches="tight", pad_inches=0)
    plt.close(fig)


def save_shared_legends(output_root):
    legends_dir = os.path.join(output_root, "legends")
    os.makedirs(legends_dir, exist_ok=True)

    colorbars = [
        ("attention_colorbar.pdf", CAM_CMAP, 0.0, 1.0, "CAM / attention activation"),
        ("posterior_colorbar.pdf", POSTERIOR_CMAP, 0.0, 1.0, "Posterior probability"),
        ("uncertainty_colorbar.pdf", UNCERTAINTY_CMAP, 0.0, 1.0, "Uncertainty (bits)"),
        ("llr_colorbar.pdf", LLR_CMAP, -LLR_CLIP, LLR_CLIP, "Log-likelihood ratio"),
    ]
    for filename, cmap_name, vmin, vmax, label in colorbars:
        path = os.path.join(legends_dir, filename)
        if os.path.exists(path):
            continue
        fig, ax = plt.subplots(figsize=(0.55, 2.6))
        norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
        cb = mcolorbar.ColorbarBase(ax, cmap=matplotlib.colormaps[cmap_name], norm=norm, orientation="vertical")
        cb.set_label(label, fontsize=8)
        cb.ax.tick_params(labelsize=7)
        fig.savefig(path, format="pdf", bbox_inches="tight", pad_inches=0.02)
        plt.close(fig)

    swatches_path = os.path.join(legends_dir, "category_key.pdf")
    if not os.path.exists(swatches_path):
        entries = [
            ("Ground truth / True Positive", PALETTE["f1"]),
            ("Attention (CAM) mask / False Negative", PALETTE["recall"]),
            ("Posterior-refined mask / False Positive", PALETTE["precision"]),
        ]
        fig, ax = plt.subplots(figsize=(3.4, 0.35 * len(entries) + 0.2))
        ax.set_xlim(0, 1)
        ax.set_ylim(0, len(entries))
        ax.set_axis_off()
        for i, (label, hex_color) in enumerate(entries):
            y = len(entries) - i - 1
            ax.add_patch(plt.Rectangle((0.0, y + 0.15), 0.08, 0.7, color=hex_color))
            ax.text(0.12, y + 0.5, label, va="center", fontsize=9)
        fig.savefig(swatches_path, format="pdf", bbox_inches="tight", pad_inches=0.02)
        plt.close(fig)


def save_paper_figures(result, output_root, scan_id, cfg):
    save_shared_legends(output_root)

    dpi = cfg.get("paper_vis_dpi", DEFAULT_DPI)
    size_in = cfg.get("paper_vis_size_in", DEFAULT_SIZE_IN)
    thickness = cfg.get("paper_vis_line_thickness", DEFAULT_LINE_THICKNESS)

    case_dir = os.path.join(output_root, scan_id)
    os.makedirs(case_dir, exist_ok=True)

    def panel(img, label):
        _save_panel(img, os.path.join(case_dir, f"{label}.pdf"), figsize, dpi)

    base_rgb = _to_gray_rgb(result["input"])
    h, w = base_rgb.shape[:2]
    figsize = _figsize_for(h, w, size_in)
    gt_mask = result.get("ground_truth_mask", np.zeros((h, w), dtype=np.uint8))
    gt_bboxes = result.get("gt_bboxes")

    panel(base_rgb, "01_original_ct")

    cam = result.get("cam")
    cam_mask = result.get("cam_mask")
    if cam is not None:
        panel(_heatmap_overlay(base_rgb, cam, CAM_CMAP, 0.0, 1.0), "02_attention_map")
    if cam_mask is not None:
        panel(_solid_mask_overlay(base_rgb, cam_mask, PALETTE["recall"]), "03_attention_map_binarized")

    posterior = result.get("hu_posterior")
    llr = result.get("hu_llr")
    final_mask = cam_mask
    if llr is not None:
        panel(_heatmap_overlay(base_rgb, np.clip(llr, -LLR_CLIP, LLR_CLIP), LLR_CMAP, -LLR_CLIP, LLR_CLIP), "04_llr_map")
    if posterior is not None:
        panel(_heatmap_overlay(base_rgb, posterior, POSTERIOR_CMAP, 0.0, 1.0), "05_posterior_map")
        panel(_heatmap_overlay(base_rgb, _binary_entropy(posterior), UNCERTAINTY_CMAP, 0.0, 1.0), "06_posterior_uncertainty")

        final_mask = result.get("hu_mask")
        if final_mask is not None:
            panel(_solid_mask_overlay(base_rgb, final_mask, PALETTE["precision"]), "07_posterior_map_binarized")

    panel(_solid_mask_overlay(base_rgb, gt_mask, PALETTE["f1"]), "08_ground_truth_mask")

    if final_mask is not None:
        boundary = _contour_overlay(
            base_rgb,
            [(gt_mask, PALETTE["f1"]), (final_mask, PALETTE["precision"])],
            thickness,
        )
        panel(boundary, "09_boundary_comparison")

    prompt_bboxes = result.get("hu_bboxes") or result.get("cam_bboxes") or []
    if gt_bboxes or prompt_bboxes:
        bbox_img = _bbox_overlay(
            base_rgb,
            [(gt_bboxes, PALETTE["f1"]), (prompt_bboxes, PALETTE["precision"])],
            thickness,
        )
        panel(bbox_img, "10_bbox_overlay")

    if final_mask is not None:
        panel(_error_map(base_rgb, final_mask, gt_mask), "11_error_analysis_map")
