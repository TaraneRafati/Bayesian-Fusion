import os

os.environ["NO_ALBUMENTATIONS_UPDATE"] = "1"

import albumentations as A
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from pytorch_grad_cam import AblationCAM, GradCAM, HiResCAM

from .bbox import get_bounding_box
from .cam_metrics import compute_segmentation_metrics
from .hu import run_hu_refinement
from .subtype_metrics import compute_subtype_segmentation_metrics

CAM_METHODS = {
    "GradCAM": GradCAM,
    "HiResCAM": HiResCAM,
    "AblationCAM": AblationCAM,
}


def create_resize_only(cfg):
    h, w = cfg['image_size']
    return A.Compose([
        A.Resize(height=h, width=w)
    ])


def load_and_resize_mask(masks_dir, filename, target_size, verbose=True):
    if not filename:
        return np.zeros(target_size, dtype=np.uint8)

    mask_path = os.path.join(masks_dir, filename)
    if not os.path.exists(mask_path):
        if verbose:
            print(f"Mask file not found for {filename}, using empty mask.")
        return np.zeros(target_size, dtype=np.uint8)

    mask = (np.load(mask_path) > 0).astype(np.uint8)
    gt_tensor = torch.from_numpy(mask).unsqueeze(0).unsqueeze(0).float()
    gt_tensor = F.interpolate(gt_tensor, size=target_size, mode='nearest')
    return gt_tensor.squeeze().byte().numpy()


def binarize_cam_mask(cam_map, cam_threshold, out_size=None):
    if out_size is not None and cam_map.shape != out_size:
        cam_map = cv2.resize(cam_map, (out_size[1], out_size[0]), interpolation=cv2.INTER_LINEAR)

    return (cam_map > cam_threshold).astype(np.uint8)


def normalize_image(input_tensor):
    input_image = np.transpose(input_tensor.detach().cpu().numpy().astype(np.float32), (1, 2, 0))
    if input_image.max() > input_image.min():
        input_image = (input_image - input_image.min()) / (input_image.max() - input_image.min())
    else:
        input_image = np.zeros_like(input_image, dtype=np.float32)
    return input_image


def cam_mask_refine(results, cfg):
    fixed_thresh = cfg.get("cam_threshold", 0.5)
    dino_attn_thresh = cfg.get("dino_attention_threshold", fixed_thresh)

    for r in results:
        feature_type = r.get('feature_type', 'cam')
        thresh = dino_attn_thresh if feature_type in ("attention", "rollout") else fixed_thresh

        mask = binarize_cam_mask(r['cam'], thresh)
        r['cam_mask'] = (mask > 0).astype(np.uint8)
        r['cam_bboxes'] = get_bounding_box(r['cam_mask'])

        if cfg.get("hu_filter", {}).get("enabled"):
            run_hu_refinement(r, cfg)

        if 'ground_truth_mask' in r:
            r['ground_truth_mask'] = (r['ground_truth_mask'] > 0).astype(np.uint8)

    compute_segmentation_metrics(results, cfg)

    compute_subtype_segmentation_metrics(results, cfg)

    return results
