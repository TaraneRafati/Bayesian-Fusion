import numpy as np

from .bbox import get_bounding_box
from .hu_prob import probabilistic_hu_cam_fusion_gmm


def run_hu_refinement(result, cfg):
    refined_mask, posterior_map, llr_map = probabilistic_hu_cam_fusion_gmm(result['cam'], result.get('original_ct'), cfg)
    result['hu_posterior'] = posterior_map
    result['hu_llr'] = llr_map
    result['hu_mask'] = refined_mask
    result['hu_bboxes'] = get_bounding_box(refined_mask)
    return result


def evaluate_hu_performance(results, eval_func):
    print("\n" + "=" * 45 + "\n[HU FILTERING METRICS]\n" + "=" * 45)
    eval_func(results, mask_key='hu_mask', bbox_key='hu_bboxes')


def get_hu_plot_config(result, create_mask_overlay_fn):
    h, w = result['ground_truth_mask'].shape[:2]
    configs = []

    posterior_map = result.get('hu_posterior', None)
    if posterior_map is not None:
        llr_map = result.get('hu_llr', None)
        configs.append((
            np.clip(llr_map, -20.0, 20.0),
            "Log-Likelihood Ratio (LLR)"
        ))

        configs.append((
            np.clip(posterior_map, 0.0, 1.0),
            "Probabilistic Posterior Landscape"
        ))

        mask = result.get('hu_mask', np.zeros((h, w), dtype=np.uint8))
        configs.append((
            create_mask_overlay_fn(result['ground_truth_mask'], mask, h, w),
            "Probabilistic Fusion vs GT"
        ))

    return configs
