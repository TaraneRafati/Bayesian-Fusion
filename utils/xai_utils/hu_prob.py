import cv2
import numpy as np

from .skull_stripping import extract_intracranial_mask


def compute_gmm_log_likelihood(ct_slice, weights, means, stds):
    weights = np.array(weights)
    means = np.array(means)
    stds = np.array(stds)
    eps = 1e-6
    ct_expanded = ct_slice[..., np.newaxis]

    log_component_likelihoods = (
        np.log(weights + eps)
        - 0.5 * ((ct_expanded - means) ** 2) / (stds ** 2 + eps)
        - np.log(stds + eps)
        - 0.5 * np.log(2 * np.pi)
    )

    m_c = np.max(log_component_likelihoods, axis=-1, keepdims=True)
    stabilized_sum = np.sum(np.exp(log_component_likelihoods - m_c), axis=-1)

    return np.squeeze(m_c, axis=-1) + np.log(stabilized_sum + eps)


def generate_fallback_skull_prior(valid_brain_mask, sigma=15.0):
    outside_mask = (valid_brain_mask == 0).astype(np.uint8)
    dist_transform = cv2.distanceTransform(valid_brain_mask.astype(np.uint8), cv2.DIST_L2, 5)
    spatial_prior_k = np.exp(-(dist_transform ** 2) / (2 * (sigma ** 2)))
    spatial_prior_k = np.where(outside_mask == 1, 1.0, spatial_prior_k)
    return spatial_prior_k


def compute_multiclass_log_priors(cam_prob_h, spatial_prior_k, eps=1e-6):
    total_prior = cam_prob_h + spatial_prior_k
    scale_factor = np.where(total_prior > 1.0 - eps, (1.0 - 2 * eps) / (total_prior + 1e-12), 1.0)

    cam_prob_h = np.clip(cam_prob_h * scale_factor, eps, 1.0 - eps)
    spatial_prior_k = np.clip(spatial_prior_k * scale_factor, eps, 1.0 - eps)
    prob_b = np.clip(1.0 - (cam_prob_h + spatial_prior_k), eps, 1.0 - eps)

    log_g_H = np.log(cam_prob_h)
    log_g_K = np.log(spatial_prior_k)
    log_g_B = np.log(prob_b)

    return log_g_H, log_g_K, log_g_B


def hysteresis_threshold_posterior(posterior_map, tau_high, tau_low, domain_mask=None):
    seed = (posterior_map >= tau_high).astype(np.uint8)
    candidate = (posterior_map >= tau_low).astype(np.uint8)

    if domain_mask is not None:
        domain_binary = (domain_mask > 0).astype(np.uint8)
        candidate = cv2.bitwise_and(candidate, domain_binary)

    num_labels, labels = cv2.connectedComponents(candidate)
    grown = np.zeros_like(seed)

    for label_id in range(1, num_labels):
        component = (labels == label_id)
        if np.any(component & (seed > 0)):
            grown[component] = 1

    return grown


def probabilistic_hu_cam_fusion_gmm(cam_prob_h, original_ct, cfg):
    hu_cfg = cfg.get("hu_filter", {})
    fusion_threshold = hu_cfg.get("fusion_threshold", 0.5)
    sigma_prior = hu_cfg.get("spatial_prior_k_sigma", 15.0)

    valid_brain_mask = extract_intracranial_mask(original_ct)

    gmm_params = cfg["gmm_parameters"]

    log_p_I_H = compute_gmm_log_likelihood(original_ct, **gmm_params["hemorrhage"])
    log_p_I_K = compute_gmm_log_likelihood(original_ct, **gmm_params["skull"])
    log_p_I_B = compute_gmm_log_likelihood(original_ct, **gmm_params["background"])

    spatial_prior_k = generate_fallback_skull_prior(valid_brain_mask, sigma=sigma_prior)

    log_g_H, log_g_K, log_g_B = compute_multiclass_log_priors(cam_prob_h, spatial_prior_k)

    a_H = log_p_I_H + log_g_H
    a_K = log_p_I_K + log_g_K
    a_B = log_p_I_B + log_g_B

    logits_max = np.maximum(a_H, np.maximum(a_K, a_B))
    exp_a_H = np.exp(a_H - logits_max)
    exp_a_K = np.exp(a_K - logits_max)
    exp_a_B = np.exp(a_B - logits_max)
    posterior_hemorrhage = exp_a_H / (exp_a_H + exp_a_K + exp_a_B + 1e-12)

    tau_low = hu_cfg.get("tau_low", 0.35)
    use_hysteresis = hu_cfg.get("use_hysteresis", True)

    if use_hysteresis:
        brain_mask = (valid_brain_mask > 0).astype(np.uint8)
        final_mask = hysteresis_threshold_posterior(
            posterior_hemorrhage,
            tau_high=fusion_threshold,
            tau_low=tau_low,
            domain_mask=brain_mask
        )
    else:
        final_mask = (posterior_hemorrhage > fusion_threshold).astype(np.uint8)

    return final_mask, posterior_hemorrhage, log_p_I_H - np.logaddexp(log_p_I_K, log_p_I_B)
