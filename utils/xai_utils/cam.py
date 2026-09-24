import os

import torch
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import BinaryClassifierOutputTarget
from tqdm import tqdm

from .bbox import get_bounding_box
from .cam_utils import CAM_METHODS, create_resize_only, load_and_resize_mask, normalize_image
from .dino_utils import generate_dinov2_cam_batch, is_dinov2_model
from .visualization import plot_result


def process_batch_item(cfg, input_tensor, cam_map, target, filename, batch_idx, idx, pred, original_ct, feature_type='cam'):
    input_image = normalize_image(input_tensor)
    ground_truth_mask = load_and_resize_mask(cfg.get('masks_dir'), filename, cam_map.shape)

    return {
        'cam': cam_map,
        'feature_type': feature_type,
        'ground_truth_mask': ground_truth_mask,
        'gt_bboxes': get_bounding_box(ground_truth_mask),
        'input_tensor': input_tensor,
        'input': input_image,
        'target': target.item() if target is not None else None,
        'pred': pred.item(),
        'filename': filename if filename is not None else f'image_{batch_idx}_{idx}.png',
        'original_ct': original_ct,
        'overlay': show_cam_on_image(input_image, cam_map, use_rgb=True)
    }


def generate_cam_batch(cfg, model, dataloader, target_layers, max_inferences=None, device=None):
    device = device or (torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))
    model.to(device).eval()

    if is_dinov2_model(cfg, model):
        return generate_dinov2_cam_batch(cfg, model, dataloader, target_layers, max_inferences, device)

    if len(target_layers) != 1:
        raise ValueError(f"Exactly one target layer is supported, got {target_layers}")
    layer = model.model.blocks[target_layers[0]]
    results = []
    inference_count = 0

    cam_method = cfg.get("cam_method", "HiResCAM")
    print(f"info: cam method is:{cam_method}")
    for batch_idx, data in tqdm(enumerate(dataloader), total=len(dataloader), desc="CAM Inference"):
        input_tensor = data['input_data'].to(device)
        original_cts = data['input_cts']
        targets = data.get('targets', None)
        filenames = data.get('filename', [None] * input_tensor.shape[0])

        with torch.no_grad():
            outputs = model(input_tensor)
            preds = torch.sigmoid(outputs).gt(0.5).float()
        labels = [BinaryClassifierOutputTarget(int(target.item())) for target in targets]

        cam = CAM_METHODS[cam_method](model=model, target_layers=[layer])
        cam_maps = cam(input_tensor=input_tensor, targets=labels)

        for i in range(input_tensor.shape[0]):
            if max_inferences is not None and inference_count >= max_inferences:
                break

            pred, target = preds[i], targets[i]
            if pred.item() == 0 and target.item() == 0:
                continue

            image_np = original_cts[i].cpu().numpy() if isinstance(original_cts[i], torch.Tensor) else original_cts[i]
            ct_resized = create_resize_only(cfg)(image=image_np)['image']

            result = process_batch_item(cfg, input_tensor[i], cam_maps[i], targets[i], filenames[i], batch_idx, i, pred, original_ct=ct_resized)
            results.append(result)

            inference_count += 1

    print(f"[INFO] Processed {inference_count} inferences.")
    return results


def visualize_results(results, visualize_dir, cfg):
    save_dir = visualize_dir
    os.makedirs(save_dir, exist_ok=True)

    for result in tqdm(results, desc="Visualizing Results"):
        filename = os.path.splitext(result['filename'])[0] + '.png'
        save_path = os.path.join(save_dir, filename)
        plot_result(result, save_path, cfg)

    print(f"[INFO] Saved GradCAM figures to {save_dir}")
