import contextlib

import cv2
import numpy as np
import torch
from pytorch_grad_cam.utils.model_targets import BinaryClassifierOutputTarget
from tqdm import tqdm

from .cam_utils import CAM_METHODS, create_resize_only


def is_dinov2_model(cfg, model=None):
    if model is not None:
        cls_name = model.__class__.__name__.lower()
        if "dinov2" in cls_name:
            return True
        base = getattr(model, "model", None)
        if base is not None and hasattr(base, "blocks") and hasattr(base, "patch_embed"):
            return True
    return "dinov2" in cfg.get("backbone", "").lower()


def get_num_prefix_tokens(model, cfg):
    base = getattr(model, "model", model)

    if hasattr(base, "n_storage_tokens"):
        return 1 + int(base.n_storage_tokens)
    if hasattr(base, "num_register_tokens"):
        return 1 + int(base.num_register_tokens)
    if "num_register_tokens" in cfg:
        return 1 + int(cfg["num_register_tokens"])

    model_name = getattr(model, "model_name", "") or cfg.get("backbone", "")
    if "reg" in model_name.lower():
        return 1 + 4

    return 1


def get_patch_size(model):
    base = getattr(model, "model", model)
    if hasattr(base, "patch_embed") and hasattr(base.patch_embed, "patch_size"):
        ps = base.patch_embed.patch_size
        return ps[0] if isinstance(ps, (tuple, list)) else ps
    return 14


def get_dinov2_layers(model, target_layers):
    blocks = model.model.blocks
    if target_layers is None:
        return [blocks[-1].norm1]

    layers = []
    for layer_idx in target_layers:
        block = blocks[layer_idx] if isinstance(layer_idx, int) else layer_idx
        layer = block.norm1 if hasattr(block, "norm1") else block
        layers.append(layer)
    return layers


def make_vit_reshape_transform(height, width, num_prefix_tokens):
    def reshape_transform(tensor):
        result = tensor[:, num_prefix_tokens:, :]
        expected = height * width
        if result.shape[1] != expected:
            raise ValueError(
                f"[DINOv2 reshape_transform] Token count mismatch: got {result.shape[1]} "
                f"patch tokens but expected {height}x{width}={expected}. "
                f"Check num_prefix_tokens (currently {num_prefix_tokens}) and image_size/patch_size."
            )
        result = result.reshape(tensor.size(0), height, width, tensor.size(-1))
        result = result.permute(0, 3, 1, 2)
        return result
    return reshape_transform


def _patched_attn_forward(self, x, *args, **kwargs):
    B, N, C = x.shape
    qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
    q, k, v = qkv[0], qkv[1], qkv[2]
    attn = (q @ k.transpose(-2, -1)) * self.scale
    attn = attn.softmax(dim=-1)
    self._cached_attn = attn.detach()

    if isinstance(self.attn_drop, torch.nn.Module):
        attn = self.attn_drop(attn)
    else:
        attn = torch.nn.functional.dropout(attn, p=float(self.attn_drop), training=self.training)

    x = (attn @ v).transpose(1, 2).reshape(B, N, C)
    x = self.proj(x)
    x = self.proj_drop(x)
    return x


@contextlib.contextmanager
def capture_attention(blocks):
    originals = [blk.attn.forward for blk in blocks]
    for blk in blocks:
        blk.attn.forward = _patched_attn_forward.__get__(blk.attn, blk.attn.__class__)
    try:
        yield
    finally:
        for blk, orig_forward in zip(blocks, originals):
            blk.attn.forward = orig_forward
            if hasattr(blk.attn, "_cached_attn"):
                del blk.attn._cached_attn


def _patch_scores_to_image(patch_scores, grid_h, grid_w, img_height, img_width):
    B = patch_scores.shape[0]
    grid = patch_scores.detach().float().cpu().numpy().reshape(B, grid_h, grid_w)
    out = np.zeros((B, img_height, img_width), dtype=np.float32)
    for i in range(B):
        g = grid[i]
        g = (g - g.min()) / (g.max() - g.min() + 1e-8)
        out[i] = cv2.resize(g, (img_width, img_height), interpolation=cv2.INTER_LINEAR)
    return out


def compute_raw_attention_batch(model, input_tensor, num_prefix_tokens, grid_h, grid_w,
                                 img_height, img_width, head_reduction="mean"):
    base = getattr(model, "model", model)
    last_block = base.blocks[-1]

    with capture_attention([last_block]):
        with torch.no_grad():
            _ = model(input_tensor)
        attn = last_block.attn._cached_attn

    cls_to_patch = attn[:, :, 0, num_prefix_tokens:]
    cls_to_patch = cls_to_patch.max(dim=1).values if head_reduction == "max" else cls_to_patch.mean(dim=1)

    return _patch_scores_to_image(cls_to_patch, grid_h, grid_w, img_height, img_width)


def compute_attention_rollout_batch(model, input_tensor, num_prefix_tokens, grid_h, grid_w,
                                     img_height, img_width, head_reduction="mean", discard_ratio=0.0,
                                     layer_indices=None):
    base = getattr(model, "model", model)
    all_blocks = base.blocks

    if layer_indices is not None:
        num_layers = len(all_blocks)
        valid_indices = [i for i in layer_indices if 0 <= i < num_layers]
        if not valid_indices:
            raise ValueError(f"No valid layer indices provided. Valid range: 0-{num_layers-1}")
        blocks = [all_blocks[i] for i in valid_indices]
        print(f"[INFO] Using layers {valid_indices} for attention rollout (out of {num_layers} total)")
    else:
        blocks = all_blocks
        print(f"[INFO] Using all {len(blocks)} layers for attention rollout")

    with capture_attention(blocks):
        with torch.no_grad():
            _ = model(input_tensor)
        attentions = [blk.attn._cached_attn for blk in blocks]

    B, _, N, _ = attentions[0].shape
    device = attentions[0].device
    rollout = torch.eye(N, device=device).unsqueeze(0).expand(B, -1, -1).clone()
    eye = torch.eye(N, device=device).unsqueeze(0)

    for attn in attentions:
        attn_h = attn.max(dim=1).values if head_reduction == "max" else attn.mean(dim=1)

        if discard_ratio > 0:
            flat = attn_h.reshape(B, -1)
            k = int(flat.shape[-1] * discard_ratio)
            if k > 0:
                _, low_idx = flat.topk(k, dim=-1, largest=False)
                flat = flat.scatter(-1, low_idx, 0.0)
                attn_h = flat.reshape(B, N, N)

        attn_aug = attn_h + eye
        attn_aug = attn_aug / attn_aug.sum(dim=-1, keepdim=True)
        rollout = attn_aug @ rollout

    cls_to_patch = rollout[:, 0, num_prefix_tokens:]
    return _patch_scores_to_image(cls_to_patch, grid_h, grid_w, img_height, img_width)


def generate_dinov2_cam_batch(cfg, model, dataloader, target_layers, max_inferences=None, device=None):
    from .cam import process_batch_item

    device = device or (torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))
    model.to(device).eval()

    dino_cfg = cfg.get("dino", {})
    use_cam = dino_cfg.get("use_cam", False)
    use_attention = dino_cfg.get("use_attention", False)
    use_rollout = dino_cfg.get("use_rollout", False)

    if not any([use_cam, use_attention, use_rollout]):
        raise ValueError(
            "At least one of use_cam, use_attention, or use_rollout must be True in dino config"
        )

    if sum([use_cam, use_attention, use_rollout]) > 1:
        raise ValueError(
            "Only one of use_cam, use_attention, or use_rollout can be True in dino config"
        )

    img_height, img_width = cfg.get('image_size', [518, 518])
    patch_size = get_patch_size(model)
    grid_h, grid_w = img_height // patch_size, img_width // patch_size
    num_prefix_tokens = get_num_prefix_tokens(model, cfg)

    if use_cam:
        feature_name = "cam"
        print("[INFO] Using GradCAM/HiResCAM/AblationCAM method")
    elif use_attention:
        feature_name = "attention"
        print("[INFO] Using raw attention method (last-layer CLS-to-patch self-attention)")
    else:
        feature_name = "rollout"
        print("[INFO] Using attention rollout method")

    print(f"[INFO] DINOv2 features={feature_name}: grid={grid_h}x{grid_w}, patch_size={patch_size}, "
          f"num_prefix_tokens={num_prefix_tokens}")

    if use_cam:
        reshape_transform = make_vit_reshape_transform(grid_h, grid_w, num_prefix_tokens)
        layers = get_dinov2_layers(model, target_layers)
        if len(layers) != 1:
            raise ValueError(f"Exactly one target layer is supported, got {target_layers}")
        cam_method = cfg.get("cam_method", "HiResCAM")
        cam_class = CAM_METHODS[cam_method]
        print(f"[INFO] Using CAM method: {cam_method}")
    elif use_attention:
        head_reduction = dino_cfg.get("dino_head_reduction", "mean")
        print(f"[INFO] Head reduction: {head_reduction}")
    else:
        head_reduction = dino_cfg.get("dino_head_reduction", "mean")
        print(f"[INFO] Head reduction: {head_reduction}")
        rollout_layers = dino_cfg.get("rollout_layers", None)
        if rollout_layers is not None:
            print(f"[INFO] Using specific layers for rollout: {rollout_layers}")

    results = []
    inference_count = 0

    for batch_idx, data in tqdm(enumerate(dataloader), total=len(dataloader), desc="DINOv2 CAM Inference"):
        input_tensor = data['input_data'].to(device)
        original_cts = data['input_cts']
        targets = data.get('targets', None)
        filenames = data.get('filename', [None] * input_tensor.shape[0])

        with torch.no_grad():
            outputs = model(input_tensor)
            preds = torch.sigmoid(outputs).gt(0.5).float()
        labels = [BinaryClassifierOutputTarget(int(t.item())) for t in targets]

        if use_cam:
            cam = cam_class(model=model, target_layers=[layers[0]], reshape_transform=reshape_transform)
            cam_maps = cam(input_tensor=input_tensor, targets=labels)
        elif use_attention:
            cam_maps = compute_raw_attention_batch(
                model, input_tensor, num_prefix_tokens, grid_h, grid_w,
                img_height, img_width, head_reduction=head_reduction
            )
        else:
            cam_maps = compute_attention_rollout_batch(
                model, input_tensor, num_prefix_tokens, grid_h, grid_w,
                img_height, img_width, head_reduction=head_reduction,
                discard_ratio=dino_cfg.get("rollout_discard_ratio", 0.0),
                layer_indices=dino_cfg.get("rollout_layers", None)
            )

        for i in range(input_tensor.shape[0]):
            if max_inferences is not None and inference_count >= max_inferences:
                break

            pred, target = preds[i], targets[i] if targets is not None else None
            if target is not None and pred.item() == 0 and target.item() == 0:
                continue

            image_np = original_cts[i].cpu().numpy() if isinstance(original_cts[i], torch.Tensor) else original_cts[i]
            ct_resized = create_resize_only(cfg)(image=image_np)['image']

            result = process_batch_item(
                cfg, input_tensor[i], cam_maps[i], target, filenames[i], batch_idx, i, pred,
                original_ct=ct_resized, feature_type=feature_name
            )
            results.append(result)
            inference_count += 1

    method_label = cam_method if use_cam else f"{feature_name} (heads={head_reduction})"
    print(f"[INFO] Processed {inference_count} DINOv2 inferences with {method_label}.")
    return results
