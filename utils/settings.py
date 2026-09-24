import copy

IMAGE_SIZES = {
    "EfficientNetV2B3": 512,
    "ResNet18": 512,
    "DINOv2ViTS14": 518,
}

TRAIN_AUGMENTATIONS = {
    "flip": {"kwargs": {"p": 0.5}, "args": {"apply_on_pos_only": 0}},
    "rotate": {"kwargs": {"p": 0.0, "limit": 15}, "args": {"apply_on_pos_only": 0}},
    "shift_scale": {
        "kwargs": {"p": 0.5, "shift_limit": 0.1, "scale_limit": 0.1, "rotate_limit": 15},
        "args": {"apply_on_pos_only": 0},
    },
    "color_jitter": {
        "kwargs": {"p": 0.0, "brightness": 0.2, "contrast": 0.2, "saturation": 0.2, "hue": 0.2},
        "args": {"apply_on_pos_only": 0},
    },
}


def base_config(backbone, image_size=None, batch_size=16):
    size = image_size or IMAGE_SIZES[backbone]
    return copy.deepcopy({
        "plugin": "brainhemorrhage",
        "task": "binary classification",
        "backbone": backbone,
        "image_size": [size, size],
        "batch_size": batch_size,
        "loss": "BCEWithLogitsLoss",
        "loss_args": {},
        "optimizer": "Adam",
        "input_mode": "Repeat",
        "window_widths": [80],
        "window_centers": [40],
        "pretrained": 1,
        "frozen_layers_ratio": 0,
        "train_augmentations": TRAIN_AUGMENTATIONS,
        "val_augmentations": {},
    })
