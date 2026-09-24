from models.tasks.classification.dinov2 import DINOv2ViTS14
from models.tasks.classification.efficientnet import EfficientNetV2B3
from models.tasks.classification.resnet import ResNet18

backbones = {
    "EfficientNetV2B3": EfficientNetV2B3,
    "ResNet18": ResNet18,
    "DINOv2ViTS14": DINOv2ViTS14,
}

__all__ = (
    "backbones",
)
