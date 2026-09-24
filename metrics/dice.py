import torch
import numpy as np
from metrics.basemetric import Metric


class BinaryDice(Metric):
    def __init__(self, name='binary_dice', threshold=0.5, from_logits=True):
        super(BinaryDice, self).__init__(name=f'{name}@{threshold}')
        self.threshold = threshold
        self.true_positive = 0.0
        self.false_positive = 0.0
        self.false_negative = 0.0
        self.from_logits = from_logits

    def update_states(self, y_pred, y_true, sample_weights=None):
        if sample_weights is None:
            sample_weights = torch.ones_like(y_pred)

        if self.from_logits:
            y_pred = torch.sigmoid(y_pred)

        y_pred = (y_pred.detach().cpu().numpy() >= self.threshold).astype('int32')
        y_true = y_true.detach().cpu().numpy().astype('int32')

        y_pred = y_pred.flatten()
        y_true = y_true.flatten()

        self.true_positive += np.sum((y_pred == y_true) & (y_true == 1))
        self.false_positive += np.sum((y_pred != y_true) & (y_pred == 1))
        self.false_negative += np.sum((y_pred != y_true) & (y_true == 1))

    def reset_states(self):
        self.true_positive = 0.0
        self.false_positive = 0.0
        self.false_negative = 0.0

    def result(self):
        intersection = self.true_positive
        return 2 * intersection / (2 * intersection + self.false_positive + self.false_negative + 1e-9)


class BboxDice(Metric):
    def __init__(self, name='bbox_dice', mask_size=(512, 512)):
        super(BboxDice, self).__init__(name=name)
        self.intersection = 0.0
        self.area_sum = 0.0
        self.mask_size = mask_size

    def boxes_to_mask(self, boxes):
        mask = np.zeros(self.mask_size, dtype=np.uint8)
        boxes = np.array(boxes).reshape(-1, 4)
        for box in boxes:
            x1, y1, x2, y2 = map(int, box)
            mask[y1:y2, x1:x2] = 1
        return mask

    def update_states(self, cam_boxes, gt_boxes, task=None, sample_weights=None):
        if not cam_boxes and not gt_boxes:
            return

        cam_mask = self.boxes_to_mask(cam_boxes)
        gt_mask = self.boxes_to_mask(gt_boxes)

        intersection = np.logical_and(cam_mask, gt_mask).sum()
        area_sum = cam_mask.sum() + gt_mask.sum()

        if sample_weights is None:
            sample_weights = 1.0

        self.intersection += intersection * sample_weights
        self.area_sum += area_sum * sample_weights

    def reset_states(self):
        self.intersection = 0.0
        self.area_sum = 0.0

    def result(self):
        return (2 * self.intersection) / (self.area_sum + 1e-9)
