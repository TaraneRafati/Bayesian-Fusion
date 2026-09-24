from metrics.basemetric import Metric
import torch
import numpy as np


class BinaryRecall(Metric):
    def __init__(self, name='binary_recall', threshold=0.5, from_logits=True):
        super(BinaryRecall, self).__init__(name=f'{name}@{threshold}')
        self.threshold = threshold
        self.true_positive = 0.0
        self.false_negative = 0.0
        self.from_logits = from_logits

    def update_states(self, y_pred, y_true, sample_weights=None):
        if sample_weights is None:
            sample_weights = torch.ones_like(y_pred)

        if self.from_logits:
            y_pred = torch.sigmoid(y_pred)

        y_pred = (y_pred.detach().cpu().numpy() >= self.threshold).astype('int32')
        y_true = y_true.detach().cpu().numpy().astype('int32')

        self.true_positive += np.sum((y_pred == y_true) & (y_true == 1))
        self.false_negative += np.sum((y_pred != y_true) & (y_true == 1))

    def reset_states(self):
        self.true_positive = 0.0
        self.false_negative = 0.0

    def result(self):
        return self.true_positive / (self.true_positive + self.false_negative + 1e-9)
