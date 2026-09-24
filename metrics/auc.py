from sklearn.metrics import roc_auc_score
import torch
import numpy as np
from metrics.basemetric import Metric
import warnings
from sklearn.exceptions import UndefinedMetricWarning


class BinaryAUC(Metric):
    def __init__(self, name='binary_auc', from_logits=True, **kwargs):
        super(BinaryAUC, self).__init__(name=name)
        self.from_logits = from_logits
        self.reset_states()

    def update_states(self, y_pred, y_true):
        if self.from_logits:
            y_pred = torch.sigmoid(y_pred)

        self.true_labels.append(y_true.detach().cpu().numpy().ravel())
        self.pred_scores.append(y_pred.detach().cpu().numpy().ravel())

    def reset_states(self):
        self.true_labels = []
        self.pred_scores = []

    def result(self):
        try:
            y_true = np.concatenate(self.true_labels)
            y_pred = np.concatenate(self.pred_scores)

            if len(np.unique(y_true)) < 2:
                return float('nan')

            warnings.simplefilter("ignore", category=UndefinedMetricWarning)
            return roc_auc_score(y_true, y_pred)
        except Exception:
            return float('nan')
