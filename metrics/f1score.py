from metrics.basemetric import Metric
from metrics.precision import BinaryPrecision
from metrics.recall import BinaryRecall


class BinaryF1(Metric):
    def __init__(self, name='binary_f1', threshold=0.5, from_logits=True):
        super(BinaryF1, self).__init__(name=f'{name}@{threshold}')
        self.precision_metric = BinaryPrecision(name='binary_precision', threshold=threshold, from_logits=from_logits)
        self.recall_metric = BinaryRecall(name='binary_recall', threshold=threshold, from_logits=from_logits)

    def update_states(self, y_pred, y_true, sample_weights=None):
        self.precision_metric.update_states(y_pred, y_true, sample_weights)
        self.recall_metric.update_states(y_pred, y_true, sample_weights)

    def reset_states(self):
        self.precision_metric.reset_states()
        self.recall_metric.reset_states()

    def result(self):
        precision = self.precision_metric.result()
        recall = self.recall_metric.result()
        return 2 * (precision * recall) / (precision + recall + 1e-9)
