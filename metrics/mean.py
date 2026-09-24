from metrics.basemetric import Metric
import torch


class Mean(Metric):
    def __init__(self, name='mean'):
        super(Mean, self).__init__(name=name)
        self.counter = 0.0
        self.running_mean = 0.0

    def update_states(self, value):
        if isinstance(value, torch.Tensor):
            value = value.detach().cpu().numpy().mean()
        elif isinstance(value, (int, float)):
            value = float(value)
        else:
            raise TypeError(f"Unsupported type for 'value': {type(value)}")

        self.running_mean += value
        self.counter += 1.0

    def reset_states(self):
        self.counter = 0.0
        self.running_mean = 0.0

    def result(self):
        return self.running_mean / (self.counter + 1e-9)
