import numpy as np
from metrics.basemetric import Metric


class LooseHit(Metric):
    def __init__(self, name='loose_hit'):
        super(LooseHit, self).__init__(name=name)
        self.hits = 0
        self.total = 0

    def update_states(self, cam_mask, gt_mask):
        cam_mask = cam_mask.astype(np.bool_)
        gt_mask = gt_mask.astype(np.bool_)
        self.hits += int(np.any(np.logical_and(cam_mask, gt_mask)))
        self.total += 1

    def reset_states(self):
        self.hits = 0
        self.total = 0

    def result(self):
        return self.hits / (self.total + 1e-9)
