class Metric(object):
    def __init__(self, name='metric',**kwargs):
        self.name = name

    def update_states(self, y_true, y_pred, sample_weights=None):
        pass

    def reset_states(self):
        pass

    def result(self):
        return 0
