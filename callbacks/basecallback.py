class Callback(object):
    def __init__(self, name='callback'):
        self.name = name

    def on_epoch_begin(self, epoch=None, logs=None):
        pass

    def on_train_begin(self, epoch=None, logs=None):
        pass

    def on_train_batch_begin(self, step=None, logs=None):
        pass

    def on_test_batch_begin(self, step=None, logs=None):
        pass

    def on_train_batch_end(self, step=None, logs=None):
        pass

    def on_test_batch_end(self, step=None, logs=None):
        pass

    def on_epoch_end(self, epoch=None, logs=None):
        pass

    def on_train_end(self, logs=None):
        pass
