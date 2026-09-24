import time
from callbacks.basecallback import Callback


class TimeLimit(Callback):
    def __init__(self, max_seconds, verbose=True):
        super(TimeLimit, self).__init__(name="TimeLimitCallback")
        self.max_seconds = max_seconds
        self.verbose = verbose
        self.start_time = None
        self.epoch_start_time = None
        self.epoch_durations = []

    def on_train_begin(self, epoch=None, logs=None):
        self.start_time = time.time()
        if self.verbose:
            print(f"[TimeLimitCallback] Training started. Time limit: {self.max_seconds} seconds.")

    def on_epoch_begin(self, epoch=None, logs=None):
        self.epoch_start_time = time.time()

    def on_epoch_end(self, epoch=None, logs=None):
        duration = time.time() - self.epoch_start_time
        self.epoch_durations.append(duration)

        elapsed = time.time() - self.start_time
        avg_epoch_time = sum(self.epoch_durations) / len(self.epoch_durations)
        estimated_next_total = elapsed + avg_epoch_time

        if self.verbose:
            print(f"[TimeLimitCallback] Epoch {epoch} took {duration:.2f} seconds. "
                  f"Estimated total with next: {estimated_next_total:.2f}/{self.max_seconds}")

        if estimated_next_total > self.max_seconds:
            if self.verbose:
                print(f"[TimeLimitCallback] Stopping early to stay under time limit ({self.max_seconds}s).")
            raise StopIteration
