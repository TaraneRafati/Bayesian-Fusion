import numpy as np


def load_numpy(file_path):
    return np.load(file_path).astype(np.float32)


def normalize_channel(channel):
    channel_min = channel.min()
    channel_max = channel.max()

    if channel_max == channel_min:
        return np.zeros_like(channel, dtype=np.float32)

    return ((channel - channel_min) / (channel_max - channel_min) * 255.0).astype(np.float32)


class CreateChannels(object):
    def __init__(self, mode='Repeat', window_widths=[80], window_centers=[40]):
        assert mode == 'Repeat', "Invalid mode: only 'Repeat' is supported"
        assert len(window_centers) == len(window_widths), "number of elements in window_centers and window_widths must be equal"
        assert len(window_centers) == 1, "When using 'Repeat' mode, only one window center and width are supported"
        self.mode = mode
        self.window_widths = window_widths
        self.window_centers = window_centers

    def apply_windowing(self, image, center, width):
        lower_bound = center - (width / 2)
        upper_bound = center + (width / 2)
        return np.clip(image, lower_bound, upper_bound)

    def __call__(self, image):
        windowed_image = normalize_channel(self.apply_windowing(image, self.window_centers[0], self.window_widths[0]))
        return np.stack([windowed_image] * 3, axis=-1)
