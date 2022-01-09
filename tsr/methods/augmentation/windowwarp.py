from tsr.methods.augmentation import Augmentation
from tsr.methods.augmentation.common import resize_time_series, cut_time_series, check_proba
import tensorflow as tf
from typing import Union


class WindowWarp(Augmentation):
    def __init__(
            self,
            batch_size: int,
            do_prob: float,
            sequence_shape: Union[list, tuple],
            min_window_size: int,
            max_window_size: int,
            scale_factor: float,
            method: str = 'bilinear',
    ):
        """
        Use bilinear interpolation (as if it was an image) to resize a window and insert the window back.

        Args:
                min_window_size:
                max_window_size:
                batch_size:
                do_prob:
                sequence_shape: in the form of [Length, Channels]
        """

        super().__init__()
        self.min_window_size = min_window_size
        self.max_window_size = max_window_size
        self.scale_factor = scale_factor
        self.sequence_shape = sequence_shape
        self.batch_size = batch_size
        self.do_prob = do_prob
        self.method = method

    def call(self, example: dict) -> dict:
        x = example["input"]

        # apply the function across a tensor with shape [batchsize] to return
        # a tensor with shape [batchsize, length, channels]
        x = tf.map_fn(self.singular_call, x, dtype = tf.float32)

        example["input"] = x

        return example

    @tf.function
    def singular_call(self, input: tf.Tensor) -> tf.Tensor:

        if check_proba(self.do_prob):
            start, end = self.get_window()
            window_size = end - start

            target_window_size = max(int((float(window_size) * self.scale_factor)), 2)

            window = input[start:end]
            window = resize_time_series(window, target_window_size, method = self.method)

            input = cut_time_series(input, start, end, window)

        return input

    def get_window(self):
        # max val is exclusive
        start = tf.random.uniform((), maxval = self.sequence_shape[0] - self.max_window_size + 1, dtype = tf.int64)
        end = start + tf.random.uniform((), minval = self.min_window_size, maxval = self.max_window_size + 1,
                                        dtype = tf.int64)

        return start, end

    @classmethod
    def from_config(cls, config):
        raise NotImplementedError