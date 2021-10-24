from abc import ABC, abstractmethod
import tensorflow as tf

tfkl = tf.keras.layers
tfk = tf.keras


class Augmentation(ABC, tfkl.Layer):
    @abstractmethod
    def __init__(self):
        """
        Augmentations should be defined as a callable class with an initialization function specifying
        the parameters for the callable. For example, a resize augmentation would specify the target
        X and Y shapes, if it is for an image.

        Note that the callable function should always accept a dictionary, as that is the preferred unit
        for tf datasets.

        Augmentations should operate on batches, rather than single examples.
        """

    @abstractmethod
    def call(self, example: dict) -> dict:
        """
        This is the batch wise call of the function.

        Args:
                example: a batched time series

        Returns:
                dict
        """

    @abstractmethod
    def singular_call(self, input: tf.Tensor) -> tf.Tensor:
        """
        This is a call for a single tensor, not batched. Not all augmentations can execute a singular call.
        Where it cannot, simple pass on the singular call.

        Args:
                input: tf.Tensor

        Returns:
                tf.Tensor
        """
