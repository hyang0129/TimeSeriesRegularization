from abc import ABC, abstractmethod
import tensorflow as tf
from tsr.datasets.common import Transform

tfkl = tf.keras.layers
tfk = tf.keras


class Augmentation(ABC, Transform):
    '''
    Functionally, augmentations are the same as transforms
    '''
