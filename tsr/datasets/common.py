import tensorflow as tf


def fix_type(x, y):
    return tf.cast(x, tf.float32), tf.cast(y, tf.float32)
