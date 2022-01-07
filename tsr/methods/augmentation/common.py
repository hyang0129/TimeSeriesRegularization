import tensorflow as tf


def resize_time_series(series, new_length, method='bilinear'):
    '''
    Resize a time series, as if it was a 1 dimensional image
    Args:
        series:
        new_length:
        method:

    Returns:

    '''
    image_like = tf.expand_dims(series, axis = 0)
    return tf.image.resize(images=image_like, size= [1, new_length], method=method)

def cut_time_series(series, cut_start, cut_end):
    raise NotImplementedError
