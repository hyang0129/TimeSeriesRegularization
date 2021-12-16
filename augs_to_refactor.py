import tensorflow as tf


def get_augs(x, DO_PROB = 0.5):
    SHAPE = x.shape
    BATCH_SIZE = 64

    def random_shift(x, y):
        x = tf.pad(x, [[64, 64], [0, 0]])
        start = tf.random.uniform(shape = [], minval = 0, maxval = 64, dtype = tf.int64)
        x = x[start: start + SHAPE[0]]
        x = tf.reshape(x, (SHAPE[0], 23))
        return x, y

    # def random_slice(x, y):
    #     start = tf.random.uniform(shape = [], minval = 0, maxval = 4096 - BLOCK_SHAPE[0], dtype = tf.int64)
    #     x = x[start: start + BLOCK_SHAPE[0]]
    #     x = tf.reshape(x, (BLOCK_SHAPE))
    #     return x, y

    def batch_random_linear_mixup(x):
        LINEAR_MIX_MIN = 0.1
        LINEAR_MIX_MAX = 0.6
        LINEAR_MIX_PROBA = DO_PROB
        batch_size = BATCH_SIZE
        new = tf.random.uniform((), minval = LINEAR_MIX_MIN, maxval = LINEAR_MIX_MAX)
        prob = LINEAR_MIX_PROBA
        do_aug = tf.cast(tf.random.uniform((batch_size, 1, 1), minval = 0.0, maxval = 1.0) > prob, tf.float32)

        take_from_b_percetnage = do_aug * new

        a = x
        b = tf.random.shuffle(x)

        x = a * (1 - take_from_b_percetnage) + b * (take_from_b_percetnage)

        return x

    def get_batch_random_cutout():
        '''Drops a t by n component of the series, t is time steps and n is a selection of variables'''

        batch = BATCH_SIZE
        length = SHAPE[1]
        nvar = SHAPE[2]
        max_cutout_length = SHAPE[1]
        min_coutout_length = SHAPE[1] // 2
        elem_drop_prob = 0.5 if nvar > 4 else 0.1
        do_prob = DO_PROB

        def time_wise_cut(nothing):
            time = tf.range(0, length, dtype = tf.float32)
            start = tf.random.uniform((), maxval = length - max_cutout_length)
            end = start + tf.random.uniform((), minval = min_coutout_length, maxval = max_cutout_length)
            do = tf.cast(tf.random.uniform((), ) < do_prob, tf.float32)
            return tf.cast(tf.logical_and(time > start, time < end), tf.float32) * do

        def element_wise_cut(nothing):
            return tf.cast(tf.random.uniform((nvar,)) < elem_drop_prob, tf.float32)

        def cutout_mask(nothing):
            time = tf.reshape(time_wise_cut(nothing), (length, 1))
            elem = tf.reshape(element_wise_cut(nothing), (1, nvar))
            return tf.cast(time * elem < 1, tf.float32)

        def batch_random_cutout(x):
            x = x * tf.map_fn(cutout_mask, tf.zeros((batch,)), dtype = tf.float32)
            return x

        return batch_random_cutout

    def get_batch_cutmix():
        batch = BATCH_SIZE
        length = SHAPE[1]
        nvar = SHAPE[2]
        max_cutout_length = SHAPE[1]
        min_coutout_length = SHAPE[1] // 2
        elem_drop_prob = 0.5 if nvar > 4 else 0.1
        do_prob = DO_PROB

        def time_wise_cut(nothing):
            time = tf.range(0, length, dtype = tf.float32)
            start = tf.random.uniform((), maxval = length - max_cutout_length)
            end = start + tf.random.uniform((), minval = min_coutout_length, maxval = max_cutout_length)
            do = tf.cast(tf.random.uniform((), ) < do_prob, tf.float32)
            return tf.cast(tf.logical_and(time > start, time < end), tf.float32) * do

        def element_wise_cut(nothing):
            return tf.cast(tf.random.uniform((nvar,)) < elem_drop_prob, tf.float32)

        def cutout_mask(nothing):
            time = tf.reshape(time_wise_cut(nothing), (length, 1))
            elem = tf.reshape(element_wise_cut(nothing), (1, nvar))
            return tf.cast(time * elem < 1, tf.float32)

        def batch_cutmix(x):
            a = x
            b = tf.random.shuffle(x)
            mask = tf.map_fn(cutout_mask, tf.zeros((batch,))) > 0
            return tf.where(mask, a, b)

        return batch_cutmix

    def get_batch_aug():
        batch_random_cutout = get_batch_random_cutout()
        batch_cutmix = get_batch_cutmix()

        def batch_aug(x, y):
            x = batch_cutmix(x)
            x = batch_cutmix(x)
            x = batch_random_cutout(x)
            x = batch_random_linear_mixup(x)
            return x, y

        return batch_aug

    return get_batch_aug()
