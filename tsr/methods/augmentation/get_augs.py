from tsr.methods.augmentation import Cutmix, Mixup, Cutout, WindowWarp


def get_augs(SHAPE, BATCH_SIZE = 64, DO_PROB = 0.5, element_prob = 0.5, version = 0):
    if version == -1:

        def batch_aug(x, y):
            return x, y

    elif version == 0:

        DO_PROB = 0.7
        element_prob = 0.3

        mixup = Mixup(batch_size = BATCH_SIZE,
                      do_prob = DO_PROB,
                      sequence_shape = SHAPE[1:],
                      linear_mix_min = 0.1,
                      linear_mix_max = 0.5)

        cutmix = Cutmix(batch_size = BATCH_SIZE,
                        do_prob = DO_PROB,
                        sequence_shape = SHAPE[1:],
                        min_cutmix_len = SHAPE[1] // 2,
                        max_cutmix_len = SHAPE[1],
                        channel_replace_prob = element_prob,

                        )
        cutmix.batch = BATCH_SIZE
        cutout = Cutout(
            batch_size = BATCH_SIZE,
            do_prob = DO_PROB,
            sequence_shape = SHAPE[1:],
            min_cutout_len = SHAPE[1] // 2,
            max_cutout_len = SHAPE[1],
            channel_drop_prob = element_prob,
        )

        def batch_aug(x, y):
            example = {'input': x, 'target': y}
            example = cutmix(example)
            example = cutout(example)
            example = mixup(example)
            x, y = example['input'], example['target']
            return x, y



    elif version == 1:
        DO_PROB = 0.8
        element_prob = 0.5

        mixup = Mixup(batch_size = BATCH_SIZE,
                      do_prob = DO_PROB,
                      sequence_shape = SHAPE[1:],
                      linear_mix_min = 0.1,
                      linear_mix_max = 0.5)

        cutmix = Cutmix(batch_size = BATCH_SIZE,
                        do_prob = DO_PROB,
                        sequence_shape = SHAPE[1:],
                        min_cutmix_len = SHAPE[1] // 2,
                        max_cutmix_len = SHAPE[1],
                        channel_replace_prob = element_prob,

                        )
        cutmix.batch = BATCH_SIZE
        cutout = Cutout(
            batch_size = BATCH_SIZE,
            do_prob = DO_PROB,
            sequence_shape = SHAPE[1:],
            min_cutout_len = SHAPE[1] // 2,
            max_cutout_len = SHAPE[1],
            channel_drop_prob = element_prob,
        )

        def batch_aug(x, y):
            example = {'input': x, 'target': y}
            example = cutmix(example)
            example = cutout(example)
            example = mixup(example)
            x, y = example['input'], example['target']
            return x, y

    elif version == 2:
        DO_PROB = 0.8
        element_prob = 0.2

        mixup = Mixup(batch_size = BATCH_SIZE,
                      do_prob = DO_PROB,
                      sequence_shape = SHAPE[1:],
                      linear_mix_min = 0.1,
                      linear_mix_max = 0.5)

        cutmix = Cutmix(batch_size = BATCH_SIZE,
                        do_prob = DO_PROB,
                        sequence_shape = SHAPE[1:],
                        min_cutmix_len = SHAPE[1] // 2,
                        max_cutmix_len = SHAPE[1],
                        channel_replace_prob = element_prob,

                        )
        cutmix.batch = BATCH_SIZE
        cutout = Cutout(
            batch_size = BATCH_SIZE,
            do_prob = DO_PROB,
            sequence_shape = SHAPE[1:],
            min_cutout_len = SHAPE[1] // 2,
            max_cutout_len = SHAPE[1],
            channel_drop_prob = element_prob,
        )

        def batch_aug(x, y):
            example = {'input': x, 'target': y}
            example = cutmix(example)
            example = cutmix(example)
            example = cutout(example)
            example = mixup(example)
            x, y = example['input'], example['target']
            return x, y

    elif version == 3:
        DO_PROB = 0.5
        element_prob = 0.3

        cutout = Cutout(
            batch_size = BATCH_SIZE,
            do_prob = DO_PROB,
            sequence_shape = SHAPE[1:],
            min_cutout_len = SHAPE[1] // 2,
            max_cutout_len = SHAPE[1],
            channel_drop_prob = element_prob,
        )

        def batch_aug(x, y):
            example = {'input': x, 'target': y}
            example = cutout(example)
            example = cutout(example)
            x, y = example['input'], example['target']
            return x, y

    elif version == 4:
        DO_PROB = 0.5
        element_prob = 0.3

        mixup = Mixup(batch_size = BATCH_SIZE,
                      do_prob = DO_PROB,
                      sequence_shape = SHAPE[1:],
                      linear_mix_min = 0.1,
                      linear_mix_max = 0.5)

        def batch_aug(x, y):
            example = {'input': x, 'target': y}
            example = mixup(example)
            x, y = example['input'], example['target']
            return x, y

    elif version == 5:
        DO_PROB = 0.5
        element_prob = 0.3

        cutmix = Cutmix(batch_size = BATCH_SIZE,
                        do_prob = DO_PROB,
                        sequence_shape = SHAPE[1:],
                        min_cutmix_len = SHAPE[1] // 2,
                        max_cutmix_len = SHAPE[1],
                        channel_replace_prob = element_prob,

                        )

        def batch_aug(x, y):
            example = {'input': x, 'target': y}
            example = cutmix(example)
            example = cutmix(example)
            x, y = example['input'], example['target']
            return x, y

    elif version == 6:
        DO_PROB = 0.5

        expand_window = WindowWarp(batch_size = BATCH_SIZE,
                        do_prob = DO_PROB,
                        sequence_shape = SHAPE[1:],
                        min_window_size = SHAPE[1] // 8,
                        max_window_size = SHAPE[1] // 3,
                        scale_factor = 2
                                   )
        shrink_window = WindowWarp(batch_size = BATCH_SIZE,
                        do_prob = DO_PROB,
                        sequence_shape = SHAPE[1:],
                        min_window_size = SHAPE[1] // 8,
                        max_window_size = SHAPE[1] // 3,
                        scale_factor = 1
                                   )

        def batch_aug(x, y):
            example = {'input': x, 'target': y}
            example = shrink_window(example)
            example = expand_window(example)
            x, y = example['input'], example['target']
            return x, y

    else:
        raise KeyError('Augmentation Version Not Specified')

    return batch_aug

