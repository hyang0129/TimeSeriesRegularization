from tsr.methods.augmentation import Cutmix, Mixup, Cutout


def get_augs(SHAPE, BATCH_SIZE = 64, DO_PROB = 0.5, element_prob = 0.5, version = 0):
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
    cutout.batch = BATCH_SIZE

    if version == 0:

        def batch_aug(x, y):
            example = {'input': x, 'target': y}
            example = cutmix(example)
            example = cutout(example)
            example = mixup(example)
            x, y = example['input'], example['target']
            return x, y

    else:
        raise KeyError('Augmentation Version Not Specified')

    return batch_aug

