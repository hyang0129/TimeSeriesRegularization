import tensorflow as tf



class InceptionBlock(tf.keras.layers.Layer):


    def __init__(self,
                 nb_filters=32,
                 use_bottleneck=True,
                 kernel_size=41,
                 stride = 1,
                 depth = 3):

        super().__init__()
        self.nb_filters = nb_filters
        self.use_bottleneck = use_bottleneck
        self.kernel_size = kernel_size - 1
        self.callbacks = None
        self.bottleneck_size = 32
        self.stride = stride
        self.depth = depth


    def call(self, x):

        shortcut_x = x

        for d in range(self.depth):
            x = InceptionLayer(
                nb_filters =  self.nb_filters,
                use_bottleneck = self.use_bottleneck,
                kernel_size = self.kernel_size,
                stride = self.stride,
            )(x)

        x = ShortcutLayer()(x, shortcut_x)
        return x


class InceptionLayer(tf.keras.layers.Layer):

    def __init__(self,
                 nb_filters=32,
                 use_bottleneck=True,
                 kernel_size=41,
                 stride = 1):

        super().__init__()
        self.nb_filters = nb_filters
        self.use_bottleneck = use_bottleneck
        self.kernel_size = kernel_size - 1
        self.callbacks = None
        self.bottleneck_size = 32
        self.stride = stride

    def call(self, input_tensor):
        activation = 'linear'
        stride = self.stride
        if self.use_bottleneck and int(input_tensor.shape[-1]) > 1:
            input_inception = tf.keras.layers.Conv1D(filters=self.bottleneck_size, kernel_size=1,
                                                  padding='same', activation=activation, use_bias=False)(input_tensor)
        else:
            input_inception = input_tensor

        # kernel_size_s = [3, 5, 8, 11, 17]
        kernel_size_s = [self.kernel_size // (2 ** i) for i in range(3)]

        conv_list = []

        for i in range(len(kernel_size_s)):
            conv_list.append(tf.keras.layers.Conv1D(filters=self.nb_filters, kernel_size=kernel_size_s[i],
                                                 strides=stride, padding='same', activation=activation, use_bias=False)(
                input_inception))

        max_pool_1 = tf.keras.layers.MaxPool1D(pool_size=3, strides=stride, padding='same')(input_tensor)

        conv_6 = tf.keras.layers.Conv1D(filters=self.nb_filters, kernel_size=1,
                                     padding='same', activation=activation, use_bias=False)(max_pool_1)

        conv_list.append(conv_6)

        x = tf.keras.layers.Concatenate(axis=2)(conv_list)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation(activation='relu')(x)
        return x

class ShortcutLayer(tf.keras.layers.Layer):

    def __init__(self):
        super().__init__()

    def call(self, input_tensor, shortcut_tensor):
        shortcut_y = tf.keras.layers.Conv1D(filters = int(shortcut_tensor.shape[-1]), kernel_size = 1,
                                            padding = 'same', use_bias = False)(input_tensor)
        shortcut_y = tf.keras.layers.normalization.BatchNormalization()(shortcut_y)
        x = tf.keras.layers.Add()([shortcut_y, shortcut_tensor])
        x = tf.keras.layers.Activation('relu')(x)

        return x
