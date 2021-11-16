import tensorflow as tf

from tsr.models.attention import EncoderLayer


def get_mhsa_model(
    strategy: tf.distribute.Strategy,
    input_shape=(4096, 23),
    conv_filters=[128, 128, 256, 256, 512],
    kernel=[7, 7, 3, 7, 7],
    strides=[1, 2, 1, 2, 2],
    num_heads=8,
    d_model=512,
    dff=512,
    output: tf.keras.layers.Layer = tf.keras.layers.Dense(1, activation="sigmoid"),
) -> tf.keras.Model:
    """

    Constructs a Conv-MHSA model, with default settings configured for the IAAI 2022 Paper.

    Args:
            strategy:
            input_shape:
            conv_filters:
            kernel:
            strides:
            num_heads:
            d_model:
            dff:
            output:

    Returns:

    """
    with strategy.scope():

        model = tf.keras.Sequential(
            [
                tf.keras.Input(shape=input_shape),
                tf.keras.layers.Conv1D(
                    filters=conv_filters[0],
                    kernel_size=kernel[0],
                    strides=strides[0],
                    padding="same",
                    activation="relu",
                ),
                tf.keras.layers.Conv1D(
                    filters=conv_filters[1],
                    kernel_size=kernel[1],
                    strides=strides[1],
                    padding="same",
                    activation="relu",
                ),
                tf.keras.layers.Conv1D(
                    filters=conv_filters[2],
                    kernel_size=kernel[2],
                    strides=strides[2],
                    padding="same",
                    activation="relu",
                ),
                tf.keras.layers.Conv1D(
                    filters=conv_filters[3],
                    kernel_size=kernel[3],
                    strides=strides[3],
                    padding="same",
                    activation="relu",
                ),
                tf.keras.layers.Conv1D(
                    filters=conv_filters[4],
                    kernel_size=kernel[4],
                    strides=strides[4],
                    padding="same",
                    activation="relu",
                ),
                EncoderLayer(d_model=d_model, num_heads=num_heads, dff=dff),
                EncoderLayer(d_model=d_model, num_heads=num_heads, dff=dff),
                EncoderLayer(d_model=d_model, num_heads=num_heads, dff=dff),
                EncoderLayer(d_model=d_model, num_heads=num_heads, dff=dff),
                tf.keras.layers.GlobalAveragePooling1D(),
                output,
            ]
        )

    return model
