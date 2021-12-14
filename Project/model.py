import tensorflow as tf
from tensorflow.keras.layers import Conv1D, PReLU, Dropout, Lambda, concatenate, Input, add
from tensorflow.keras.models import Model


def subpixel1d(input_shape, r, color=False):
    def _phase_shift(I, r=2):
        x = tf.transpose(I, [2, 1, 0])
        x = tf.batch_to_space(x, [r], [[0, 0]])
        x = tf.transpose(x, [2, 1, 0])
        return x

    def subpixel_shape(input_shape):
        dims = [input_shape[0],
                input_shape[1] * r,
                int(input_shape[2] / (r))]
        output_shape = tuple(dims)
        return output_shape

    def subpixel(x):
        x_upsampled = _phase_shift(x, r)
        return x_upsampled

    return Lambda(subpixel, output_shape=subpixel_shape)


def create_downsampling_block(x, filters, kernel_size, stride=1):
    x = Conv1D(filters, kernel_size, strides=stride, padding='same')(x)
    x = PReLU()(x)
    return x


def create_upsampling_block(x, filters, kernel_size, corresponding_downsample_block, stride=1):
    x = Conv1D(filters, kernel_size, strides=stride, padding='same')(x)
    x = Dropout(rate=0.5)(x)
    x = PReLU()(x)
    x = subpixel1d(x.shape, r=2)(x)
    x = concatenate([x, corresponding_downsample_block])
    return x


def create_model(number_of_blocks, input_size=256):
    x = Input((input_size, 1))
    x_input = x
    downsampling_blocks = []

    for layer_index in range(0, number_of_blocks):
        x = create_downsampling_block(x, 32, 8, stride=2)
        downsampling_blocks.append(x)

    x = Conv1D(padding='same', filters=32, kernel_size=8, strides=2)(x)
    x = PReLU()(x)

    for layer_index in range(0, number_of_blocks):
        x = create_upsampling_block(x, 64, 8, downsampling_blocks[-1])
        downsampling_blocks.pop()

    x = Conv1D(padding='same', kernel_initializer='he_normal', filters=2, kernel_size=8)(x)
    x = subpixel1d(x.shape, r=2)(x)
    output = add([x, x_input])
    model = Model(x_input, output)

    model.summary()
    return model

