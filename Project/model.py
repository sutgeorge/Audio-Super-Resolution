import tensorflow as tf
from tensorlayer.layers import SubpixelConv1d
from tensorflow.keras.layers import Conv1D, LeakyReLU, Dropout, Lambda, concatenate, Input, add, Activation
from tensorflow.keras.models import Model
from constants import *


def subpixel1d(input_shape, r):
    def _phase_shift(I, r=2):
        X = tf.transpose(I)#, [2, 1, 0])
        X = tf.batch_to_space(X, [r], [[0, 0]])
        X = tf.transpose(X)#, [2, 1, 0])
        return X

    def subpixel_shape(input_shape):
        dims = [input_shape[0],
                input_shape[1] * r,
                int(input_shape[2] / r)]
        output_shape = tuple(dims)
        return output_shape

    def subpixel(x):
        x_upsampled = _phase_shift(x, r)
        return x_upsampled

    return Lambda(subpixel, output_shape=subpixel_shape)


def create_downsampling_block(x, filters, kernel_size, stride=1):
    x = Conv1D(filters, kernel_size, strides=stride, padding='same')(x)
    x = LeakyReLU()(x)
    return x


def create_upsampling_block(x, filters, kernel_size, corresponding_downsample_block, stride=1):
    x = Conv1D(filters, kernel_size, strides=stride, padding='same')(x)
    x = LeakyReLU()(x)
    x = Dropout(rate=0.5)(x)
    x = subpixel1d(x.shape, r=2)(x)
    x = concatenate([x, corresponding_downsample_block])
    return x


def create_model(number_of_blocks, batch_size=BATCH_SIZE, input_size=SAMPLE_DIMENSION):
    x = Input((input_size, 1), batch_size=batch_size)
    x_input = x
    downsampling_blocks = []

    number_of_filters, kernel_size = 0, 0
    for layer_index in range(0, number_of_blocks):
        number_of_filters = 16 if layer_index == 0 else 32
        kernel_size = 4 if layer_index > 1 else (8 if layer_index == 1 else 16)
        x = create_downsampling_block(x, number_of_filters, kernel_size, stride=2)
        downsampling_blocks.append(x)

    x = Conv1D(padding='same', filters=32, kernel_size=4, strides=2)(x)
    x = LeakyReLU()(x)

    for layer_index in range(0, number_of_blocks):
        number_of_filters = 16 if layer_index == number_of_blocks - 1 else 32
        kernel_size = 4 if layer_index < number_of_blocks - 2 else (8 if layer_index == 1 else 16)
        x = create_upsampling_block(x, 2*number_of_filters, kernel_size, downsampling_blocks[-1])
        downsampling_blocks.pop()

    x = Conv1D(padding='same', kernel_initializer='he_normal', filters=2, kernel_size=8)(x)
    x = subpixel1d(x.shape, r=2)(x)
    output = add([x, x_input])
    model = Model(x_input, output)

    model.summary()
    return model

