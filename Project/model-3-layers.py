import tensorflow as tf
from tensorlayer.layers import SubpixelConv1d
from tensorflow.keras.layers import Conv1D, LeakyReLU, Dropout, Lambda, concatenate, Input, add, Activation, SeparableConv1D, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.utils import plot_model
from constants import *


def subpixel1d(input_shape, r):
    def _phase_shift(I, r=2):
        X = tf.transpose(I)
        X = tf.batch_to_space(X, [r], [[0, 0]])
        X = tf.transpose(X)
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


def create_downsampling_block(x, filters, kernel_size, stride=2):
    x = Conv1D(filters, kernel_size, kernel_initializer='Orthogonal', strides=stride, padding='same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)
    return x


def create_upsampling_block(x, filters, kernel_size, padding='same', stride=1, corresponding_downsample_block=None):
    x = Conv1D(filters, kernel_size, kernel_initializer='orthogonal', strides=stride, padding=padding)(x)
    x = LeakyReLU()(x)
    x = Dropout(rate=0.5)(x)
    x = subpixel1d(x.shape, r=2)(x)
    if corresponding_downsample_block is not None:
        x = concatenate([x, corresponding_downsample_block])
    return x


def create_model(batch_size=BATCH_SIZE, input_size=SAMPLE_DIMENSION // RESAMPLING_FACTOR):
    x = Input((input_size, 1), batch_size=batch_size)
    x_input = x
    downsampling_blocks = []

    x = create_downsampling_block(x, filters=64, kernel_size=32, stride=2)
    downsampling_blocks.append(x)
    x = create_downsampling_block(x, filters=128, kernel_size=32, stride=2)
    downsampling_blocks.append(x)
    x = create_downsampling_block(x, filters=256, kernel_size=32, stride=2)
    downsampling_blocks.append(x)

    x = Conv1D(padding='same', kernel_initializer='Orthogonal', filters=256, kernel_size=32, strides=2)(x)
    x = LeakyReLU()(x)

    x = create_upsampling_block(x, filters=256, kernel_size=32, corresponding_downsample_block=downsampling_blocks[-1])
    downsampling_blocks.pop()
    x = create_upsampling_block(x, filters=128, kernel_size=32, corresponding_downsample_block=downsampling_blocks[-1])
    downsampling_blocks.pop()
    x = create_upsampling_block(x, filters=64, kernel_size=32, corresponding_downsample_block=downsampling_blocks[-1])
    downsampling_blocks.pop()
    x = create_upsampling_block(x, filters=64, kernel_size=32)
    x = add([x, x_input])

    x = create_upsampling_block(x, filters=64, kernel_size=32)
    x = create_upsampling_block(x, filters=64, kernel_size=32)
    x = Conv1D(filters=1, kernel_initializer='Orthogonal', kernel_size=1)(x)

    model = Model(x_input, x)
    plot_model(model, to_file="model_stage_" + str(STAGE) + ".png", show_shapes=True, show_layer_names=True)
    return model

