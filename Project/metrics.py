from tensorflow.keras import backend as K
import tensorflow as tf
from sklearn.metrics import mean_squared_error
from constants import *


def signal_to_noise_ratio(actual_signal, predicted_signal):
    noise = predicted_signal - actual_signal
    noise_power = K.mean(noise ** 2)
    signal_power = K.mean(actual_signal ** 2)
    ratio = 10 * K.log(signal_power) / K.log(noise_power)
    return K.mean(ratio)


def root_mean_squared_error(actual_signal, predicted_signal):
    return tf.sqrt(tf.losses.mean_squared_error(actual_signal, predicted_signal))


def normalised_root_mean_squared_error(actual_signal, predicted_signal, interquartile_range):
    return tf.sqrt(tf.losses.mean_squared_error(actual_signal, predicted_signal)) / interquartile_range


def normalised_root_mean_squared_error_training(actual_signal, predicted_signal):
    return normalised_root_mean_squared_error(actual_signal, predicted_signal,
                                              TRAINING_SET_THIRD_QUANTILE - TRAINING_SET_FIRST_QUANTILE)


def normalised_root_mean_squared_error_validation(actual_signal, predicted_signal):
    return normalised_root_mean_squared_error(actual_signal, predicted_signal,
                                              VALIDATION_SET_THIRD_QUANTILE - VALIDATION_SET_FIRST_QUANTILE)


def normalised_root_mean_squared_error_testing(actual_signal, predicted_signal):
    return normalised_root_mean_squared_error(actual_signal, predicted_signal,
                                              TESTING_SET_THIRD_QUANTILE - TESTING_SET_FIRST_QUANTILE)


def normalised_root_mean_squared_error_for_single_examples(actual_signal, predicted_signal, option):
    minimum_value = tf.reduce_min(actual_signal)
    maximum_value = tf.reduce_max(actual_signal)
    if option == "mean":
        return tf.sqrt(tf.losses.mean_squared_error(actual_signal, predicted_signal)) / tf.math.reduce_mean(actual_signal)
    elif option == "sd":
        return tf.sqrt(tf.losses.mean_squared_error(actual_signal, predicted_signal)) / tf.math.reduce_std(actual_signal)
    elif option == "range":
        return tf.sqrt(tf.losses.mean_squared_error(actual_signal, predicted_signal)) / (maximum_value - minimum_value)
    raise Exception("Unexpected option parameter.")
