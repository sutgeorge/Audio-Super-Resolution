from tensorflow.keras import backend as K
import tensorflow as tf
from sklearn.metrics import mean_squared_error


def signal_to_noise_ratio(actual_signal, predicted_signal):
    noise = predicted_signal - actual_signal
    loss = K.sqrt(K.mean(noise ** 2))
    l2_norm = K.sqrt(K.mean(predicted_signal ** 2))
    ratio = 20 * K.log(l2_norm / loss) / K.log(10.0)
    return K.mean(ratio)


def normalised_root_mean_squared_error(actual_signal, predicted_signal):
    minimum_value = tf.reduce_min(actual_signal)
    maximum_value = tf.reduce_max(actual_signal)
    if maximum_value != minimum_value:
        return tf.sqrt(tf.losses.mean_squared_error(actual_signal, predicted_signal)) / (maximum_value - minimum_value)
    return tf.sqrt(tf.losses.mean_squared_error(actual_signal, predicted_signal)) / tf.math.reduce_mean(actual_signal)
