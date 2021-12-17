import numpy as np
from tensorflow.keras import backend as K


def signal_to_noise_ratio(actual_signal, predicted_signal):
    noise = predicted_signal - actual_signal
    loss = K.sqrt(K.mean(noise**2))
    l2_norm = K.sqrt(K.mean(predicted_signal**2))
    ratio = 20 * K.log(l2_norm / loss) / K.log(10.0)
    return K.mean(ratio)
