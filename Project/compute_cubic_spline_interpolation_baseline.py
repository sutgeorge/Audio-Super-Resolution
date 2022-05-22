import tensorflow as tf
from constants import *
from DatasetGenerator import DatasetGenerator
import numpy as np
from metrics import *
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Lambda
from model import create_model

_, _, (input_test_files, target_test_files) = DatasetGenerator.split_list_of_files()
input_test_data, target_test_data = [], []

number_of_testing_batches = int(NUMBER_OF_TESTING_TENSORS / BATCH_SIZE)
snr_sum, mse_sum, nrmse_sum = 0, 0, 0
number_of_samples = number_of_testing_batches*BATCH_SIZE

for index in range(0, number_of_samples, BATCH_SIZE):
    for i in range(0, BATCH_SIZE):
        low_resolution_patch = np.load("preprocessed_dataset/low_res/" + input_test_files[index])
        high_resolution_patch = np.load("preprocessed_dataset/high_res/" + target_test_files[index])
        low_resolution_patch = low_resolution_patch.reshape(low_resolution_patch.shape[0])
        high_resolution_patch = high_resolution_patch.reshape(high_resolution_patch.shape[0])
        cubic_spline_patch = DatasetGenerator.upsample(low_resolution_patch, RESAMPLING_FACTOR)

        snr = signal_to_noise_ratio(K.constant(high_resolution_patch), K.constant(cubic_spline_patch))
        mse = tf.losses.mean_squared_error(K.constant(high_resolution_patch), K.constant(cubic_spline_patch))
        nrmse = normalised_root_mean_squared_error_for_single_examples(high_resolution_patch, cubic_spline_patch, "range")

    print("Low-res length: {}".format(low_resolution_patch.shape))
    print("High-res length: {}".format(high_resolution_patch.shape))
    print("Cubic spline length: {}".format(cubic_spline_patch.shape))

    snr_sum += snr
    mse_sum += mse
    nrmse_sum += nrmse

    print("Loaded testing sample {}".format(index))

snr_mean = snr_sum / number_of_samples
mse_mean = mse_sum / number_of_samples
nrmse_mean = nrmse_sum / number_of_samples

print("SNR: {}".format(snr_mean))
print("MSE: {}".format(mse_mean))
print("NRMSE: {}".format(nrmse_mean))
