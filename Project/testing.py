import tensorflow as tf
from constants import *
from DatasetGenerator import DatasetGenerator
import numpy as np
from metrics import *
from tensorflow.keras.models import load_model
from model import create_model

_, _, (input_test_files, target_test_files) = DatasetGenerator.split_list_of_files()
input_test_data, target_test_data = [], []

number_of_testing_batches = int(NUMBER_OF_TESTING_TENSORS / BATCH_SIZE)
for index in range(0, number_of_testing_batches*BATCH_SIZE):
    input_test_data.append(np.load("preprocessed_dataset/low_res/" + input_test_files[index]))
    target_test_data.append(np.load("preprocessed_dataset/high_res/" + target_test_files[index]))
    print("Loaded testing sample {}".format(index))

model = create_model()
model.load_weights(MODEL_PATH)

adam_optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)
model.compile(loss="mean_squared_error", optimizer=adam_optimizer,
              metrics=[signal_to_noise_ratio, normalised_root_mean_squared_error])

input_test_data = np.array(input_test_data)
target_test_data = np.array(target_test_data)

print("Input test data shape: {}".format(input_test_data.shape))
print("Target test data shape: {}".format(target_test_data.shape))

mean_squared_error_value, signal_to_noise_ratio_value, nrmse_value = model.evaluate(input_test_data, target_test_data, batch_size=BATCH_SIZE, verbose=True)

