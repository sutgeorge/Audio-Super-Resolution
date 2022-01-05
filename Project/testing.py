from model import create_model
from constants import *
from DatasetGenerator import DatasetGenerator
import numpy as np
from metrics import *

_, _, (input_test_files, target_test_files) = DatasetGenerator.split_list_of_files()
input_test_data, target_test_data = [], []

number_of_testing_batches = int(NUMBER_OF_TESTING_TENSORS / BATCH_SIZE)
for index in range(0, number_of_testing_batches*BATCH_SIZE):
    input_test_data.append(np.load("preprocessed_dataset/low_res/" + input_test_files[index]))
    target_test_data.append(np.load("preprocessed_dataset/high_res/" + target_test_files[index]))
    print("Loaded validation sample {}".format(index))

model = create_model(NUMBER_OF_RESIDUAL_BLOCKS)
model.load_weights(CHECKPOINT_PATH)
model.compile(loss=normalised_root_mean_squared_error, optimizer='Adam',
              metrics=[signal_to_noise_ratio])

input_test_data = np.array(input_test_data)
target_test_data = np.array(target_test_data)

print("Input test data shape: {}".format(input_test_data.shape))
print("Target test data shape: {}".format(target_test_data.shape))

nrmse_loss, snr_metric = model.evaluate(input_test_data, target_test_data, batch_size=BATCH_SIZE, verbose=True)

print("NRMSE: {}".format(nrmse_loss))
print("SNR: {}".format(snr_metric))
