import tensorflow as tf
from model import create_model
from constants import *
from DatasetGenerator import DatasetGenerator
import numpy as np
from metrics import signal_to_noise_ratio

model = create_model(NUMBER_OF_RESIDUAL_BLOCKS)
model.summary()
model.compile(loss='mse', optimizer='Adam', metrics=[signal_to_noise_ratio])
(input_data_files, target_data_files), (input_validation_files, target_validation_files), _ = DatasetGenerator.split_list_of_files()
input_data, target_data = [], []


print("Loading the .npy files...")

for index in range(0, BATCH_SIZE*NUMBER_OF_BATCHES):
    input_data.append(np.load("preprocessed_dataset/low_res/" + input_data_files[index]))
    target_data.append(np.load("preprocessed_dataset/high_res/" + target_data_files[index]))
    print("Loaded sample {}".format(index))

input_data = np.array(input_data)
target_data = np.array(target_data)

print("Some input tensor shape: {}".format(input_data[0].shape))
print("Some target tensor shape: {}".format(target_data[0].shape))
print("Input data: {}".format(input_data.shape))
print("Target data: {}".format(target_data.shape))
print("Training started...")

model.fit(input_data, target_data,
          batch_size=BATCH_SIZE,
          epochs=1,
          validation_data=(input_validation_files, target_validation_files),
          verbose=True)
