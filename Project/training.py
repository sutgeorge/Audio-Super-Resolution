import tensorflow as tf
from model import create_model
from constants import *
from DatasetGenerator import DatasetGenerator
import numpy as np
from metrics import signal_to_noise_ratio, normalised_root_mean_squared_error
from tensorflow.keras.callbacks import ModelCheckpoint
import os

model = create_model(NUMBER_OF_RESIDUAL_BLOCKS)
model.summary()
model.compile(loss=normalised_root_mean_squared_error, optimizer='Adam',
              metrics=[signal_to_noise_ratio, normalised_root_mean_squared_error])
(input_data_files, target_data_files), (input_validation_files, target_validation_files), _ \
    = DatasetGenerator.split_list_of_files()
input_data, target_data, input_validation_data, target_validation_data = [], [], [], []

print("Loading the .npy files...")

for index in range(0, BATCH_SIZE*NUMBER_OF_BATCHES):
    input_data.append(np.load("preprocessed_dataset/low_res/" + input_data_files[index]))
    target_data.append(np.load("preprocessed_dataset/high_res/" + target_data_files[index]))
    print("Loaded training sample {}".format(index))

for index in range(0, 100):
    input_validation_data.append(np.load("preprocessed_dataset/low_res/" + input_validation_files[index]))
    target_validation_data.append(np.load("preprocessed_dataset/high_res/" + target_validation_files[index]))
    print("Loaded validation sample {}".format(index))

input_data = np.array(input_data)
target_data = np.array(target_data)
input_validation_data = np.array(input_validation_data)
target_validation_data = np.array(target_validation_data)

print("Some input tensor shape: {}".format(input_data[0].shape))
print("Some target tensor shape: {}".format(target_data[0].shape))
print("Input data: {}".format(input_data.shape))
print("Target data: {}".format(target_data.shape))
print("Input validation data: {}".format(input_validation_data.shape))
print("Target validation data: {}".format(target_validation_data.shape))
print("Training started...")

checkpoint_callback = ModelCheckpoint(filepath=CHECKPOINT_PATH,
                                      save_weights_only=True,
                                      verbose=True)

model.fit(input_data, target_data,
          batch_size=BATCH_SIZE,
          epochs=1,
          validation_data=(input_validation_data, target_validation_data),
          callbacks=[checkpoint_callback],
          verbose=True)
