import tensorflow as tf
from model import create_model
from constants import *
from DatasetGenerator import DatasetGenerator
import numpy as np
from metrics import signal_to_noise_ratio, normalised_root_mean_squared_error
from tensorflow.keras.callbacks import ModelCheckpoint
import matplotlib.pyplot as plt
import os

model = create_model(NUMBER_OF_RESIDUAL_BLOCKS)
model.summary()
"""
model.compile(loss=normalised_root_mean_squared_error, optimizer='RMSprop',
              metrics=[signal_to_noise_ratio, normalised_root_mean_squared_error])
"""
model.compile(loss="mean_squared_error", optimizer='Adam',
              metrics=[signal_to_noise_ratio, normalised_root_mean_squared_error])
(input_data_files, target_data_files), (input_validation_files, target_validation_files), _ \
    = DatasetGenerator.split_list_of_files()
input_data, target_data, input_validation_data, target_validation_data = [], [], [], []

print("Loading the .npy files...")

print("Number of input data files: {}".format(len(input_data_files)))
print("Number of target data files: {}".format(len(target_data_files)))
number_of_input_batches = int(NUMBER_OF_TRAINING_TENSORS / BATCH_SIZE)
for index in range(0, number_of_input_batches*BATCH_SIZE):
    input_data.append(np.load("preprocessed_dataset/low_res/" + input_data_files[index]))
    target_data.append(np.load("preprocessed_dataset/high_res/" + target_data_files[index]))
    print("Loaded training sample {}".format(index))

number_of_validation_batches = int(NUMBER_OF_VALIDATION_TENSORS / BATCH_SIZE)
for index in range(0, number_of_validation_batches*BATCH_SIZE):
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
print("Number of input batches: {}".format(number_of_input_batches))
print("Number of validation batches: {}".format(number_of_validation_batches))
print("Number of input data files: {}".format(len(input_data_files)))  #112388
print("Number of validation data files: {}".format(len(input_validation_files)))  #6242
print("Training started...")

checkpoint_callback = ModelCheckpoint(filepath=CHECKPOINT_PATH,
                                      save_weights_only=True,
                                      verbose=True)

history = model.fit(input_data, target_data,
                    batch_size=BATCH_SIZE,
                    epochs=NUMBER_OF_EPOCHS,
                    validation_data=(input_validation_data, target_validation_data),
                    callbacks=[checkpoint_callback],
                    verbose=True)

print("model.fit history:")
print(list(history.history.keys()))
print(history.history)

fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 6), sharex=True)
# fig.tight_layout(pad=2.0)
axes[0].plot(history.history['loss'], color=(255/255.0, 0/255.0, 0/255.0))
axes[0].set_title("Training loss")
axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("Loss (MSE)")
axes[1].plot(history.history['val_loss'], color=(0/255.0, 255/255.0, 0/255.0))
axes[1].set_title("Validation loss")
axes[1].set_xlabel("Epoch")
plt.savefig("training_validation_plot.png", bbox_inches='tight')
plt.show()
