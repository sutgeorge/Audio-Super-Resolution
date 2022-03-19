from model import create_model
from constants import *
import tensorflow_datasets as tfds
import numpy as np
from metrics import *
from tensorflow.keras.models import Model
import librosa.display
import soundfile as sf
import matplotlib.pyplot as plt
import random
import matplotlib.pylab as pl
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

print("Loading and compiling model...")
model = create_model()
model.summary()

model.compile(loss="mean_squared_error", optimizer='Adam',
              metrics=[signal_to_noise_ratio, normalised_root_mean_squared_error],
              run_eagerly=True)
model.load_weights(MODEL_PATH)

print("Model layers: {}".format(model.layers))
print("Loading the sample vocal recording from the VCTK dataset...")
dataset = tfds.load("vctk", with_info=False)
sample_array = None
transcript = None
recording_index = 0

chosen_recording = random.randint(AMOUNT_OF_TRACKS_USED_FOR_DATA_GENERATION + 1, AMOUNT_OF_TRACKS_USED_FOR_DATA_GENERATION + 100)

for sample in dataset['train']:
    if recording_index == chosen_recording:
        transcript = sample['text']
        print("Recording transcript: {}".format(transcript))
        sample_array = np.array(sample['speech'], dtype=float)
        break
    recording_index += 1

print("Downsampling the audio...")

sample_array_length = len(sample_array)
sample_array = sample_array[:sample_array_length - (sample_array_length % RESAMPLING_FACTOR)]

high_resolution_chunk = sample_array[(sample_array_length // 2):(sample_array_length // 2) + SAMPLE_DIMENSION]
low_resolution_chunk = high_resolution_chunk[::RESAMPLING_FACTOR]
low_resolution_chunk = np.reshape(low_resolution_chunk, (len(low_resolution_chunk), 1))
input_batch = BATCH_SIZE * [low_resolution_chunk]
input_batch = tf.constant(input_batch)

x = np.linspace(1, 5, SAMPLE_DIMENSION)

pl.figure(figsize=(20,20))
ax = pl.subplot(projection='3d')

number_of_layers = len(model.layers)
subset_of_layers = []

for layer_index in range(0, number_of_layers):
    print("Plotting the output of layer {}...".format(layer_index))
    subset_of_layers.append(model.layers[layer_index].output)
    auxiliary_model = Model(inputs=model.inputs, outputs=model.outputs + subset_of_layers)
    print(model.layers[layer_index].output)
    intermediate_layer_output = auxiliary_model.predict(input_batch)

    if type(intermediate_layer_output) != list:
        print("Intermediate layer output shape: {}".format(intermediate_layer_output[0].shape))
        intermediate_layer_output = intermediate_layer_output[0]
        print("Intermediate layer output shape after cropping array: {}".format(intermediate_layer_output.shape))
        intermediate_layer_output = np.reshape(intermediate_layer_output, intermediate_layer_output.shape[0])
        print("Intermediate layer output shape after reshaping: {}".format(intermediate_layer_output.shape))
    else:
        print("Intermediate layer output shape: {}".format(intermediate_layer_output[0].shape))
        intermediate_layer_output = intermediate_layer_output[0]
        print("Intermediate layer output shape after cropping array: {}".format(intermediate_layer_output.shape))
        print(intermediate_layer_output)
        intermediate_layer_output = np.reshape(intermediate_layer_output[0], intermediate_layer_output.shape[1])
        print("Intermediate layer output shape after reshaping: {}".format(intermediate_layer_output.shape))


    z_layer_output = intermediate_layer_output # np.reshape(intermediate_layer_output, intermediate_layer_output.shape[0])
    remaining_amount_of_zeroes = (SAMPLE_DIMENSION - len(z_layer_output)) // 2
    zeroes = np.zeros(remaining_amount_of_zeroes)
    z_layer_output = np.concatenate([zeroes, z_layer_output, zeroes])
    y = np.ones(x.size)*layer_index
    ax.plot(x, y, z_layer_output, color='r')
    print("Plotted layer {}'s output.".format(layer_index))

ax.set_xlabel('Time axis')
ax.set_zlabel('Amplitude')

plt.savefig("outputs/intermediate-layer-outputs/layer-outputs.png")
plt.show()


