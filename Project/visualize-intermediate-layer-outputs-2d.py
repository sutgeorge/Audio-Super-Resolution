from model import create_model
from constants import *
import tensorflow_datasets as tfds
import tensorflow.keras.backend as K
import numpy as np
from metrics import *
from tensorflow.keras.models import Model
import soundfile as sf
import matplotlib.pyplot as plt
import random
import matplotlib.pylab as pl
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import math

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

if "layer_outputs" not in os.listdir("./"):
    os.mkdir("./layer_outputs")

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

super_resolution_chunk = model.predict(input_batch)[0]

number_of_layers = len(model.layers)
subset_of_layers = []

plot_index = 0
residual_block_index = 0

figure, axes = plt.subplots(3, 1, figsize=(20, 20))
axes[0].set_title("Low-res")
axes[0].plot(low_resolution_chunk)
axes[1].set_title("High-res")
axes[1].plot(high_resolution_chunk)
axes[2].set_title("Super-res")
axes[2].plot(super_resolution_chunk)
figure.suptitle("Chunks")
plt.savefig("./layer_outputs/chunks.png")
plt.close()

figure, axes = plt.subplots(3, 1, figsize=(20, 20))
axes[0].set_title("Low-res")
axes[0].plot(low_resolution_chunk[(len(low_resolution_chunk) // 2):(len(low_resolution_chunk) // 2) + 25])
axes[1].set_title("High-res")
axes[1].plot(high_resolution_chunk[(len(high_resolution_chunk) // 2):(len(high_resolution_chunk) // 2) + 100])
axes[2].set_title("Super-res")
axes[2].plot(super_resolution_chunk[(len(super_resolution_chunk) // 2):(len(super_resolution_chunk) // 2) + 100])
figure.suptitle("Small parts of the chunks")
plt.savefig("./layer_outputs/minuscule-parts-of-the-chunks.png")
plt.close()

for layer_index in range(0, number_of_layers):
    if 'conv' not in model.layers[layer_index].name:
        continue
    print("Plotting the output of layer {}...".format(layer_index))
    subset_of_layers.append(model.layers[layer_index].output)
    auxiliary_model = Model(inputs=model.input, outputs=model.layers[layer_index].output)
    auxiliary_model.summary()
    intermediate_layer_output = auxiliary_model.predict(input_batch)

    intermediate_layer_output = intermediate_layer_output[0]
    number_of_samples = intermediate_layer_output.shape[0]
    number_of_filters = intermediate_layer_output.shape[1]
    colors = ['r', 'g', 'b', 'm', 'c', 'y', 'k']
    plot_size = int(math.sqrt(number_of_filters))

    for current_filter_index in range(0, number_of_filters, 4):
        try:
            figure, axes = plt.subplots(2, 2, figsize=(20, 20))
            axes[0, 0].set_title("Block {} | Filter {}".format(residual_block_index, current_filter_index))
            axes[0, 0].plot(intermediate_layer_output[:, current_filter_index], color=colors[current_filter_index % len(colors)])

            axes[0, 1].set_title("Block {} | Filter {}".format(residual_block_index, current_filter_index + 1))
            axes[0, 1].plot(intermediate_layer_output[:, current_filter_index + 1], color=colors[(current_filter_index + 1) % len(colors)])

            axes[1, 0].set_title("Block {} | Filter {}".format(residual_block_index, current_filter_index + 2))
            axes[1, 0].plot(intermediate_layer_output[:, current_filter_index + 2], color=colors[(current_filter_index + 2) % len(colors)])

            axes[1, 1].set_title("Block {} | Filter {}".format(residual_block_index, current_filter_index + 3))
            axes[1, 1].plot(intermediate_layer_output[:, current_filter_index + 3], color=colors[(current_filter_index + 3) % len(colors)])
            plt.savefig("./layer_outputs/{}".format(str(plot_index) + ".png"))
            plot_index += 1
        except Exception as e:
            continue

    print("Plotted layer {}'s output.".format(layer_index))
    residual_block_index += 1

