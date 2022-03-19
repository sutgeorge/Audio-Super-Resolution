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
model.compile(loss="mean_squared_error", optimizer='Adam',
              metrics=[signal_to_noise_ratio, normalised_root_mean_squared_error],
              run_eagerly=True)
model.load_weights(MODEL_PATH)

# auxiliary_model = Model(inputs=model.inputs, outputs=model.outputs + [model.layers[1]])

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

x = np.linspace(1, 5, SAMPLE_DIMENSION)
y = np.ones(x.size)
y2 = np.ones(x.size)*2

remaining_amount_of_zeroes = (SAMPLE_DIMENSION - LOW_RESOLUTION_DIMENSION) // 2
z_low_res = low_resolution_chunk.tolist()
zeroes = np.zeros(remaining_amount_of_zeroes).tolist()
z_low_res = zeroes + z_low_res + zeroes
z_high_res = high_resolution_chunk

pl.figure(figsize=(20,20))
ax = pl.subplot(projection='3d')
ax.plot(x, y, z_high_res, color='r')
ax.plot(x, y2, z_low_res, color='g')

ax.set_xlabel('Time')
ax.set_zlabel('Amplitude')

plt.savefig("outputs/intermediate-layer-outputs/layer-outputs.png")
plt.show()

