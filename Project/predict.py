from model import create_model
from constants import *
import tensorflow_datasets as tfds
import tensorflow as tf
from DatasetGenerator import DatasetGenerator
import numpy as np
from metrics import *
import librosa
import librosa.display
import soundfile as sf
import datetime
import matplotlib.pyplot as plt
import random
import re

print("Loading and compiling model...")
model = create_model()
model.compile(loss="mean_squared_error", optimizer='Adam',
              metrics=[signal_to_noise_ratio, normalised_root_mean_squared_error],
              run_eagerly=True)
# model.load_weights(CHECKPOINT_PATH).expect_partial()
model.load_weights(MODEL_PATH)

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
downsampled_array = np.array(sample_array[0::RESAMPLING_FACTOR])

downsampled_array = np.reshape(downsampled_array, (len(downsampled_array), 1))
sample_array = np.reshape(sample_array, (len(sample_array), 1))

print("Sample array length: {}".format(len(sample_array)))
print("Downsampled array length: {}".format(len(downsampled_array)))

super_resolution_output_chunks = []

print("Cutting chunks from the downsampled audio (length={}) and feeding batches to the model...".format(len(downsampled_array)))

for sample_index in range(0, len(downsampled_array), LOW_RESOLUTION_DIMENSION * BATCH_SIZE):
    print("> Sample index {}".format(sample_index))
    input_batch = []

    for batch_sample_index in range(0, BATCH_SIZE):
        print("--- Batch sample index {}".format(batch_sample_index))
        if sample_index + (batch_sample_index + 1) * LOW_RESOLUTION_DIMENSION >= len(downsampled_array):
            chunk = downsampled_array[sample_index + batch_sample_index * LOW_RESOLUTION_DIMENSION:]
            zeroes_to_add = SAMPLE_DIMENSION // RESAMPLING_FACTOR - len(chunk)
            array_of_zeroes = np.zeros(zeroes_to_add)
            chunk = np.append(chunk, array_of_zeroes)
            chunk = chunk.reshape((chunk.shape[0], 1))
            print("Chunk shape (1):")
            print(chunk.shape)
        else:
            chunk = downsampled_array[sample_index + batch_sample_index * LOW_RESOLUTION_DIMENSION:
                                      sample_index + (batch_sample_index + 1) * LOW_RESOLUTION_DIMENSION]
            print("Chunk shape (2):")
            print(chunk.shape)
        print(chunk)
        input_batch.append(chunk)

    print("> Feeding the low-res batch to the model...")
    prediction = model.predict(np.array(input_batch))
    super_resolution_output_chunks.append(prediction.reshape((prediction.shape[0]*prediction.shape[1])))
    print("Prediction done. Output has been appended to the result.")

output = np.array([element for sublist in super_resolution_output_chunks for element in sublist])
print(output)
print("Output shape: {}".format(output.shape))
output = output[:sample_array.size]

downsampled_array = downsampled_array.reshape(downsampled_array.shape[0])
sample_array = sample_array.reshape(sample_array.shape[0])
low_resolution_signal_spectrogram = librosa.feature.melspectrogram(y=downsampled_array, sr=DOWNSAMPLED_RATE)
high_resolution_signal_spectrogram = librosa.feature.melspectrogram(y=sample_array, sr=VCTK_DATASET_SAMPLING_RATE)
super_resolution_signal_spectrogram = librosa.feature.melspectrogram(y=output, sr=VCTK_DATASET_SAMPLING_RATE)

print("Computed the spectrograms.")

fig, ax = plt.subplots(3, 1, figsize=(10, 10))
low_res_decibel_units = librosa.power_to_db(low_resolution_signal_spectrogram, ref=np.max)
high_res_decibel_units = librosa.power_to_db(high_resolution_signal_spectrogram, ref=np.max)
super_res_decibel_units = librosa.power_to_db(super_resolution_signal_spectrogram, ref=np.max)
ax[0].set_title("Low-res ({} samples)".format(downsampled_array.shape[0]))
first_subplot_spectrogram = librosa.display.specshow(low_res_decibel_units, x_axis='time',
                               y_axis='mel', sr=DOWNSAMPLED_RATE, ax=ax[0])
ax[1].set_title("High-res ({} samples)".format(sample_array.shape[0]))
second_subplot_spectrogram = librosa.display.specshow(high_res_decibel_units, x_axis='time',
                               y_axis='mel', sr=VCTK_DATASET_SAMPLING_RATE, ax=ax[1])
ax[2].set_title("Super-res ({} samples)".format(output.size))
third_subplot_spectrogram = librosa.display.specshow(super_res_decibel_units, x_axis='time',
                               y_axis='mel', sr=VCTK_DATASET_SAMPLING_RATE, ax=ax[2])

fig.tight_layout()
fig.colorbar(third_subplot_spectrogram, ax=[ax[0], ax[1], ax[2]], format='%+2.0f dB')
plt.savefig("outputs/spectrograms-track-{}-transcript-{}.png".format(chosen_recording, transcript))
plt.show()

sf.write("outputs/track-no-{}-high-res.wav".format(chosen_recording), np.int16(sample_array), VCTK_DATASET_SAMPLING_RATE)
sf.write("outputs/track-no-{}-low-res.wav".format(chosen_recording), np.int16(downsampled_array), DOWNSAMPLED_RATE)
sf.write("outputs/track-no-{}-super-res.wav".format(chosen_recording), np.int16(output.reshape(output.shape)), VCTK_DATASET_SAMPLING_RATE)

