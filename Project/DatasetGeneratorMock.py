import tensorflow as tf
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
import librosa
import librosa.display
import numpy as np
import soundfile as sf
import time
from scipy import interpolate
import random
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

VCTK_DATASET_SAMPLING_RATE = 48000
RESAMPLING_FACTOR = 4
DOWNSAMPLED_RATE = int(VCTK_DATASET_SAMPLING_RATE / RESAMPLING_FACTOR)


class DatasetGeneratorMock:
    def __init__(self):
        self.__training_dataset = None
        self.__testing_dataset = None

    def downsampling_demo(self):
        dataset, dataset_information = tfds.load(
            "vctk",
            with_info=True
        )

        sample = next(iter(dataset['train']))
        print(sample)
        start_time = time.process_time()
        sample_array = np.array(sample['speech'], dtype=np.float)
        text = str(tf.keras.backend.get_value(sample['text']))
        print("Text read by the person: {}".format(text))
        downsampled_array = librosa.resample(
            sample_array, VCTK_DATASET_SAMPLING_RATE, DOWNSAMPLED_RATE, res_type='linear')
        sample_array = np.int16(sample_array)
        downsampled_array = np.int16(downsampled_array)
        sf.write(
            'normal_audio_{}.wav'.format(text), sample_array, VCTK_DATASET_SAMPLING_RATE)
        sf.write(
            'downsampled_audio_{}.wav'.format(text), downsampled_array, DOWNSAMPLED_RATE)
        end_time = time.process_time()
        print("Elapsed time: {} seconds".format(end_time - start_time))
        print("Normal audio length: {}".format(len(sample_array)))
        print("Downsampled audio length: {}".format(len(downsampled_array)))

    def upsample(self, low_resolution_array, resampling_factor):
        low_resolution_array = low_resolution_array.flatten()
        high_resolution_length = len(low_resolution_array) * resampling_factor
        x_sp = np.zeros(high_resolution_length)

        knots = np.arange(high_resolution_length, step=resampling_factor)
        high_resolution_range = np.arange(high_resolution_length)

        spline = interpolate.splrep(knots, low_resolution_array)
        evaluated_spline = interpolate.splev(high_resolution_range, spline)

        return evaluated_spline

    def decimate_and_interpolate(self):
        dataset, dataset_information = tfds.load(
            "vctk",
            with_info=True
        )

        resampling_factor = 4

        sample = next(iter(dataset['train']))
        print(sample)
        start_time = time.process_time()
        sample_array = np.array(sample['speech'], dtype=np.float)
        text = str(tf.keras.backend.get_value(sample['text']))
        print("Text read by the person: {}".format(text))
        sample_array_length = len(sample_array)
        sample_array = sample_array[:sample_array_length - (sample_array_length % resampling_factor)]
        downsampled_array = np.array(sample_array[0::resampling_factor])
        downsampled_array = self.upsample(downsampled_array, resampling_factor)

        downsampled_array = np.reshape(downsampled_array, (len(downsampled_array), 1))
        sample_array = np.reshape(sample_array, (len(sample_array), 1))
        sf.write(
            'normal_audio_{}.wav'.format(text), np.int16(sample_array), VCTK_DATASET_SAMPLING_RATE)
        sf.write(
            'downsampled_audio_{}.wav'.format(text), np.int16(downsampled_array), DOWNSAMPLED_RATE)
        end_time = time.process_time()
        print("Elapsed time: {} seconds".format(end_time - start_time))
        print("Normal audio length: {}".format(len(sample_array)))
        print("Downsampled audio length: {}".format(len(downsampled_array)))
