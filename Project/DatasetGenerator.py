import tensorflow as tf
import tensorflow_datasets as tfds
import librosa
import librosa.display
import numpy as np
import soundfile as sf
import time
from scipy import interpolate

VCTK_DATASET_SAMPLING_RATE = 48000
RESAMPLING_FACTOR = 4
DOWNSAMPLED_RATE = int(VCTK_DATASET_SAMPLING_RATE / RESAMPLING_FACTOR)
OVERLAP = 128


class DatasetGenerator:
    def __init__(self):
        self.__training_dataset = None
        self.__testing_dataset = None

    def upsample(self, low_resolution_array, resampling_factor):
        low_resolution_array = low_resolution_array.flatten()
        high_resolution_length = len(low_resolution_array) * resampling_factor

        knots = np.arange(high_resolution_length, step=resampling_factor)
        high_resolution_range = np.arange(high_resolution_length)

        spline = interpolate.splrep(knots, low_resolution_array)
        evaluated_spline = interpolate.splev(high_resolution_range, spline)

        return evaluated_spline

    def generate_dataset(self, sample_dimension=256):
        dataset = tfds.load("vctk", with_info=False)
        sample_index = 0

        file_index = 0
        for sample in dataset['train']:
            sample_array = np.array(sample['speech'], dtype=np.float)
            sample_array_length = len(sample_array)
            sample_array = sample_array[:sample_array_length - (sample_array_length % RESAMPLING_FACTOR)]

            downsampled_array = np.array(sample_array[0::RESAMPLING_FACTOR])
            downsampled_array = self.upsample(downsampled_array, RESAMPLING_FACTOR)

            downsampled_array = np.reshape(downsampled_array, (len(downsampled_array), 1))
            sample_array = np.reshape(sample_array, (len(sample_array), 1))

            for index in range(0, len(sample_array) - sample_dimension, OVERLAP):
                low_resolution_chunk = downsampled_array[index:index+sample_dimension]
                high_resolution_chunk = sample_array[index:index+sample_dimension]
                pair = np.array([low_resolution_chunk, high_resolution_chunk])
                filename = "lr_hr_pair_sample_index_{}_chunk_index_{}_length_{}"\
                    .format(sample_index, index, sample_dimension)
                print("File {} generated - sample {}.".format(file_index, sample_index))
                np.save("preprocessed_dataset/" + filename, pair)
                file_index += 1

            sample_index += 1

