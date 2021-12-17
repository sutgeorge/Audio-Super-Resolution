import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
from scipy import interpolate
import os
from constants import *


class DatasetGenerator:
    @staticmethod
    def generate_dataset(sample_dimension=SAMPLE_DIMENSION):
        dataset = tfds.load("vctk", with_info=False)
        sample_index, file_index = 0, 0

        for sample in dataset['train']:
            if sample_index == 100:
                break

            sample_array = np.array(sample['speech'], dtype=np.float)
            sample_array_length = len(sample_array)
            sample_array = sample_array[:sample_array_length - (sample_array_length % RESAMPLING_FACTOR)]

            downsampled_array = np.array(sample_array[0::RESAMPLING_FACTOR])
            downsampled_array = DatasetGenerator.upsample(downsampled_array, RESAMPLING_FACTOR)

            downsampled_array = np.reshape(downsampled_array, (len(downsampled_array), 1))
            sample_array = np.reshape(sample_array, (len(sample_array), 1))

            for index in range(0, len(sample_array) - sample_dimension, OVERLAP):
                low_resolution_chunk = downsampled_array[index:index + sample_dimension]
                high_resolution_chunk = sample_array[index:index + sample_dimension]
                filename = "sample_index_{}_chunk_index_{}_length_{}" \
                    .format(sample_index, index, sample_dimension)
                print("File {} generated - sample {}.".format(file_index, sample_index))
                np.save("preprocessed_dataset/low_res/lr_" + filename, low_resolution_chunk)
                np.save("preprocessed_dataset/high_res/hr_" + filename, high_resolution_chunk)
                file_index += 1

            sample_index += 1

    @staticmethod
    def split_list_of_files(training_percentage=0.9):
        low_resolution_files = np.sort(np.array(os.listdir("preprocessed_dataset/low_res")))
        high_resolution_files = np.sort(np.array(os.listdir("preprocessed_dataset/high_res")))
        final_index_for_training_chunk = int(len(low_resolution_files) * training_percentage) - 1
        validation_chunk_size = int(len(low_resolution_files) * (1 - training_percentage) / 2) - 1
        training_set = (low_resolution_files[:final_index_for_training_chunk],
                        high_resolution_files[:final_index_for_training_chunk])
        validation_set = (low_resolution_files[final_index_for_training_chunk:final_index_for_training_chunk+validation_chunk_size],
                          high_resolution_files[final_index_for_training_chunk:final_index_for_training_chunk+validation_chunk_size])
        testing_set = (low_resolution_files[final_index_for_training_chunk+validation_chunk_size:],
                       high_resolution_files[final_index_for_training_chunk+validation_chunk_size:])
        return training_set, validation_set, testing_set

    @staticmethod
    def upsample(low_resolution_array, resampling_factor):
        low_resolution_array = low_resolution_array.flatten()
        high_resolution_length = len(low_resolution_array) * resampling_factor

        knots = np.arange(high_resolution_length, step=resampling_factor)
        high_resolution_range = np.arange(high_resolution_length)

        spline = interpolate.splrep(knots, low_resolution_array)
        evaluated_spline = interpolate.splev(high_resolution_range, spline)

        return evaluated_spline
