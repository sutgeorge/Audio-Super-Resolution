from multiprocessing import Process
import tensorflow_datasets as tfds
from scipy import interpolate
from constants import *
import tensorflow as tf
import numpy as np
import datetime
import os


class DatasetGenerator:
    def __init__(self):
        self.__data = []
        self.__process_list = []
        self.__start_time, self.__end_time = None, None

    def run_generation_task(self, start_index, end_index, process_index, group_index):
        for recording_index in range(start_index, end_index):
            file_index = 0
            sample_array = self.__data[recording_index]
            sample_array_length = len(sample_array)
            sample_array = sample_array[:sample_array_length - (sample_array_length % RESAMPLING_FACTOR)]

            for sample_index in range(0, len(sample_array) - SAMPLE_DIMENSION, OVERLAP):
                high_resolution_chunk = sample_array[sample_index:sample_index + SAMPLE_DIMENSION]
                low_resolution_chunk = high_resolution_chunk[0::RESAMPLING_FACTOR]
                high_resolution_chunk = np.reshape(high_resolution_chunk, (len(high_resolution_chunk), 1))
                low_resolution_chunk = np.reshape(low_resolution_chunk, (len(low_resolution_chunk), 1))
                filename = "recording_index_{}_sample_index_{}_group_index_{}" \
                    .format(group_index*AMOUNT_OF_TRACKS_IN_A_DATA_GENERATION_BATCH+recording_index, sample_index, group_index)
                print("Process {} generated the filepair no. {} from recording {} (group {})".format(
                    process_index, file_index, group_index*AMOUNT_OF_TRACKS_IN_A_DATA_GENERATION_BATCH+recording_index, group_index))
                np.save("preprocessed_dataset/low_res/lr_" + filename + "_length_" + str(LOW_RESOLUTION_DIMENSION), low_resolution_chunk)
                np.save("preprocessed_dataset/high_res/hr_" + filename + "_length_" + str(SAMPLE_DIMENSION), high_resolution_chunk)
                file_index += 1

    def find_workload_interval(self, process_index):
        return ((process_index * len(self.__data)) // NUMBER_OF_PROCESSES,
                ((process_index + 1) * len(self.__data)) // NUMBER_OF_PROCESSES)

    def generate_dataset(self):
        self.__start_time = datetime.datetime.now()
        print("Data generation started at {}".format(self.__start_time.strftime("%Y-%m-%d %H:%M:%S")))

        dataset = tfds.load("vctk", with_info=False)
        dataset = dataset['train'].take(AMOUNT_OF_TRACKS_USED_FOR_DATA_GENERATION)
        sample_index, tracks_processing_batch_index = 0, 0

        for sample in dataset:
            print("Adding sample no. {} to the list...".format(sample_index))
            self.__data.append(np.array(sample['speech'], dtype=np.float))
            sample_index += 1
            if sample_index % AMOUNT_OF_TRACKS_IN_A_DATA_GENERATION_BATCH == 0:
                for process_index in range(0, NUMBER_OF_PROCESSES):
                    workload_interval = self.find_workload_interval(process_index)
                    process = Process(target=self.run_generation_task,
                                      args=(workload_interval[0], workload_interval[1], process_index, tracks_processing_batch_index))
                    process.start()
                    self.__process_list.append(process)

                for process in self.__process_list:
                    process.join()

                self.__data = []
                self.__process_list = []
                tracks_processing_batch_index += 1

        self.__end_time = datetime.datetime.now()
        print("Data generation started at {}".format(self.__start_time.strftime("%Y-%m-%d %H:%M:%S")))
        print("Data generation ended at {}".format(self.__end_time.strftime("%Y-%m-%d %H:%M:%S")))

    @staticmethod
    def split_list_of_files():
        low_resolution_files = np.sort(np.array(os.listdir("preprocessed_dataset/low_res")))
        high_resolution_files = np.sort(np.array(os.listdir("preprocessed_dataset/high_res")))
        training_set = (low_resolution_files[:NUMBER_OF_TRAINING_TENSORS],
                        high_resolution_files[:NUMBER_OF_TRAINING_TENSORS])
        validation_set = (low_resolution_files[NUMBER_OF_TRAINING_TENSORS:NUMBER_OF_TRAINING_TENSORS+NUMBER_OF_VALIDATION_TENSORS],
                          high_resolution_files[NUMBER_OF_TRAINING_TENSORS:NUMBER_OF_TRAINING_TENSORS+NUMBER_OF_VALIDATION_TENSORS])
        testing_set = (low_resolution_files[NUMBER_OF_TRAINING_TENSORS+NUMBER_OF_VALIDATION_TENSORS:],
                       high_resolution_files[NUMBER_OF_TRAINING_TENSORS+NUMBER_OF_VALIDATION_TENSORS:])
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
