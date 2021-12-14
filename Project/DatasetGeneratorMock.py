import tensorflow as tf
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
import librosa
import librosa.display
import numpy as np
import soundfile as sf
import time
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
