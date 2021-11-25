import tensorflow as tf
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
import librosa
import numpy as np
import soundfile as sf
import random
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

VCTK_DATASET_SAMPLING_RATE = 44100
RESAMPLING_FACTOR = 4
DOWNSAMPLED_RATE = int(VCTK_DATASET_SAMPLING_RATE / RESAMPLING_FACTOR)


class DatasetGenerator:
    def __init__(self):
        self.__training_dataset = None
        self.__testing_dataset = None

    def downsampling_demo(self):
        dataset, dataset_information = tfds.load(
            "vctk",
            #split=["train", "test"],
            #shuffle_files=True,
            with_info=True
        )

        print(dataset_information)
        number_of_samples = len(dataset['train'])
        sample_index = 0
        chosen_sample_index = 10 # random.randint(0, number_of_samples)
        print("Data samples: {}".format(number_of_samples))
        print("Chosen demo sample index: {}".format(chosen_sample_index))

        for sample in dataset['train']:
            if sample_index == chosen_sample_index:
                print("Stopped at sample index {}".format(sample_index))
                sample_array = np.array(sample['speech'], dtype=np.float)
                text = str(tf.keras.backend.get_value(sample['text']))
                print("Text read by the person: {}".format(text))
                downsampled_array = librosa.resample(
                    sample_array, VCTK_DATASET_SAMPLING_RATE, DOWNSAMPLED_RATE)
                sf.write(
                    'normal_audio_{}.wav'.format(text), sample_array, VCTK_DATASET_SAMPLING_RATE)
                sf.write(
                    'downsampled_audio_{}.wav'.format(text), downsampled_array, DOWNSAMPLED_RATE)

                print("Audio sample length: {}".format(len(sample_array)))
                print("Downsampled audio length: {}".format(len(downsampled_array)))

            sample_index += 1
