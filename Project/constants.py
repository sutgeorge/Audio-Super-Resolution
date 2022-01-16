import os
VCTK_DATASET_SAMPLING_RATE = 48000
RESAMPLING_FACTOR = 4
DOWNSAMPLED_RATE = int(VCTK_DATASET_SAMPLING_RATE / RESAMPLING_FACTOR)
OVERLAP = 128
SAMPLE_DIMENSION = 256
NUMBER_OF_EPOCHS = 100
NUMBER_OF_FILES = len(os.listdir("preprocessed_dataset/low_res/"))
NUMBER_OF_TRAINING_TENSORS = int(0.9 * NUMBER_OF_FILES)
NUMBER_OF_VALIDATION_TENSORS = int(0.05 * NUMBER_OF_FILES)
NUMBER_OF_TESTING_TENSORS = int(0.05 * NUMBER_OF_FILES)
BATCH_SIZE = 16  # The number of input tensors should be divisible by the batch size
NUMBER_OF_RESIDUAL_BLOCKS = 6
CHECKPOINT_PATH = "checkpoints/checkpoint.ckpt"
