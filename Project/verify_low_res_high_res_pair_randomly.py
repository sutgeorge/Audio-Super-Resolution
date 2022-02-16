import numpy as np
import matplotlib.pyplot as plt
import os
import random

low_resolution_files = np.sort(np.array(os.listdir("preprocessed_dataset/low_res")))
high_resolution_files = np.sort(np.array(os.listdir("preprocessed_dataset/high_res")))

chosen_file_index = random.randint(0, len(low_resolution_files))

low_res_array = np.load("preprocessed_dataset/low_res/" + low_resolution_files[chosen_file_index])
high_res_array = np.load("preprocessed_dataset/high_res/" + high_resolution_files[chosen_file_index])

figure, axes = plt.subplots(2, 1, figsize=(10, 8))

axes[0].set_title(low_resolution_files[chosen_file_index])
axes[0].plot(low_res_array)

axes[1].set_title(high_resolution_files[chosen_file_index])
axes[1].plot(high_res_array)

plt.show()
