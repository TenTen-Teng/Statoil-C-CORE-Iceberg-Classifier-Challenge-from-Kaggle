import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from Data.constant import TEST_JSON_PATH, TEST_PATH

json_path = TEST_JSON_PATH

image_path = TEST_PATH

if not os.path.exists(image_path):
    os.makedirs(image_path)

test = pd.read_json(json_path)

# calculate band3
count = 1
for i in range(len(test)):
    print(i, "/", len(test))

    band_1 = np.array(test["band_1"][i])
    band_2 = np.array(test["band_2"][i])

    num_band1 = band_1
    num_band2 = band_2
    band3 = num_band1 + num_band2

    plt.imshow(band3.reshape(75, 75))

    if count in range(1, 10):
        prefix = "000"
        plt.savefig(image_path + prefix + str(count) + '.jpg')
        count += 1
    else:
        if count in range(10, 100):
            prefix = "00"
            plt.savefig(image_path + prefix + str(count) + '.jpg')
            count += 1
        else:
            if count in range(100, 1000):
                prefix = "0"
                plt.savefig(image_path + prefix + str(count) + '.jpg')
                count += 1
            else:
                plt.savefig(image_path + str(count) + '.jpg')
                count += 1


