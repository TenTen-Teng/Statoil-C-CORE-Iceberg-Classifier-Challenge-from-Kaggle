import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from Data.constant import ALL_DATA_ICE, ALL_DATA_SHIP, TRAIN_AUG_JSON

ice_file_path = ALL_DATA_ICE
ship_file_path = ALL_DATA_SHIP

if not os.path.exists(ice_file_path):
    os.makedirs(ice_file_path)
if not os.path.exists(ship_file_path):
    os.makedirs(ship_file_path)

file_aug_path = TRAIN_AUG_JSON

train = pd.read_json(file_aug_path)

# set index of image
num_ice = 0
num_ship = 0

# generate images from 2nd file(band_h, band_v, band_r)
for i in range(len(train['is_iceberg'])):
    print(i+1, "/", len(train['is_iceberg']))
    plt.figure(0)
    plt.xticks(())
    plt.yticks(())
    if str(train['is_iceberg'][i]) == "0":
        plt.imshow(np.array(train["band_h_3"][i]).reshape(75, 75))
        plt.savefig(ship_file_path + 'ship_2_' + str(num_ship) + '.jpg')
        num_ship += 1

        plt.imshow(np.array(train["band_v_3"][i]).reshape(75, 75))
        plt.savefig(ship_file_path + 'ship_2_' + str(num_ship) + '.jpg')
        num_ship += 1

        plt.imshow(np.array(train["band_r_3"][i]).reshape(75, 75))
        plt.savefig(ship_file_path + 'ship_2_' + str(num_ship) + '.jpg')
    else:
        if str(train['is_iceberg'][i]) == "1":
            plt.imshow(np.array(train["band_h_3"][i]).reshape(75, 75))
            plt.savefig(ice_file_path + 'ice_2_' + str(num_ice) + '.jpg')
            num_ice += 1

            plt.imshow(np.array(train["band_v_3"][i]).reshape(75, 75))
            plt.savefig(ice_file_path + 'ice_2_' + str(num_ice) + '.jpg')
            num_ice += 1

            plt.imshow(np.array(train["band_r_3"][i]).reshape(75, 75))
            plt.savefig(ice_file_path + 'ice_2_' + str(num_ice) + '.jpg')
