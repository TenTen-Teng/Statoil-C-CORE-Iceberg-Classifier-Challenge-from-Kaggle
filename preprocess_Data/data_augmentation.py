import pandas as pd
import numpy as np
import json
import cv2
import os
from Data.constant import TRAIN_BAND3_JSON, TRAIN_AUG_JSON, PRE_DATA_ROOT

# file_path is the path of the json file with 3 band data
# new_file_path is the new file path
file_path = TRAIN_BAND3_JSON
new_file_path = TRAIN_AUG_JSON

if not os.path.exists(PRE_DATA_ROOT):
    os.makedirs(PRE_DATA_ROOT)

# read data
train = pd.read_json(file_path)

# data augmentation
list = []
list_final = []
data_aug = {}

for i in range(len(train["is_iceberg"])):
    print(i + 1, '/', len(train["is_iceberg"]))
    # add is_iceberg to dict
    data_aug["is_iceberg"] = str(train["is_iceberg"][i])

    band_3 = np.array(train["band_3"][i])
    # horizontal flip
    image_h_3 = cv2.flip(band_3, 0)
    # vertical filp
    image_v_3 = cv2.flip(band_3, 1)

    data_aug["band_h_3"] = np.array(image_h_3).tolist()
    data_aug["band_v_3"] = np.array(image_v_3).tolist()

    # rotate image
    band_3_reshape = np.array(train["band_3"][i]).reshape(75, 75)
    # rotate band3 image
    angle = 90
    rows, cols = band_3_reshape.shape
    M = cv2.getRotationMatrix2D((cols / 2, rows / 2), 90, 1)
    image_r = cv2.warpAffine(band_3_reshape, M, (cols, rows))
    data_aug["band_r_3"] = np.array(image_r).tolist()

    list.append(data_aug.copy())

# write to a new json file
new_data = json.dumps(list)
loader = json.loads(new_data)

with open(new_file_path, 'w+') as f:
    f.writelines(json.dumps(loader))
    f.close()