import pandas as pd
import numpy as np
import json
import os
from Data.constant import TRAIN_JSON_PATH, TRAIN_BAND3_JSON, PRE_DATA_ROOT


# file_path is the path of the raw train json file
# new_file_path is the new json file with 3 band data of each image
file_path = TRAIN_JSON_PATH
new_file_path = TRAIN_BAND3_JSON

if not os.path.exists(PRE_DATA_ROOT):
    os.makedirs(PRE_DATA_ROOT)

train = pd.read_json(file_path)

# calculate band3
new_band_3 = []
for i in range(len(train)):
    band_1 = np.array(train["band_1"][i])
    band_2 = np.array(train["band_2"][i])

    num_band1 = band_1
    num_band2 = band_2
    num_band3 = num_band1 + num_band2

    new_band_3.append(num_band3.tolist())

with open(file_path, 'r') as f:
    raw_data = json.load(f)

for i in range(len(raw_data)):
    raw_data[i]["band_3"] = new_band_3[i]

# write to a new json file
with open(new_file_path, 'w+') as f:
    f.writelines(json.dumps(raw_data))