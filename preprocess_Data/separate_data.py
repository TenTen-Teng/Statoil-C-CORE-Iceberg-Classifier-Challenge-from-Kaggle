import os
import shutil
from Data.constant import SEP_TRAIN_ICE, SEP_TRAIN_SHIP, SEP_VAL_ICE, SEP_VAL_SHIP, ALL_DATA_ICE, ALL_DATA_SHIP
from Data.constant import DATA_ZIP_PATH, SEP_DATA_ROOT, zipfolder
# separate data
# train:validation -> 8:2

train_ice_path = SEP_TRAIN_ICE
train_ship_path = SEP_TRAIN_SHIP
val_ice_path = SEP_VAL_ICE
val_ship_path = SEP_VAL_SHIP

if not os.path.exists(train_ice_path):
    os.makedirs(train_ice_path)
if not os.path.exists(train_ship_path):
    os.makedirs(train_ship_path)
if not os.path.exists(val_ice_path):
    os.makedirs(val_ice_path)
if not os.path.exists(val_ship_path):
    os.makedirs(val_ship_path)

ice_image_path = ALL_DATA_ICE
ship_image_path = ALL_DATA_SHIP

# ice images
num_ice = 0
for dirpath, subdirs, files in os.walk(ice_image_path):
    for file in files:
        num_ice += 1

print("number of ice image: ", num_ice)

train_num = 0.8 * num_ice
test_num = 0.2 * num_ice

for dirpath, subdirs, files in os.walk(ice_image_path):
    count = 0
    for file in files:
        if count < train_num:
            shutil.copy2(ice_image_path + file, train_ice_path)
        else:
            shutil.copy2(ice_image_path + file, val_ice_path)

        count += 1

# ship images
num_ship = 0
for dirpath, subdirs, files in os.walk(ship_image_path):
    for file in files:
        num_ship += 1

print("number of ship images: ", num_ship)

train_num = 0.8 * num_ship
test_num = 0.2 * num_ship

for dirpath, subdirs, files in os.walk(ship_image_path):
    count = 0
    for file in files:
        if count < train_num:
            shutil.copy2(ship_image_path + file, train_ship_path)
        else:
            shutil.copy2(ship_image_path + file, val_ship_path)
        count += 1

# zip file
zipfolder(DATA_ZIP_PATH, SEP_DATA_ROOT)

# print number of image
num_train_ice = 0
num_val_ice = 0
num_train_ship = 0
num_val_ship = 0
for dirpath, subdirs, files in os.walk(train_ice_path):
    for file in files:
        num_train_ice += 1
print("train data of ice = ", num_train_ice)

for dirpath, subdirs, files in os.walk(val_ice_path):
    for file in files:
        num_val_ice += 1
print("val data of ice = ", num_val_ice)

for dirpath, subdirs, files in os.walk(train_ship_path):
    for file in files:
        num_train_ship += 1
print("train data of ship = ", num_train_ship)

for dirpath, subdirs, files in os.walk(val_ship_path):
    for file in files:
        num_val_ship += 1
print("val data of ship= ", num_val_ship)