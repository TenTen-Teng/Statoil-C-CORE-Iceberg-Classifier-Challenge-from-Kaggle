import os
import random
import numpy as np
import fnmatch
import cv2
import caffe
import lmdb
import os
from Data.constant import TRAIN_PATH, VAL_PATH, check_file

# check data file
check_file(TRAIN_PATH)
check_file(VAL_PATH)

train_lmdb = 'train_lmdb'
test_lmdb = 'test_lmdb'

JPG_train_path = 'train'
JPG_test_path = 'test'

IMAGE_WIDTH = 75
IMAGE_HEIGHT = 75


def transform_img(img, img_width, img_height):
    # Histogram Equalization
    img[:, :, 0] = cv2.equalizeHist(img[:, :, 0])
    img[:, :, 1] = cv2.equalizeHist(img[:, :, 1])
    img[:, :, 2] = cv2.equalizeHist(img[:, :, 2])

    # Image Resizing
    img = cv2.resize(img, (img_width, img_height), interpolation=cv2.INTER_CUBIC)

    return img


def make_datum(img, label):
    return caffe.proto.caffe_pb2.Datum(
        channels=3,
        width=IMAGE_WIDTH,
        height=IMAGE_HEIGHT,
        label=label,
        data=np.rollaxis(img, axis=2, start=0).tostring())


# If already exsit previous lmdb folders, remove them
os.system('rm -rf  ' + train_lmdb)
os.system('rm -rf  ' + test_lmdb)


# univarsal way
# a list to store all images' path
i = 0
train_data = []
for root, dirnames, filenames in os.walk(JPG_train_path):
    i = i + 1
    for filename in fnmatch.filter(filenames, '*.jpg'):
        train_data.append(os.path.join(root, filename))
num_train = len(train_data)
num_label_train = i - 1

k = 0
test_data = []
for root, dirnames, filenames in os.walk(JPG_test_path):
    k = k + 1
    for filename in fnmatch.filter(filenames, '*.jpg'):
        test_data.append(os.path.join(root, filename))
num_test = len(test_data)
num_label_test = k - 1

# Shuffle train_data
print('shuffling train data')
random.shuffle(train_data)

print('\nCreating train_lmdb')
env_db = lmdb.open(train_lmdb, map_size=int(1e12))
with env_db.begin(write=True) as txn:
    for idx, img_path in enumerate(train_data):
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        img = transform_img(img, img_width=IMAGE_WIDTH, img_height=IMAGE_HEIGHT)
        label = int(img_path.split('/')[-2])
        datum = make_datum(img, label)
        txn.put('{:0>5d}'.format(idx), datum.SerializeToString())
        print('{:0>5d}'.format(idx) + ':' + img_path)
env_db.close()


print('\nCreating test_lmdb')
env_db = lmdb.open(test_lmdb, map_size=int(1e12))
with env_db.begin(write=True) as txn:
    for idx, img_path in enumerate(test_data):
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        img = transform_img(img, img_width=IMAGE_WIDTH, img_height=IMAGE_HEIGHT)
        label = int(img_path.split('/')[-2])
        datum = make_datum(img, label)
        txn.put('{:0>5d}'.format(idx), datum.SerializeToString())
        print('{:0>5d}'.format(idx) + ':' + img_path)
env_db.close()

print('\nFinished processing all images')
print('\nTraining data has {} images in {} labels'.format(num_train, num_label_train))
print('\nTest data has {} images in {} labels'.format(num_test, num_label_test))
