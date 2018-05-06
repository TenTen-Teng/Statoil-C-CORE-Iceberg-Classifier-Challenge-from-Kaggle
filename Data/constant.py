import os
import zipfile

DATA_PATH_ROOT = '../Data/'
# json file paths
TRAIN_JSON_PATH = DATA_PATH_ROOT + 'train.json'
TEST_JSON_PATH = DATA_PATH_ROOT + 'test.json'

PRE_DATA_ROOT = '../Data/raw/'
# preprocess new data file paths
TRAIN_BAND3_JSON = PRE_DATA_ROOT + 'train_add_band3.json'
TRAIN_AUG_JSON = PRE_DATA_ROOT + 'train_addDA.json'
ALL_DATA_ICE = PRE_DATA_ROOT + 'all/data/1/'
ALL_DATA_SHIP = PRE_DATA_ROOT + 'all/data/0/'

SEP_DATA_ROOT = PRE_DATA_ROOT + 'processed'
SEP_TRAIN_ICE = SEP_DATA_ROOT + '/train/1/'
SEP_TRAIN_SHIP = SEP_DATA_ROOT + '/train/0/'
SEP_VAL_ICE = SEP_DATA_ROOT + '/validation/1/'
SEP_VAL_SHIP = SEP_DATA_ROOT + '/validation/0/'

# zip file paths
DATA_ZIP_PATH = DATA_PATH_ROOT + 'data.zip'
# zip test file
TEST_ZIP_PATH = DATA_PATH_ROOT + 'data_test.zip'
# test file
# Keras
KERAS_PATH = '../Iceberg_Keras/'
KERAS_TEST_PATH = KERAS_PATH + 'data_test'
# Pytorch
PYTORCH_PATH = '../Iceberg_Pytorch/'

# data path
PREPROCESSED_DATA_ROOT = PRE_DATA_ROOT + 'processed/'
TRAIN_PATH = PREPROCESSED_DATA_ROOT + 'train'
VAL_PATH = PREPROCESSED_DATA_ROOT + 'validation'
TEST_PATH = DATA_PATH_ROOT + 'data_test/'

# Keras output file path
KERAS_OUTPUT_PATH_ROOT = KERAS_PATH + 'output/'
KERAS_SUBMISSION_PATH = KERAS_OUTPUT_PATH_ROOT + 'submission.csv'
KERAS_MODEL_PATH = KERAS_OUTPUT_PATH_ROOT + 'my_model.h5'
KERAS_MODEL_PLOT = KERAS_OUTPUT_PATH_ROOT + 'model_plot.png'

# Pytorch output file path
PYTORCH_OUTPUT_PATH_ROOT = PYTORCH_PATH + 'output/'
PYTORCH_SUBMISSION_PATH = PYTORCH_OUTPUT_PATH_ROOT + 'submission.csv'
PYTORCH_MODEL_PATH = PYTORCH_OUTPUT_PATH_ROOT + 'cnn.pt'


def check_file(path):
    print('checking file ...')
    if not os.path.exists(path):
        print('preparing data ...')

        zip_data = zipfile.ZipFile(DATA_ZIP_PATH, 'r')
        zip_data.extractall(DATA_PATH_ROOT)
        zip_data.close()
    else:
        print('data exists')


def check_test_file(path, framework):
    print('checking file ...')
    if not os.path.exists(path):
        print('preparing data ...')
        zip_data = zipfile.ZipFile(TEST_ZIP_PATH, 'r')

        if framework == 'Keras':
            zip_data.extractall(KERAS_PATH)
            zip_data.close()
    else:
        print('data exists')


def zipfolder(foldername, target_dir):
    zipobj = zipfile.ZipFile(foldername, 'w', zipfile.ZIP_DEFLATED)
    rootlen = len(target_dir) + 1
    for base, dirs, files in os.walk(target_dir):
        for file in files:
            fn = os.path.join(base, file)
            zipobj.write(fn, fn[rootlen:])