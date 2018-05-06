'preprocess_Data' is for preprocessing data, data augmentation, generate images, and separate data.
1. 'preprocess.py' is for preprocessing data.
    Add two bands from 'train.json', then create a new json file contains 3 bands information.
    input: train.json
    output: train_add_band3.json
2. 'data_augmentation.py' is for data augmentation.
    Flip each image horizontally, vertically, and rotate 90 degrees using band3 data, then save to a new json file.
    input: train_add_band3.json
    output: train_addDA.json
3. 'generate_json_band3.py' is for generating images from the 'train_addDA.json' file.
    input: train_addDA.json
    output: all images using training and validation
    output path: './Data/raw/all'
4. 'separate_data.py' is for separating all data to validation data and training data.
    train:validation = 8:2
    input: step 3 output file
    output: train and validation
5. 'generate_json_test.py' is for generating test images from the test.json file.
    input: test.json
    output: test data image
    output path: './Data/data_test'

The order of running this folder follows: 1 -> 2 -> 3 -> 4 -> 5

NOTE: Before running these scripts, please check the raw json files saved in './Data'

Pass the preprocessing, download data zip file from (save them in 'Data' folder):
https://drive.google.com/open?id=1Zz_rbZw5Iv1IAt73JSCF1qYfJU9n-900

Kaggle data source:
https://www.kaggle.com/c/statoil-iceberg-classifier-challenge/data
