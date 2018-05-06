import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
import os
from Data.constant import TEST_JSON_PATH, KERAS_SUBMISSION_PATH, KERAS_MODEL_PATH, KERAS_OUTPUT_PATH_ROOT, \
    check_test_file, KERAS_TEST_PATH


# perpare data
df_test = pd.read_json(TEST_JSON_PATH)

# upzip test image file, save it in the same location as keras train.py file
check_test_file(KERAS_TEST_PATH, 'Keras')

if not os.path.exists(KERAS_OUTPUT_PATH_ROOT):
    os.makedirs(KERAS_OUTPUT_PATH_ROOT)

# generate test data
datagen = ImageDataGenerator()
generator = datagen.flow_from_directory(
        KERAS_TEST_PATH,
        target_size=(75, 75),
        batch_size=32,
        class_mode=None,  # only data, no labels
        shuffle=False)  # keep data in same order as labels

# build Model
def getModel():
    # Build keras model
    model = Sequential()

    # CNN 1
    model.add(Conv2D(64, kernel_size=(3, 3), activation='relu', input_shape=(75, 75, 3)))
    model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))
    model.add(Dropout(0.2))

    # CNN 2
    model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Dropout(0.2))

    # CNN 3
    model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Dropout(0.2))

    # CNN 4
    model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Dropout(0.2))

    # You must flatten the data for the dense layers
    model.add(Flatten())

    # Dense 1
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.2))

    # Dense 2
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.2))

    # Output
    model.add(Dense(1, activation="sigmoid"))

    optimizer = Adam(lr=0.0001, decay=0.0)
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])

    return model


model = getModel()
model.load_weights(KERAS_MODEL_PATH)

# pred_test = model.predict(generator)
probabilities = model.predict_generator(generator)

submission = pd.DataFrame({'id': df_test["id"], 'is_iceberg': probabilities.reshape((probabilities.shape[0]))})
print(submission.head(10))

submission.to_csv(KERAS_SUBMISSION_PATH, index=False)


