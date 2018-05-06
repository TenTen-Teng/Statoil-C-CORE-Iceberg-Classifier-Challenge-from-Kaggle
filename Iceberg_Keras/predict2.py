import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.optimizers import Adam
import os
from Data.constant import TRAIN_JSON_PATH, TEST_JSON_PATH, KERAS_SUBMISSION_PATH, KERAS_OUTPUT_PATH_ROOT


np.random.seed(1990)

df_train = pd.read_json(TRAIN_JSON_PATH)

if not os.path.exists(KERAS_OUTPUT_PATH_ROOT):
    os.makedirs(KERAS_OUTPUT_PATH_ROOT)

def norm_imgs(df):
    imgs = []
    for i, row in df.iterrows():
        # Reshape image into 75*75
        band_1 = np.array(row['band_1']).reshape(75, 75)
        band_2 = np.array(row['band_2']).reshape(75, 75)
        band_3 = band_1 + band_2

        # Normalization
        x1 = (band_1 - band_1.mean()) / np.std(band_1) # (band_1.max() - band_1.min())
        x2 = (band_2 - band_2.mean()) / np.std(band_2) # (band_2.max() - band_2.min())
        x3 = (band_3 - band_3.mean()) / np.std(band_3) # (band_3.max() - band_3.min())

        imgs.append(np.dstack((x1, x2, x3)))
    return np.array(imgs)


Xtrain = norm_imgs(df_train)
Xtrain.shape
Ytrain = np.array(df_train['is_iceberg'])


# set na incident angle to 0
df_train.inc_angle = df_train.inc_angle.replace('na',0)
idx_use = np.where(df_train.inc_angle>0) # find the indices where the incident angle is >0


# train with only the known incident angles
Ytrain = Ytrain[idx_use[0]]
Xtrain = Xtrain[idx_use[0],...]


# Build Model
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


# Train the model
model = getModel()
model.summary()

batch_size = 32
earlyStopping = EarlyStopping(monitor='val_loss', patience=10, verbose=0, mode='min')
mcp_save = ModelCheckpoint('.mdl_wts.hdf5', save_best_only=True, monitor='val_loss', mode='min')
reduce_lr_loss = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=7, verbose=1, epsilon=1e-4, mode='min')

model.fit(Xtrain, Ytrain, batch_size=batch_size, epochs=10, verbose=1, callbacks=[earlyStopping, mcp_save, reduce_lr_loss], validation_split=0.25)


df_test = pd.read_json(TEST_JSON_PATH)
df_test.inc_angle = df_test.inc_angle.replace('na',0)
Xtest = (norm_imgs(df_test))
pred_test = model.predict(Xtest)

submission = pd.DataFrame({'id': df_test["id"], 'is_iceberg': pred_test.reshape((pred_test.shape[0]))})
print(submission.head(10))

submission.to_csv(KERAS_SUBMISSION_PATH, index=False)