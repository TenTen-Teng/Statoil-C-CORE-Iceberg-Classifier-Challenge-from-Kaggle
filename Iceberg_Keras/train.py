import matplotlib.pyplot as plt
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from keras.utils.vis_utils import plot_model
import time
import os
from Data.constant import TRAIN_PATH, VAL_PATH, KERAS_MODEL_PATH, check_file, KERAS_MODEL_PLOT, KERAS_OUTPUT_PATH_ROOT

plt.switch_backend('agg')

# check data file
check_file(TRAIN_PATH)
check_file(VAL_PATH)

if not os.path.exists(KERAS_OUTPUT_PATH_ROOT):
    os.makedirs(KERAS_OUTPUT_PATH_ROOT)

# generate training data
datagen = ImageDataGenerator()
train_generator = datagen.flow_from_directory(
    TRAIN_PATH,  # training dataset
        target_size=(75, 75),
        batch_size=32,
        class_mode='binary')

# generate validation data
val_generator = datagen.flow_from_directory(
        VAL_PATH,   # validation dataset
        target_size=(75, 75),
        batch_size=32,
        class_mode='binary')


# Build Model
def getModel():
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

    # flatten the data for the dense layers
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
plot_model(model, to_file=KERAS_MODEL_PLOT, show_shapes=True, show_layer_names=True)

earlyStopping = EarlyStopping(monitor='val_loss', patience=10, verbose=0, mode='min')
mcp_save = ModelCheckpoint('.mdl_wts.hdf5', save_best_only=True, monitor='val_loss', mode='min')
reduce_lr_loss = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=10, verbose=1, epsilon=1e-4, mode='min')

ep = 150
start = time.time() # count time
history = model.fit_generator(
        train_generator,
        epochs=ep,
        callbacks=[earlyStopping, mcp_save, reduce_lr_loss],
        validation_data = val_generator)
total_time = time.time()- start

# save trained model
model.save(KERAS_MODEL_PATH)

# calculate average accuracy
train_acc = np.mean(history.history['acc'])
val_acc = np.mean(history.history['val_acc'])
print("The average training accuracy is {}".format(train_acc))
print("The average testing accuracy is {}".format(val_acc))

# calculate average loss
train_loss = np.mean(history.history['loss'])
val_loss = np.mean(history.history['val_loss'])
print("The average training loss is {}".format(train_loss))
print("The average testing loss is {}".format(val_loss))

print("The training time is {}".format(total_time))

# summarize history for accuracy
plt.figure(1)
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')

# summarize history for loss
plt.figure(2)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()