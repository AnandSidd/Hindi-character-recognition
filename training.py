import os
import numpy as np
import skimage.io as io
import keras
import pandas as pd
import matplotlib.pyplot as plt
from keras.layers import *
from keras.utils import *
from keras.optimizers import Adam
from keras.models import *
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import *


X_train = pd.read_csv('X_train.csv')
X_train = X_train.drop(X_train.columns[0], axis = 1)
X_train = np.array(X_train)

Y_train = pd.read_csv('Y_train.csv')
Y_train = Y_train.drop(Y_train.columns[0], axis = 1)
Y_train = np.array(Y_train)

X_test = pd.read_csv('X_test.csv')
X_test = X_test.drop(X_test.columns[0], axis = 1)
X_test = np.array(X_test)

Y_test = pd.read_csv('Y_test.csv')
Y_test = Y_test.drop(Y_test.columns[0], axis = 1)
Y_test = np.array(Y_test)

X_train = X_train.reshape((61200,32,32,1))
print(X_train.shape)
Y_train = Y_train.reshape((61200,1))
print(Y_train.shape)
X_test = X_test.reshape((10800,32,32,1))
print(X_test.shape)
Y_test = Y_test.reshape((10800,1))
print(Y_test.shape)


X_train = X_train/255
X_test = X_test/255
X_train, Y_train = shuffle(X_train, Y_train, random_state = 2)
X_test, Y_test = shuffle(X_test, Y_test, random_state = 0)
# plt.imshow(X_test[0])
# print(ref[int(Y_test[0])])
X_test, X_val, Y_test, Y_val = train_test_split(X_test, Y_test, test_size = 0.6, random_state = 1)
print(X_test.shape)
print(X_val.shape)
Y_test = to_categorical(Y_test)
print(Y_test.shape)
Y_val = to_categorical(Y_val)
print(Y_val.shape)
Y_train = to_categorical(Y_train)
print(Y_train.shape)
inputs = Input(shape = (32,32,1))
conv0 = Conv2D(64, 3, padding = 'same', activation = 'relu')(inputs)
conv1 = Conv2D(64, 3, padding='same', activation='relu')(conv0)
conv2 = Conv2D(128, 3, padding='same', activation='relu')(conv1)
pool2 = MaxPooling2D((2,2))(conv2)
conv3 = Conv2D(128, 3, padding='same', activation='relu')(pool2)
conv4 = Conv2D(256, 5, padding='same', activation='relu')(conv3)
pool4 = MaxPooling2D((2,2))(conv4)
conv5 = Conv2D(256, 5, padding='same', activation='relu')(pool4)
flat = Flatten()(conv5)
dense0 = Dense(512, activation='relu')(flat)
dense1 = Dense(128, activation='relu')(dense0)
dense2 = Dense(64, activation='relu')(dense1)
dense3 = Dense(37, activation='softmax')(dense2)

model1 = Model(inputs, dense3)
print(model1.summary())
datagen = ImageDataGenerator(rotation_range = 10, zoom_range = 0.1, width_shift_range = 0.1, height_shift_range = 0.1)
datagen.fit(X_train)
model1.compile(Adam(lr = 10e-4), loss = 'categorical_crossentropy', metrics = ['accuracy'])
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.8,
                              patience=3)
history = model1.fit_generator(datagen.flow(X_train, Y_train, batch_size = 200), epochs = 25, steps_per_epoch = 300, validation_data = (X_val, Y_val), callbacks = [reduce_lr])

model1.save('model1.h5')