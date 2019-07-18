# -*- coding: utf-8 -*-
"""
Created on Wed Jun 26 14:49:51 2019

@author: ziqian
"""

import tensorflow as tf
import pickle
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.callbacks import TensorBoard

NAME = "Comax_vs_Asti"

tensorboard = TensorBoard(log_dir='.logs/{}'.format(NAME))


pickle_in = open("X.pickle", "rb")
X = pickle.load(pickle_in)

pickle_in = open("y.pickle", "rb")
y = pickle.load(pickle_in)

model = tf.keras.models.Sequential()

model.add(Conv2D(256, (3, 3), input_shape=X.shape[1:]))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(256, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
s = X.shape

model.add(Flatten(input_shape=(s[1], s[2], s[3])))  # convert 3D feature maps to 1D feature vectors

#model.add(Dense(256, activation=tf.nn.relu))
#model.add(Dense(128, activation=tf.nn.relu))
model.add(Dense(50, activation=tf.nn.softmax))

model.compile(loss='sparse_categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])


history = model.fit(X, y, batch_size=10, epochs=3, validation_split=0.2, callbacks=[tensorboard]) 

# save model
save_path = "E:\A_files\optics_and_photonics\summer_project\summerproject\codes"
path = save_path + "model_trained"
model.save(path)
#return model, loss, acc
#%% test

