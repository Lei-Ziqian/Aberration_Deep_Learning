# -*- coding: utf-8 -*-

import tensorflow as tf
import matplotlib.pyplot as plt
import os
import cv2

mnist = tf.keras.datasets.mnist 
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# normalize a numpy array
# numpy reshape to flatten the images
x_train = tf.keras.utils.normalize(x_train, axis=1).reshape(x_train.shape[0], -1)
x_test = tf.keras.utils.normalize(x_test, axis=1).reshape(x_test.shape[0], -1)

#build network
#softmax--probability distribution
#use dense function
model = tf.keras.models.Sequential()
#model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu, input_shape= x_train.shape[1:]))
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
#set dense = 10 to output 10 numbers from 0 to 9
model.add(tf.keras.layers.Dense(10, activation=tf.nn.softmax))

#define parameters training in the model
model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
model.fit(x_train, y_train, epochs=3)


## combine pyplot with numpy into a single namespace as matlab
#import matplotlib.pyplot as plt 
##monocolour
#plt.imshow(x_train[0], cmap = plt.cm.binary) 

#evaluate the sample data with model
val_loss, val_acc = model.evaluate(x_test, y_test)
print(val_loss)
print(val_acc)


