# -*- coding: utf-8 -*-
"""
Created on Tue Jul  9 20:21:13 2019

@author: ziqian
"""

import scipy.io as sio
import os
import tensorflow as tf
import numpy as np
import random
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Dense, Activation, Flatten, LeakyReLU
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.layers import Conv2D, MaxPooling2D

path = "E:/A_files/optics_and_photonics/summer_project/summerproject/figures/ast/"
train_path = path + "train/"  +"0710diff/"
direc_train = [name for name in os.listdir(train_path)]
Num_train = len(direc_train)

listname = []
x_train = []
y_train = []
training_data = []
# get name
#for name in os.listdir(train_path):
#   if os.path.splitext(name)[1] == '.mat':
#        name = os.path.splitext(name)[0]
#        listname.append(name)

for order in range(Num_train):
    filename = train_path + direc_train[order]
    dic = sio.loadmat(filename)
    y = float(os.path.splitext(direc_train[order])[0])
    training_data.append([dic['psf_save'], y])
    
random.shuffle(training_data)

for X,Y in training_data:
    x_train.append(X)
    y_train.append(Y)
    
x_train = np.array(x_train)/255.0


#%%GenTrainModel

model = tf.keras.models.Sequential()
model.add(Flatten(input_shape=x_train.shape[1:])) 
#model.add(Dense(256, activation=LeakyReLU))
model.add(Dense(256, activation=tf.nn.relu))
#model.add(Dense(256,activation=tf.nn.tanh))
model.add(Dense(256, activation=tf.nn.relu))
model.add(Dense(128, activation="linear"))
model.add(Dense(1))

model.compile(loss='mean_squared_error',
              optimizer='adam',
              metrics=['mse'])

model.fit(x_train, y_train, batch_size=4, epochs=100) 
save_path = path + "model_trained_" 
model.save(save_path)

#%% test

test_path = path + "test/" + "0710diff/"
direc_test = [name for name in os.listdir(test_path)]
Num_test = len(direc_test)

x_test = []
y_test = []

for order in range(Num_test):
    filename = test_path + direc_test[order]
    dic = sio.loadmat(filename)
    x_test.append(dic['psf_save'])
    y_test.append(float(os.path.splitext(direc_test[order])[0]))
#print(x_test[34])    
x_test1 = np.array(x_test)/255.0
test_data = (x_test1, y_test)

#test
save_path = path + "model_trained_diff" 
modelLoad = tf.keras.models.load_model(save_path)
# predict
predictions = modelLoad.predict(x_test1)

plt.plot(y_test,predictions)