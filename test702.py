# -*- coding: utf-8 -*-
"""
Created on Tue Jul  2 21:24:57 2019

@author: ziqian
"""
import scipy.io as sio
import os
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Dense, Activation, Flatten, LeakyReLU
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.layers import Conv2D, MaxPooling2D

path = "E:/A_files/optics_and_photonics/summer_project/summerproject/figures/ast/"
train_path = path + "train/" + "1.1/"
direc_train = [name for name in os.listdir(train_path)]
Num_train = len(direc_train)
Num = 51

#training_data = []
x_train = []
y_train = []
for order in range(Num_train):
    filename = train_path + direc_train[order]
    dic = sio.loadmat(filename)
    image = dic['psf_noi']
    for ii in range(Num):
        x_train.append(image[:, :, ii])
        y_train.append(ii*0.04-1) #from -1 to 1
    
    #print (np.shape(x_train))=(Num_train*Num, 32, 32)
    #print (np.shape(y_train))=(Num_train*Num,)
x_train = np.array(x_train)/255.0
training_data = (x_train, y_train)

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
              optimizer='sgd',
              metrics=['mse'])

model.fit(x_train, y_train, batch_size=4, epochs=200) 
save_path = path + "model_trained_ast1.1_200epochs" 
model.save(save_path)

#%%GenTestData
test_path = path + "test/" + "1.1/"
direc_test = [name for name in os.listdir(test_path)]
Num_test = len(direc_test)
Num = 51
x_test = []
y_test = []
for order in range(Num_test):
    filename = test_path + direc_test[order]
    dic = sio.loadmat(filename)
    image = dic['psf_noi']
    for ii in range(Num):
        x_test.append(image[:, :, ii])
        y_test.append(ii*0.04-1) #from 0 to Num-1
#print(x_test[34])    
x_test1 = np.array(x_test)/255.0
test_data = (x_test1, y_test)

#test
modelLoad = tf.keras.models.load_model(save_path)
# predict
predictions = modelLoad.predict(x_test1)

#pre_zpos = max(predictions, key=lambda e: int(e[0]))
#print(pre_zpos)
#plt.imshow(x_test[123])
#plt.show()
#print("y_test value is", y_test[123])
#plt.plot(y_test,np.round(predictions))
plt.plot(y_test,predictions)