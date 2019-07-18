# -*- coding: utf-8 -*-
"""
Created on Tue Jul  2 14:31:54 2019

@author: ziqian
"""
import scipy.io as sio
import os
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Dense, Activation, Flatten
from tensorflow.keras.callbacks import TensorBoard


#%%
def gen_train(path, Num):
    train_path = path + "train/"
    direc = [name for name in os.listdir(train_path)]
    Num_train = len(direc)

    #training_data = []
    x_train = []
    y_train = []
    for order in range(Num_train):
        filename = path + direc[order]
        dic = sio.loadmat(filename)
        image = dic['psf']
        for ii in range(Num):
            x_train.append(image[:, :, ii])
            y_train.append(ii) #from 0 to Num-1
    
    #print (np.shape(x_train))=(Num_train*Num, 32, 32)
    #print (np.shape(y_train))=(Num_train*Num,)
    training_data = (x_train, y_train)

    x_train = tf.keras.utils.normalize(x_train, axis=1)
    training_data = (x_train, y_train)
    return  training_data
#%%   
def test_model(path, save_path, Num):
    #generate test data
    test_path = path + "test/"
    direc_test = [name for name in os.listdir(test_path)]
    Num_test = len(direc_test)

    x_test = []
    y_test = []
    for order in range(Num_test):
        filename = test_path + direc_test[order]
        dic = sio.loadmat(filename)
        image = dic['psf']
        for ii in range(Num):
            x_test.append(image[:, :, ii])
            y_test.append(ii) #from 0 to Num-1
            #print(x_test[34])    
    x_test = tf.keras.utils.normalize(x_test, axis=1)
    test_data = (x_test, y_test)

    #test
    modelLoad = tf.keras.models.load_model(save_path)
    # predict
    predictions = modelLoad.predict(x_test)
    return predictions, test_data
#%%
def train_model(path, training_data, epos, Num, basize):
    (x_train, y_train) = training_data
    model = tf.keras.models.Sequential()

    model.add(Flatten(input_shape=x_train.shape[1:]))  # convert 3D feature maps to 1D feature vectors

    model.add(Dense(256, activation=tf.nn.relu))
    model.add(Dense(128, activation=tf.nn.relu))
    model.add(Dense(1,activation='linear'))

    model.compile(loss='mean_squared_error',
              optimizer='adam',
              metrics=['mse'])

    history = model.fit(x_train, y_train, batch_size=basize, epochs=epos) 
    loss = history.history['loss']
    acc = history.history['mean_squared_error']
    save_path = path + "model_trained"
    model.save(save_path)
    return model, save_path, loss, acc