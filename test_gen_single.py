# -*- coding: utf-8 -*-
"""
Created on Mon Jun 24 21:41:19 2019

@author: ziqian
"""

import scipy.io as sio
import matplotlib.pyplot as plt
import numpy as np

#%%
# load matlab file .mat into python. scipy.io.loadmat save file as dict. 
# Here, the matrix is saved under the content called 'psf'
ast_path = r'E:/A_files/optics_and_photonics/summer_project/summerproject/figures/ast/3.mat'
ast_dic = sio.loadmat(ast_path)
ast = ast_dic['psf']

comax_path = r'E:/A_files/optics_and_photonics/summer_project/summerproject/figures/coma_x/3.mat'
comax_dic = sio.loadmat(comax_path)
comax = comax_dic['psf']

#Initialize training array
training_data = []
test_data = []
#%%
#creat training data
#Define function creat_training_data()

for n in range(51):
    new_ast_array = ast[:, :, n]
    training_data.append([new_ast_array, 1])
    #classindex: for astigmatism, zernike#= 5
#print ("Shape of asti training array is ", np.shape(training_data))

for n in range(51):
    new_comax_array = comax[:, :, n]
    training_data.append([new_comax_array, 0])
    #classindex: for coma tilte x, zernike#= 7
#print ("Shape of total training data array is ", np.shape(training_data))

#Shuffle the data
import random
random.shuffle(training_data)
random.shuffle(test_data)
#%%
#make model
X = []
y = []

for image, label in training_data:
    X.append(image)
    y.append(label)

    
print (np.shape(X))
    
X = np.array(X).reshape(-1, 32, 32, 1)
print (np.shape(X))
#size of each training aberration image is 32 * 32 
#%%
#save data
import pickle

pickle_out = open("X.pickle", "wb")
pickle.dump(X, pickle_out)
pickle_out.close()

pickle_out = open("y.pickle", "wb")
pickle.dump(y, pickle_out)
pickle_out.close()

print ("traning_data has been generated and saved successfully :)")