# -*- coding: utf-8 -*-
"""
Created on Thu Jun 27 10:18:20 2019

@author: ziqian
"""

import scipy.io as sio
import matplotlib.pyplot as plt
import numpy as np
import os

#%%

#Initialize training data
training_data = []

# This module is for generize the coma_x training data
#load matlab file .mat into python. scipy.io.loadmat save file as dict. 
# Here, the matrix is saved under the content called 'psf'
path = "E:/A_files/optics_and_photonics/summer_project/summerproject/figures/coma_x/"
direc = [name for name in os.listdir(path)]
total_data = len(direc)

for order in range(total_data):
    filename = path + direc[order]
    comax_dic = sio.loadmat(filename)
    comax = comax_dic['psf']
    i = 0
    j = 1
    while i < 5:
        training_data_comax = []
        while j < 51:
            new_comax_array = comax[:, :, j]
            j +=5
            training_data_comax.append(new_comax_array)
        #print (np.shape(training_data_comax))
        i+=1
        j=i+1
        training_data.append([training_data_comax, 1])



# This module is for generize the ast training data    
path = "E:/A_files/optics_and_photonics/summer_project/summerproject/figures/ast/"
direc = [name for name in os.listdir(path)]
total_data = len(direc)

for order in range(total_data):
    filename = path + direc[order]
    ast_dic = sio.loadmat(filename)
    ast = ast_dic['psf']
    i = 0
    j = 1
    while i < 5:
        training_data_ast = []
        while j < 51:
            new_ast_array = comax[:, :, j]
            j +=5
            training_data_ast.append(new_ast_array)
        #print (np.shape(training_data_ast))
        i+=1
        j=i+1
        training_data.append([training_data_ast, 0])   

print ("DATA Shape ", np.shape(training_data), np.shape(training_data_comax), np.shape(training_data_ast))

#%%
#Shuffle the data
import random
random.shuffle(training_data)
#%%
#make model
X = []
y = []
for image, label in training_data:
    X.append(image)
    y.append(label)
#print (np.shape(X))
    
X = np.array(X).reshape(-1, 320, 32, 1)
print ("DATA X Shape ", np.shape(X))
#print ("DATA y Shape ", np.shape(y))
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
