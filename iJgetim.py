# -*- coding: utf-8 -*-
"""
Created on Thu Jul  4 11:52:15 2019

@author: ziqian
"""
import skimage as skim
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

I = skim.io.imread(r"E:\A_files\optics_and_photonics\summer_project\summerproject\20190705_Ziqian_beads\test1.tif")
#print(I.shape)=(51, 32, 32)
#print(np.max(I[1]), np.min(I[1])) maximum = 255
Ishape = np.shape(I)
I_x = []
I_y = []
Ast_mag = []
for ii in range(Ishape[0]):
    I_x.append(I[ii])
    I_y.append(ii*0.04-1)
#I_x_nor = np.array(I_x)/np.amax(I_x)
I_x_nor = np.array(I_x)/65535
training_data = (I_x_nor, I_y)

Ast_mag = I_x_nor[0, :, :]
Ast_mag = Ast_mag.reshape(1, 32, 32)

save_path_ast = r"E:\A_files\optics_and_photonics\summer_project\summerproject\figures\ast\model_trained_step"

save_path = r"E:\A_files\optics_and_photonics\summer_project\summerproject\figures\ast\model_trained_ast1.2_200epochs"
modelLoad = tf.keras.models.load_model(save_path)
modelLoadast = tf.keras.models.load_model(save_path_ast)

# predict

for jj in range(Ishape[0]):
    Ast_mag = I_x_nor[jj, :, :]
    Ast_mag = Ast_mag.reshape(1, 32, 32)
    predictions_ast = modelLoadast.predict(Ast_mag)
   # print(jj, "th predict:", predictions_ast)

predictions = modelLoad.predict(I_x_nor)
#plt.plot(I_y,np.round(predictions))
plt.plot(I_y,predictions)
plt.plot(I_y, I_y)