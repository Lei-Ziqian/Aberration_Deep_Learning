# -*- coding: utf-8 -*-
"""
Created on Mon Jul 15 14:41:50 2019

@author: ziqian
"""
import imageio as io
import os
import tensorflow as tf
import numpy as np
from PIL import Image


path = "E:/A_files/optics_and_photonics/summer_project/summerproject/20190705_Ziqian_beads/0714save/"
direc_test = [name for name in os.listdir(path)]
Num = len(direc_test)

save_path = r"E:\A_files\optics_and_photonics\summer_project\summerproject\figures\ast\model_trained_step_15"
modelLoad = tf.keras.models.load_model(save_path)

predictions = []
for order in range(Num):
    filename = path + direc_test[order]
   # im = Image.open(filename)
   # imarray = np.array(im)
    im = io.imread(filename)
    img = np.array(im)/65535.0
    img = img.reshape(1, 32, 32)
    predictions = modelLoad.predict(img)
    print(predictions)