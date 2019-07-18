# -*- coding: utf-8 -*-
"""
Created on Thu Jul 11 16:06:44 2019

@author: ziqian
"""
import numpy as np
import matplotlib.pyplot as plt
import skimage as skim
import matplotlib.patches as patches
import skimage.feature

image_path = r"E:\A_files\optics_and_photonics\summer_project\summerproject\20190705_Ziqian_beads\astigBeads_MMStack_360360.ome.tif"
I = skim.io.imread(image_path)

img0 = I[15, :, :]
img = np.array(img0)/256

blob_log= skim.feature.blob_log(img, min_sigma=1, max_sigma=3, threshold=1, overlap=0.5)

fig,ax = plt.subplots(1)
ax.imshow(img, cmap='gray')


for ii in range(blob_log.shape[0]):
    y, x, r = blob_log[ii]
    if r<=3:
        rect = patches.Rectangle((x-16, y-16), 32, 32, edgecolor='r', fill=None)
    #y, x, r = blob_log[ii]
    #c = plt.Circle((x, y), r, color='red', linewidth=2, fill=False)
        ax.add_patch(rect)
plt.show()


#initial

[centre_y, centre_x, size] = np.hsplit(blob_log,3)
X=[] # save x_position after filtering
Y=[] # save y_position after filtering
for ii in range(16, blob_log.shape[0]-16):
    size_flag = 0
    for xx in range(-16, 16):
        if abs(centre_x[ii]-centre_x[xx+ii])<25 and abs(centre_y[ii]-centre_y[xx+ii])<25 and xx!=0:
            size_flag = size_flag + 1   
        else:
            pass
    if size_flag == 0 and 340>=centre_x[ii]>=16 and 340>=centre_y[ii]>=16:
        X.append(int(np.asscalar(centre_x[ii])))
        Y.append(int(np.asscalar(centre_y[ii])))

fig,ax = plt.subplots(1)
ax.imshow(img, cmap='gray')
for ii in range(len(X)):
    rect = patches.Rectangle((X[ii]-16, Y[ii]-16), 32, 32, edgecolor='r', fill=None)
    ax.add_patch(rect)
plt.show()

# crop
corp_beads=[]
for ii in range(len(X)):
    corp_beads.append(img0[Y[ii]-16:Y[ii]+16, X[ii]-16:X[ii]+16])

#%%save cropped images

from PIL import Image
im = Image.open(image_path)
save_path = r"E:\A_files\optics_and_photonics\summer_project\summerproject\20190705_Ziqian_beads\0714save\0"
crop_bead = []
for ii in range(len(X)):
    #for i in range(im.n_frames):
    im.seek(15)
    crop_bead= im.crop((X[ii]-16, Y[ii]-16, X[ii]+16, Y[ii]+16))
    #Image.crop((left, upper, right, lower))
    crop_bead.save(save_path + str(ii)+ ".tif", save_all=True)

