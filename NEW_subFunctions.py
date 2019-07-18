import tensorflow as tf
import matplotlib.pyplot as plt
import readTrainingStacks as rts
import cv2
import numpy as np
import os
#%%
def bioformatsToNParray(path, save_path, fac, zStep):
    # load first image and get pixel sizes (X,Y and Z->T)
    direc = [name for name in os.listdir(path) if os.path.isdir(os.path.join(path, name))]
    filename = path + direc[1] + "\\MMStack_Pos1.ome.tif" 
    #filename = path + direc[1] + "\\" + direc[1] + "_MMStack_Pos1.ome.tif" 
    pT, pX, pY = rts.getImageInfo(filename) 
    # prepare buffers for the analysed data
    ll= len(direc)
    arrX_save = np.zeros((int(pT*ll),int(pY/fac), int(pX/fac)))
    # write into np array
    for ii in range(0, ll):
        #new_filename = path + direc[ii] + "\\" + direc[ii] + "_MMStack_Pos1.ome.tif" 
        new_filename = path + direc[ii] + "\\MMStack_Pos1.ome.tif"
        print(ii)
        # prepare x_train: read each z plane and write into array
        for i in range(0, pT):
            new_img = rts.getImage(new_filename,i)
            rez_img = cv2.resize(new_img, dsize=(int(pX/fac), int(pY/fac)), interpolation=cv2.INTER_CUBIC)
            arrX_save[pT*ii+i, :, :] = rez_img    
    # save arrays    
    pp = save_path + "x_train"
    np.save(pp, arrX_save)
    return arrX_save, pT
#%%
def genY_train(path, numSlice, zStep):
    pp =path + "x_train.npy"
    xArr = np.load(pp)
    s = xArr.shape
    int(s[0]/numSlice)
    xSum1 = []
    com = []
    yHelp = np.arange(0, numSlice, 1)
    y = []
    for i in range(0, int(s[0]/numSlice)):
        print(i)
        x = xArr[i*numSlice:numSlice*(i+1)]
        for ii in range(numSlice):
            xx = x[ii]
            mm = np.min(xx)+(np.max(xx)-np.min(xx))/2
            v2 = xx >= mm
            im2 = xx*v2
            ssum1 = sum(sum(im2))/sum(sum(v2))
            xSum1.append(ssum1)
    #    plt.pause(0.05)    
    #    plt.plot(xxSum1)   
        cgx = np.sum(yHelp*xSum1)/np.sum(xSum1)
        com.append(cgx)
        xSum1 = []
        yS = 0-int(cgx)
        yE = numSlice-int(cgx)
        y1 = np.arange(yS, yE, 1)*zStep
        y[i*numSlice:(i+1)*numSlice] = y1
        #arrX_save[pT*ii+i, :, :] = rez_img
        #y.append(y1)
    #plt.plot(com)
    # save Y
    pp = path + "y_train"
    np.save(pp, y)
    return y
#%%
def trainModel(path, numSlice, epos):
    pp =path + "x_train.npy"
    xArr = np.load(pp)
    pp = path + "y_train.npy"
    yArr = np.load(pp)
    s = xArr.shape
    # define model (there are two types sequential and ???)   
    model = tf.keras.models.Sequential()
    # INPUT LAYER flatten data before feeding into network. Does not need to be a layer.
    model.add(tf.keras.layers.Flatten(input_shape=(s[1], s[2])))
    # FIRST LAYER is density layer with 128 units (neurons) and activation = activation function
    # as activation we use rectified linear (default to go), kind of probability distribution
    model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
    # SECOND LAYER
    model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu)) 
    # OUTPUT LAYER less neurons -> results
    #model.add(tf.keras.layers.Dense(pT+outputLayerBuffer, activation=tf.nn.softmax))
    model.add(tf.keras.layers.Dense(1))   
    # define training compiler
    # define parameters of training of the model
    # optimizer, most complex part of network -> did not get it
    # loss, kind of error, network tries to minimize loss. Very imortatnt for performance
    model.compile(optimizer = 'adam',
                  #loss = 'sparse_categorical_crossentropy',
                  loss = 'mean_squared_error',
                  metrics = ['mse'])
    #train
    x_train = tf.keras.utils.normalize(xArr, axis = 1)
    #history = model.fit(x_train, yArr,epochs=epos, validation_data=(x_val, y_val))
    history = model.fit(x_train, yArr,epochs=epos)   
    loss = history.history['loss']
    acc = history.history['mean_squared_error']
    # save model
    pp = path + "model_trained"
    model.save(pp)
    return model, loss, acc
#%%
def testModel(path, val_path, numSlice, zStep):
    # define parameters
    pp =path + "x_train.npy"
    xArr = np.load(pp)
    pp = path + "model_trained"
    modelLoad = tf.keras.models.load_model(pp)
    s = xArr.shape
    # load valid data
    #val_path = "D:\\User Data\\Fred\\20190419_dpTrainingSet\\run1_10\\run1_10_MMStack_Pos0.ome.tif"
    x_val = np.zeros((numSlice,s[1], s[2]))
    for i in range(0, numSlice):
        new_img = rts.getImage(val_path,i)
        x_val[i, :, :] = cv2.resize(new_img, dsize=(s[2], s[1]), interpolation=cv2.INTER_CUBIC)
    x_val1 = tf.keras.utils.normalize(x_val, axis = 1)
    # get y_val
    xSum1 = []
    yHelp = np.arange(0, numSlice, 1)
    for ii in range(numSlice):
        xx = x_val[ii]
        mm = np.min(xx)+(np.max(xx)-np.min(xx))/2
        v2 = xx >= mm
        im2 = xx*v2
        ssum1 = sum(sum(im2))/sum(sum(v2))
        xSum1.append(ssum1) 
    cgx = np.sum(yHelp*xSum1)/np.sum(xSum1)
    yS = 0-int(cgx)
    yE = numSlice-int(cgx)
    y_val1 = np.arange(yS, yE, 1)*zStep
    # predict
    predictions = modelLoad.predict(x_val1)
    # save Y
    pp = path + "x_val"
    np.save(pp, x_val1)
    pp = path + "y_val"
    np.save(pp, y_val1)
    return predictions, y_val1