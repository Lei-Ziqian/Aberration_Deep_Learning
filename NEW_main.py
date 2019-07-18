import sys
sys.path.insert(0, "D:\\User Data\\Fred\\deep_learning_af")
import matplotlib.pyplot as plt
import NEW_subFunctions as sf
#%% define parameters
path = "D:\\User Data\\Fred\\20190617_deeplearning\\"                          # path for data 
save_path = "D:\\User Data\\Fred\\deepLearningData\\20190617_cleanNewRep\\"    # path for saving
#val_path = "D:\\User Data\\Fred\\20190608_deepLearning_train\\range100um_250nmSteps_noKnife_50msIntTime_1\\range100um_250nmSteps_noKnife_50msIntTime_1_MMStack_Pos0.ome.tif"
fac = 25               # compress factor for images
zStep=0.25             # in um
epos = 30               
#%%
x_train, numSlice = sf.bioformatsToNParray(path, save_path, fac, zStep)
y_train = sf.genY_train(save_path, numSlice, zStep)
model, loss, acc = sf.trainModel(save_path, numSlice, epos)
#predictions, y_val = sf.testModel(save_path, val_path, numSlice, zStep)
#plt.plot(predictions, y_val)
#%% 
import os
val_path = "D:\\User Data\\Fred\\20190617_deeplearning_val\\"
direc = [name for name in os.listdir(val_path) if os.path.isdir(os.path.join(val_path, name))]
ll= len(direc)
ii=2
print(direc[ii])
filename = val_path + direc[ii] + "\\MMStack_Pos1.ome.tif" 
#filename = val_path + direc[ii] + "\\" + direc[ii] + "_MMStack_Pos1.ome.tif" 
predictions, y_val = sf.testModel(save_path, filename, numSlice, zStep)
plt.plot(predictions, y_val, label=direc[ii])
#%%
for ii in range(0, 1):
    print(direc[ii])
    #filename = val_path + direc[ii] + "\\" + direc[ii] + "_MMStack_Pos1.ome.tif" 
    filename = val_path + direc[ii] + "\\MMStack_Pos1.ome.tif" 
    predictions, y_val = sf.testModel(save_path, filename, numSlice, zStep)
    plt.plot(predictions, y_val, label=direc[ii])