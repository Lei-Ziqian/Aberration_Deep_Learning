import matplotlib.pyplot as plt
import numpy as np
import bioformats as bf
import javabridge

# %% start javabridge
javabridge.start_vm(class_path=bf.JARS)
# %% defining path 
#filename = "D:\\User Data\\Fred\\20190331_dpTrainingData\\train_3_1\\train_3_1_MMStack_Pos0.ome.tif"
#md = bf.get_omexml_metadata(filename)
#rdr = bf.ImageReader(filename, perform_init=True)
#ome = bf.OMEXML(md)
#pixels = ome.image().Pixels
#pZ = 250

# %% do stuff
#print('pixels in Z: ' + str(pixels.SizeZ))
#print('pixels in C: ' + str(pixels.SizeC))
#print('pixels in T: ' + str(pixels.SizeT))
#print('pixels in X: ' + str(pixels.SizeX))
#print('pixels in Y: ' + str(pixels.SizeY))
#count = pixels.SizeT

#with bf.ImageReader(filename) as reader:
#   img = reader.read(t = 50)
#   plt.imshow(img, cmap = plt.cm.binary)
#   plt.show() 
#   reader.close()
# %% define functions   
def genGraphSingle(filename):
    md = bf.get_omexml_metadata(filename)
    #rdr = bf.ImageReader(filename, perform_init=True)
    ome = bf.OMEXML(md)
    pixels = ome.image().Pixels
    count = pixels.SizeT
    vPZ = []
    pZ = 250
    for i in range(count):
           vPZ.append(i*pZ/1000)
    xSum = []              
    with bf.ImageReader(filename) as reader:
        for i in range(count):
            #print(i)
            img = reader.read(t = i)
            #img = ndimage.gaussian_filter(img, sigma=(200, 200))
            mm = np.max(img)
            v2 = img >= mm/15
            im2 = img*v2
            ssum = sum(sum(im2))/sum(sum(v2))
            xSum.append(ssum)
        reader.close()
        plt.plot(vPZ, xSum)
        plt.show()

def getMax(filename):
    md = bf.get_omexml_metadata(filename)
    #rdr = bf.ImageReader(filename, perform_init=True)
    ome = bf.OMEXML(md)
    pixels = ome.image().Pixels
    count = pixels.SizeT
    xSum = []              
    with bf.ImageReader(filename) as reader:
        for i in range(count):
            img = reader.read(t = i)
            mm = np.max(img)
            v2 = img >= mm/15
            im2 = img*v2
            ssum = sum(sum(im2))/sum(sum(v2))
            xSum.append(ssum)
        reader.close()
        mP = max(xSum)    
        idx = xSum.index(mP)
    return mP, idx    
        
def genGraph(filename):
    md = bf.get_omexml_metadata(filename)
    #rdr = bf.ImageReader(filename, perform_init=True)
    ome = bf.OMEXML(md)
    pixels = ome.image().Pixels
    count = pixels.SizeT
    xSum = []              
    with bf.ImageReader(filename) as reader:
        for i in range(count):
            img = reader.read(t = i)
            mm = np.max(img)
            v2 = img >= mm/15
            im2 = img*v2
            ssum = sum(sum(im2))/sum(sum(v2))
            xSum.append(ssum)
        reader.close()
        mP = max(xSum)    
        idx = int(np.round(xSum.index(mP)-count/2))
        xSum = np.roll(xSum, -idx)
    return xSum, idx

def getImage(filename, num):
    with bf.ImageReader(filename) as reader:
       img = reader.read(t = num)
    return img

def getImageInfo(filename):
    md = bf.get_omexml_metadata(filename)
    ome = bf.OMEXML(md)
    pixels = ome.image().Pixels
    pT = pixels.SizeT
    pX = pixels.SizeX
    pY = pixels.SizeY
    return pT, pX, pY

def showGrahp(filename, num):
    md = bf.get_omexml_metadata(filename)
    #rdr = bf.ImageReader(filename, perform_init=True)
    ome = bf.OMEXML(md)
    pixels = ome.image().Pixels
    print('pixels in Z: ' + str(pixels.SizeZ))
    print('pixels in C: ' + str(pixels.SizeC))
    print('pixels in T: ' + str(pixels.SizeT))
    print('pixels in X: ' + str(pixels.SizeX))
    print('pixels in Y: ' + str(pixels.SizeY))
    with bf.ImageReader(filename) as reader:
       img = reader.read(t = num)
       plt.imshow(img, cmap = plt.cm.binary)
       plt.show() 
       reader.close()

# %% kill bridge (not usefull)
#javabridge.kill_vm()    

  


