import numpy as np
from matplotlib import pyplot as plt
from sklearn.utils import shuffle
import cv2
from PIL import Image
import os
import shutil
import sys

def load_data(p="../", raw=True):
    try:
        X = np.load(p+"X_train.npy")
        y = np.load(p+"y_train.npy")
    except:
        X = np.loadtxt(p+"train_x.csv", delimiter=",") # load from text 
        y = np.loadtxt(p+"train_y.csv", delimiter=",") 
        y = y.reshape(-1, 1)
        np.save(p+"X_train",x)
        np.save(p+"y_train",y)
    if not raw:
        filter = np.vectorize(lambda x: 0 if x < 225-20 else x)
        X = np.vectorize(X.flatten()).reshape((-1,4096))
        X = (X - np.mean(X,axis=0))/128 #normalize
    print("X.shape",x.shape, "Y.shape",y.shape)
    X,y = shuffle(X,y)
    return X,y


def load_test_data(p = "../", raw=True):
    try:
        X = np.load(p+"X_test"+("" if raw else "_preprocessed")+".npy")
    except:
        X = np.loadtxt(p+"test_x.csv", delimiter=",") # load from text
        np.save(p+"X_test.npy",x)
    if not raw:    
        filter = np.vectorize(lambda x: 0 if x < 225-20 else x)
        X = np.vectorize(X.flatten()).reshape((-1,4096))
        X = (X - np.mean(X,axis=0))/128 #normalize
    print("X_test.shape",x.shape)
    return X



######################
### Preprocessing ####
######################

def save_as_pic(out_dir ="../pic_train/" ,in_file="../X_train.npy"):
    """convert numpy array into .jpg"""
    X = np.load(in_file)
    os.rmdir(out_dir)
    os.mkdir(out_dir)
    for i,arr in enumerate(X):
        img = Image.fromarray(arr).convert("RGB")
        img.save(out_dir+str(i)+".png")


def remove_background(out_file ="../X_train_preprocessed", out_dir = "../pic_preprocessed/",pic_dir = "../pic_train/"):
    """uses opencv grabcut to remove background:
     https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_grabcut/py_grabcut.html#grabcut """
    pics = [p for p in os.listdir(pic_dir) if p.split(".")[-1]=="png"]
    shutil.rmtree(out_dir, ignore_errors=True)
    os.mkdir(out_dir)
    X = np.zeros((len(pics), 64*64))
    print("new X,shape",X.shape)
    for i,pic in enumerate(pics):
        # print("processing",pic)
        img = cv2.imread(pic_dir+pic)

        ## Grabcut using Network flow
        mask = np.zeros(img.shape[:2],np.uint8)
        bgdModel = np.zeros((1,65),np.float64)
        fgdModel = np.zeros((1,65),np.float64)

        rect = (1,1,64,64)
        cv2.grabCut(img,mask,rect,bgdModel,fgdModel,10,cv2.GC_INIT_WITH_RECT)

        mask2 = np.where((mask==2)|(mask==0),0,1).astype('uint8')
        img = img*mask2[:,:,np.newaxis]

        gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        X[i] = gray_image.reshape((-1,))

        img = Image.fromarray(img)
        img.save(out_dir+str(i)+".png")
        
    
    np.save(out_file,X)
        

if __name__=="__main__":
    data_dir = sys.argv[1] # folder for .csv files
    load_data(p=data_dir)
    load_test_data(data_dir)
    
    if "-preprocess" in sys.argv:
        save_as_pic(out_dir =data_dir+"/pic_train/" ,in_file=data_dir+"/X_train.npy")
        remove_background(out_file = data_dir+"X_train_preprocessed", out_dir = data_dir+"/pic_preprocessed/",pic_dir = data_dir+"/pic_train/")
    
        save_as_pic(out_dir =data_dir+"/pic_test/" ,in_file=data_dir+"/X_test.npy")
        remove_background(out_file = data_dir+"X_test_preprocessed", out_dir = data_dir+"/pic_preprocessed_test/",pic_dir = data_dir+"/pic_test/")