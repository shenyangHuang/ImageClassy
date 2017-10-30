import numpy as np
from sklearn.utils import shuffle


def load_data(p="../"):
    try:
        x = np.load(p+"x_train.npy")
        y = np.load(p+"y_train.npy")
    except:
        x = np.loadtxt(p+"train_x.csv", delimiter=",") # load from text 
        y = np.loadtxt(p+"train_y.csv", delimiter=",") 
        x = x.reshape(-1, 64, 64) # reshape 
        y = y.reshape(-1, 1)
        np.save(p+"x_train",x)
        np.save(p+"y_train",y)
    print("X.shape",x.shape, "Y.shape",y.shape)
    x,y = shuffle(x,y)
    return x,y

def load_test_data(p = "../"):
    try:
        x = np.load(p+"x_test.npy")
    except:
        x = np.loadtxt(p+"test_x.csv", delimiter=",") # load from text
        x = x.reshape((-1,64,64))
        np.save(p+"x_test.npy",x)
    print("X_test.shape",x.shape)
    return x

def write_prediction(yp,p="../test_y.csv"):
    np.savetxt(p, yp, delimiter=",")
    

if __name__=="__main__":
    X,y = load_data()