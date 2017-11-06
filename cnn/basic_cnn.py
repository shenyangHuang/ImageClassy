import keras
from sklearn.utils import shuffle
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, BatchNormalization
from keras.layers import Conv2D, MaxPooling2D, AveragePooling2D
from sklearn.preprocessing import LabelEncoder
from keras.utils import np_utils
from keras.callbacks import ModelCheckpoint
import matplotlib.pyplot as plt
import pandas as pd
import sys
sys.path.append('../')
from utils import *

# hyper parameters
batch_sz = 128
epochs = 300
filter_sizes = [32,32]
hidden_sz = 256

def build_model():
    model = Sequential()
    model.add(Conv2D(32, (5, 5), padding='same', input_shape=(64,64,1)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    for fs in filter_sizes:
        model.add(Conv2D(fs, (1,1)))
        model.add(Conv2D(fs, (3,3)))
        model.add(BatchNormalization())
        model.add(Activation('relu'))

        model.add(Conv2D(fs, (1,1)))
        model.add(Conv2D(fs, (3,3)))
        model.add(BatchNormalization())
        model.add(Activation('relu'))

        model.add(AveragePooling2D(pool_size=(4, 4)))
        model.add(Dropout(0.2))

    model.add(Flatten())
    
    model.add(Dense(hidden_sz))
    # model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(40))
    model.add(Activation('softmax'))

    opt = keras.optimizers.Adam(lr=5e-3)
    model.compile(loss='categorical_crossentropy',optimizer=opt, metrics=['accuracy'])
    return model

def fit(X,y, save=False):
    X = X.reshape((-1,64,64,1))
    print("build model...filter sizes: ",filter_sizes)
    model = build_model()
    # chckpt = ModelCheckpoint("../../0.2Ddrop.{epoch:02d}-{val_loss:.2f}.hdf5", monitor='val_loss', verbose=0, save_best_only=True, save_weights_only=False, mode='auto', period=20)
    print(model.summary())

    print("start training for %s epochs. X_train shape: %s, y_train shape: %s"%(epochs, X.shape, y.shape))
    history = model.fit(X,y,batch_size=batch_sz, epochs=epochs,verbose=1, validation_split = 0.1)
    
    if save:
        train_loss = np.array(history.history["loss"]).reshape((-1,1))
        train_acc = np.array(history.history["acc"]).reshape((-1,1))
        valid_loss = np.array(history.history["val_loss"]).reshape((-1,1))
        valid_acc = np.array(history.history["val_acc"]).reshape((-1,1))
        tmp = np.concatenate([train_loss, train_acc, valid_loss, valid_acc], axis=1)
        print("write shape",tmp.shape)
        df = pd.DataFrame(tmp,columns = ["train_loss"," train_acc", "valid_loss", "valid_acc"])
        df.to_csv("../../cnn_history0.2filter.csv")
        
    return model

def filter(x):
    t = 20
    if x < 225-t:
        return 0
    else:
        return x

if __name__=="__main__":
    X,y =load_data("../../",raw=True)
    filter = np.vectorize(filter)
    X = filter(X.flatten()).reshape((-1,4096))
    X = (X - np.mean(X,axis=0))/128 #normalize
    X, y = shuffle(X,y)

    # X_test = load_test_data(p = "../../").reshape((-1,4096))
    # X_test = (X_test - np.mean(X_test,axis=0))/128 #normalize

    encoder = LabelEncoder()
    encoded_Y = encoder.fit_transform(y.reshape((-1,)))
    one_hot_y = np_utils.to_categorical(encoded_Y)
    model = fit(np.expand_dims(X, axis=-1),one_hot_y, save=True)
    
    # yp = model.predict(np.expand_dims(X_test.reshape(()), axis=-1))
    # write_prediction(yp,"../../basic_cnn.csv")