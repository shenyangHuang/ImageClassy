import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D,Conv1D, MaxPooling1D
from sklearn.preprocessing import LabelEncoder
from keras.utils import np_utils
import sys
sys.path.append('../')
from utils import *

# hyper parameters
batch_sz = 32
epochs = 400
filter_sizes = [32,64]
hidden_sz =128

def build_model():
    model = Sequential()
    model.add(Conv1D(32, (3,), padding='same', input_shape=(64,64)))
    model.add(Activation('relu'))

    for fs in filter_sizes:
        model.add(Conv1D(fs, (2,)))
        model.add(Conv1D(fs, (2,)))
        model.add(Activation('relu'))
        model.add(MaxPooling1D(pool_size=(2, )))
        model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(hidden_sz))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(40))
    model.add(Activation('softmax'))

    opt = keras.optimizers.Adam()
    model.compile(loss='categorical_crossentropy',optimizer=opt, metrics=['accuracy'])
    return model

def fit(X,y):
    print("build model...filter sizes: ",filter_sizes)
    model = build_model()
    X_train, X_valid = X[:int(0.1*(X.shape[0]))], X[int(0.1*(X.shape[0])):]
    y_train, y_valid = y[:int(0.1*(X.shape[0]))], y[int(0.1*(X.shape[0])):]

    print("start training for %s epochs. X shape: %s, y shape: %s"%(epochs, X_train.shape, y_train.shape))
    history = model.fit(X_train,y_train,batch_size=batch_sz, epochs=epochs,verbose=0)
    scores = model.evaluate(X_valid, y_valid)
    
    print('\nValidation loss:', scores[0],'Validation accuracy:', scores[1])
    print("Training loss",history.history["loss"][-1],"training accuracy",history.history["acc"][-1])
    return model

if __name__=="__main__":
    X,y =load_data("../../")
    # X_test = load_test_data(p = "../../")

    encoder = LabelEncoder()
    encoded_Y = encoder.fit_transform(y.reshape((-1,)))
    one_hot_y = np_utils.to_categorical(encoded_Y)
    # model = fit(np.expand_dims(X, axis=-1),one_hot_y)
    model = fit(X,one_hot_y)
    exit(0)
    yp = model.predict(np.expand_dims(X, axis=-1))
    write_prediction(yp,"../../basic_cnn.csv")
