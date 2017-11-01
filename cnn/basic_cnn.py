from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from sklearn.preprocessing import LabelEncoder
from keras.utils import np_utils
import sys
sys.path.append('../')
from utils import *

# hyper parameters
batch_sz = 32
epochs = 200
filter_sizes = [64,32,32,32]

def build_model():
    model = Sequential()
    model.add(Conv2D(32, (3, 3), padding='same' input_shape=(batch_sz, 64,64,1)))
    model.add(Activation('relu'))

    for fs in filter_sizes:
        model.add(Conv2D(fs, (3, 3)))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(40))
    model.add(Activation('softmax'))

    opt = keras.optimizers.Adam()
    model.compile(loss='categorical_crossentropy',optimizer=opt, metrics=['accuracy'])
    return model

def fit(X,Y,epochs=50):
    model = build_model()
    X_train, X_valid = X[:int(0.1*(X.shape[0]))], X[int(0.1*(X.shape[0])):]
    y_train, y_valid = y[:int(0.1*(X.shape[0]))], y[int(0.1*(X.shape[0])):]

    model.fit(X_train,y_train)
    scores = model.evaluate(X_test, y_test,batch_size=batch_sz, epochs=epochs)
    print('Test loss:', scores[0])
    print('Test accuracy:', scores[1])

if __name__=="__main__":
    X,y =load_data("../../")
    encoder = LabelEncoder()
    encoded_Y = encoder.fit_transform(Y)
    one_hot_y = np_utils.to_categorical(encoded_Y)
    fit(X,one_hot_y)