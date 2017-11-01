import sys
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, SVR

sys.path.append('../')
from utils import *

if __name__=="__main__":
    p="../../" # path to data files

    # model = LogisticRegression()
    model = SVR()
    
    X,y = load_data(p=p)

    X_train, X_valid = X[:int(0.1*(X.shape[0]))], X[int(0.1*(X.shape[0])):]
    y_train, y_valid = y[:int(0.1*(X.shape[0]))], y[int(0.1*(X.shape[0])):]
    model.fit(X_train.reshape((-1,4096)),y_train.reshape((-1,)))
    print("model score on validation set",model.score(X_valid.reshape((-1,4096)),y_valid.reshape((-1,))))
    # x_test = load_test_data(p=p)
    # yp = model.predict()
    # write_prediction(yp)
