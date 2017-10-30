import sys
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

sys.path.append('../')
from utils import *

if __name__=="__main__":
    p="../../" # path to data files

    # model = LogisticRegression()
    model = SVC()
    
    X,y = load_data(p=p)
    X_train, X_valid = X[:0.1*(X.shape[0])], X[0.1*(X.shape[0]):]
    y_train, y_valid = y[:0.1*(X.shape[0])], y[0.1*(X.shape[0]):]
    model.fit(X_train.reshape((-1,4096)),y_train)
    print("model score on validation set",model.score(X_valid,y_valid))

    # x_test = load_test_data(p=p)
    # yp = model.predict()
    # write_prediction(yp)
