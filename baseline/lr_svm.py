import sys
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsClassifier
sys.path.append('../')
from utils import *


def eval(X,y,clf, c = 1):
    if clf == "LogisticRegression":
        model = LogisticRegression()
    elif clf == "SVM":
        model = SVC(C=c)
    else:
        print("Input name has to be either LogisticRegression or SVM!")
        exit(0)

    X_train, X_valid = X[:int(0.1*(X.shape[0]))], X[int(0.1*(X.shape[0])):]
    y_train, y_valid = y[:int(0.1*(X.shape[0]))], y[int(0.1*(X.shape[0])):]
    model.fit(X_train.reshape((-1,4096)),y_train.reshape((-1,)))
    print("model score on validation set",model.score(X_valid.reshape((-1,4096)),y_valid.reshape((-1,))))
    # x_test = load_test_data(p=p)
    # yp = model.predict()
    # write_prediction(yp)




if __name__=="__main__":
    
    p=sys.argv[1] # path to data files
    X,y = load_data(p=p, raw=True)
    clf = sys.argv[2]

    eval(X,y,clf)