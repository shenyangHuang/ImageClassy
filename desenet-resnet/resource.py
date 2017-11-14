import numpy
import pickle

def load_category_map():
    path = 'category_map.pickle'
    f = open(path, 'rb')
    category_map = pickle.load(f)
    f.close()
    return category_map

def load_reverse_category_map():
    path = 'reverse_category_map.pickle'
    f = open(path, 'rb')
    reverse_category_map = pickle.load(f)
    f.close()
    return reverse_category_map

def load_train_x(t=255):
    def f(x):
        if x <= 255 - t:
            return 0
        else:
            return x
    
    train_x = numpy.load('train_x.npy')
    f = numpy.vectorize(f)
    train_x = f(train_x.flatten()).reshape(-1, 64, 64, 1)
    return train_x

def load_train_y():
    return numpy.load('train_y.npy')

def load_test_x(t=255):
    def f(x):
        if x <= 255 - t:
            return 0
        else:
            return x
    
    test_x = numpy.load('test_x.npy')
    f = numpy.vectorize(f)
    test_x = f(test_x.flatten()).reshape(-1, 64, 64, 1)
    return test_x
