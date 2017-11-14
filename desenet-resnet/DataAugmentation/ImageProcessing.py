import numpy as np
import cv2

def smooth(img):
    blur = ((1, 1), 1)
    erode_ = (2, 2)
    dilate_ = (3, 3)
    img = cv2.dilate(cv2.erode(cv2.GaussianBlur(img/255, blur[0], blur[1]),
                               np.ones(erode_)), np.ones(dilate_))
    g = np.vectorize(lambda x : 0 if x == 0 else 255)
    return np.uint8(g(img))

def filter(img, t=20):
    f = np.vetorize(lambda x : 0 if x <= 255 - t else x)
    return f(img)
