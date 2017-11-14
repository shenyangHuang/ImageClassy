import random
from PIL import Image, ImageChops
import numpy as np
import skimage.util as sk_util

class Digits:
    def __init__(self):
        self.digit_map = {}
        for digit in '0123456789AM':
            f_handle = open(str(digit) + '.csv', 'r')
            lines = f_handle.readlines()
            lines = map(lambda x: x.replace('\n', ''), lines)
            lines = map(lambda x: './%s/%s' % (str(digit), x), lines)
            lines = list(lines)
            self.digit_map[str(digit)] = lines

    def trim(self, im):
        bg = Image.new(im.mode, im.size, im.getpixel((0,0)))
        diff = ImageChops.difference(im, bg)
        diff = ImageChops.add(diff, diff, 2.0, -100)
        bbox = diff.getbbox()
        if bbox:
            return im.crop(bbox)

    def reverse(self, img):
        mat = np.copy(np.asarray(img, dtype = np.uint8))
        mat[:,:,0] = 255 - mat[:,:,0]
        mat[:,:,1] = 255 - mat[:,:,1]
        mat[:,:,2] = 255 - mat[:,:,2]
        return Image.fromarray(mat)
    
    def get_digit(self, digit):
        index = random.randint(0, len(self.digit_map[str(digit)]) - 1)
        path =  self.digit_map[str(digit)][index]
        img = Image.open(path)
        img = self.reverse(img)
        img = img.rotate(random.randint(-45, 45))
        img = self.trim(img)

        new_size = random.randint(20, 25)
        ratio = max(img.size[0]/new_size, img.size[1]/new_size)
        img = img.resize((int(img.size[0] / ratio),
                          int(img.size[1] / ratio)))
        img_arr = np.asarray(img, dtype = np.uint8)
        mat = img_arr[:,:,0]
        return mat


def place_character(shapes):
    open_box = [(0,0,64,64)]
    result = []
    for shape in shapes:
        width = shape[0]
        height = shape[1]
        for slot_index in range(len(open_box)):
            (pos_x, pos_y, open_x, open_y) = open_box[slot_index]
            if open_x >= width and open_y >= height:
                select_x = random.randint(0, open_x - width)
                select_y = random.randint(0, open_y - height)
                open_box.pop(slot_index)
                open_box.append((pos_x, pos_y, select_x, select_y))
                open_box.append((select_x + width,
                                 pos_y,
                                 open_x - (select_x + width),
                                 select_y))
                open_box.append((pos_x,
                                 select_y + height,
                                 select_x,
                                 open_y - (select_y + height)))
                open_box.append((select_x + width,
                                 select_y + height,
                                 open_x - (select_x + width),
                                 open_y - (select_y + height)))
                result.append([pos_x + select_x, pos_y + select_y])
                break
    return result
    
def build_train_sample(gen=Digits()):
    c1 = 'AM'[random.randint(0,1)]
    d1 = '0123456789'[random.randint(0,9)]
    d2 = '0123456789'[random.randint(0,9)]
    operation = gen.get_digit(c1)
    operand0 = gen.get_digit(d1)
    operand1 = gen.get_digit(d2)

    canvas = np.uint8(np.zeros((64, 64)))
    pos = []
    c = 0
    while len(pos) < 3:
        pos = place_character([operation.shape, operand0.shape, operand1.shape])
        if c >= 1000:
            return None
        c += 1
    pos0 = pos[0]
    pos1 = pos[1]
    pos2 = pos[2]

    canvas[pos0[0]:pos0[0] + operation.shape[0],
           pos0[1]:pos0[1] + operation.shape[1]] += operation
    canvas[pos1[0]:pos1[0] + operand0.shape[0],
           pos1[1]:pos1[1] + operand0.shape[1]] += operand0
    canvas[pos2[0]:pos2[0] + operand1.shape[0],
           pos2[1]:pos2[1] + operand1.shape[1]] += operand1

    return (canvas, c1, int(d1), int(d2))

from tqdm import tqdm
def build_data_set(size):
    dataset = np.uint8(np.zeros((size, 64, 64)))
    labels = np.uint8(np.zeros((size, 3)))
    generator = Digits()
    for i in tqdm(range(size)):
        sample = None
        while sample is None:
            sample = build_train_sample(generator)
        dataset[i,:,:] = sample[0][:,:]
        if sample[1] == 'A':
            labels[i,0] = 1
        else:
            labels[i,0] = 0
        labels[i,1] = np.uint8(sample[2])
        labels[i,2] = np.uint8(sample[3])    
    return (dataset, labels)
