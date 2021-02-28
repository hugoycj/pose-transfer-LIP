import os
import random

import cv2 as cv
import numpy as np

def get_category(categories_folder, name):
    filename = os.path.join(categories_folder, name + '.png')
    semantic = cv.imread(filename, 0)
    return semantic


def to_bgr(y_pred):
    ret = np.zeros((im_size, im_size, 3), np.float32)
    for r in range(320):
        for c in range(320):
            color_id = y_pred[r, c]
            # print("color_id: " + str(color_id))
            ret[r, c, :] = color_map[color_id]
    ret = ret.astype(np.uint8)
    return ret


def random_choice(image_size):
    height, width = image_size
    crop_height, crop_width = 320, 320
    x = random.randint(0, max(0, width - crop_width))
    y = random.randint(0, max(0, height - crop_height))
    return x, y


def safe_crop(mat, x, y):
    crop_height, crop_width = 320, 320
    if len(mat.shape) == 2:
        ret = np.zeros((crop_height, crop_width), np.uint8)
    else:
        ret = np.zeros((crop_height, crop_width, 3), np.uint8)
    crop = mat[y:y + crop_height, x:x + crop_width]
    h, w = crop.shape[:2]
    ret[0:h, 0:w] = crop
    return ret