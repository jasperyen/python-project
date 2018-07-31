import cv2
import os
import subprocess

import numpy as np

from keras.models import Model, load_model
from keras.layers import DepthwiseConv2D
from keras_applications.mobilenet import relu6
from keras.utils.generic_utils import CustomObjectScope

captcha_height = 128
captcha_width = 128

letter_str = '0123456789_'
letter_amount = len(letter_str)

model_path = '..\\..\\model\\captcha_model.h5'
dir_path = '..\\..\\data\\captcha\\'

print('Loading model : ' + model_path)
with CustomObjectScope({'relu6': relu6,'DepthwiseConv2D': DepthwiseConv2D}):
    model = load_model(model_path)

model.summary()
print('Load model success\n')

data = np.zeros([1, captcha_height, captcha_width, 3]).astype('float32')

for file in os.listdir(dir_path) :

    src = cv2.imread(dir_path + file)    
    img = np.zeros((captcha_height, captcha_width, 3), np.uint8)
    img[0:60, 14:114] = src[0:60, 0:100]
    img[68:128, 14:114] = src[0:60, 100:200]
    
    data[0] = img
    data = data / 255
    
    predict = model.predict(data)
    captcha = ''
    if predict[0][0].argmax(axis=-1) == 1 :
        for i in range(1, 6) :
            index = predict[i][0].argmax(axis=-1)
            captcha = captcha + letter_str[index]
    elif predict[0][0].argmax(axis=-1) == 0 :
        for i in range(6, 12) :
            index = predict[i][0].argmax(axis=-1)
            captcha = captcha + letter_str[index]
    
    print('result : ' + captcha)
    
    cv2.imshow('src', src)
    cv2.waitKey(0)

