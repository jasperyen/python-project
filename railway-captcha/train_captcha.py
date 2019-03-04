import cv2
import csv
import base64
import os
import subprocess

import numpy as np
import pandas as pd
import tensorflow as tf

from keras.models import Model
from keras.layers import Input, Dense, Dropout
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.applications.mobilenetv2 import MobileNetV2

def to_onehot(text):
    tag_list = []
    for letter in text:
        onehot = [0 for _ in range(letter_amount)]
        num = letter_str.find(letter)
        onehot[num] = 1
        tag_list.append(onehot)
    return tag_list

def to_taglist(text):
    if len(text) == 5 :
        tag56_list = [0,1]
        tag5_list = to_onehot(text)
        tag6_list = to_onehot('______')
    elif len(text) == 6 :
        tag56_list = [1,0]
        tag5_list = to_onehot('_____')
        tag6_list = to_onehot(text)
    return (tag56_list, tag5_list, tag6_list)

def to_label(read_tag, size):
    label = [[] for _ in range(size)]
    for arr in read_tag:
        for index in range(size):
            label[index].append(arr[index])
    label = [arr for arr in np.asarray(label)]
    return label


train_amount = 40000
test_amount = 10000
valid_amount = 10000

captcha_height = 128
captcha_width = 128

letter_str = "0123456789ABCDEFGHJKLMNPQRSTUVWXYZ_"
letter_amount = len(letter_str)


train_data = np.zeros([train_amount, captcha_height, captcha_width, 3]).astype('float32')
test_data = np.zeros([test_amount, captcha_height, captcha_width, 3]).astype('float32')
valid_data = np.zeros([valid_amount, captcha_height, captcha_width, 3]).astype('float32')

train_56_tag = [[]] * train_amount
test_56_tag = [[]] * test_amount
valid_56_tag = [[]] * valid_amount

train_5d_tag = [[]] * train_amount
test_5d_tag = [[]] * test_amount
valid_5d_tag = [[]] * valid_amount

train_6d_tag = [[]] * train_amount
test_6d_tag = [[]] * test_amount
valid_6d_tag = [[]] * valid_amount
            

print('\nLoading data : ', end='')

#   開始讀檔
csv_path = "C:\\workspace\\temp\\correct_en.csv"

count = 0
train_count = 0
test_count = 0
valid_count = 0

with open(csv_path, newline='') as csvfile:
    for row in csv.reader(csvfile) :
        text = row[1]
        
        if len(text) != 5 and len(text) != 6 :
            continue
        
        nparr = np.fromstring(base64.b64decode(row[2]), np.uint8)
        src = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        img = np.zeros((captcha_height, captcha_width, 3), np.uint8)
        img[0:60, 14:114] = src[0:60, 0:100]
        img[68:128, 14:114] = src[0:60, 100:200]

        if count < train_amount :   # 前 N 張圖
            train_data[train_count] = img
            train_56_tag[train_count], train_5d_tag[train_count], train_6d_tag[train_count] = to_taglist(text)
            train_count += 1

        elif count < train_amount + test_amount :   # 前 N+M 張圖
            test_data[test_count] = img
            test_56_tag[test_count], test_5d_tag[test_count], test_6d_tag[test_count] = to_taglist(text)
            test_count += 1

        elif count < train_amount + test_amount + valid_amount :   # 前 N+M+O 張圖
            valid_data[valid_count] = img
            valid_56_tag[valid_count], valid_5d_tag[valid_count], valid_6d_tag[valid_count] = to_taglist(text)
            valid_count += 1

        else :
            break

        count += 1

print('\n\ntrain amount : %d' %train_count)
print('\ntest amount : %d' %test_count)
print('\nvalid amount : %d' %valid_count)

#  標準化資料
train_data /= 255
test_data /= 255
valid_data /= 255

train_onehot = {'digit5or6' : np.asarray(train_56_tag)}
test_onehot = {'digit5or6' : np.asarray(test_56_tag)}
valid_onehot = {'digit5or6' : np.asarray(valid_56_tag)}

train_5d_onehot = to_label(train_5d_tag, 5)
test_5d_onehot = to_label(test_5d_tag, 5)
valid_5d_onehot = to_label(valid_5d_tag, 5)

train_6d_onehot = to_label(train_6d_tag, 6)
test_6d_onehot = to_label(test_6d_tag, 6)
valid_6d_onehot = to_label(valid_6d_tag, 6)

for i in range(5) :
    train_onehot['5digit_' + str(i)] = train_5d_onehot[i]
    test_onehot['5digit_' + str(i)] = test_5d_onehot[i]
    valid_onehot['5digit_' + str(i)] = valid_5d_onehot[i]

for i in range(6) :    
    train_onehot['6digit_' + str(i)] = train_6d_onehot[i]
    test_onehot['6digit_' + str(i)] = test_6d_onehot[i]
    valid_onehot['6digit_' + str(i)] = valid_6d_onehot[i]
    

#   建置神經網路
base_model = MobileNetV2(weights='imagenet', include_top=False, pooling='avg',
                        input_shape=(captcha_height, captcha_width, 3))

out = base_model.output

out = Dropout(0.5)(out)

final_layer = []

final_layer.append(Dense(2, name='digit5or6', activation='softmax')(out))

for i in range (5) :
    final_layer.append(Dense(letter_amount, name='5digit_' + str(i), activation='softmax')(out))

for i in range (6) :
    final_layer.append(Dense(letter_amount, name='6digit_' + str(i), activation='softmax')(out))

model = Model(inputs=base_model.input, outputs=final_layer)

model.summary()
print('create model success')

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
print('compile model success')


checkpoint = ModelCheckpoint("C:\\workspace\\temp\\train\\captcha_model.h5", 
                             monitor = 'val_5digit_2_acc', 
                             verbose = 1, save_best_only = True, mode = 'max')
earlystop = EarlyStopping(monitor = 'val_5digit_2_acc', patience = 100, verbose = 1, mode='auto')
callbacks_list = [checkpoint, earlystop]

print('training model . . . ')

train_history = model.fit( x = train_data, y = train_onehot,
                           validation_data = (valid_data, valid_onehot),
                           epochs = 1000, batch_size = 16, verbose = 2, callbacks = callbacks_list)

evaluate = model.evaluate(test_data, test_onehot)
print('\ntest evaluate: ', evaluate)


subprocess.call("pause",shell=True)
