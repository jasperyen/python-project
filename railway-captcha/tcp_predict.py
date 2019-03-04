import time
import socket
import threading
import cv2

import numpy as np

from keras import backend as K
from keras.models import Model, load_model

global lock, graph, model, letter_str, letter_amount, captcha_height, captcha_width

graph = K.get_session().graph

bind_ip = ''
bind_port = 8900

captcha_height = 128
captcha_width = 128

letter_str = '0123456789ABCDEFGHJKLMNPQRSTUVWXYZ_'
letter_amount = len(letter_str)

model_path = '..\\..\\model\\captcha_model.h5'

print('Loading model : ' + model_path)
model = load_model(model_path)
print('Load model success\n')

server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server.bind((bind_ip, bind_port))
server.listen(10)  # max backlog of connections

lock = threading.Lock()

print ('Listening on {}:{}'.format(bind_ip, bind_port))


def handle_client_connection(client_socket, address):
    global lock, graph, model, letter_str, letter_amount, captcha_height, captcha_width

    request = client_socket.recv(8192)
    src = cv2.imdecode(np.fromstring(request, np.uint8), cv2.IMREAD_COLOR)

    img = np.zeros((captcha_height, captcha_width, 3), np.uint8)
    img[0:60, 14:114] = src[0:60, 0:100]
    img[68:128, 14:114] = src[0:60, 100:200]
    
    data = np.zeros([1, captcha_height, captcha_width, 3]).astype('float32')
    data[0] = img
    data = data / 255
    
    with lock, graph.as_default():
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
    

    print('Accepted connection from {}:{}'.format(address[0], address[1]) + ' , result : ' + captcha)
    client_socket.send(captcha.encode('utf-8'))
    client_socket.close()

while True:
    client_sock, address = server.accept()
    client_handler = threading.Thread(
        target=handle_client_connection,
        args=(client_sock,address,)
    )
    client_handler.start()