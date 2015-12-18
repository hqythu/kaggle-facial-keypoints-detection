import csv
import theano
from theano import tensor as T
from theano import function
import scipy.io as sio
import numpy as np

from load import mnist


class Model():
    def __init__(self, learning_rate, momentum, batch_size, epoch_time):
        self.layers = []
        self.params = []
        self.input = T.fmatrix()
        self.output = self.input
        self.label = T.fmatrix()

        self.learning_rate = learning_rate
        self.momentum = momentum
        self.batch_size = batch_size
        self.epoch_time = epoch_time

    def add_layer(self, layer):
        self.layers.append(layer)
        self.params += layer.params
        self.output = layer.get_output(self.output)

    def set_loss_function(self, loss_function):
        self.loss_function = loss_function

    def build(self):
        self.cost = self.loss_function(self.output, self.label)
        self.label_predict = self.output #T.argmax(self.output, axis=1)
        self.grad_params = [T.grad(self.cost, param) for param in self.params]
        self.last_delta = T.tensor4()
        updates = [(param, param - grad_param * self.learning_rate)
            for param, grad_param in zip(self.params, self.grad_params)]

        self.train = function([self.input, self.label], self.cost,
            updates=updates, allow_input_downcast=True)
        self.predict = function([self.input], self.label_predict,
            allow_input_downcast=True)

    def train_model(self, train_x, train_y):
        teX = np.asarray(train_x)
        teY = np.asarray(train_y)
        for i in range(self.epoch_time):
            print 'epoch:', i+1, ',',
            cost = []
            for start, end in zip(range(0, len(train_x), self.batch_size),
                range(self.batch_size, len(train_x), self.batch_size)):
                cost += [self.train(train_x[start:end], train_y[start:end])]
            tmp = self.predict(teX) - teY
            # tmp = (self.predict(teX) - teY) * (teY != -1)
            accuracy = np.sqrt(np.mean(tmp * tmp))
            # accuracy = np.mean(np.argmax(test_y, axis=1) == self.predict(test_x))
            print 'cost:', np.mean(cost), ',', 'accuracy:', accuracy * 48

    def save_test_result(self, test_x):
        dic = {
                'left_eye_center_x'         :  0,
                'left_eye_center_y'         :  1,
                'right_eye_center_x'        :  2,
                'right_eye_center_y'        :  3,
                'left_eye_inner_corner_x'   :  4,
                'left_eye_inner_corner_y'   :  5,
                'left_eye_outer_corner_x'   :  6,
                'left_eye_outer_corner_y'   :  7,
                'right_eye_inner_corner_x'  :  8,
                'right_eye_inner_corner_y'  :  9,
                'right_eye_outer_corner_x'  : 10,
                'right_eye_outer_corner_y'  : 11,
                'left_eyebrow_inner_end_x'  : 12,
                'left_eyebrow_inner_end_y'  : 13,
                'left_eyebrow_outer_end_x'  : 14,
                'left_eyebrow_outer_end_y'  : 15,
                'right_eyebrow_inner_end_x' : 16,
                'right_eyebrow_inner_end_y' : 17,
                'right_eyebrow_outer_end_x' : 18,
                'right_eyebrow_outer_end_y' : 19,
                'nose_tip_x'                : 20,
                'nose_tip_y'                : 21,
                'mouth_left_corner_x'       : 22,
                'mouth_left_corner_y'       : 23,
                'mouth_right_corner_x'      : 24,
                'mouth_right_corner_y'      : 25,
                'mouth_center_top_lip_x'    : 26,
                'mouth_center_top_lip_y'    : 27,
                'mouth_center_bottom_lip_x' : 28,
                'mouth_center_bottom_lip_y' : 29
              }

        answer = self.predict(test_x)
        answer = answer * 48.0 + 48
        sio.savemat('result.mat', {'test_x': test_x,
                                   'test_y': answer} )
        csvfile = file('result.csv', 'wb')
        writer = csv.writer(csvfile)
        writer.writerow(['RowId', 'Location'])

        csvmodel = file('IdLookupTable.csv', 'rb')
        csvmodel.readline()
        text = np.loadtxt(csvmodel, dtype=np.str, delimiter=",")
        csvmodel.close()
        for i in range(len(text)):
            writer.writerow([ int(text[i][0]), answer[ int(text[i][1])-1 ][ dic[text[i][2]] ] ])

        csvfile.close()
