import theano
from theano import tensor as T
from theano import function
import numpy as np

from load import mnist


class Model():
    def __init__(self, learning_rate, batch_size, epoch_time):
        self.layers = []
        self.params = []
        self.input = T.fmatrix()
        self.output = self.input
        self.label = T.fmatrix()

        self.learning_rate = learning_rate
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
        self.label_predict = T.argmax(self.output, axis=1)
        self.grad_params = [T.grad(self.cost, param) for param in self.params]
        updates = [(param, param - grad_param * self.learning_rate)
            for param, grad_param in zip(self.params, self.grad_params)]

        self.train = function([self.input, self.label], self.cost,
            updates=updates, allow_input_downcast=True)
        self.predict = function([self.input], self.label_predict,
            allow_input_downcast=True)

    def train_model(self, train_x, train_y, test_x, test_y):
        for i in range(self.epoch_time):
            print 'epoch:', i+1, ',',
            cost = []
            for start, end in zip(range(0, len(train_x), self.batch_size),
                range(self.batch_size, len(train_x), self.batch_size)):
                cost += [self.train(train_x[start:end], train_y[start:end])]
            accuracy = np.mean(np.argmax(test_y, axis=1) == self.predict(test_x))
            print 'cost:', np.mean(cost), ',', 'accuracy:', accuracy

