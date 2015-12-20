import math
import theano
from theano import tensor as T
from theano import function
import theano.tensor.signal.downsample
import numpy as np


def sigmoid(X):
    return T.nnet.sigmoid(X)


def hyperbolic(X):
    return T.tanh(X)


def rectify(X):
    return T.maximum(X, 0.0)


def softmax(X):
    return T.nnet.softmax(X)


def EuclideanLoss(output, label):
    return T.mean(T.sqr(output - label))
    # return T.mean(T.sqr(T.dot(output - label, 1 * (label != -1) )))


def CrossEntropyLoss(output, label):
    return T.mean(T.nnet.categorical_crossentropy(output, label))


class SoftmaxLayer(object):
    def __init__(self):
        self.params = []
        self.regularization = []

    def get_output(self, input):
        self.input = input
        self.output = softmax(self.input)
        return self.output


class FullConnectedLayer(object):
    def __init__(self, num_in, num_out, rand_std, activation=None):
        bound = math.sqrt(6.0) / math.sqrt(num_in + num_out)
        w_values = np.asarray(np.random.uniform(-bound, bound, (num_in, num_out)), dtype='float32')
        b_values = np.zeros(num_out, dtype='float32')

        self.w = theano.shared(w_values)
        self.b = theano.shared(b_values)
        self.params = [self.w, self.b]
        self.regularization = [T.mean(T.sqr(self.w))]
        self.activation = activation

    def get_output(self, input):
        self.input = input.flatten(2)
        lin_output = T.dot(self.input, self.w) + self.b
        self.output = lin_output if self.activation is None else self.activation(lin_output)
        return self.output


class ReshapeLayer(object):
    def __init__(self, num_channel, image_width, image_height):
        self.num_channel = num_channel
        self.image_width = image_width
        self.image_height = image_height
        self.params = []
        self.regularization = []

    def get_output(self, input):
        self.input = input
        self.output = self.input.reshape((-1, self.num_channel,
            self.image_width, self.image_height))
        return self.output


class ConvolutionLayer(object):
    def __init__(self, kernel_size, num_out, num_in, rand_std, activation=None):
        w_shape = (num_out, num_in) + kernel_size
        w_values = np.asarray(np.random.randn(*w_shape) * rand_std,
            dtype='float32')
        b_values = np.asarray(np.random.randn(num_out), dtype='float32')

        self.kernel_size = kernel_size
        self.w = theano.shared(w_values)
        self.b = theano.shared(b_values)
        self.params = [self.w, self.b]
        self.regularization = [T.mean(T.sqr(self.w))]
        self.activation = activation

    def get_output(self, input):
        self.input = input
        conv_output = T.nnet.conv.conv2d(self.input, self.w) + self.b.dimshuffle('x', 0, 'x', 'x')
        self.output = conv_output if self.activation is None else self.activation(conv_output)
        return self.output


class PoolingLayer(object):
    def __init__(self, kernel_size, activation=None):
        self.params = []
        self.regularization = []
        self.kernel_size = kernel_size
        self.activation = activation

    def get_output(self, input):
        self.input = input
        pool_output = T.signal.downsample.max_pool_2d(self.input,
            self.kernel_size, ignore_border=True)
        self.output = pool_output if self.activation is None else self.activation(pool_output)
        return self.output


class DropoutLayer(object):
    def __init__(self, p_drop):
        self.p_drop = p_drop
        self.params = []

    def get_output(input):
        self.input = input
        self.output = output
        if self.p_drop > 0:
            retain_p = 1 - self.p_drop
        return self.output
