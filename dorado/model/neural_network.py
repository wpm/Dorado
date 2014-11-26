from math import sqrt

import numpy
import theano
import theano.tensor as T

from dorado import random_matrix
from dorado.model.theano_model import TheanoModel
from dorado.model.parameters import ModelParameters


class NeuralNetwork(TheanoModel):
    @staticmethod
    def initial_parameters(dimension, classes, hidden):
        w = numpy.zeros((dimension, classes), dtype=theano.config.floatX)
        b = numpy.zeros((classes,), dtype=theano.config.floatX)
        bound = 4 * sqrt(6.0 / (dimension + hidden))
        h = random_matrix(dimension, hidden, bound)
        d = numpy.zeros((hidden,), dtype=theano.config.floatX)
        bound = 4 * sqrt(6.0 / (hidden + classes))
        u = random_matrix(hidden, classes, bound)
        return ModelParameters(w, b, h, d, u)

    def __init__(self, w, b, h, d, u, l1=0.0, l2=0.0):
        dimension, classes = w.shape
        self.w = theano.shared(w, name='W')
        self.b = theano.shared(b, name='b')
        self.h = theano.shared(h, name='H')
        self.d = theano.shared(d, name='d')
        self.u = theano.shared(u, name='U')
        super(NeuralNetwork, self).__init__(dimension, classes, l1, l2)

    def dimension(self):
        return self.w.shape[0]

    def classes(self):
        return self.w.shape[1]

    def parameters(self):
        return [self.w, self.b, self.h, self.d, self.u]

    def regularized_parameters(self):
        return [self.h, self.u, self.w]

    def p_y_given_x(self):
        return T.nnet.softmax(T.dot(T.dot(self.x, self.h) + self.d, self.u) + T.dot(self.x, self.w) + self.b)

    def __repr__(self):
        return "<%s, %d dimensions, %d classes, hidden layer size %d>" % \
               (self.__class__.__name__, self.dimension(), self.classes(), self.d.shape)

