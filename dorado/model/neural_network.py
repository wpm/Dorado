from math import sqrt

import numpy
import theano
import theano.tensor as T

from dorado.model.theano_model import TheanoModel


class NeuralNetwork(TheanoModel):
    def __init__(self, dimension, classes, h, l1=0.0, l2=0.0):
        self.h = h
        b = 4 * sqrt(6.0 / (dimension + self.h))
        self.H = theano.shared(self._random_matrix(dimension, self.h, b), name='H')
        self.d = theano.shared(numpy.zeros((self.h,), dtype=theano.config.floatX), name='d')
        b = 4 * sqrt(6.0 / (self.h + classes))
        self.U = theano.shared(self._random_matrix(self.h, classes, b), name='U')
        self.W = theano.shared(numpy.zeros((dimension, classes), dtype=theano.config.floatX), name='W')
        self.b = theano.shared(numpy.zeros((classes,), dtype=theano.config.floatX), name='b')
        super(NeuralNetwork, self).__init__(dimension, classes, l1, l2)

    def parameters(self):
        return [self.W, self.b, self.H, self.d, self.U]

    def regularized_parameters(self):
        return [self.H, self.U, self.W]

    def p_y_given_x(self):
        return T.nnet.softmax(T.dot(T.dot(self.x, self.H) + self.d, self.U) + T.dot(self.x, self.W) + self.b)

    def __repr__(self):
        return "<%s, %d dimensions, %d classes, hidden layer size %d>" % \
               (self.__class__.__name__, self.dimension(), self.classes(), self.h)

    def _random_matrix(self, r, c, b=1):
        """
        Matrix with random elements selected uniformly from [-b, b].

        :type r: int
        :param r: rows

        :type c: int
        :param c: columns

        :type b: float
        :param b: bound

        :returns: randomly generated matrix
        """
        return numpy.random.uniform(-b, b, (r, c))
