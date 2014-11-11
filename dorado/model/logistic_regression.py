import numpy

import theano
import theano.tensor as T

from dorado.model.theano_model import TheanoModel


class LogisticRegression(TheanoModel):
    @classmethod
    def create(cls, W, b, l1=0.0, l2=0.0):
        dimension, classes = W.shape
        model = cls(dimension, classes, l1, l2)
        model.W.set_value(W)
        model.b.set_value(b)
        return model

    def __init__(self, dimension, classes, l1=0.0, l2=0.0):
        self.W = theano.shared(numpy.zeros((dimension, classes), dtype=theano.config.floatX), name='W')
        self.b = theano.shared(numpy.zeros((classes,), dtype=theano.config.floatX), name='b')
        super(LogisticRegression, self).__init__(dimension, classes, l1, l2)

    def parameters(self):
        return [self.W, self.b]

    def regularized_parameters(self):
        return [self.W]

    def p_y_given_x(self):
        return T.nnet.softmax(T.dot(self.x, self.W) + self.b)

    def copy(self):
        return LogisticRegression.create(self.W.get_value(), self.b.get_value(), self._l1, self._l2)


