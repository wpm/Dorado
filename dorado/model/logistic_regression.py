import numpy
import theano
import theano.tensor as T
from dorado.model.parameters import ModelParameters

from dorado.model.theano_model import TheanoModel


class LogisticRegression(TheanoModel):
    @staticmethod
    def initial_parameters(dimension, classes):
        w = numpy.zeros((dimension, classes), dtype=theano.config.floatX)
        b = numpy.zeros((classes,), dtype=theano.config.floatX)
        return ModelParameters(w, b)

    def __init__(self, w, b, l1=0.0, l2=0.0):
        self.w = theano.shared(w, name='W')
        self.b = theano.shared(b, name='b')
        super(LogisticRegression, self).__init__(l1, l2, w, b)

    def dimension(self):
        return self.w.shape[0]

    def classes(self):
        return self.w.shape[1]

    def p_y_given_x(self):
        return T.nnet.softmax(T.dot(self.x, self.w) + self.b)

    def parameters(self):
        return [self.w, self.b]

    def regularized_parameters(self):
        return [self.w]
