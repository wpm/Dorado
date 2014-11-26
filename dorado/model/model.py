import numpy
import theano

from dorado.model.parameters import ModelParameters


class Model(object):
    @staticmethod
    def initial_parameters(dimension, classes):
        w = numpy.zeros((dimension, classes), dtype=theano.config.floatX)
        b = numpy.zeros((classes,), dtype=theano.config.floatX)
        return ModelParameters(w, b)

    @classmethod
    def factory(cls):
        return lambda parameters: cls(*parameters)

    def __init__(self, *parameters):
        self.set_parameters(ModelParameters(*parameters))

    def __repr__(self):
        return "<%s, %d dimensions, %d classes>" % (self.__class__.__name__, self.dimension(), self.classes())

    def __eq__(self, other):
        return self.get_parameters() == other.get_parameters()

    def __ne__(self, other):
        return not self == other

    def dimension(self):
        raise NotImplementedError()

    def classes(self):
        raise NotImplementedError()

    def get_parameters(self):
        raise NotImplementedError()

    def set_parameters(self, parameters):
        raise NotImplementedError()

    def train(self, data, learning_rate):
        raise NotImplementedError()

    def error_rate(self, data):
        raise NotImplementedError()

    def predict(self, data):
        raise NotImplementedError()
