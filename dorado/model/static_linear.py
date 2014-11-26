import numpy

from dorado.model.model import Model
from dorado.model.parameters import ModelParameters


class StaticLinearModel(Model):
    @classmethod
    def initial_parameters(cls, dimension, classes):
        w = numpy.random.rand(dimension, classes)
        b = numpy.random.rand(classes)
        return ModelParameters(w, b)

    def __repr__(self):
        return "<%s, %d dimensions, %d classes>" % (self.__class__.__name__, self.dimension(), self.classes())

    def dimension(self):
        return self.w.shape[0]

    def classes(self):
        return self.w.shape[1]

    def get_parameters(self):
        return self._parameters

    def set_parameters(self, parameters):
        self._parameters = parameters

    def _get_w(self):
        return self._parameters[0]

    w = property(_get_w, None, None)

    def _get_b(self):
        return self._parameters[1]

    b = property(_get_b, None, None)

    def train(self, data, learning_rate):
        pass

    def error_rate(self, data):
        predictions = self.predict(data)
        correct = numpy.count_nonzero(predictions == data.labels)
        n = float(len(data.vectors))
        return (n - correct) / n

    def predict(self, data):
        p_y_x = self._p_y_given_x(data.vectors)
        return numpy.argmax(p_y_x, axis=1)

    def _p_y_given_x(self, vectors):
        p = numpy.dot(vectors, self.w) + self.b
        norm_transform = p / p.sum(axis=1)[:, numpy.newaxis]
        return numpy.log(norm_transform)
