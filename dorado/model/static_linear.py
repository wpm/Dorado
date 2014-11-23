import copy

import numpy
from dorado.model.model import Model
from dorado.model.model_parameters import ModelParameters


class StaticLinearModel(Model):
    """A linear model that does not change in response to training
    """

    @classmethod
    def random(cls, dimension, classes):
        return cls(numpy.random.rand(dimension, classes), numpy.random.rand(classes))

    def __init__(self, W, b):
        self.W = W
        self.b = b

    def __eq__(self, other):
        return (self.W == other.W).all() and (self.b == other.b).all()

    def __hash__(self):
        return hash((self.W, self.b))

    def zero(self):
        return StaticLinearModel(numpy.zeros((self.dimension(), self.classes())), numpy.zeros(self.classes()))

    def __add__(self, other):
        return StaticLinearModel(self.W + other.W, self.b + other.b)

    def __div__(self, n):
        return StaticLinearModel(self.W / n, self.b / n)

    def dimension(self):
        return self.W.shape[0]

    def classes(self):
        return self.W.shape[1]

    def train(self, data, **parameters):
        return copy.deepcopy(self)

    def parameter_values(self):
        return ModelParameters(self.W.copy(), self.b.copy())

    def set_parameter_values(self, values):
        W, b = values.parameters
        self.W = W.copy()
        self.b = b.copy()

    def error_rate(self, data):
        p_y_x = self._p_y_given_x(data.vectors)
        predictions = numpy.argmax(p_y_x, axis=1)
        correct = numpy.count_nonzero(numpy.equal(predictions, data.labels))
        n = float(len(data.vectors))
        return (n - correct) / n

    def _p_y_given_x(self, vectors):
        p = numpy.dot(vectors, self.W) + self.b
        norm_transform = p / p.sum(axis=1)[:, numpy.newaxis]
        return numpy.log(norm_transform)

    def _p(self, vectors, labels):
        return self._p_y_given_x(vectors)[numpy.arange(labels.shape[0]), labels]