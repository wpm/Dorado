import copy

import numpy

from dorado.model.model import Model


class RandomModel(Model):
    def __init__(self, dimension, classes, W=None):
        if not W == None:
            self.W = W
        else:
            self.W = numpy.random.rand(dimension, classes)

    def zero(self):
        return RandomModel(None, None, numpy.zeros((self.dimension(), self.classes())))

    def __eq__(self, other):
        return (self.W == other.W).all()

    def __add__(self, other):
        return RandomModel(None, None, W=self.W + other.W)

    def __div__(self, n):
        return RandomModel(None, None, W=self.W / n)

    def train(self, data, **parameters):
        return copy.deepcopy(self)

    def dimension(self):
        return self.W.shape[0]

    def classes(self):
        return self.W.shape[1]

    def error_rate(self, data):
        p_y_x = self._p_y_given_x(data.vectors)
        predictions = numpy.argmax(p_y_x, axis=1)
        correct = numpy.count_nonzero(numpy.equal(predictions, data.labels))
        n = float(len(data.vectors))
        return (n - correct) / n

    def _p_y_given_x(self, vectors):
        p = numpy.dot(vectors, self.W)
        norm = p / p.sum(axis=1)[:, numpy.newaxis]
        return numpy.log(norm)

    def _p(self, vectors, labels):
        return self._p_y_given_x(vectors)[numpy.arange(labels.shape[0]), labels]