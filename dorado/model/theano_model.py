import itertools

import numpy
import theano
import theano.tensor as T

from dorado.model.model import Model
from dorado.model.model_parameters import ModelParameters


class TheanoModel(Model):
    def __init__(self, dimension, classes, l1=0.0, l2=0.0):
        self._l1 = l1
        self._l2 = l2
        self.y = T.lvector('y')
        self.x = T.matrix('x')
        self.l1 = T.dscalar('l1')
        self.l2 = T.dscalar('l2')
        e = T.scalar('e')
        self._dimension = dimension
        self._classes = classes
        y_pred = T.argmax(self.p_y_given_x(), axis=1)
        updates = [(p, p - e * T.grad(self.cost() - self.regularization(), p))
                   for p in self.parameters()]
        self.sgd_training_iteration = theano.function(
            [self.y, self.x, e],
            self.cost(),
            updates=updates,
            givens={self.l1: self._l1, self.l2: self._l2})
        self.L1 = sum([numpy.abs(p.flatten()).sum() for p in self.regularized_parameters()])
        self.L2 = sum([(p.flatten() ** 2).sum() for p in self.regularized_parameters()])
        self._error_rate = theano.function([self.y, self.x], T.mean(T.neq(y_pred, self.y)))
        self.predict = theano.function([self.x], y_pred)

    def __eq__(self, other):
        for p_a, p_b in itertools.izip(self.parameters(), other.parameters()):
            if not (p_a.get_value() == p_b.get_value()).all():
                return False
        return self._l1 == other._l1 and self._l2 == other._l2

    def dimension(self):
        return self._dimension

    def classes(self):
        return self._classes

    def parameters(self):
        raise NotImplementedError()

    def error_rate(self, data):
        return self._error_rate(data.labels, data.vectors)

    def regularized_parameters(self):
        raise NotImplementedError()

    def regularization(self):
        return sum([
            self.l1 * numpy.abs(p.flatten()).sum() +
            self.l2 * (p.flatten() ** 2).sum()
            for p in self.regularized_parameters()])

    def zero(self):
        z = self.copy()
        for p in z.parameters():
            value = numpy.zeros(p.get_value().shape)
            p.set_value(value)
        return z

    def train(self, data, rate=0.13):
        self.sgd_training_iteration(data.labels, data.vectors, rate)
        return self

    def parameter_values(self):
        return ModelParameters(*[p.get_value() for p in self.parameters()])

    def set_parameter_values(self, values):
        for p, v in zip(self.parameters(), values.parameters):
            p.set_value(v)

    def __add__(self, other):
        s = self.copy()
        ps = itertools.izip(s.parameters(), other.parameters())
        new_values = [a.get_value() + b.get_value() for (a, b) in ps]
        for p, v in itertools.izip(s.parameters(), new_values):
            p.set_value(v)
        return s

    def __div__(self, n):
        q = self.copy()
        for p in q.parameters():
            p.set_value(p.get_value() / n)
        return q

    def cost(self):
        return -T.mean(T.log(self.p_y_given_x())[T.arange(self.y.shape[0]), self.y])

    def p_y_given_x(self):
        raise NotImplementedError()

    def copy(self):
        raise NotImplementedError()