import numpy
import theano
import theano.tensor as T

from dorado.model.model import Model
from dorado.model.parameters import ModelParameters


class TheanoModel(Model):
    @classmethod
    def factory(cls, l1=0.0, l2=0.0):
        return lambda parameters: cls(l1, l2, *parameters)

    def __init__(self, l1=0.0, l2=0.0, *params):
        super(TheanoModel, self).__init__(*params)
        self.y = T.lvector('y')
        self.x = T.matrix('x')
        self.l1 = T.dscalar('l1')
        self.l2 = T.dscalar('l2')
        e = T.scalar('e')
        y_pred = T.argmax(self.p_y_given_x(), axis=1)
        updates = [(p, p - e * T.grad(self.cost() - self.regularization(), p))
                   for p in self.parameters()]
        self.sgd_training_iteration = theano.function(
            [self.y, self.x, e],
            self.cost(),
            updates=updates,
            givens={self.l1: l1, self.l2: l2})
        self.L1 = sum([numpy.abs(p.flatten()).sum() for p in self.regularized_parameters()])
        self.L2 = sum([(p.flatten() ** 2).sum() for p in self.regularized_parameters()])
        self._error_rate = theano.function([self.y, self.x], T.mean(T.neq(y_pred, self.y)))
        self.predict = theano.function([self.x], y_pred)

    def get_parameters(self):
        return ModelParameters(*[parameter.get_value() for parameter in self.parameters()])

    def set_parameters(self, parameters):
        for parameter, value in zip(self.parameters(), parameters):
            parameter.set_value(value)

    def train(self, data, learning_rate):
        self.sgd_training_iteration(data.labels, data.vectors, learning_rate)

    def error_rate(self, data):
        return self._error_rate(data.labels, data.vectors)

    def cost(self):
        return -T.mean(T.log(self.p_y_given_x())[T.arange(self.y.shape[0]), self.y])

    def regularization(self):
        return sum([
            self.l1 * numpy.abs(p.flatten()).sum() +
            self.l2 * (p.flatten() ** 2).sum()
            for p in self.regularized_parameters()])

    def p_y_given_x(self):
        raise NotImplementedError()

    def parameters(self):
        raise NotImplementedError()

    def regularized_parameters(self):
        raise NotImplementedError()

