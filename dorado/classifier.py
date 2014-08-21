from copy import deepcopy
from math import sqrt
import theano.tensor as T
import theano
from theano import function, shared
import numpy as np


class Classifier(object):
    """An abstract machine learning classifier"""

    def __init__(self, dim, classes, l1 = 0.0, l2 = 0.0):
        self.y = T.lvector('y')
        self.x = T.matrix('x')
        self.l1 = T.dscalar('l1')
        self.l2 = T.dscalar('l2')
        e = T.scalar('e')
        self.dim = dim
        self.classes = classes
        y_pred = T.argmax(self.p_y_given_x(), axis=1)
        updates = [(p, p - e * T.grad(self.cost() - self.regularization(), p)) 
                    for p in self.parameters()]
        self.sgd_training_iteration = function(
            [self.y, self.x, e],
            self.cost(), 
            updates = updates,
            givens = {self.l1: l1, self.l2:l2})
        self.L1 = sum([np.abs(p.flatten()).sum() for p in self.regularized_parameters()])
        self.L2 = sum([(p.flatten() ** 2).sum() for p in self.regularized_parameters()])
        self.error_rate = function([self.y, self.x], T.mean(T.neq(y_pred, self.y)))
        self.predict = function([self.x], y_pred)

    def parameters(self):
        raise NotImplementedError()

    def regularized_parameters(self):
        raise NotImplementedError()

    def p_y_given_x(self):
        raise NotImplementedError()

    def cost(self):
        raise NotImplementedError()

    def regularization(self):
        return sum([
            self.l1 * np.abs(p.flatten()).sum() + 
            self.l2 * (p.flatten() ** 2).sum()
            for p in self.regularized_parameters()])

    def zero(self):
        for p in self.parameters():
            z = np.zeros(p.get_value().shape)
            p.set_value(z)
        return self

    def __add__(self, other):
        sum = deepcopy(self)
        ps = zip(self.parameters(), other.parameters())
        new_values = [a.get_value() + b.get_value() for (a,b) in ps]
        for p,v in zip(sum.parameters(), new_values):
            p.set_value(v)
        return sum

    def __div__(self, k):
        for p in self.parameters():
            p.set_value(p.get_value()/k)
        return self


class LogisticRegression(Classifier):
    """Logistic regression"""
    def __init__(self, dim, classes, l1 = 0.0, l2 = 0.0):
        self.W = shared(np.zeros((dim, classes), dtype=theano.config.floatX), name = 'W')
        self.b = shared(np.zeros((classes,), dtype=theano.config.floatX), name = 'b')
        super(LogisticRegression, self).__init__(dim, classes, l1, l2)

    def parameters(self):
        return [self.W, self.b]

    def regularized_parameters(self):
        return [self.W]

    def p_y_given_x(self):
        return T.nnet.softmax(T.dot(self.x, self.W) + self.b)

    def cost(self):
        return -T.mean(T.log(self.p_y_given_x())[T.arange(self.y.shape[0]), self.y])

    def __repr__(self):
        dim, classes = self.W.get_value().shape
        return "%s(dim = %d, classes = %d)" % (self.__class__.__name__, dim, classes)


class NeuralNetwork(Classifier):
    def __init__(self, dim, classes, h, l1 = 0.0, l2 = 0.0):
        self.h = h
        b = 4 * sqrt(6.0/(dim + self.h))
        self.H = shared(random_matrix(dim, self.h, b), name = 'H')
        self.d = shared(np.zeros((self.h,), dtype = theano.config.floatX), name = 'd')
        b = 4 * sqrt(6.0/(self.h + classes))
        self.U = shared(random_matrix(self.h, classes, b), name = 'U')
        self.W = shared(np.zeros((dim, classes), dtype=theano.config.floatX), name = 'W')
        self.b = shared(np.zeros((classes,), dtype=theano.config.floatX), name = 'b')
        super(NeuralNetwork, self).__init__(dim, classes, l1, l2)

    def parameters(self):
        return [self.H, self.d, self.U, self.W, self.b]

    def regularized_parameters(self):
        return [self.H, self.U, self.W]

    def p_y_given_x(self):
        return T.nnet.softmax(T.dot(T.dot(self.x, self.H) + self.d, self.U) + T.dot(self.x, self.W) + self.b)

    def cost(self):
        return -T.mean(T.log(self.p_y_given_x())[T.arange(self.y.shape[0]), self.y])

    def __repr__(self):
        dim, classes = self.W.get_value().shape
        return "%s(dim = %d, classes = %d, h = %d)" % (self.__class__.__name__, dim, classes, self.h)


def random_matrix(r, c, b = 1):
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
    return np.random.uniform(-b, b, (r, c))
