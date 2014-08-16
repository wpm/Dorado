
from copy import deepcopy
import logging
from math import sqrt
import numpy as np
import theano.tensor as T
import theano
from theano import function, shared


class LabeledData(object):
    """
    A set of training example vectors with associated labels
    """
    def __init__(self, y, x):
        assert y.shape[0] == x.shape[0], \
            "Unmatched number of labels (%d) and training points (%d)" % (y.shape[0], x.shape[0])
        self.y = y
        self.x = x

    def __len__(self):
        """
        Numbered of labeled examples
        """
        return len(self.y)

    def dim(self):
        """
        The dimensionality of the example vectors
        """
        return self.x.shape[1]

    def classes(self):
        """The number of unique classes"""
        return np.unique(self.y).size

    def partition(self, batch_size):
        batches = []
        n = len(self)/batch_size
        for batch in xrange(n):
            i = batch * batch_size
            y_batch = self.y[i : i + batch_size]
            x_batch = self.x[i : i + batch_size]
            batches.append(LabeledData(y_batch, x_batch))
        return batches

    def shuffle(self):
        # TODO Randomly shuffle the data.
        return LabeledData(self.y, self.x)


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


def sgd_train(model, training_set, validation_set, batch_size, patience_rate,
                epochs, rate, validation_frequency):
    logging.info(model)
    logging.info("Train on %d labeled examples" % len(training_set))
    # The default patience rate and validation frequency are both a single epoch.
    if patience_rate == None:
        patience_rate = len(training_set)
    if validation_frequency == None:
        validation_frequency = len(training_set)/batch_size
    training_set = training_set.shuffle()
    batches = training_set.partition(batch_size)
    patience = patience_rate
    training_points = 0
    iterations = 0
    best_error_rate = np.inf
    early_stop = False
    epoch = 0
    best_model = model
    while epoch < epochs and not early_stop:
        epoch += 1
        logging.info("Epoch %d" % epoch)
        for b, batch in enumerate(batches):
            logging.debug("Batch %d" % (b + 1))
            iterations += 1
            p = model.sgd_training_iteration(batch.y, batch.x, rate)
            logging.debug("Training NLL %04f" % p)
            training_points += len(batch)
            if iterations % validation_frequency == 0:
                error_rate = model.error_rate(validation_set.y, validation_set.x)
                logging.info("%d. Validation error rate %04f" % (iterations, error_rate))
                if error_rate < best_error_rate:
                    best_error_rate = error_rate
                    best_model = deepcopy(model)
                    patience = training_points + patience_rate
                elif training_points > patience:
                    early_stop = True
                    break
    logging.info("Best validation error rate %04f" % best_error_rate)
    return best_model, best_error_rate


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
