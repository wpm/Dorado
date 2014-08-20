import argparse
from copy import deepcopy
import cPickle
import gzip
import logging
from math import sqrt
import numpy as np
import operator
import theano.tensor as T
import theano
from theano import function, shared


def model_training_arguments(description):
    parser = argparse.ArgumentParser(description)
    parser.add_argument('train', type = load_compressed, help = 'training data')
    parser.add_argument('validation', type = load_compressed,
        help = 'validation data')
    parser.add_argument('type', choices = ['lg', 'nn'], 
        help = 'Classifier type: logistic regression or neural network')
    parser.add_argument('model', help = 'Trained model file')
    parser.add_argument('--classes', type = int, 
        help = 'number of classes, default is the number of unique labels in training')
    parser.add_argument('--l1', type = float, default = 0.0, help = 'L1 regularization constant')
    parser.add_argument('--l2', type = float, default = 0.0, help = 'L2 regularization constant')
    parser.add_argument('--hidden', type = int, default = 1000,
        help = 'Number of neural network hidden nodes')
    parser.add_argument('--batch', type = int, default = 100, help = 'batch size')
    parser.add_argument('--epochs', type = int, default = 1000, help = 'maximum training epochs')
    parser.add_argument('--patience', type = int,
        help = 'number of training examples to see before an early stop, default is the entire set')
    parser.add_argument('--frequency', type = int,
        help = 'how often to check the validation set, default is once per epoch')
    parser.add_argument('--rate', type = float, default = 0.13, help = 'learning rate')
    parser.add_argument('--log', default = 'CRITICAL', help = 'logging level')

    args = parser.parse_args()
    if args.classes == None:
        args.classes = args.train.classes()
    args.classifier = {
        'lg':LogisticRegression(args.train.dim(), args.classes, args.l1, args.l2),
        'nn':NeuralNetwork(args.train.dim(), args.classes, args.hidden, args.l1, args.l2)
    }[args.type]

    return args

def load_compressed(filename):
    with gzip.open(filename) as file:
        return cPickle.load(file)

def dump_compressed(obj, filename):
    with gzip.open(filename, 'w') as file:
        return cPickle.dump(obj, file)


class LabeledData(object):
    """
    A set of training example vectors with associated labels
    """
    def __init__(self, y, x):
        assert y.shape[0] == x.shape[0], \
            "Unmatched number of labels (%d) and training points (%d)" % (y.shape[0], x.shape[0])
        self.y = y
        self.x = x

    def __repr__(self):
        return "<%s %d examples, dimension %d, %d classes>" % \
            (self.__class__.__name__, len(self), self.dim(), self.classes())

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


def parallel_train(sc, model, training_data, batch_size, rate):
    def map_train(batch):
        model.sgd_training_iteration(batch.y, batch.x, rate)
        return model

    zero = sc.broadcast(deepcopy(model).zero())
    batches = sc.broadcast(training_data.partition(batch_size))
    k = len(batches.value)
    while True:
        ensemble = sc.parallelize(batches.value)
        model = ensemble.map(map_train).fold(zero.value, operator.add) / k
        yield model


def train(epochs, validation_data, min_epochs = 1, freq = 1, patience = 1):
    best_error = np.inf
    best_model = None
    wait = patience
    for epoch, model in enumerate(epochs, 1):
        if epoch > min_epochs and epoch % freq == 0:
            e = model.error_rate(validation_data.y, validation_data.x)
            logging.info("%d. %04f" % (epoch, e))
            if e < best_error:
                best_error = e
                best_model = deepcopy(model)
                wait = patience
            elif wait == 0:
                break
        wait -= 1
    return best_error, best_model


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
            logging.debug("Training cost %04f" % p)
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
