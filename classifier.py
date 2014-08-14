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


class LogisticRegressionClassifier(object):
    """Logistic regression classifier"""
    def __init__(self, n_in, n_out, y = T.lvector('y'), x = T.matrix('x')):
        e = T.scalar('e')
        self.W = shared(np.zeros((n_in, n_out), dtype=theano.config.floatX), name = 'W')
        self.b = shared(np.zeros((n_out,), dtype=theano.config.floatX), name = 'b')
        p_y_given_x = T.nnet.softmax(T.dot(x, self.W) + self.b)
        negative_log_likelihood = -T.mean(T.log(p_y_given_x)[T.arange(y.shape[0]), y])
        y_pred = T.argmax(p_y_given_x, axis=1)
        g_W = T.grad(negative_log_likelihood, self.W)
        g_b = T.grad(negative_log_likelihood, self.b)
        self.training_iteration = function([y, x, e],
            negative_log_likelihood,
            updates = [
                (self.W, self.W - e * g_W),
                (self.b, self.b - e * g_b)
                ])
        self.error_rate = function([y, x], T.mean(T.neq(y_pred, y)))
        self.predict = function([x], y_pred)

    def __repr__(self):
        n_in, n_out = self.W.get_value().shape
        return "%s(n_in = %d, n_out = %d)" % (self.__class__.__name__, n_in, n_out)
