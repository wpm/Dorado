import cPickle
import gzip

import numpy as np


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
        n = len(self) / batch_size
        for batch in xrange(n):
            i = batch * batch_size
            y_batch = self.y[i: i + batch_size]
            x_batch = self.x[i: i + batch_size]
            batches.append(LabeledData(y_batch, x_batch))
        return batches

    def shuffle(self):
        """Randomly shuffle this data set"""
        i = np.arange(len(self))
        np.random.shuffle(i)
        return LabeledData(self.y[i], self.x[i])
