import numpy as np
import theano


class Index(object):
    """
    Mapping of objects to integer indexes

    0 is reserved as an out-of-vocabularly index.
    """

    def __init__(self, items=[]):
        self._indexes = {}
        for item in items:
            if not item in self._indexes:
                self._indexes[item] = len(self)

    def __repr__(self):
        return "<Index %d items>" % len(self)

    def __len__(self):
        return len(self._indexes) + 1

    def __getitem__(self, items):
        return np.asarray([self._indexes.get(item, 0) for item in items])


class Embedding(object):
    """Distributed embedding of a set of objects"""

    def __init__(self, index, m, b=1):
        self.index = index
        self.m = m
        self.C = theano.shared(
            np.random.uniform(-b, b, (len(self.index), self.m)),
            name='C')

    def __repr__(self):
        return repr(self.C.get_value())

    def __getitem__(self, items):
        return self.C.get_value()[self.index[items]].flatten()
