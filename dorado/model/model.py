class Model(object):
    def __repr__(self):
        return "<%s, %d dimensions, %d classes>" % (self.__class__.__name__, self.dimension(), self.classes())

    def __ne__(self, other):
        return not self.__eq__(other)

    def zero(self):
        raise NotImplementedError()

    def __eq__(self, other):
        raise NotImplementedError()

    def __add__(self, other):
        raise NotImplementedError()

    def __div__(self, n):
        raise NotImplementedError()

    def train(self, data, **parameters):
        raise NotImplementedError()

    def dimension(self):
        raise NotImplementedError()

    def classes(self):
        raise NotImplementedError()

    def error_rate(self, data):
        raise NotImplementedError()

    def predict(self, data):
        raise NotImplementedError()