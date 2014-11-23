import itertools
import numpy


class ModelParameters(object):
    def __init__(self, *parameters):
        self.parameters = list(parameters)

    def __repr__(self):
        return "\n".join(str(p) for p in self.parameters)

    def __ne__(self, other):
        return not self.__eq__(other)

    def __eq__(self, other):
        for p_a, p_b in self._parameter_pairs(other):
            if not (p_a == p_b).all():
                return False
        return True

    def __hash__(self):
        return hash(self.parameters)

    def __add__(self, other):
        return ModelParameters(*[p_a + p_b for p_a, p_b in self._parameter_pairs(other)])

    def __div__(self, n):
        return ModelParameters(*[p / n for p in self.parameters])

    def zero(self):
        return ModelParameters(*[numpy.zeros(p.shape) for p in self.parameters])

    def _parameter_pairs(self, other):
        return itertools.izip(self.parameters, other.parameters)