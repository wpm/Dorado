import numpy


class ModelParameters(tuple):
    def __new__(cls, *parameter_values):
        return super(ModelParameters, cls).__new__(cls, tuple(parameter_values))

    def __eq__(self, other):
        for p_a, p_b in zip(self, other):
            if not (p_a == p_b).all():
                return False
        return True

    def __add__(self, other):
        return ModelParameters(*[p_a + p_b for p_a, p_b in zip(self, other)])

    def __div__(self, n):
        return ModelParameters(*[p / n for p in self])

    def zero(self):
        return ModelParameters(*[numpy.zeros(p.shape) for p in self])
