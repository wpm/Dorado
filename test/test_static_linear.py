import unittest

import numpy
import numpy.testing

from dorado.model.parameters import ModelParameters
from dorado.model.static_linear import StaticLinearModel


class StaticModelTestCase(unittest.TestCase):
    def setUp(self):
        w = numpy.array(
            [
                [0.1, 0.4],
                [0.2, 0.5],
                [0.3, 0.6]
            ]
        )
        b = numpy.array([0.7, 0.8])
        self.model = StaticLinearModel(w, b)
        self.vectors = numpy.array(
            [
                [1, 2, 3],
                [4, 5, 6]
            ]
        )
        self.labels = numpy.array([0, 1])

    def test_sizes(self):
        self.assertEqual(self.model.dimension(), 3)
        self.assertEqual(self.model.classes(), 2)

    def test_zero(self):
        zero = StaticLinearModel(*self.model.get_parameters().zero())
        self.assertEqual(zero, StaticLinearModel(numpy.zeros((3, 2)), numpy.zeros(2)))

    def test_random(self):
        parameters = StaticLinearModel.initial_parameters(5, 10)
        factory = StaticLinearModel.factory()
        model = factory(parameters)
        self.assertEqual(model.dimension(), 5)
        self.assertEqual(model.classes(), 10)

    def test_parameter_values(self):
        actual = self.model.get_parameters()
        expected = ModelParameters(
            numpy.array(
                [
                    [0.1, 0.4],
                    [0.2, 0.5],
                    [0.3, 0.6]
                ]
            ),
            numpy.array([0.7, 0.8])
        )
        self.assertEqual(actual, expected)


if __name__ == '__main__':
    unittest.main()
