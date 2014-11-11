import unittest

import numpy
import numpy.testing

from dorado.model.static_linear import StaticLinearModel


class StaticModelTestCase(unittest.TestCase):
    def setUp(self):
        W = numpy.array(
            [
                [0.1, 0.4],
                [0.2, 0.5],
                [0.3, 0.6]
            ]
        )
        b = numpy.array([0.7, 0.8])
        self.model = StaticLinearModel(W, b)
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
        self.assertEqual(self.model.zero(), StaticLinearModel(numpy.zeros((3, 2)), numpy.zeros(2)))

    def test_log_prob(self):
        numpy.testing.assert_allclose(
            self.model._p_y_given_x(self.vectors),
            [
                [-1.06635143, -0.42199441],
                [-1.15671992, -0.37763031]
            ]
        )
        numpy.testing.assert_allclose(self.model._p(self.vectors, self.labels), [-1.06635143, -0.37763031])

    def test_average(self):
        m1 = StaticLinearModel(numpy.arange(4).reshape(2, 2), numpy.arange(2))
        m2 = StaticLinearModel(numpy.arange(4).reshape(2, 2) + 2, numpy.arange(2) + 2)
        mean = (m1 + m2)/2
        self.assertEqual(mean, StaticLinearModel(numpy.arange(4).reshape(2, 2) + 1, numpy.arange(2) + 1))


if __name__ == '__main__':
    unittest.main()
