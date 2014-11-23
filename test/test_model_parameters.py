from unittest import TestCase

import numpy

from dorado.model.model_parameters import ModelParameters


class TestModelParameters(TestCase):
    def setUp(self):
        self.theta1 = ModelParameters(numpy.arange(4).reshape(2, 2), numpy.arange(2))
        self.theta2 = ModelParameters(numpy.arange(4, 8).reshape(2, 2), numpy.arange(2, 4))

    def test_add(self):
        actual = self.theta1 + self.theta2
        expected = ModelParameters(
            numpy.array(
                [
                    [0 + 4, 1 + 5],
                    [2 + 6, 3 + 7]
                ]
            ),
            numpy.array([0 + 2, 1 + 3])
        )
        self.assertEqual(actual, expected)

    def test_zero(self):
        actual = self.theta1.zero()
        expected = ModelParameters(
            numpy.array(
                [
                    [0, 0],
                    [0, 0]
                ]),
            numpy.array([0, 0]
            )
        )
        self.assertEqual(actual, expected)

    def test_div(self):
        actual = self.theta1 / 2.0
        expected = ModelParameters(
            numpy.array(
                [
                    [0 / 2.0, 1 / 2.0],
                    [2 / 2.0, 3 / 2.0]
                ]
            ),
            numpy.array([0 / 2.0, 1 / 2.0])
        )
        self.assertEqual(actual, expected)