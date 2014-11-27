import copy
import unittest

import numpy
import theano

from dorado.data import LabeledData
from dorado.model.logistic_regression import LogisticRegression
from dorado.model.parameters import ModelParameters


class LogisticRegressionTestCase(unittest.TestCase):
    def setUp(self):
        numpy.random.seed(0)
        self.train_data = LabeledData(
            numpy.array(
                [
                    [0.10, 0.20, 0.30, 0.40],
                    [0.50, 0.60, 0.70, 0.80],
                    [0.90, 0.10, 0.11, 0.12],
                    [0.13, 0.14, 0.15, 0.16]
                ]
            ),
            numpy.array([1, 0, 1, 0])
        )
        self.validate_data = LabeledData(
            numpy.array(
                [
                    [0.17, 0.18, 0.19, 0.20],
                    [0.21, 0.22, 0.23, 0.24]
                ]
            ),
            numpy.array([1, 0])
        )
        w = numpy.arange(8, dtype=theano.config.floatX).reshape(4, 2)
        b = numpy.arange(2, dtype=theano.config.floatX)
        self.parameters = ModelParameters(w, b)
        self.model = LogisticRegression(0.0, 0.0, w, b)

    def test_training_iteration(self):
        original = copy.deepcopy(self.parameters)
        self.model.train(self.train_data, 0.13)
        self.assertNotEqual(original, self.model.get_parameters())

    def test_model_parameters(self):
        actual = self.model.get_parameters()
        self.assertEqual(actual, self.parameters)

    def test_set_model_parameters(self):
        w = self.parameters[0] * 10
        b = self.parameters[1] * 10
        self.model.set_parameters(ModelParameters(w, b))
        actual_w, actual_b = self.model.get_parameters()
        numpy.testing.assert_almost_equal(w, actual_w)
        numpy.testing.assert_almost_equal(b, actual_b)

    def test_factory(self):
        factory = LogisticRegression.factory(0.01, 0.02)
        model = factory(self.parameters)
        self.assertEqual(model, LogisticRegression(0.01, 0.02, self.parameters[0], self.parameters[1]))


if __name__ == '__main__':
    unittest.main()
