import unittest
import numpy
from dorado.data import LabeledData
from dorado.model.logistic_regression import LogisticRegression
from dorado.train import train_model, averaged_epochs


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
        self.model = LogisticRegression(self.train_data.dimension(), self.validate_data.classes())

    def test_average(self):
        m1 = LogisticRegression.create(numpy.arange(4).reshape(2, 2), numpy.arange(2))
        m2 = LogisticRegression.create(numpy.arange(4).reshape(2, 2) + 2, numpy.arange(2) + 2)
        mean = (m1 + m2) / 2
        self.assertEqual(mean, LogisticRegression.create(numpy.arange(4).reshape(2, 2) + 1, numpy.arange(2) + 1))

    def test_something(self):
        model, error = train_model(averaged_epochs, self.model, self.train_data, self.validate_data, 4, 5)
        print(model)
        print(error)


if __name__ == '__main__':
    unittest.main()
