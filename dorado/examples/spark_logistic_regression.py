"""Train a logistic regression model on Spark

A script that can be passed to spark-submit. Running

    spark-submit spark_logistic_regression.py training.gz validation.gz model.gz

is equivalent to running

    dorado train logreg --spark spark_logistic_regression.py training.gz validation.gz model.gz
"""
import logging
import numpy

import dorado
from dorado import initialize_logging, load_compressed, write_compressed
from dorado.epochs import SparkParallelAveragedEpochs
from dorado.model.logistic_regression import LogisticRegression


def spark_logistic_regression_train(training_data, validation_data, l1=0.0, l2=0.0,
                                    batches=100, learning_rate=0.13, patience=5):
    """Train a logistic regression model using Spark

    :param training_data: training data
    :type training_data: dorado.data.LabeledData

    :param validation_data: validation data
    :type validation_data: dorado.data.LabeledData

    :param l1: l1 regularization
    :type l1: float

    :param l2: l2 regularization
    :type l2: float

    :param batches: number of batches into which to divide the training data
    :type batches: int

    :param learning_rate: optimization learning rate
    :type learning_rate: float

    :param patience: number of epochs without improvement before search stops
    :type patience: int

    :return: trained parameters and their validation error rate
    """
    import pyspark

    dimension, classes = validation_data.dimension(), validation_data.classes()
    model_factory = LogisticRegression.factory(l1, l2)
    initial_parameters = LogisticRegression.initial_parameters(dimension, classes)
    epochs = SparkParallelAveragedEpochs(pyspark.SparkContext(), batches, learning_rate, patience)
    return dorado.train(model_factory, initial_parameters, training_data, validation_data, epochs)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('training', help='training data')
    parser.add_argument('validation', help='validation data')
    parser.add_argument('model', help='compressed model file')
    parser.add_argument('--l1', type=float, default=0.0, help='l1 regularization, default 0.0')
    parser.add_argument('--l2', type=float, default=0.0, help='l2 regularization, default 0.0')
    parser.add_argument('--batches', type=int, default=100, help='number of batches, default 100')
    parser.add_argument('--learning-rate', type=float, default=0.13, help='learning rate, default 0.13')
    parser.add_argument('--patience', type=int, default=5, help='epochs of patience, default 5')
    parser.add_argument('--seed', type=int, help='random number seed')
    parser.add_argument('--log', default='INFO', help='logging level')
    args = parser.parse_args()

    initialize_logging(args.log)

    logging.info("Begin")
    if not args.seed is None:
        numpy.random.seed(args.seed)

    training_data = load_compressed(args.training)
    validation_data = load_compressed(args.validation)
    model, validation_error = \
        spark_logistic_regression_train(training_data, validation_data, args.l1, args.l2,
                                        args.batches, args.learning_rate, args.patience)
    logging.info("Validation error %0.4f" % validation_error)
    write_compressed(model, args.model)
    logging.info("Done")
