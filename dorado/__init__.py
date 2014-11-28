"""
Distributed machine learning with Theano and Spark
"""
import cPickle
import gzip
import logging

import numpy


def train(model_factory, initial_parameters, training_data, validation_data, epochs):
    model = model_factory(initial_parameters)
    logging.info("Model %s" % model)
    logging.info("Training data %s" % training_data)
    logging.info("Validation data %s" % validation_data)
    logging.info("Epochs %s" % epochs)
    best_error = numpy.Inf
    best_parameters = initial_parameters
    wait = epochs.patience
    logging.info("Initial validation error %0.4f" % model.error_rate(validation_data))
    for i, parameters in enumerate(epochs(model_factory, initial_parameters, training_data), 1):
        model.set_parameters(parameters)
        validation_error = model.error_rate(validation_data)
        logging.info("Epoch %d (%d), validation error %0.4f" % (i, wait, validation_error))
        if validation_error >= best_error:
            wait -= 1
            if not wait:
                break
        else:
            best_parameters = parameters
            best_error = validation_error
            wait = epochs.patience
    model.set_parameters(best_parameters)
    return model, best_error


def random_matrix(r, c, b=1):
    """
    Matrix with random elements selected uniformly from [-b, b].

    :type r: int
    :param r: rows

    :type c: int
    :param c: columns

    :type b: float
    :param b: bound

    :returns: randomly generated matrix
    """
    return numpy.random.uniform(-b, b, (r, c))


def initialize_logging(level):
    logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s', datefmt='%y/%m/%d %H:%M:%S',
                        level=getattr(logging, level.upper()))


def load_compressed(filename):
    logging.info("Read %s" % filename)
    with gzip.open(filename) as f:
        return cPickle.load(f)


def write_compressed(o, filename):
    logging.info("Write %s" % filename)
    with gzip.open(filename, 'w') as f:
        cPickle.dump(o, f)